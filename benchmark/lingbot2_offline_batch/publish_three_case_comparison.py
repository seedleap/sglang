#!/usr/bin/env python3
"""Publish a small, shareable LingBot2 input/output comparison set to S3."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import boto3


DEFAULT_SAMPLE_IDS = (
    "thirdperson-remaining3699x5/TPV/gvs2_00002102-action-00",
    "thirdperson-remaining3699x5/TPV/gvs2_00002162-action-04",
    "thirdperson-remaining3699x5/TPV/gvs2_00002255-action-03",
)
DEFAULT_BUCKET = "leap-world-us-east-2"
DEFAULT_PREFIX = (
    "world-model/eval/lingbot2/eval_results/minWM/"
    "comparison_sets/three_cases_20260716"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent
        / "results/actual_video_actions/batch3_3699x5_generated_wasd_ijkl/"
        "video_action.actual.jsonl",
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--aws-profile", default="spot")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--expires", type=int, default=604800)
    return parser.parse_args()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def load_rows(path: Path, sample_ids: tuple[str, ...]) -> list[dict]:
    wanted = set(sample_ids)
    found: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in wanted:
                found[row["sample_id"]] = row
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"samples are not actual outputs: {sorted(missing)}")
    return [found[sample_id] for sample_id in sample_ids]


def load_prompt_records(s3, rows: list[dict]) -> dict[str, dict]:
    image_to_labels: dict[str, tuple[str, str]] = {}
    for row in rows:
        image_uri = row["image"]["s3_uri"]
        bucket, key = parse_s3_uri(image_uri)
        if "/images/" not in key:
            raise RuntimeError(f"cannot derive labels path from {image_uri}")
        labels_key = key.split("/images/", 1)[0] + "/labels/prompts.jsonl"
        image_to_labels[row["image"]["image_id"]] = (bucket, labels_key)

    wanted = set(image_to_labels)
    records: dict[str, dict] = {}
    for bucket, key in sorted(set(image_to_labels.values())):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        for raw_line in body.iter_lines():
            record = json.loads(raw_line)
            if record.get("id") in wanted:
                records[record["id"]] = record
    missing = wanted - records.keys()
    if missing:
        raise RuntimeError(f"prompt records not found: {sorted(missing)}")
    return records


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def copy_asset(s3, source_uri: str, bucket: str, key: str) -> None:
    source_bucket, source_key = parse_s3_uri(source_uri)
    s3.copy_object(
        Bucket=bucket,
        Key=key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )


def presign(s3, bucket: str, key: str, content_type: str, expires: int) -> str:
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentType": content_type,
            "ResponseContentDisposition": "inline",
        },
        ExpiresIn=expires,
    )


def build_cases(s3, rows: list[dict], prompts: dict[str, dict], args) -> list[dict]:
    cases = []
    for number, row in enumerate(rows, 1):
        image_id = row["image"]["image_id"]
        image_ext = Path(parse_s3_uri(row["image"]["s3_uri"])[1]).suffix or ".png"
        image_key = f"{args.prefix}/assets/case-{number:02d}-{image_id}{image_ext}"
        video_key = f"{args.prefix}/assets/case-{number:02d}-{row['case_id']}.mp4"
        copy_asset(s3, row["image"]["s3_uri"], args.bucket, image_key)
        copy_asset(s3, row["video"]["s3_uri"], args.bucket, video_key)

        prompt_record = prompts[image_id]
        case = {
            "schema_version": 1,
            "case_number": number,
            "sample_id": row["sample_id"],
            "image": {
                "image_id": image_id,
                "original_s3_uri": row["image"]["s3_uri"],
                "packaged_s3_uri": s3_uri(args.bucket, image_key),
                "http_url": presign(
                    s3, args.bucket, image_key, f"image/{image_ext.lstrip('.')}", args.expires
                ),
            },
            "prompt": {
                "generation_prompt": prompt_record["generation_prompt"],
                "negative_prompt": prompt_record.get("negative_prompt"),
                "viewpoint": prompt_record.get("viewpoint"),
                "subject_detail": prompt_record.get("subject_detail"),
                "scene_signature": prompt_record.get("scene_signature"),
                "art_style_primary": prompt_record.get("art_style_primary"),
            },
            "action_trajectory": row["action_trajectory"],
            "model_controls": row["model_controls"],
            "result_video": {
                **row["video"],
                "original_s3_uri": row["video"]["s3_uri"],
                "packaged_s3_uri": s3_uri(args.bucket, video_key),
                "http_url": presign(
                    s3, args.bucket, video_key, "video/mp4", args.expires
                ),
            },
            "actual_output": row["actual_output"],
        }
        case["result_video"].pop("s3_uri", None)
        cases.append(case)
    return cases


def render_html(cases: list[dict], jsonl_uri: str, expires: int) -> str:
    cards = []
    for case in cases:
        action = case["action_trajectory"]
        prompt = case["prompt"]
        frame_actions = json.dumps(action["frame_actions"], ensure_ascii=False)
        segment_names = ("movement", "noop", "camera")
        segments = "".join(
            "<tr>"
            f"<td>{segment_names[index]}</td>"
            f"<td>{segment['start_frame']}–{segment['end_frame']}</td>"
            f"<td>{html.escape(str(segment['key'] or 'noop'))}</td>"
            "</tr>"
            for index, segment in enumerate(action["segments"])
        )
        cards.append(
            f"""
<article class="case">
  <h2>Case {case['case_number']}: {html.escape(case['image']['image_id'])}</h2>
  <div class="badges">
    <span>movement: <b>{html.escape(action['movement_key'])}</b></span>
    <span>camera: <b>{html.escape(action['camera_key'])}</b></span>
    <span>seed: <b>{action['action_seed']}</b></span>
    <span>720p · 24 fps · 129 frames</span>
  </div>
  <div class="media-grid">
    <figure><img src="{html.escape(case['image']['http_url'], quote=True)}" alt="input image"><figcaption>Input image</figcaption></figure>
    <figure><video src="{html.escape(case['result_video']['http_url'], quote=True)}" controls preload="metadata"></video><figcaption>LingBot2 result</figcaption></figure>
  </div>
  <h3>Prompt</h3>
  <pre>{html.escape(prompt['generation_prompt'])}</pre>
  <h3>Action trajectory</h3>
  <table><thead><tr><th>Segment</th><th>Frames</th><th>Key</th></tr></thead><tbody>{segments}</tbody></table>
  <details><summary>完整 129 帧 action JSON</summary><pre>{html.escape(frame_actions)}</pre></details>
  <details><summary>S3 paths / metadata</summary><pre>{html.escape(json.dumps({'sample_id': case['sample_id'], 'image': case['image']['packaged_s3_uri'], 'video': case['result_video']['packaged_s3_uri'], 'trajectory_id': action['trajectory_id']}, ensure_ascii=False, indent=2))}</pre></details>
</article>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LingBot2 三个对比 Case</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#0b0d12; color:#e8ebf2 }}
body {{ max-width:1400px; margin:auto; padding:28px }}
h1,h2,h3 {{ color:#fff }} .note {{ color:#aeb7c8 }}
.case {{ background:#141821; border:1px solid #293143; border-radius:14px; padding:22px; margin:24px 0 }}
.media-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px }}
figure {{ margin:0 }} img,video {{ width:100%; aspect-ratio:16/9; object-fit:contain; background:#050608; border-radius:8px }}
figcaption {{ margin-top:7px; color:#aeb7c8 }} .badges {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px }}
.badges span {{ background:#20283a; border-radius:999px; padding:6px 10px }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#090b10; padding:14px; border-radius:8px; line-height:1.45 }}
table {{ border-collapse:collapse; width:100%; margin-bottom:12px }} th,td {{ border:1px solid #384258; padding:8px; text-align:left }}
summary {{ cursor:pointer; padding:10px 0; color:#9cc5ff }} a {{ color:#9cc5ff }}
@media(max-width:800px) {{ .media-grid {{ grid-template-columns:1fr }} body {{ padding:14px }} }}
</style></head><body>
<h1>LingBot2 第三人称：三个可复现对比 Case</h1>
<p class="note">每个 case 包含同一输入图片、完整生成 prompt、确定性 action trajectory 和实际结果视频。媒体链接为 S3 预签名 URL，有效期约 {expires // 86400} 天。机器可读数据：<code>{html.escape(jsonl_uri)}</code></p>
{''.join(cards)}
</body></html>"""


def main() -> None:
    args = parse_args()
    sample_ids = tuple(args.sample_ids or DEFAULT_SAMPLE_IDS)
    if len(sample_ids) != 3:
        raise ValueError("exactly three --sample-id values are required")
    session = boto3.Session(profile_name=args.aws_profile, region_name=args.region)
    s3 = session.client("s3")
    rows = load_rows(args.manifest, sample_ids)
    prompts = load_prompt_records(s3, rows)
    cases = build_cases(s3, rows, prompts, args)

    jsonl_key = f"{args.prefix}/cases.jsonl"
    jsonl_body = "".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
        for case in cases
    ).encode()
    s3.put_object(
        Bucket=args.bucket,
        Key=jsonl_key,
        Body=jsonl_body,
        ContentType="application/x-ndjson; charset=utf-8",
    )
    jsonl_uri = s3_uri(args.bucket, jsonl_key)
    html_key = f"{args.prefix}/index.html"
    html_body = render_html(cases, jsonl_uri, args.expires).encode()
    s3.put_object(
        Bucket=args.bucket,
        Key=html_key,
        Body=html_body,
        ContentType="text/html; charset=utf-8",
        CacheControl="no-store",
    )
    html_url = presign(s3, args.bucket, html_key, "text/html", args.expires)
    print(
        json.dumps(
            {
                "s3_prefix": s3_uri(args.bucket, args.prefix) + "/",
                "html_s3_uri": s3_uri(args.bucket, html_key),
                "html_presigned_url": html_url,
                "jsonl_s3_uri": jsonl_uri,
                "jsonl_rows": len(cases),
                "sample_ids": sample_ids,
                "expires_seconds": args.expires,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
