#!/usr/bin/env python3
"""Build a paired visual and performance report for LingBot VAE A/B tests."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_fixture(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["sample_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def target_video(row: dict[str, Any]) -> dict[str, Any]:
    targets = [
        message
        for message in row["messages"]
        if message.get("role") == "target" and message.get("type") == "video"
    ]
    if len(targets) != 1:
        raise ValueError(f"{row['sample_id']}: expected exactly one target video")
    return targets[0]


def prompt_of(row: dict[str, Any]) -> str:
    prompts = [
        message.get("content", "")
        for message in row["messages"]
        if message.get("role") == "user" and message.get("type") == "text"
    ]
    if len(prompts) != 1:
        raise ValueError(f"{row['sample_id']}: expected exactly one text prompt")
    return str(prompts[0])


def action_summary(row: dict[str, Any]) -> str:
    controls = target_video(row).get("controls", [])
    keyboard = next(
        (
            item
            for item in controls
            if item.get("type") == "keyboard_direction_frame_interval"
        ),
        None,
    )
    if keyboard is None:
        return "(no keyboard action)"
    keys = keyboard.get("action_keys", [])
    frames = keyboard.get("actions", [])
    labels = [
        "+".join(key for key, enabled in zip(keys, frame) if enabled) or "noop"
        for frame in frames
    ]
    runs: list[tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return " | ".join(f"{label} x{count}" for label, count in runs)


def output_relative_path(result: dict[str, Any]) -> str | None:
    output = result.get("output")
    if not output:
        return None
    marker = "cases/videos/"
    if marker not in output:
        raise ValueError(f"unrecognized result output path: {output}")
    return output.split(marker, 1)[1]


def read_variant(root: Path) -> dict[str, Any]:
    summary = read_json(root / "cases" / "summary.json")
    runtime_path = root / "taehv-runtime.json"
    startup_path = root / "server-startup-seconds"
    return {
        "root": root,
        "summary": summary["summary"],
        "results": {row["sample_id"]: row for row in summary["results"]},
        "runtime": (
            read_json(runtime_path) if runtime_path.exists() else {"enabled": False}
        ),
        "startup_sec": (
            float(startup_path.read_text().strip()) if startup_path.exists() else None
        ),
    }


def numeric_improvement(
    baseline: float | None, candidate: float | None, *, lower_is_better: bool
) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    change = (candidate - baseline) / baseline * 100.0
    return -change if lower_is_better else change


def performance_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    definitions = {
        "server_startup_sec": (baseline["startup_sec"], candidate["startup_sec"], True),
        "warmup_wall_sec": (
            baseline["summary"].get("warmup_wall_sec"),
            candidate["summary"].get("warmup_wall_sec"),
            True,
        ),
        "measured_wall_sec": (
            baseline["summary"].get("measured_wall_sec"),
            candidate["summary"].get("measured_wall_sec"),
            True,
        ),
        "node_videos_per_hour": (
            baseline["summary"].get("node_videos_per_hour_this_run"),
            candidate["summary"].get("node_videos_per_hour_this_run"),
            False,
        ),
        "videos_per_gpu_hour": (
            baseline["summary"].get("videos_per_gpu_hour_this_run"),
            candidate["summary"].get("videos_per_gpu_hour_this_run"),
            False,
        ),
        "request_p50_sec": (
            baseline["summary"].get("request_persisted_end_to_end_sec", {}).get("p50"),
            candidate["summary"].get("request_persisted_end_to_end_sec", {}).get("p50"),
            True,
        ),
        "request_p95_sec": (
            baseline["summary"].get("request_persisted_end_to_end_sec", {}).get("p95"),
            candidate["summary"].get("request_persisted_end_to_end_sec", {}).get("p95"),
            True,
        ),
    }
    return {
        name: {
            "baseline": baseline_value,
            "taehv": candidate_value,
            "improvement_percent": numeric_improvement(
                baseline_value, candidate_value, lower_is_better=lower_is_better
            ),
            "lower_is_better": lower_is_better,
        }
        for name, (
            baseline_value,
            candidate_value,
            lower_is_better,
        ) in definitions.items()
    }


def presign(client: Any, uri: str, expires_in: int) -> str:
    bucket, key = parse_s3_uri(uri)
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.strip('/')}"


def render_html(
    rows: list[dict[str, Any]], performance: dict[str, Any], expires_in: int
) -> str:
    cards = []
    for row in rows:
        cards.append(
            """
<article>
  <div class="case-header"><strong>{sample_id}</strong><span>{status}</span></div>
  <img src="{image_url}" alt="{sample_id} original image" loading="lazy">
  <p class="prompt">{prompt}</p>
  <pre>{action}</pre>
  <div class="videos">
    <section><h2>原 VAE</h2>{baseline_video}</section>
    <section><h2>TAEHV</h2>{candidate_video}</section>
  </div>
</article>""".format(
                sample_id=html.escape(row["sample_id"]),
                status=html.escape(row["status"]),
                image_url=html.escape(row["image_url"]),
                prompt=html.escape(row["prompt"]),
                action=html.escape(row["action"]),
                baseline_video=(
                    f'<video controls preload="metadata" src="{html.escape(row["baseline_video_url"])}"></video>'
                    if row.get("baseline_video_url")
                    else '<p class="missing">缺少基线视频</p>'
                ),
                candidate_video=(
                    f'<video controls preload="metadata" src="{html.escape(row["candidate_video_url"])}"></video>'
                    if row.get("candidate_video_url")
                    else '<p class="missing">缺少 TAEHV 视频</p>'
                ),
            )
        )
    metric_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(name),
            "-" if value["baseline"] is None else f"{value['baseline']:.3f}",
            "-" if value["taehv"] is None else f"{value['taehv']:.3f}",
            (
                "-"
                if value["improvement_percent"] is None
                else f"{value['improvement_percent']:+.2f}%"
            ),
        )
        for name, value in performance["metrics"].items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LingBot TAEHV A/B</title><style>
:root{{color-scheme:dark;--bg:#090d13;--panel:#121a24;--line:#29394c;--text:#eef5ff;--muted:#a7b5c7;--accent:#69d9b8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}}
header{{padding:26px max(24px,calc((100vw - 1600px)/2));border-bottom:1px solid var(--line)}}h1{{margin:0;font-size:25px}}p{{color:var(--muted)}}
table{{border-collapse:collapse;margin-top:14px;width:min(920px,100%);background:var(--panel)}}td,th{{padding:8px 10px;border:1px solid var(--line);text-align:left}}main{{max-width:1700px;margin:auto;padding:22px}}
#cases{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}article{{border:1px solid var(--line);background:var(--panel);overflow:hidden;border-radius:8px}}
.case-header{{padding:10px 12px;display:flex;justify-content:space-between;gap:12px;overflow-wrap:anywhere}}article>img{{display:block;width:100%;max-height:430px;object-fit:contain;background:#05070a}}
.prompt{{padding:0 12px;margin:10px 0;color:var(--text)}}pre{{margin:0 12px 12px;padding:8px;white-space:pre-wrap;word-break:break-word;background:#0b1119;border:1px solid var(--line);color:var(--muted)}}
.videos{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}}.videos section{{background:var(--panel);padding:10px}}h2{{font-size:14px;margin:0 0 8px;color:var(--accent)}}video{{display:block;width:100%;aspect-ratio:832/480;background:#05070a}}.missing{{min-height:150px;padding:20px}}
@media(max-width:700px){{header,main{{padding-left:12px;padding-right:12px}}#cases,.videos{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>LingBot 原 VAE vs TAEHV A/B</h1><p>固定 testset100_v2 前 100 个 case；预签名链接有效期 {expires_in // 3600} 小时。TAEHV 只用于该测试 Job，线上服务未切换。</p>
<table><thead><tr><th>指标</th><th>原 VAE</th><th>TAEHV</th><th>TAEHV 改善</th></tr></thead><tbody>{metric_rows}</tbody></table></header>
<main><div id="cases">{''.join(cards)}</div></main></body></html>"""


def build_report(
    *,
    baseline_root: Path,
    candidate_root: Path,
    fixture_path: Path,
    bucket: str,
    output_prefix: str,
    output_dir: Path,
    profile: str | None,
    region: str,
    expires_in: int,
) -> dict[str, Any]:
    baseline = read_variant(baseline_root)
    candidate = read_variant(candidate_root)
    fixture = read_fixture(fixture_path)
    client = boto3.Session(profile_name=profile, region_name=region).client("s3")
    rows = []
    for sample_id, source in fixture.items():
        baseline_result = baseline["results"].get(sample_id)
        candidate_result = candidate["results"].get(sample_id)
        image_url = presign(client, target_video(source)["uri"], expires_in)
        baseline_relative = (
            output_relative_path(baseline_result)
            if baseline_result and baseline_result.get("success")
            else None
        )
        candidate_relative = (
            output_relative_path(candidate_result)
            if candidate_result and candidate_result.get("success")
            else None
        )
        baseline_uri = (
            s3_uri(
                bucket,
                f"{output_prefix.strip('/')}/baseline/cases/videos/{baseline_relative}",
            )
            if baseline_relative
            else None
        )
        candidate_uri = (
            s3_uri(
                bucket,
                f"{output_prefix.strip('/')}/taehv/cases/videos/{candidate_relative}",
            )
            if candidate_relative
            else None
        )
        rows.append(
            {
                "sample_id": sample_id,
                "prompt": prompt_of(source),
                "action": action_summary(source),
                "image_url": image_url,
                "baseline_video_uri": baseline_uri,
                "candidate_video_uri": candidate_uri,
                "baseline_video_url": (
                    presign(client, baseline_uri, expires_in) if baseline_uri else None
                ),
                "candidate_video_url": (
                    presign(client, candidate_uri, expires_in)
                    if candidate_uri
                    else None
                ),
                "status": (
                    "paired"
                    if baseline_uri and candidate_uri
                    else "unpaired / failed output"
                ),
            }
        )
    metrics = performance_metrics(baseline, candidate)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paired_successes": sum(1 for row in rows if row["status"] == "paired"),
        "total_cases": len(rows),
        "variants": {
            "baseline": {
                "enabled": bool(baseline["runtime"].get("enabled")),
                **baseline["summary"],
            },
            "taehv": {
                "enabled": bool(candidate["runtime"].get("enabled")),
                **candidate["summary"],
            },
        },
        "metrics": metrics,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        render_html(rows, report, expires_in), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--bucket", default="leap-world-us-east-2")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--expires-in", type=int, default=7 * 24 * 60 * 60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        fixture_path=args.fixture,
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        profile=args.profile,
        region=args.region,
        expires_in=args.expires_in,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("total_cases", "paired_successes", "metrics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
