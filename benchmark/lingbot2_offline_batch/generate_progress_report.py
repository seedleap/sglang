#!/usr/bin/env python3
"""Build a shareable HTML report for LingBot2 offline batch outputs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import boto3


STATUS_RE = re.compile(r"shard-(\d+)-(benchmark|upload)-summary\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="LingBot2 第三人称批量推理进度")
    parser.add_argument("--subtitle", default="第三人称 720p · 当前已上传样例快照")
    parser.add_argument("--sample-limit", type=int, default=60)
    parser.add_argument("--profile", default="wms")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--expires-in", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--total-cases", type=int, default=5000)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def even_sample(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    out = []
    last = -1
    for i in range(limit):
        idx = round(i * (len(items) - 1) / (limit - 1))
        if idx == last and idx + 1 < len(items):
            idx += 1
        out.append(items[idx])
        last = idx
    dedup: dict[str, dict[str, Any]] = {}
    for item in out:
        dedup[item["sample_id"]] = item
    if len(dedup) < limit:
        for item in items:
            dedup.setdefault(item["sample_id"], item)
            if len(dedup) >= limit:
                break
    selected = list(dedup.values())
    selected.sort(key=lambda row: row["case_index"])
    return selected[:limit]


def read_needed_messages(
    result_dir: Path, selected: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    needed_by_shard: dict[int, set[str]] = defaultdict(set)
    for row in selected:
        needed_by_shard[row["shard"]].add(row["sample_id"])

    out: dict[str, dict[str, Any]] = {}
    input_dir = result_dir / "input"
    for shard, sample_ids in needed_by_shard.items():
        path = input_dir / f"messages-shard-{shard:02d}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample_id = row["sample_id"]
                if sample_id in sample_ids:
                    out[sample_id] = row
    return out


def list_status_objects(s3: Any, bucket: str, prefix: str) -> dict[int, dict[str, str]]:
    status_prefix = f"{prefix}/status/"
    token = None
    shard_keys: dict[int, dict[str, str]] = defaultdict(dict)
    while True:
        kwargs = {"Bucket": bucket, "Prefix": status_prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            match = STATUS_RE.search(key)
            if not match:
                continue
            shard = int(match.group(1))
            kind = match.group(2)
            shard_keys[shard][kind] = key
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return shard_keys


def s3_get_json(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def fmt_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_html(payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{payload["title"]}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg:#091019;
      --bg2:#101a27;
      --panel:#132030;
      --panel2:#1a2b40;
      --line:#29425e;
      --text:#ecf4ff;
      --muted:#9fb3ca;
      --accent:#81e6c7;
      --warn:#f7c86b;
      --danger:#ff8a8a;
    }}
    * {{ box-sizing:border-box }}
    body {{
      margin:0;
      color:var(--text);
      background:
        radial-gradient(circle at top right, rgba(77,120,255,.18), transparent 28%),
        radial-gradient(circle at top left, rgba(75,205,170,.16), transparent 24%),
        linear-gradient(180deg, #0a1018 0%, #091019 100%);
      font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif;
    }}
    header {{
      position:sticky;
      top:0;
      z-index:3;
      backdrop-filter: blur(14px);
      background:rgba(9,16,25,.92);
      border-bottom:1px solid var(--line);
      padding:22px 24px 18px;
    }}
    h1 {{ margin:0 0 6px; font-size:24px; line-height:1.2 }}
    .sub {{ color:var(--muted) }}
    .stats,.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px }}
    .pill,.chip {{
      border:1px solid var(--line);
      border-radius:999px;
      background:#0f1823;
      padding:6px 10px;
      white-space:nowrap;
    }}
    .chip.warn {{ border-color:#7c6130; color:var(--warn) }}
    .chip.danger {{ border-color:#7a3737; color:var(--danger) }}
    main {{ max-width:1680px; margin:auto; padding:20px 24px 40px }}
    .summary {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
      gap:16px;
      margin-bottom:18px;
    }}
    .panel {{
      border:1px solid var(--line);
      border-radius:16px;
      background:linear-gradient(180deg, rgba(20,32,48,.95), rgba(17,28,41,.95));
      padding:16px 16px 14px;
      box-shadow:0 10px 30px rgba(0,0,0,.16);
    }}
    .panel h2 {{ margin:0 0 10px; font-size:16px }}
    .kv {{ display:grid; grid-template-columns:132px 1fr; gap:6px 10px; color:var(--muted) }}
    .kv strong {{ color:var(--text); font-weight:600 }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all }}
    .section-title {{ margin:22px 0 10px; font-size:18px }}
    table {{
      width:100%;
      border-collapse:collapse;
      overflow:hidden;
      border-radius:14px;
      border:1px solid var(--line);
      background:rgba(17,27,40,.92);
    }}
    th,td {{ padding:10px 12px; border-bottom:1px solid rgba(41,66,94,.7); text-align:left }}
    th {{ color:#bfd3e8; background:rgba(24,38,56,.96); font-weight:600 }}
    tr:last-child td {{ border-bottom:0 }}
    #grid {{
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
      gap:16px;
      margin-top:14px;
    }}
    article {{
      overflow:hidden;
      border:1px solid var(--line);
      border-radius:16px;
      background:linear-gradient(180deg, rgba(20,32,48,.98), rgba(16,25,36,.98));
      box-shadow:0 10px 32px rgba(0,0,0,.18);
    }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#02060b }}
    .body {{ padding:13px 14px 15px }}
    .title-row {{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:flex-start;
    }}
    .title-row strong {{ overflow-wrap:anywhere }}
    .badge {{
      flex:none;
      color:var(--accent);
      border:1px solid #2e6f5c;
      border-radius:999px;
      padding:3px 8px;
      font-size:12px;
    }}
    .meta {{
      display:flex;
      flex-wrap:wrap;
      gap:5px 12px;
      color:var(--muted);
      margin:8px 0;
    }}
    .timeline {{
      display:flex;
      height:30px;
      margin:10px 0 7px;
      overflow:hidden;
      border-radius:8px;
      font:12px ui-monospace,monospace;
      background:#0d1520;
    }}
    .segment {{
      display:flex;
      align-items:center;
      justify-content:center;
      min-width:44px;
      border-right:1px solid #162131;
      padding:0 6px;
    }}
    .move {{ background:#245349 }}
    .none {{ background:#313b4a }}
    .camera {{ background:#314a74 }}
    .segments {{ color:#c1cddd; font:12px ui-monospace,monospace }}
    details {{ margin-top:10px }}
    summary {{ cursor:pointer; color:#8dc8ff }}
    .prompt {{ margin-top:8px; font-size:12px; white-space:pre-wrap; color:#d6e1ef }}
    .links {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:10px }}
    a,button {{
      color:#8dc8ff;
      border:0;
      padding:0;
      background:none;
      font:inherit;
      cursor:pointer;
      text-decoration:none;
    }}
    .foot {{ color:var(--muted); margin-top:18px; font-size:12px }}
    @media (max-width: 720px) {{
      header,main {{ padding-left:12px; padding-right:12px }}
      #grid {{ grid-template-columns:1fr }}
      .kv {{ grid-template-columns:1fr }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{payload["title"]}</h1>
  <div class="sub">{payload["subtitle"]}</div>
  <div class="stats" id="stats"></div>
  <div class="chips" id="chips"></div>
</header>
<main>
  <section class="summary">
    <div class="panel">
      <h2>批次概览</h2>
      <div class="kv" id="overview"></div>
    </div>
    <div class="panel">
      <h2>性能摘要</h2>
      <div class="kv" id="perf"></div>
    </div>
    <div class="panel">
      <h2>说明</h2>
      <div class="kv" id="notes"></div>
    </div>
  </section>

  <h2 class="section-title">已完成 Shard</h2>
  <table>
    <thead>
      <tr>
        <th>Shard</th>
        <th>样本</th>
        <th>上传</th>
        <th>节点吞吐</th>
        <th>请求 p50</th>
        <th>上传耗时</th>
      </tr>
    </thead>
    <tbody id="shards"></tbody>
  </table>

  <h2 class="section-title">样例视频</h2>
  <div class="sub">从当前已上传成功的视频里均匀抽样 {payload["selected_count"]} 条，直接播放 S3 上的 MP4。</div>
  <div id="grid"></div>

  <div class="foot">构建时间 {payload["built_at"]}；视频签名到期 {payload["url_expires_at"]}。这是静态快照，不会自动刷新后续进度。</div>
</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False, separators=(",", ":"))};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt = n => Number(n).toFixed(2);
const fmtInt = n => new Intl.NumberFormat('en-US').format(Number(n || 0));
const fmtMiB = n => `${{(Number(n || 0) / 1048576).toFixed(1)}} MiB`;

document.querySelector('#stats').innerHTML = DATA.stats.map(x => `<span class="pill">${{esc(x)}}</span>`).join('');
document.querySelector('#chips').innerHTML = DATA.chips.map(x => {{
  const cls = x.kind ? `chip ${{x.kind}}` : 'chip';
  return `<span class="${{cls}}">${{esc(x.text)}}</span>`;
}}).join('');

function renderKv(el, rows) {{
  el.innerHTML = rows.map(([k, v]) => `<><div>${{esc(k)}}</div><strong class="${{String(v).includes('/') || String(v).includes('s3://') ? 'mono' : ''}}">${{esc(v)}}</strong></>`).join('');
}}

renderKv(document.querySelector('#overview'), DATA.overview);
renderKv(document.querySelector('#perf'), DATA.performance);
renderKv(document.querySelector('#notes'), DATA.notes);

document.querySelector('#shards').innerHTML = DATA.shards.map(s => `<tr>
  <td class="mono">${{esc(s.index)}}</td>
  <td>${{fmtInt(s.samples)}}</td>
  <td>${{fmtInt(s.uploaded)}} / ${{fmtInt(s.attempted)}}</td>
  <td>${{fmt(s.videos_per_hour)}} videos/h</td>
  <td>${{fmt(s.p50_sec)}} s</td>
  <td>${{fmt(s.upload_wall_sec)}} s</td>
</tr>`).join('');

function segmentHtml(s, i) {{
  const kind = s.key === null ? 'none' : i === 0 ? 'move' : 'camera';
  const label = s.key === null ? 'none' : s.key;
  return `<div class="segment ${{kind}}" style="flex:${{s.num_frames}}">${{esc(label)}} · ${{s.num_frames}}f</div>`;
}}

function card(x) {{
  const detail = x.segments.map(s => `${{s.key ?? 'none'}}: ${{s.start_frame}}–${{s.end_frame}} (${{s.num_frames}}f)`).join(' · ');
  return `<article>
    <video controls playsinline preload="metadata" src="${{esc(x.url)}}"></video>
    <div class="body">
      <div class="title-row">
        <strong>${{esc(x.sample_id)}}</strong>
        <span class="badge">${{esc(x.movement_key)}} → none → ${{esc(x.camera_key)}}</span>
      </div>
      <div class="meta">
        <span>case #${{x.case_index}}</span>
        <span>shard-${{String(x.shard).padStart(2, '0')}}</span>
        <span>${{x.width}}×${{x.height}}</span>
        <span>${{x.fps}}fps</span>
        <span>${{x.frames}} frames</span>
        <span>${{fmt(x.duration_sec)}}s</span>
        <span>E2E ${{fmt(x.latency_sec)}}s</span>
        <span>${{fmtMiB(x.bytes)}}</span>
      </div>
      <div class="timeline">${{x.segments.map(segmentHtml).join('')}}</div>
      <div class="segments">${{esc(detail)}}</div>
      <details>
        <summary>查看 prompt / trajectory</summary>
        <div class="prompt">${{esc(x.prompt)}}</div>
        <div class="segments">${{esc(x.trajectory)}}</div>
      </details>
      <div class="links">
        <a href="${{esc(x.url)}}" target="_blank" rel="noreferrer">单独打开视频</a>
        <button data-uri="${{esc(x.s3_uri)}}">复制 S3 URI</button>
      </div>
    </div>
  </article>`;
}}

document.querySelector('#grid').innerHTML = DATA.items.map(card).join('');
document.addEventListener('click', e => {{
  if (e.target.matches('button[data-uri]')) {{
    navigator.clipboard.writeText(e.target.dataset.uri);
  }}
}});
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    input_dir = result_dir / "input"
    artifact = load_json(result_dir / "artifact-urls.json")
    case_rows = load_jsonl(input_dir / "case-index.jsonl")
    case_by_sample = {row["sample_id"]: row for row in case_rows}

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    bucket = artifact["bucket"]
    prefix = artifact["prefix"].strip("/")
    status_keys = list_status_objects(s3, bucket, prefix)
    completed_shards = sorted(
        shard
        for shard, kinds in status_keys.items()
        if "benchmark" in kinds and "upload" in kinds
    )

    benchmark_summaries: dict[int, dict[str, Any]] = {}
    upload_summaries: dict[int, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []

    for shard in completed_shards:
        bench = s3_get_json(s3, bucket, status_keys[shard]["benchmark"])
        upload = s3_get_json(s3, bucket, status_keys[shard]["upload"])
        benchmark_summaries[shard] = bench
        upload_summaries[shard] = upload
        summary = bench["summary"]
        up_summary = upload["summary"]
        shard_rows.append(
            {
                "index": shard,
                "samples": summary["successful_samples"],
                "attempted": up_summary["attempted"],
                "uploaded": up_summary["uploaded"],
                "videos_per_hour": summary["node_videos_per_hour_this_run"],
                "p50_sec": summary["request_persisted_end_to_end_sec"]["p50"],
                "upload_wall_sec": up_summary["wall_sec"],
            }
        )
        for row in bench["results"]:
            if row.get("success"):
                item = dict(row)
                item["shard"] = shard
                case = case_by_sample[item["sample_id"]]
                item["case_index"] = case["case_index"]
                item["case_id"] = case["case_id"]
                item["movement_key"] = case["movement_key"]
                item["camera_key"] = case["camera_key"]
                all_results.append(item)

    all_results.sort(key=lambda row: row["case_index"])
    selected = even_sample(all_results, args.sample_limit)
    messages_by_sample = read_needed_messages(result_dir, selected)

    built_at = datetime.now(timezone.utc)
    url_expires_at = built_at + timedelta(seconds=args.expires_in)

    items = []
    for row in selected:
        msg = messages_by_sample[row["sample_id"]]
        metadata = msg["metadata"]
        prompt = ""
        for part in msg["messages"]:
            if part.get("role") == "user" and part.get("type") == "text":
                prompt = part["content"]
                break
        key = f"{prefix}/videos/{row['case_id']}.mp4"
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=args.expires_in,
        )
        items.append(
            {
                "sample_id": row["sample_id"],
                "case_index": row["case_index"],
                "case_id": row["case_id"],
                "image_id": row["image_id"],
                "trajectory": metadata["trajectory"],
                "segments": metadata["source_segments"],
                "movement_key": row["movement_key"],
                "camera_key": row["camera_key"],
                "latent_actions": metadata.get("latent_camera_actions", []),
                "width": row["media"]["width"],
                "height": row["media"]["height"],
                "fps": 24,
                "frames": row["media"]["frames"],
                "duration_sec": row["media"]["duration_sec"],
                "latency_sec": row["persisted_end_to_end_sec"],
                "bytes": row["media"]["bytes"],
                "prompt": prompt,
                "s3_uri": f"s3://{bucket}/{key}",
                "url": url,
                "shard": row["shard"],
            }
        )

    latencies = [row["persisted_end_to_end_sec"] for row in all_results]
    bytes_list = [row["media"]["bytes"] for row in all_results]
    total_uploaded = sum(upload["summary"]["uploaded"] for upload in upload_summaries.values())
    total_uploaded_bytes = sum(
        upload["summary"]["uploaded_bytes"] for upload in upload_summaries.values()
    )
    pending_shards = [
        idx for idx in range(20) if idx not in completed_shards
    ]

    payload = {
        "title": args.title,
        "subtitle": args.subtitle,
        "built_at": fmt_dt(built_at),
        "url_expires_at": fmt_dt(url_expires_at),
        "selected_count": len(items),
        "stats": [
            f"S3 已上传 {total_uploaded}/{args.total_cases}",
            f"已完成 shard {len(completed_shards)}/20",
            f"抽样展示 {len(items)} 条",
            f"累计体积 {total_uploaded_bytes / 1024 ** 3:.1f} GiB",
            f"视频 URL 有效期 7 天",
        ],
        "chips": [
            {"text": f"完成 shard: {', '.join(f'{x:02d}' for x in completed_shards)}"},
            {
                "text": f"未完成 shard: {', '.join(f'{x:02d}' for x in pending_shards)}",
                "kind": "warn" if pending_shards else "",
            },
        ],
        "overview": [
            ["批次前缀", prefix],
            ["构建时间", fmt_dt(built_at)],
            ["当前可见视频", f"{total_uploaded} / {args.total_cases}"],
            ["已完成 shard", f"{len(completed_shards)} / 20"],
            ["结果目录", str(result_dir)],
        ],
        "performance": [
            ["E2E 平均", f"{mean(latencies):.2f} s" if latencies else "0.00 s"],
            ["E2E p50", f"{percentile(latencies, 0.50):.2f} s"],
            ["E2E p95", f"{percentile(latencies, 0.95):.2f} s"],
            ["平均文件体积", f"{mean(bytes_list) / 1024 ** 2:.1f} MiB" if bytes_list else "0.0 MiB"],
            [
                "中位单节点吞吐",
                f"{percentile([row['videos_per_hour'] for row in shard_rows], 0.50):.1f} videos/h"
                if shard_rows
                else "0.0 videos/h",
            ],
        ],
        "notes": [
            ["输入图片/消息", "网页只读 S3 上的已上传 MP4，不内嵌视频数据"],
            ["抽样策略", "按 case_index 从已上传结果里均匀抽样"],
            ["可转发性", "同一个 HTML 文件内含所有所需视频签名 URL"],
            ["静态性质", "页面不会自动刷新后续新完成的 shard"],
            ["签名到期", fmt_dt(url_expires_at)],
        ],
        "shards": shard_rows,
        "items": items,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bucket": bucket,
                "prefix": prefix,
                "completed_shards": completed_shards,
                "uploaded": total_uploaded,
                "selected_items": len(items),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
