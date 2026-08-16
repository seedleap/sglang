#!/usr/bin/env python3
"""Build a standalone HTML gallery for the balanced-action preview videos."""

from __future__ import annotations

import argparse
import gzip
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="wms")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--expires-in", type=int, default=7 * 24 * 60 * 60)
    return parser.parse_args()


def split_s3_uri(uri: str) -> tuple[str, str]:
    bucket_and_key = uri.removeprefix("s3://")
    return tuple(bucket_and_key.split("/", 1))  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    with gzip.open(args.messages, "rt", encoding="utf-8") as file:
        messages = [json.loads(line) for line in file if line.strip()]
    outputs = [
        json.loads(line)
        for line in args.output_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: len(messages)]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = {row["sample_id"]: row for row in summary["results"]}
    output_by_id = {row["sample_id"]: row for row in outputs}
    client = boto3.Session(
        profile_name=args.profile, region_name=args.region
    ).client("s3")
    items = []
    for message in messages:
        sample_id = message["sample_id"]
        output = output_by_id[sample_id]
        bucket, key = split_s3_uri(output["s3_uri"])
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=args.expires_in,
        )
        target = message["messages"][1]
        metadata = message["metadata"]
        items.append(
            {
                "sample_id": sample_id,
                "image_id": metadata["image_id"],
                "prompt": message["messages"][0]["content"],
                "trajectory": metadata["source_trajectory_id"],
                "segments": metadata["source_segments"],
                "movement_key": metadata["trajectory_balance"]["movement_key"],
                "camera_key": metadata["trajectory_balance"]["camera_key"],
                "latent_actions": metadata["latent_camera_actions"],
                "width": target["output"]["width"],
                "height": target["output"]["height"],
                "fps": target["metadata"]["fps"],
                "frames": target["metadata"]["output_video_frames"],
                "duration_sec": results[sample_id]["video_seconds"],
                "latency_sec": results[sample_id]["persisted_end_to_end_sec"],
                "bytes": results[sample_id]["media"]["bytes"],
                "s3_uri": output["s3_uri"],
                "url": url,
            }
        )

    generated = datetime.now(timezone.utc)
    expires = generated + timedelta(seconds=args.expires_in)
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    summary_data = json.dumps(summary["summary"], ensure_ascii=False).replace(
        "</", "<\\/"
    )
    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>第三人称 720p balanced-action preview 10</title>
  <style>
    :root {{ color-scheme:dark; --bg:#090d13; --panel:#121923; --line:#28364a; --muted:#91a2b9; --accent:#72e1be; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:var(--bg); color:#edf4ff; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif }}
    header {{ padding:22px 24px 17px; border-bottom:1px solid var(--line); background:#0d131c; position:sticky; top:0; z-index:2 }}
    h1 {{ margin:0 0 6px; font-size:22px }}
    .sub {{ color:var(--muted) }}
    .stats {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:5px 9px; background:#111a25 }}
    main {{ max-width:1600px; margin:auto; padding:18px 24px 36px }}
    #grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:15px }}
    article {{ overflow:hidden; border:1px solid var(--line); border-radius:13px; background:var(--panel) }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#020407 }}
    .body {{ padding:12px 13px 14px }}
    .title {{ display:flex; gap:8px; justify-content:space-between; align-items:flex-start }}
    .title strong {{ overflow-wrap:anywhere }}
    .badge {{ flex:none; color:var(--accent); border:1px solid #326b5d; border-radius:999px; padding:2px 8px }}
    .meta {{ color:var(--muted); display:flex; flex-wrap:wrap; gap:5px 13px; margin:8px 0 }}
    .timeline {{ display:flex; height:30px; margin:10px 0 7px; overflow:hidden; border-radius:7px; font:12px ui-monospace,monospace }}
    .segment {{ display:flex; align-items:center; justify-content:center; min-width:46px; border-right:1px solid #17202c }}
    .move {{ background:#275b50 }} .none {{ background:#30394a }} .camera {{ background:#344f79 }}
    .segments {{ color:#bdc9da; font:12px ui-monospace,monospace }}
    details {{ margin-top:9px; color:#c8d3e3 }} summary {{ cursor:pointer; color:#8ccfff }}
    .prompt {{ margin-top:7px; white-space:pre-wrap; font-size:12px }}
    .links {{ display:flex; gap:12px; margin-top:10px }}
    a,button {{ color:#8ccfff; border:0; padding:0; background:none; font:inherit; cursor:pointer; text-decoration:none }}
    @media(max-width:600px) {{ header,main {{ padding-left:12px; padding-right:12px }} #grid {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body>
<header>
  <h1>第三人称 720p · 129 帧 · Balanced Actions · Preview 10</h1>
  <div class="sub">视频直接读取 S3，不内嵌 MP4。签名生成于 {html.escape(generated.isoformat())}，到期时间 {html.escape(expires.isoformat())}。</div>
  <div class="stats" id="stats"></div>
</header>
<main><div id="grid"></div></main>
<script>
const ITEMS={data};
const SUMMARY={summary_data};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt=n=>Number(n).toFixed(2);
document.querySelector('#stats').innerHTML=[
  `成功 ${{SUMMARY.successful_samples}}/${{SUMMARY.selected_samples}}`,
  `节点吞吐 ${{fmt(SUMMARY.node_videos_per_hour_this_run)}} videos/h`,
  `实测 wall ${{fmt(SUMMARY.measured_wall_sec)}}s`,
  `p50 ${{fmt(SUMMARY.request_persisted_end_to_end_sec.p50)}}s`,
  `输出 1280×720 · 24fps · 129f`
].map(x=>`<span class="pill">${{x}}</span>`).join('');
function segmentHtml(s,i) {{
  const kind=s.key===null?'none':i===0?'move':'camera';
  const label=s.key===null?'none':s.key;
  return `<div class="segment ${{kind}}" style="flex:${{s.num_frames}}">${{esc(label)}} · ${{s.num_frames}}f</div>`;
}}
function card(x) {{
  const detail=x.segments.map(s=>`${{s.key??'none'}}: ${{s.start_frame}}–${{s.end_frame}} (${{s.num_frames}}f)`).join(' · ');
  return `<article>
    <video controls playsinline preload="metadata" src="${{esc(x.url)}}"></video>
    <div class="body">
      <div class="title"><strong>${{esc(x.sample_id)}}</strong><span class="badge">${{esc(x.movement_key)}} → none → ${{esc(x.camera_key)}}</span></div>
      <div class="meta"><span>${{x.width}}×${{x.height}}</span><span>${{x.fps}}fps</span><span>${{x.frames}} frames</span><span>${{fmt(x.duration_sec)}}s</span><span>E2E ${{fmt(x.latency_sec)}}s</span><span>${{(x.bytes/1048576).toFixed(1)}}MiB</span></div>
      <div class="timeline">${{x.segments.map(segmentHtml).join('')}}</div>
      <div class="segments">${{esc(detail)}}</div>
      <details><summary>查看 prompt / trajectory</summary><div class="prompt">${{esc(x.prompt)}}</div><div class="segments">${{esc(x.trajectory)}}</div></details>
      <div class="links"><a href="${{esc(x.url)}}" target="_blank" rel="noreferrer">单独打开视频</a><button data-uri="${{esc(x.s3_uri)}}">复制 S3 URI</button></div>
    </div>
  </article>`;
}}
document.querySelector('#grid').innerHTML=ITEMS.map(card).join('');
document.addEventListener('click',e=>{{
  if(e.target.matches('button[data-uri]')) navigator.clipboard.writeText(e.target.dataset.uri);
}});
</script>
</body>
</html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(items),
                "output": str(args.output),
                "expires_at": expires.isoformat(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
