#!/usr/bin/env python3
"""Build a standalone, shareable HTML gallery for LingBot2 batch outputs on S3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3


DEFAULT_BUCKET = "leap-world-us-east-2"
DEFAULT_PREFIX = "world-model/eval/lingbot2/eval_results/minWM/lingbot2_20260715"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--profile", default="wms")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--expires-in", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument(
        "--dataset",
        choices=("all", "thirdperson50", "testset100_v2"),
        default="all",
    )
    parser.add_argument(
        "--thirdperson-summary",
        type=Path,
        default=root
        / "2026-07-15-thirdperson50-run20260713"
        / "priority-thirdperson50"
        / "summary.json",
    )
    parser.add_argument(
        "--thirdperson-inputs",
        type=Path,
        default=root
        / "2026-07-15-thirdperson50-run20260713"
        / "input-messages-with-actions.jsonl",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=root / "2026-07-15-testset100-v2-full" / "full" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "2026-07-15-lingbot2-s3-gallery.html",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    thirdperson = read_json(args.thirdperson_summary)
    eval_set = read_json(args.eval_summary)
    action_inputs = {
        item["sample_id"]: item.get("metadata", {})
        for item in read_jsonl(args.thirdperson_inputs)
    }
    client = boto3.Session(
        profile_name=args.profile, region_name=args.region
    ).client("s3")

    rows: list[dict] = []
    for dataset, payload in (
        ("thirdperson50", thirdperson),
        ("testset100_v2", eval_set),
    ):
        for result in payload["results"]:
            relative_video = result["output"].split("/videos/", 1)[1]
            key = f"{args.prefix.rstrip('/')}/{dataset}/videos/{relative_video}"
            signed_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": args.bucket, "Key": key},
                ExpiresIn=args.expires_in,
            )
            metadata = action_inputs.get(result["sample_id"], {})
            label = metadata.get("label_metadata", {})
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": result["sample_id"],
                    "group": result["group"],
                    "view": result.get("view", {}),
                    "trajectory": result.get("trajectory", ""),
                    "action_plan": metadata.get("action_plan", []),
                    "subject": label.get("subject_detail", ""),
                    "scene": label.get("scene_signature", ""),
                    "style": label.get("art_style_detail", ""),
                    "duration_sec": result.get("video_seconds"),
                    "frames": result.get("persisted_frames"),
                    "e2e_sec": result.get("persisted_end_to_end_sec"),
                    "realtime_factor": result.get("realtime_factor"),
                    "width": result.get("media", {}).get("width"),
                    "height": result.get("media", {}).get("height"),
                    "fps": result.get("media", {}).get("avg_frame_rate"),
                    "bytes": result.get("media", {}).get("bytes"),
                    "s3_uri": f"s3://{args.bucket}/{key}",
                    "url": signed_url,
                }
            )

    if args.dataset != "all":
        rows = [row for row in rows if row["dataset"] == args.dataset]

    gallery_labels = {
        "all": "LingBot2 离线批量推理视频画廊",
        "thirdperson50": "第三人称 Action 50 条视频画廊",
        "testset100_v2": "minWM testset100_v2（337 条）视频画廊",
    }
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=args.expires_in)
        ).isoformat(),
        "bucket": args.bucket,
        "prefix": args.prefix.rstrip("/"),
        "dataset": args.dataset,
        "gallery_label": gallery_labels[args.dataset],
        "counts": {
            "all": len(rows),
            "thirdperson50": sum(r["dataset"] == "thirdperson50" for r in rows),
            "testset100_v2": sum(r["dataset"] == "testset100_v2" for r in rows),
        },
        "thirdperson_summary": thirdperson["summary"],
        "eval_summary": eval_set["summary"],
    }
    return rows, meta


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LingBot2 batch inference gallery</title>
  <style>
    :root { color-scheme: dark; --bg:#0a0d12; --panel:#121721; --line:#263044; --text:#ecf2ff; --muted:#91a0b8; --accent:#68d7b6; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif; }
    header { position:sticky; top:0; z-index:3; padding:18px 24px 14px; background:rgba(10,13,18,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
    h1 { margin:0 0 5px; font-size:21px; letter-spacing:.1px; }
    .sub,.hint { color:var(--muted); }
    .stats { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .pill,.filter { border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:#111722; color:var(--text); }
    .filter { cursor:pointer; }
    .filter.active { border-color:var(--accent); color:var(--accent); }
    .tools { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:12px; }
    input,select,button { font:inherit; }
    input,select { color:var(--text); background:#0d121a; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
    input { width:min(430px,100%); }
    main { padding:18px 24px 32px; max-width:1800px; margin:auto; }
    #grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:14px; }
    article { min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
    video { display:block; width:100%; aspect-ratio:832/480; background:#05070a; }
    .body { padding:11px 12px 13px; }
    .title { display:flex; gap:8px; align-items:flex-start; justify-content:space-between; }
    .title strong { overflow-wrap:anywhere; }
    .badge { flex:none; color:var(--accent); border:1px solid #2e6458; border-radius:999px; padding:2px 7px; font-size:12px; }
    .meta { display:flex; flex-wrap:wrap; gap:5px 12px; margin:8px 0; color:var(--muted); font-size:12px; }
    .trajectory { margin:7px 0; padding:7px 8px; background:#0b1017; border-radius:7px; overflow-wrap:anywhere; font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
    .actions { display:flex; flex-wrap:wrap; gap:4px; margin:7px 0; }
    .action { border:1px solid #334057; border-radius:5px; padding:2px 5px; font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }
    .desc { color:#bbc6d8; font-size:12px; margin-top:6px; }
    .links { display:flex; gap:10px; margin-top:9px; }
    a,button.link { color:#8ccfff; text-decoration:none; background:none; border:0; padding:0; cursor:pointer; }
    .pager { display:flex; justify-content:center; align-items:center; gap:12px; padding:20px 0 4px; }
    .pager button { color:var(--text); background:#151c28; border:1px solid var(--line); border-radius:7px; padding:7px 12px; cursor:pointer; }
    .pager button:disabled { opacity:.35; cursor:default; }
    .empty { grid-column:1/-1; padding:50px; text-align:center; color:var(--muted); }
    @media (max-width:600px) { header,main { padding-left:12px; padding-right:12px; } #grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header>
  <h1 id="galleryTitle">LingBot2 离线批量推理视频画廊</h1>
  <div class="sub" id="subtitle"></div>
  <div class="stats" id="stats"></div>
  <div class="tools">
    <button class="filter active" data-filter="all">全部</button>
    <button class="filter" data-filter="thirdperson50">第三人称 50</button>
    <button class="filter" data-filter="testset100_v2">testset100_v2</button>
    <button class="filter" data-filter="G1">G1</button>
    <button class="filter" data-filter="G2">G2</button>
    <button class="filter" data-filter="G3">G3</button>
    <button class="filter" data-filter="G4">G4</button>
    <button class="filter" data-filter="G5">G5</button>
    <input id="search" type="search" placeholder="搜索 sample id / action / subject / scene">
    <label class="hint">每页 <select id="pageSize"><option>8</option><option selected>12</option><option>24</option><option>48</option></select></label>
  </div>
</header>
<main>
  <div id="grid"></div>
  <div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
</main>
<script>
const ITEMS = __ITEMS__;
const META = __META__;
let filter = 'all', page = 1, pageSize = 12, query = '';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n,d=2) => Number.isFinite(Number(n)) ? Number(n).toFixed(d) : '-';
const filtered = () => ITEMS.filter(x => {
  const inGroup = filter === 'all' || x.dataset === filter || x.group === filter;
  const hay = [x.sample_id,x.group,x.trajectory,x.subject,x.scene,x.style].join(' ').toLowerCase();
  return inGroup && (!query || hay.includes(query));
});
function card(x) {
  const actions = x.action_plan?.map(a => `<span class="action">${fmt(a.start_sec,1)}s ${esc(a.keys.join('+'))}</span>`).join('') || '';
  const desc = [x.subject,x.scene,x.style].filter(Boolean).map(esc).join(' · ');
  return `<article>
    <video controls playsinline preload="metadata" src="${esc(x.url)}"></video>
    <div class="body">
      <div class="title"><strong>${esc(x.sample_id)}</strong><span class="badge">${esc(x.group)}</span></div>
      <div class="meta"><span>${x.width}×${x.height}</span><span>${esc(x.fps)} fps</span><span>${fmt(x.duration_sec)}s</span><span>E2E ${fmt(x.e2e_sec)}s</span><span>RTF ${fmt(x.realtime_factor)}</span></div>
      ${x.trajectory ? `<div class="trajectory">${esc(x.trajectory)}</div>` : ''}
      ${actions ? `<div class="actions">${actions}</div>` : ''}
      ${desc ? `<div class="desc">${desc}</div>` : ''}
      <div class="links"><a href="${esc(x.url)}" target="_blank" rel="noreferrer">单独打开</a><button class="link" data-copy="${esc(x.s3_uri)}">复制 S3 URI</button></div>
    </div>
  </article>`;
}
function render() {
  document.querySelectorAll('video').forEach(v => v.pause());
  const rows = filtered(), pages = Math.max(1, Math.ceil(rows.length/pageSize));
  page = Math.min(page,pages);
  const start = (page-1)*pageSize, shown = rows.slice(start,start+pageSize);
  $('#grid').innerHTML = shown.length ? shown.map(card).join('') : '<div class="empty">没有匹配的 case</div>';
  $('#page').textContent = `${page} / ${pages}（${rows.length} 条）`;
  $('#prev').disabled = page <= 1; $('#next').disabled = page >= pages;
  document.querySelectorAll('[data-copy]').forEach(b => b.onclick = async () => {
    await navigator.clipboard.writeText(b.dataset.copy); const old=b.textContent; b.textContent='已复制'; setTimeout(()=>b.textContent=old,900);
  });
}
$('#subtitle').textContent = `HTTPS 视频地址有效至 ${new Date(META.expires_at).toLocaleString()}；HTML 为独立文件，可直接发送。`;
$('#galleryTitle').textContent = META.gallery_label;
$('#stats').innerHTML = `<span class="pill">总计 ${META.counts.all}</span><span class="pill">第三人称 ${META.counts.thirdperson50}</span><span class="pill">评测集 ${META.counts.testset100_v2}</span><span class="pill">S3: ${esc(META.bucket)}</span>`;
document.querySelectorAll('.filter').forEach(b => {
  const f = b.dataset.filter;
  if (f !== 'all' && !ITEMS.some(x => x.dataset === f || x.group === f)) b.hidden = true;
  b.onclick = () => { document.querySelector('.filter.active').classList.remove('active'); b.classList.add('active'); filter=f; page=1; render(); };
});
$('#search').oninput = e => { query=e.target.value.trim().toLowerCase(); page=1; render(); };
$('#pageSize').onchange = e => { pageSize=Number(e.target.value); page=1; render(); };
$('#prev').onclick = () => { page--; render(); scrollTo({top:0,behavior:'smooth'}); };
$('#next').onclick = () => { page++; render(); scrollTo({top:0,behavior:'smooth'}); };
render();
</script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    if not 1 <= args.expires_in <= 7 * 24 * 60 * 60:
        raise SystemExit("--expires-in must be between 1 second and 7 days")
    rows, meta = build_rows(args)
    rendered = HTML_TEMPLATE.replace(
        "__ITEMS__", json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    ).replace(
        "__META__", json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "items": len(rows),
                "bytes": args.output.stat().st_size,
                "expires_at": meta["expires_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
