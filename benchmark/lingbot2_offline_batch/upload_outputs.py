#!/usr/bin/env python3
"""Upload successful benchmark MP4s through per-sample presigned S3 PUT URLs."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--put-urls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def upload_one(sample_id: str, path: Path, url: str, retries: int) -> dict:
    payload = path.read_bytes()
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            method="PUT",
            headers={"Content-Type": "video/mp4"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = response.status
            if 200 <= status < 300:
                return {
                    "sample_id": sample_id,
                    "success": True,
                    "path": str(path),
                    "bytes": len(payload),
                    "attempts": attempt,
                    "status": status,
                }
            last_error = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < retries:
            time.sleep(2 ** (attempt - 1))
    return {
        "sample_id": sample_id,
        "success": False,
        "path": str(path),
        "bytes": len(payload),
        "attempts": retries,
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    put_urls = json.loads(args.put_urls.read_text(encoding="utf-8"))
    successful = [row for row in summary["results"] if row.get("success")]
    tasks = []
    for row in successful:
        sample_id = row["sample_id"]
        if sample_id not in put_urls:
            raise ValueError(f"missing PUT URL for {sample_id}")
        path = Path(row["output"])
        if not path.is_file():
            raise FileNotFoundError(f"missing MP4 for {sample_id}: {path}")
        tasks.append((sample_id, path, put_urls[sample_id]))

    if len(tasks) != len(put_urls):
        missing = sorted(set(put_urls) - {sample_id for sample_id, _, _ in tasks})
        raise ValueError(
            f"summary has {len(tasks)} successful cases for {len(put_urls)} URLs; "
            f"missing first: {missing[:5]}"
        )

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upload_one, sample_id, path, url, args.retries): sample_id
            for sample_id, path, url in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "sample_id": result["sample_id"],
                        "success": result["success"],
                        "attempts": result["attempts"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results.sort(key=lambda row: row["sample_id"])
    uploaded = [row for row in results if row["success"]]
    failed = [row for row in results if not row["success"]]
    payload = {
        "summary": {
            "attempted": len(results),
            "uploaded": len(uploaded),
            "failed": len(failed),
            "uploaded_bytes": sum(row["bytes"] for row in uploaded),
            "wall_sec": time.monotonic() - started,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
