#!/usr/bin/env python3
"""Upload each completed MP4 as soon as benchmark progress records success."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--put-urls", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def upload_one(sample_id: str, path: Path, url: str, retries: int) -> dict:
    size = path.stat().st_size
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with path.open("rb") as payload:
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="PUT",
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(size),
                    },
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    status = response.status
                    etag = response.headers.get("ETag")
            if 200 <= status < 300:
                return {
                    "sample_id": sample_id,
                    "success": True,
                    "path": str(path),
                    "bytes": size,
                    "attempts": attempt,
                    "status": status,
                    "etag": etag,
                }
            last_error = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < retries:
            time.sleep(min(30, 2 ** (attempt - 1)))
    return {
        "sample_id": sample_id,
        "success": False,
        "path": str(path),
        "bytes": size,
        "attempts": retries,
        "error": last_error,
    }


def write_summary(
    output: Path,
    started: float,
    expected: int,
    results: dict[str, dict],
    generation_failures: dict[str, str],
    final: bool,
) -> None:
    uploaded = [row for row in results.values() if row["success"]]
    failed = [row for row in results.values() if not row["success"]]
    payload = {
        "summary": {
            "expected": expected,
            "attempted": len(results),
            "uploaded": len(uploaded),
            "failed": len(failed),
            "generation_failures_seen": len(generation_failures),
            "uploaded_bytes": sum(row["bytes"] for row in uploaded),
            "wall_sec": time.monotonic() - started,
            "streaming": True,
            "final": final,
        },
        "results": sorted(results.values(), key=lambda row: row["sample_id"]),
        "generation_failures": generation_failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    put_urls = json.loads(args.put_urls.read_text(encoding="utf-8"))
    expected = len(put_urls)
    results: dict[str, dict] = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        results = {
            row["sample_id"]: row
            for row in previous.get("results", [])
            if row.get("success")
        }

    started = time.monotonic()
    generation_failures: dict[str, str] = {}
    in_flight: dict[Future, str] = {}
    submitted: set[str] = set(results)
    offset = 0
    partial = ""
    completions_since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            if args.progress.exists():
                with args.progress.open(encoding="utf-8") as progress:
                    progress.seek(offset)
                    chunk = progress.read()
                    offset = progress.tell()
                if chunk:
                    partial += chunk
                    lines = partial.split("\n")
                    partial = lines.pop()
                    for line in lines:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        sample_id = row["sample_id"]
                        if not row.get("success"):
                            generation_failures[sample_id] = row.get("error", "unknown")
                            continue
                        if sample_id in submitted:
                            continue
                        if sample_id not in put_urls:
                            raise ValueError(f"missing PUT URL for {sample_id}")
                        path = Path(row["output"])
                        if not path.is_file():
                            raise FileNotFoundError(f"completed MP4 is missing: {path}")
                        future = executor.submit(
                            upload_one,
                            sample_id,
                            path,
                            put_urls[sample_id],
                            args.retries,
                        )
                        in_flight[future] = sample_id
                        submitted.add(sample_id)

            for future in list(in_flight):
                if not future.done():
                    continue
                sample_id = in_flight.pop(future)
                result = future.result()
                results[sample_id] = result
                completions_since_checkpoint += 1
                print(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "success": result["success"],
                            "attempts": result["attempts"],
                            "uploaded": sum(row["success"] for row in results.values()),
                            "expected": expected,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if completions_since_checkpoint >= 10:
                write_summary(
                    args.output,
                    started,
                    expected,
                    results,
                    generation_failures,
                    final=False,
                )
                completions_since_checkpoint = 0

            if args.done_file.exists() and not in_flight:
                if args.progress.exists():
                    current_size = args.progress.stat().st_size
                    if current_size > offset:
                        continue
                break
            time.sleep(args.poll_seconds)

    write_summary(
        args.output,
        started,
        expected,
        results,
        generation_failures,
        final=True,
    )
    uploaded = sum(row["success"] for row in results.values())
    failed = [row for row in results.values() if not row["success"]]
    if uploaded != expected or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
