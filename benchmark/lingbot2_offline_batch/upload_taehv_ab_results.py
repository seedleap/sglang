#!/usr/bin/env python3
"""Upload selected test-only A/B artifacts while excluding signed input URLs."""

from __future__ import annotations

import argparse
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.strip("/")


def artifact_paths(root: Path) -> Iterable[Path]:
    ignored_parts = {"server-cache", "input"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        relative = path.relative_to(root)
        if ignored_parts.intersection(relative.parts):
            continue
        if path.name == "image-urls.json":
            continue
        yield path


def content_type(path: Path) -> str:
    if path.suffix == ".mp4":
        return "video/mp4"
    if path.suffix in {".json", ".jsonl"}:
        return "application/json"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def upload_one(
    client: Any, root: Path, bucket: str, prefix: str, path: Path
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    key = f"{prefix.rstrip('/')}/{relative}"
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type(path)},
    )
    return {
        "relative_path": relative,
        "s3_uri": f"s3://{bucket}/{key}",
        "bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    bucket, prefix = parse_s3_uri(args.s3_uri)
    client = boto3.client("s3")
    paths = list(artifact_paths(args.root))
    uploaded: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upload_one, client, args.root, bucket, prefix, path): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                uploaded.append(future.result())
            except Exception as error:
                failures.append(
                    f"{path.relative_to(args.root)}: {type(error).__name__}: {error}"
                )
    payload = {
        "root": str(args.root),
        "destination": args.s3_uri.rstrip("/"),
        "uploaded_files": len(uploaded),
        "uploaded_bytes": sum(item["bytes"] for item in uploaded),
        "failures": sorted(failures),
    }
    (args.root / "taehv-ab-upload.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    client.upload_file(
        str(args.root / "taehv-ab-upload.json"),
        bucket,
        f"{prefix.rstrip('/')}/taehv-ab-upload.json",
        ExtraArgs={"ContentType": "application/json"},
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
