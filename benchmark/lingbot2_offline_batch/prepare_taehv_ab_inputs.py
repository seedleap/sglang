#!/usr/bin/env python3
"""Prepare one deterministic minWM fixture for a LingBot VAE A/B run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

DEFAULT_SOURCE_S3_URI = (
    "s3://leap-world-us-east-2/world-model/eval/platform/eval_sets/minWM/"
    "testset100_v2/messages.jsonl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def target_video(row: dict[str, Any]) -> dict[str, Any]:
    targets = [
        message
        for message in row.get("messages", [])
        if message.get("role") == "target" and message.get("type") == "video"
    ]
    if len(targets) != 1:
        raise ValueError(f"{row.get('sample_id')}: expected exactly one target video")
    return targets[0]


def build_fixture(
    source: Path,
    output_dir: Path,
    *,
    limit: int,
    presign_image_uri: Callable[[str], str],
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(rows) < limit:
        raise ValueError(f"source has {len(rows)} rows, fewer than requested {limit}")

    selected = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "fixture.jsonl"
    messages_path = output_dir / "messages.jsonl"
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in selected
    )
    fixture_path.write_text(serialized, encoding="utf-8")
    messages_path.write_text(serialized, encoding="utf-8")

    image_urls: dict[str, str] = {}
    for row in selected:
        image_id = str(row.get("metadata", {}).get("image_id", ""))
        if not image_id:
            raise ValueError(f"{row.get('sample_id')}: missing metadata.image_id")
        uri = str(target_video(row).get("uri", ""))
        parse_s3_uri(uri)
        existing = image_urls.get(image_id)
        if existing is None:
            image_urls[image_id] = presign_image_uri(uri)

    (output_dir / "image-urls.json").write_text(
        json.dumps(image_urls, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "fixture_sha256": sha256_file(fixture_path),
        "selected_samples": len(selected),
        "first_sample_id": selected[0]["sample_id"],
        "last_sample_id": selected[-1]["sample_id"],
        "image_count": len(image_urls),
    }
    (output_dir / "fixture-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--source-s3-uri", default=DEFAULT_SOURCE_S3_URI)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--expires-in", type=int, default=24 * 60 * 60)
    parser.add_argument("--profile")
    parser.add_argument("--region", default="us-east-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source:
        source_path = args.source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        import boto3

        client = boto3.Session(
            profile_name=args.profile, region_name=args.region
        ).client("s3")
    else:
        import boto3

        client = boto3.Session(
            profile_name=args.profile, region_name=args.region
        ).client("s3")
        bucket, key = parse_s3_uri(args.source_s3_uri)
        source_path = args.output_dir / "source-messages.jsonl"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(source_path))

    def presign_image_uri(uri: str) -> str:
        bucket, key = parse_s3_uri(uri)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=args.expires_in,
        )

    result = build_fixture(
        source_path,
        args.output_dir,
        limit=args.limit,
        presign_image_uri=presign_image_uri,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
