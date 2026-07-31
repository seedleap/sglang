#!/usr/bin/env python3
"""Publish shard fixtures and build a compact presigned artifact manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", default="leap-world-us-east-2")
    parser.add_argument(
        "--prefix",
        default=(
            "world-model/eval/lingbot2/eval_results/minWM/"
            "third_person_all_1000x5_720p_129f_20260715"
        ),
    )
    parser.add_argument("--profile", default="wms")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--expires-in", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--shards", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = boto3.Session(
        profile_name=args.profile, region_name=args.region
    ).client("s3")
    prefix = args.prefix.strip("/")

    def publish(path: Path) -> dict[str, str]:
        key = f"{prefix}/inputs/{path.name}"
        extra = {
            "ContentType": (
                "application/gzip" if path.name.endswith(".gz") else "application/json"
            )
        }
        client.upload_file(str(path), args.bucket, key, ExtraArgs=extra)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": args.bucket, "Key": key},
            ExpiresIn=args.expires_in,
        )
        return {"s3_uri": f"s3://{args.bucket}/{key}", "get_url": url}

    image_urls = publish(args.input_dir / "image-urls.json")
    shards = []
    for index in range(args.shards):
        messages = publish(args.input_dir / f"messages-shard-{index:02d}.jsonl.gz")
        put_urls = publish(args.input_dir / f"put-urls-shard-{index:02d}.json")

        def status_put_url(name: str) -> str:
            key = f"{prefix}/status/{name}"
            return client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": args.bucket,
                    "Key": key,
                    "ContentType": "application/json",
                },
                ExpiresIn=args.expires_in,
            )

        shards.append(
            {
                "index": index,
                "messages": messages,
                "put_urls": put_urls,
                "benchmark_summary_put_url": status_put_url(
                    f"shard-{index:02d}-benchmark-summary.json"
                ),
                "upload_summary_put_url": status_put_url(
                    f"shard-{index:02d}-upload-summary.json"
                ),
            }
        )

    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=args.expires_in)).isoformat(),
        "bucket": args.bucket,
        "prefix": prefix,
        "image_urls": image_urls,
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "prefix": prefix,
                "shards": len(shards),
                "expires_at": payload["expires_at"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
