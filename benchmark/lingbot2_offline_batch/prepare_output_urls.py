#!/usr/bin/env python3
"""Generate per-case S3 output indexes and presigned MP4 PUT URLs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    rows = [
        json.loads(line)
        for line in args.case_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    client = boto3.Session(
        profile_name=args.profile, region_name=args.region
    ).client("s3")
    prefix = args.prefix.strip("/")
    shard_urls: list[dict[str, str]] = [{} for _ in range(args.shards)]
    expected_shard_counts = Counter(row["shard"] for row in rows)
    output_rows = []
    for row in rows:
        key = f"{prefix}/videos/{row['case_id']}.mp4"
        put_url = client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": args.bucket,
                "Key": key,
                "ContentType": "video/mp4",
            },
            ExpiresIn=args.expires_in,
        )
        shard_urls[row["shard"]][row["sample_id"]] = put_url
        output_rows.append(
            {
                "case_index": row["case_index"],
                "shard": row["shard"],
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "s3_uri": f"s3://{args.bucket}/{key}",
                "http_url": (
                    f"https://{args.bucket}.s3.{args.region}.amazonaws.com/"
                    f"{quote(key, safe='/')}"
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "output-s3-index.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row in output_rows:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    for index, urls in enumerate(shard_urls):
        if len(urls) != expected_shard_counts[index]:
            raise ValueError(f"shard {index} has {len(urls)} URLs")
        (args.output_dir / f"put-urls-shard-{index:02d}.json").write_text(
            json.dumps(urls, indent=2) + "\n", encoding="utf-8"
        )
    preview_urls = {
        row["sample_id"]: shard_urls[row["shard"]][row["sample_id"]]
        for row in rows[:10]
    }
    (args.output_dir / "put-urls-preview10.json").write_text(
        json.dumps(preview_urls, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(rows),
                "bucket": args.bucket,
                "prefix": prefix,
                "shards": args.shards,
                "preview": len(preview_urls),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
