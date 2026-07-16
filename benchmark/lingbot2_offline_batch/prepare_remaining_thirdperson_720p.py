#!/usr/bin/env python3
"""Build 3699 x 5 720p cases from the remaining TPV manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import boto3

from prepare_capacity_smoke_720p import (
    ACTION_KEYS,
    FPS,
    GENERATED_LATENT_FRAMES,
    HEIGHT,
    OUTPUT_VIDEO_FRAMES,
    WIDTH,
    quantize_actions,
)
from thirdperson_actions import (
    ACTION_PATTERN,
    ACTION_SEED,
    build_action_trajectory,
    validate_assignment,
)


IMAGE_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`(gvs2_\d+)`\s*\|\s*`(s3://[^`]+\.png)`\s*\|",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--manifest-s3-uri", required=True)
    parser.add_argument("--action-seed", type=int, default=ACTION_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=3699)
    parser.add_argument("--cases-per-image", type=int, default=5)
    parser.add_argument("--shards", type=int, default=75)
    parser.add_argument("--profile", default="wms")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--url-expires-in", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--sample-prefix", default="thirdperson-remaining3699x5")
    parser.add_argument("--group", default="TPV_REMAINING_3699X5_720P")
    return parser.parse_args()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def load_markdown_images(path: Path, count: int) -> list[tuple[str, str]]:
    matches = IMAGE_ROW.findall(path.read_text(encoding="utf-8"))
    if len(matches) != count:
        raise ValueError(f"expected {count} image rows, found {len(matches)}")
    indices = [int(index) for index, _, _ in matches]
    if indices != list(range(1, count + 1)):
        raise ValueError("Markdown image row indices are not contiguous")
    rows = [(image_id, uri) for _, image_id, uri in matches]
    if len({image_id for image_id, _ in rows}) != count:
        raise ValueError("Markdown contains duplicate image IDs")
    return rows


def load_manifest(client, uri: str, image_ids: set[str]) -> dict[str, dict]:
    bucket, key = parse_s3_uri(uri)
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    labels = {}
    for raw in body.iter_lines():
        if not raw:
            continue
        row = json.loads(raw)
        image_id = row.get("id")
        if image_id in image_ids:
            labels[image_id] = row
    missing = sorted(image_ids - labels.keys())
    if missing:
        raise ValueError(f"manifest is missing labels for {missing[:10]}")
    return labels


def build_case(
    image_index: int,
    case_slot: int,
    case_index: int,
    image_id: str,
    image_uri: str,
    label: dict,
    action_seed: int,
    sample_prefix: str,
    group: str,
) -> dict:
    trajectory = build_action_trajectory(case_index, action_seed)
    actions, latent_keys = quantize_actions(trajectory)
    case_id = f"{image_id}-action-{case_slot:02d}"
    return {
        "schema_version": 2,
        "sample_id": f"{sample_prefix}/TPV/{case_id}",
        "source_name": label["dataset_id"],
        "metadata": {
            "case_id": case_id,
            "group": group,
            "image_index": image_index,
            "image_id": image_id,
            "duration_tier": "129f",
            "trajectory": trajectory["action_id"],
            "seed": case_slot,
            "action_id": trajectory["action_id"],
            "movement_key": trajectory["movement_key"],
            "ending_movement_key": trajectory["ending_movement_key"],
            "movement_pair": trajectory["movement_pair"],
            "camera_key": trajectory["camera_key"],
            "action_seed": trajectory["action_seed"],
            "action_pattern": trajectory["action_pattern"],
            "source_segments": trajectory.get("segments", []),
            "latent_camera_actions": latent_keys,
            "action_source_policy": "fixed-seed deterministic global schedule; per-image sampling without replacement",
            "action_quantization": "drop reference frame; majority vote per 4 generated frames",
            "view": {"tier": "tpv", "sub": "dataset_label"},
            "source_run": label.get("source_run"),
            "label_metadata": {
                "subject_primary": label.get("subject_primary"),
                "subject_detail": label.get("subject_detail"),
                "scene_primary": label.get("scene_primary"),
                "scene_signature": label.get("scene_signature"),
                "art_style_primary": label.get("art_style_primary"),
                "art_style_detail": label.get("art_style_detail"),
            },
        },
        "messages": [
            {"role": "user", "type": "text", "content": label["generation_prompt"]},
            {
                "role": "target",
                "type": "video",
                "uri": image_uri,
                "reference_frame_count": 1,
                "output": {
                    "latent_frames": GENERATED_LATENT_FRAMES + 1,
                    "height": HEIGHT,
                    "width": WIDTH,
                },
                "controls": [
                    {
                        "type": "keyboard_direction_frame_interval",
                        "actions": actions,
                        "action_keys": ACTION_KEYS,
                    }
                ],
                "metadata": {
                    "fps": FPS,
                    "source_image": f"{image_id}.png",
                    "source_video_frames": 1,
                    "output_video_frames": OUTPUT_VIDEO_FRAMES,
                    "generated_latent_frames": GENERATED_LATENT_FRAMES,
                    "negative_prompt": label.get("negative_prompt"),
                },
            },
        ],
    }


def main() -> None:
    args = parse_args()
    total_cases = args.images * args.cases_per_image
    if args.shards <= 0 or args.shards > total_cases:
        raise ValueError("invalid shard count")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("s3")
    images = load_markdown_images(args.markdown, args.images)
    labels = load_manifest(
        client, args.manifest_s3_uri, {image_id for image_id, _ in images}
    )
    for image_id, image_uri in images:
        label = labels[image_id]
        if label.get("viewpoint") != "third_person":
            raise ValueError(f"{image_id}: viewpoint is not third_person")
        if label.get("image_s3_uri") != image_uri:
            raise ValueError(f"{image_id}: Markdown/manifest S3 URI mismatch")
        if not label.get("generation_prompt"):
            raise ValueError(f"{image_id}: generation_prompt is empty")
    balance = validate_assignment(args.images, args.cases_per_image, args.action_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_urls = {}
    for image_id, image_uri in images:
        bucket, key = parse_s3_uri(image_uri)
        image_urls[image_id] = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=args.url_expires_in,
        )
    (args.output_dir / "image-urls.json").write_text(
        json.dumps(image_urls, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    shard_files = [
        gzip.open(
            args.output_dir / f"messages-shard-{index:02d}.jsonl.gz",
            "wt",
            encoding="utf-8",
        )
        for index in range(args.shards)
    ]
    preview = gzip.open(
        args.output_dir / "messages-preview10.jsonl.gz", "wt", encoding="utf-8"
    )
    index_file = (args.output_dir / "case-index.jsonl").open("w", encoding="utf-8")
    shard_counts = [0] * args.shards
    raw_action_counts: Counter[str] = Counter()
    latent_action_counts: Counter[str] = Counter()
    try:
        for image_index, (image_id, image_uri) in enumerate(images):
            label = labels[image_id]
            for case_slot in range(args.cases_per_image):
                case_index = image_index * args.cases_per_image + case_slot
                row = build_case(
                    image_index,
                    case_slot,
                    case_index,
                    image_id,
                    image_uri,
                    label,
                    args.action_seed,
                    args.sample_prefix,
                    args.group,
                )
                trajectory = build_action_trajectory(case_index, args.action_seed)
                for keys in trajectory["condition_inputs"]["camera_actions"]:
                    raw_action_counts.update(keys or ["none"])
                for keys in row["metadata"]["latent_camera_actions"]:
                    latent_action_counts.update(keys or ["none"])
                shard = case_index % args.shards
                encoded = json.dumps(row, ensure_ascii=False)
                shard_files[shard].write(encoded + "\n")
                shard_counts[shard] += 1
                if case_index < 10:
                    preview.write(encoded + "\n")
                index_file.write(
                    json.dumps(
                        {
                            "case_index": case_index,
                            "shard": shard,
                            "sample_id": row["sample_id"],
                            "case_id": row["metadata"]["case_id"],
                            "image_index": image_index,
                            "image_id": image_id,
                            "image_uri": image_uri,
                            "action_id": trajectory["action_id"],
                            "trajectory_id": trajectory["action_id"],
                            "movement_key": trajectory["movement_key"],
                            "ending_movement_key": trajectory["ending_movement_key"],
                            "movement_pair": trajectory["movement_pair"],
                            "camera_key": trajectory["camera_key"],
                            "action_seed": trajectory["action_seed"],
                            "action_pattern": trajectory["action_pattern"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        for file in shard_files:
            file.close()
        preview.close()
        index_file.close()

    if max(shard_counts) - min(shard_counts) > 1 or sum(shard_counts) != total_cases:
        raise ValueError(f"invalid shard distribution: {shard_counts}")
    print(
        json.dumps(
            {
                "images": len(images),
                "cases_per_image": args.cases_per_image,
                "cases": total_cases,
                "shards": args.shards,
                "shard_min": min(shard_counts),
                "shard_max": max(shard_counts),
                "size": f"{WIDTH}x{HEIGHT}",
                "fps": FPS,
                "frames": OUTPUT_VIDEO_FRAMES,
                "duration_sec": OUTPUT_VIDEO_FRAMES / FPS,
                "action_seed": args.action_seed,
                "action_pattern": ACTION_PATTERN,
                "action_source_policy": "fixed-seed deterministic global schedule; per-image sampling without replacement",
                "balance": balance,
                "raw_action_counts": dict(sorted(raw_action_counts.items())),
                "latent_action_counts": dict(sorted(latent_action_counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
