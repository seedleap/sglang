#!/usr/bin/env python3
"""Build deterministic third-person LingBot batch cases without trajs.jsonl."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

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
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--cases-per-image", type=int, default=5)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--action-seed", type=int, default=ACTION_SEED)
    parser.add_argument("--action-index-offset", type=int, default=0)
    parser.add_argument("--sample-prefix", default="thirdperson1000x5")
    parser.add_argument("--group", default="TPV_1000X5_720P")
    return parser.parse_args()


def load_images(path: Path, count: int) -> list[tuple[str, str]]:
    rows = IMAGE_ROW.findall(path.read_text(encoding="utf-8"))
    if len(rows) != count:
        raise ValueError(f"expected {count} image rows, found {len(rows)}")
    indices = [int(index) for index, _, _ in rows]
    ids = [image_id for _, image_id, _ in rows]
    uris = [uri for _, _, uri in rows]
    if indices != list(range(1, count + 1)):
        raise ValueError("Markdown image row indices are not contiguous")
    if len(set(ids)) != count or len(set(uris)) != count:
        raise ValueError("Markdown contains duplicate image IDs or paths")
    return list(zip(ids, uris))


def load_labels(path: Path, image_ids: set[str]) -> dict[str, dict]:
    labels = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("id") in image_ids:
                labels[row["id"]] = row
    missing = sorted(image_ids - labels.keys())
    if missing:
        raise ValueError(f"missing labels for {missing[:10]}")
    return labels


def build_case(
    image_index: int,
    image_id: str,
    image_uri: str,
    case_slot: int,
    action_index: int,
    action_seed: int,
    label: dict,
    sample_prefix: str,
    group: str,
) -> dict:
    trajectory = build_action_trajectory(action_index, action_seed)
    actions, latent_keys = quantize_actions(trajectory)
    case_id = f"{image_id}-action-{case_slot:02d}"
    return {
        "schema_version": 2,
        "sample_id": f"{sample_prefix}/TPV/{case_id}",
        "source_name": label.get(
            "dataset_id", "game_view_standard_v2_run_20260713_2000"
        ),
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
            "camera_key": trajectory["camera_key"],
            "action_seed": trajectory["action_seed"],
            "action_pattern": trajectory["action_pattern"],
            "source_segments": trajectory["segments"],
            "latent_camera_actions": latent_keys,
            "action_source_policy": "fixed-seed deterministic global schedule; per-image sampling without replacement",
            "action_quantization": "drop reference frame; majority vote per 4 generated frames",
            "view": {"tier": "tpv", "sub": "dataset_label"},
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
    images = load_images(args.markdown, args.images)
    labels = load_labels(args.labels, {image_id for image_id, _ in images})
    # A schedule offset must start at an image boundary to preserve five unique
    # combinations per image. Global pair counts remain within one for a batch.
    if args.action_index_offset % args.cases_per_image:
        raise ValueError("action-index-offset must align to an image boundary")
    balance = validate_assignment(args.images, args.cases_per_image, args.action_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    pair_counts: Counter[str] = Counter()
    try:
        for image_index, (image_id, image_uri) in enumerate(images):
            label = labels[image_id]
            for case_slot in range(args.cases_per_image):
                case_index = image_index * args.cases_per_image + case_slot
                action_index = args.action_index_offset + case_index
                row = build_case(
                    image_index,
                    image_id,
                    image_uri,
                    case_slot,
                    action_index,
                    args.action_seed,
                    label,
                    args.sample_prefix,
                    args.group,
                )
                metadata = row["metadata"]
                pair_counts[f"{metadata['movement_key']}+{metadata['camera_key']}"] += 1
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
                            "case_id": metadata["case_id"],
                            "image_index": image_index,
                            "image_id": image_id,
                            "image_uri": image_uri,
                            "action_id": metadata["action_id"],
                            "trajectory_id": metadata["action_id"],
                            "movement_key": metadata["movement_key"],
                            "camera_key": metadata["camera_key"],
                            "action_seed": metadata["action_seed"],
                            "action_pattern": metadata["action_pattern"],
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
    counts = list(pair_counts.values())
    if len(pair_counts) != 16 or max(counts) - min(counts) > 1:
        raise ValueError(f"invalid action-pair distribution: {pair_counts}")
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
                "action_index_offset": args.action_index_offset,
                "action_pattern": ACTION_PATTERN,
                "sample_prefix": args.sample_prefix,
                "group": args.group,
                "balance": {**balance, "pair_case_counts": dict(sorted(pair_counts.items()))},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
