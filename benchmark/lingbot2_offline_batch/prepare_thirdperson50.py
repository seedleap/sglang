#!/usr/bin/env python3
"""Build a native minWM-style fixture for the selected 50 TPV images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ACTION_KEYS = ["w", "a", "s", "d"]
MOVEMENT_KEYS = tuple(ACTION_KEYS)
GENERATED_LATENT_FRAMES = 20
VIDEO_ACTION_FRAMES = GENERATED_LATENT_FRAMES * 4
OUTPUT_VIDEO_FRAMES = VIDEO_ACTION_FRAMES + 1
ACTION_SEGMENTS = 10
ACTION_FRAMES_PER_SEGMENT = VIDEO_ACTION_FRAMES // ACTION_SEGMENTS
ACTION_SEGMENT_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-seed", type=int, default=20260715)
    return parser.parse_args()


def build_action_plan(sample_id: str, base_seed: int) -> tuple[list[list[int]], list[dict]]:
    """Create ten reproducible 0.5-second single-movement actions."""
    seed_material = f"{base_seed}:{sample_id}".encode()
    sample_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(sample_seed)

    video_actions: list[list[int]] = []
    plan: list[dict] = []
    previous_keys: tuple[str, ...] | None = None

    for segment_index in range(ACTION_SEGMENTS):
        while True:
            keys = (rng.choice(MOVEMENT_KEYS),)
            if keys != previous_keys:
                break
        previous_keys = keys
        action = [int(key in keys) for key in ACTION_KEYS]
        video_actions.extend([action] * ACTION_FRAMES_PER_SEGMENT)
        plan.append(
            {
                "segment": segment_index,
                "start_sec": segment_index * ACTION_SEGMENT_SECONDS,
                "end_sec": (segment_index + 1) * ACTION_SEGMENT_SECONDS,
                "kind": "movement",
                "keys": list(keys),
            }
        )

    if len(video_actions) != VIDEO_ACTION_FRAMES:
        raise AssertionError("action plan does not cover the full generated video")
    return video_actions, plan


def main() -> None:
    args = parse_args()
    selected_ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(selected_ids) != 50 or len(set(selected_ids)) != 50:
        raise ValueError("the priority fixture must contain 50 unique IDs")

    labels = {}
    with args.labels.open() as file:
        for line in file:
            label = json.loads(line)
            if label["id"] in selected_ids:
                labels[label["id"]] = label
    missing = sorted(set(selected_ids) - set(labels))
    if missing:
        raise ValueError(f"missing labels for {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for sample_id in selected_ids:
            label = labels[sample_id]
            if label.get("viewpoint") != "third_person":
                raise ValueError(
                    f"{sample_id}: expected third_person, got {label.get('viewpoint')}"
                )
            video_actions, action_plan = build_action_plan(sample_id, args.action_seed)
            trajectory = "|".join("+".join(segment["keys"]) for segment in action_plan)
            row = {
                "schema_version": 2,
                "sample_id": f"thirdperson50/TP50/{sample_id}",
                "source_name": "game_view_standard_v2_run_20260713_2000",
                "metadata": {
                    "case_id": sample_id,
                    "group": "TP50",
                    "image_id": sample_id,
                    "duration_tier": "5s",
                    "complexity": "single_movement_only",
                    "motion_pattern": "randomized_keyboard_0.5s",
                    "purpose": "third-person data synthesis preview",
                    "view": {"tier": "tpv", "sub": "dataset_label"},
                    "trajectory": trajectory,
                    "seed": 0,
                    "action_seed": args.action_seed,
                    "action_plan": action_plan,
                    "eval_set_id": "thirdperson50_run_20260713_2000",
                    "task_type": "action_ti2v",
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
                    {
                        "role": "user",
                        "type": "text",
                        "content": label["generation_prompt"],
                    },
                    {
                        "role": "target",
                        "type": "video",
                        "uri": f"{args.image_prefix.rstrip('/')}/{sample_id}.png",
                        "reference_frame_count": 1,
                        "output": {
                            "latent_frames": GENERATED_LATENT_FRAMES + 1,
                            "height": 480,
                            "width": 832,
                        },
                        "controls": [
                            {
                                "type": "keyboard_direction_frame_interval",
                                "actions": video_actions,
                                "action_keys": ACTION_KEYS,
                            }
                        ],
                        "metadata": {
                            "fps": 16,
                            "source_image": f"{sample_id}.png",
                            "source_video_frames": 1,
                            "output_video_frames": OUTPUT_VIDEO_FRAMES,
                            "generated_latent_frames": GENERATED_LATENT_FRAMES,
                            "negative_prompt": label.get("negative_prompt"),
                        },
                    },
                ],
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "samples": len(selected_ids),
                "viewpoint": "third_person",
                "size": "832x480",
                "fps": 16,
                "output_frames": OUTPUT_VIDEO_FRAMES,
                "duration_sec": OUTPUT_VIDEO_FRAMES / 16,
                "action_seed": args.action_seed,
                "action_segments_per_video": ACTION_SEGMENTS,
                "action_segment_seconds": ACTION_SEGMENT_SECONDS,
                "single_keys": ACTION_KEYS,
                "action_rule": "one movement key from wasd per segment",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
