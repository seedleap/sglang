#!/usr/bin/env python3
"""Build deterministic 720p third-person smoke cases without trajs.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from thirdperson_actions import ACTION_SEED, build_action_trajectory


ACTION_KEYS = ["w", "a", "s", "d", "i", "j", "k", "l"]
MOVEMENT_KEYS = ACTION_KEYS[:4]
CAMERA_KEYS = ACTION_KEYS[4:]
FPS = 24
WIDTH = 1280
HEIGHT = 720
GENERATED_LATENT_FRAMES = 32
OUTPUT_VIDEO_FRAMES = GENERATED_LATENT_FRAMES * 4 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--action-seed", type=int, default=ACTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=5)
    return parser.parse_args()


def load_label(path: Path, image_id: str) -> dict:
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("id") == image_id:
                return row
    raise ValueError(f"missing label for {image_id}")


def quantize_actions(row: dict) -> tuple[list[list[int]], list[list[str]]]:
    if row.get("fps") != FPS or row.get("num_frames") != OUTPUT_VIDEO_FRAMES:
        raise ValueError(f"{row.get('traj_id')}: expected {FPS} FPS and 129 frames")
    camera_actions = row.get("condition_inputs", {}).get("camera_actions", [])
    if len(camera_actions) != OUTPUT_VIDEO_FRAMES:
        raise ValueError(f"{row.get('traj_id')}: camera action length mismatch")
    for frame, keys in enumerate(camera_actions):
        if len(keys) > 1 or any(key not in ACTION_KEYS for key in keys):
            raise ValueError(f"{row.get('traj_id')}: invalid action at frame {frame}")

    # Frame zero is the conditioning image. Quantize the 128 generated frames
    # to the model's 32 latent actions with majority vote in each four-frame
    # block, then expand again for benchmark_evalset's native fixture format.
    latent_keys = []
    for offset in range(1, OUTPUT_VIDEO_FRAMES, 4):
        block = [tuple(keys) for keys in camera_actions[offset : offset + 4]]
        counts = Counter(block)
        winner = max(counts, key=lambda keys: (counts[keys], not keys, keys))
        latent_keys.append(list(winner))
    if len(latent_keys) != GENERATED_LATENT_FRAMES:
        raise AssertionError("latent action length mismatch")
    video_actions = [
        [int(key in keys) for key in ACTION_KEYS]
        for keys in latent_keys
        for _ in range(4)
    ]
    return video_actions, latent_keys


def main() -> None:
    args = parse_args()
    label = load_label(args.labels, args.image_id)
    if label.get("viewpoint") != "third_person":
        raise ValueError(f"expected third_person, got {label.get('viewpoint')}")
    if not label.get("generation_prompt"):
        raise ValueError("generation_prompt is empty")
    trajectories = [
        build_action_trajectory(case_index, args.action_seed)
        for case_index in range(args.case_count)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for case_index, trajectory in enumerate(trajectories):
            actions, latent_keys = quantize_actions(trajectory)
            row = {
                "schema_version": 2,
                "sample_id": f"capacity-smoke-720p/TPV/{args.image_id}-action-{case_index:02d}",
                "source_name": "game_view_standard_v2_run_20260713_2000",
                "metadata": {
                    "case_id": f"{args.image_id}-action-{case_index:02d}",
                    "group": "TPV_SMOKE_720P",
                    "image_id": args.image_id,
                    "duration_tier": "5s",
                    "trajectory": trajectory["traj_id"],
                    "seed": case_index,
                    "source_trajectory_index": case_index,
                    "source_trajectory_id": trajectory["traj_id"],
                    "movement_key": trajectory["movement_key"],
                    "camera_key": trajectory["camera_key"],
                    "action_seed": trajectory["action_seed"],
                    "action_pattern": trajectory["action_pattern"],
                    "source_segments": trajectory.get("segments", []),
                    "latent_camera_actions": latent_keys,
                    "action_quantization": "drop reference frame; majority vote per 4 generated frames",
                    "view": {"tier": "tpv", "sub": "dataset_label"},
                },
                "messages": [
                    {"role": "user", "type": "text", "content": label["generation_prompt"]},
                    {
                        "role": "target",
                        "type": "video",
                        "uri": args.image_uri,
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
                            "source_image": f"{args.image_id}.png",
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
                "cases": args.case_count,
                "image_id": args.image_id,
                "size": f"{WIDTH}x{HEIGHT}",
                "fps": FPS,
                "frames": OUTPUT_VIDEO_FRAMES,
                "duration_sec": OUTPUT_VIDEO_FRAMES / FPS,
                "generated_latent_frames": GENERATED_LATENT_FRAMES,
                "action_seed": args.action_seed,
                "action_ids": [row["action_id"] for row in trajectories],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
