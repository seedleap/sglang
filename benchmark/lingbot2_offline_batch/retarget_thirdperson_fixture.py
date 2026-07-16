#!/usr/bin/env python3
"""Retarget a fixed-duration third-person fixture to another video shape/FPS."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


TEMPORAL_COMPRESSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--generated-latent-frames", type=int, required=True)
    return parser.parse_args()


def target_video(row: dict) -> dict:
    targets = [
        message
        for message in row["messages"]
        if message.get("role") == "target" and message.get("type") == "video"
    ]
    if len(targets) != 1:
        raise ValueError(f"{row['sample_id']}: expected exactly one video target")
    return targets[0]


def retarget(row: dict, args: argparse.Namespace) -> dict:
    row = copy.deepcopy(row)
    target = target_video(row)
    plans = row["metadata"].get("action_plan", [])
    if not plans:
        raise ValueError(f"{row['sample_id']}: missing action_plan")

    count = len(plans)
    base, remainder = divmod(args.generated_latent_frames, count)
    latent_lengths = [base + (index < remainder) for index in range(count)]
    if any(length <= 0 for length in latent_lengths):
        raise ValueError("generated latent count is too short for the action plan")

    controls = [
        control
        for control in target.get("controls", [])
        if control.get("type") == "keyboard_direction_frame_interval"
    ]
    if len(controls) != 1:
        raise ValueError(f"{row['sample_id']}: expected exactly one keyboard control")
    control = controls[0]
    action_keys = control["action_keys"]
    video_actions = []
    elapsed_latents = 0
    for plan, latent_length in zip(plans, latent_lengths):
        vector = [int(key in plan["keys"]) for key in action_keys]
        video_actions.extend(
            [vector] * (latent_length * TEMPORAL_COMPRESSION)
        )
        plan["latent_frames"] = latent_length
        plan["video_frames"] = latent_length * TEMPORAL_COMPRESSION
        plan["start_sec"] = round(
            elapsed_latents * TEMPORAL_COMPRESSION / args.fps, 6
        )
        elapsed_latents += latent_length
        plan["end_sec"] = round(
            elapsed_latents * TEMPORAL_COMPRESSION / args.fps, 6
        )

    output_video_frames = args.generated_latent_frames * TEMPORAL_COMPRESSION + 1
    control["actions"] = video_actions
    target["output"] = {
        "latent_frames": args.generated_latent_frames + 1,
        "height": args.height,
        "width": args.width,
    }
    target["metadata"].update(
        {
            "fps": args.fps,
            "output_video_frames": output_video_frames,
            "generated_latent_frames": args.generated_latent_frames,
        }
    )
    row["metadata"]["motion_pattern"] = f"randomized_keyboard_{args.fps}fps"
    return row


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            destination.write(
                json.dumps(retarget(json.loads(line), args), ensure_ascii=False) + "\n"
            )
            count += 1
    print(
        json.dumps(
            {
                "rows": count,
                "size": f"{args.width}x{args.height}",
                "fps": args.fps,
                "generated_latent_frames": args.generated_latent_frames,
                "output_video_frames": args.generated_latent_frames * 4 + 1,
            }
        )
    )


if __name__ == "__main__":
    main()
