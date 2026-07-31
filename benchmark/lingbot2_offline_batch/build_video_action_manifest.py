#!/usr/bin/env python3
"""Join LingBot output video locations with deterministic generated actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_capacity_smoke_720p import (
    FPS,
    HEIGHT,
    OUTPUT_VIDEO_FRAMES,
    WIDTH,
    quantize_actions,
)
from thirdperson_actions import ACTION_SEED, build_action_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--action-seed", type=int, default=ACTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.case_index)
    outputs = {
        row["sample_id"]: row for row in read_jsonl(args.output_index)
    }
    if len(outputs) != len(cases):
        raise ValueError(
            f"case/output count mismatch: {len(cases)} cases, {len(outputs)} outputs"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_samples = set()
    with args.output.open("w", encoding="utf-8") as destination:
        for expected_index, case in enumerate(cases):
            if case["case_index"] != expected_index:
                raise ValueError(
                    f"case index is not contiguous at row {expected_index}: {case}"
                )
            trajectory = build_action_trajectory(expected_index, args.action_seed)
            for field in (
                "action_id",
                "movement_key",
                "ending_movement_key",
                "movement_pair",
                "camera_key",
                "action_seed",
                "action_pattern",
            ):
                if case.get(field) != trajectory[field]:
                    raise ValueError(
                        f"case {case['sample_id']} has invalid {field}: "
                        f"{case.get(field)!r}, expected {trajectory[field]!r}"
                    )
            output = outputs.get(case["sample_id"])
            if output is None:
                raise ValueError(f"missing output location for {case['sample_id']}")
            _, latent_actions = quantize_actions(trajectory)
            row = {
                "schema_version": 1,
                "sample_id": case["sample_id"],
                "case_id": case["case_id"],
                "case_index": expected_index,
                "image": {
                    "image_id": case["image_id"],
                    "image_index": case["image_index"],
                    "s3_uri": case["image_uri"],
                },
                "video": {
                    "s3_uri": output["s3_uri"],
                    "http_url": output["http_url"],
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fps": FPS,
                    "frames": OUTPUT_VIDEO_FRAMES,
                    "duration_seconds": OUTPUT_VIDEO_FRAMES / FPS,
                },
                "action_trajectory": {
                    "action_id": trajectory["action_id"],
                    "fps": trajectory["fps"],
                    "num_frames": trajectory["num_frames"],
                    "movement_key": trajectory["movement_key"],
                    "ending_movement_key": trajectory["ending_movement_key"],
                    "movement_pair": trajectory["movement_pair"],
                    "camera_key": trajectory["camera_key"],
                    "action_seed": trajectory["action_seed"],
                    "action_pattern": trajectory["action_pattern"],
                    "condition_inputs": trajectory["condition_inputs"],
                    "segments": trajectory.get("segments", []),
                },
                "model_action_mapping": {
                    "reference_frame_index": 0,
                    "generated_video_frame_range": [1, 128],
                    "temporal_compression": 4,
                    "quantization": "drop reference frame; majority vote per 4 generated frames",
                    "generated_latent_frames": 32,
                    "latent_camera_actions": latent_actions,
                },
            }
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen_samples.add(case["sample_id"])

    if len(seen_samples) != len(cases):
        raise ValueError("sample IDs are not unique")
    print(
        json.dumps(
            {
                "rows": len(cases),
                "first_sample_id": cases[0]["sample_id"],
                "last_sample_id": cases[-1]["sample_id"],
                "action_seed": args.action_seed,
                "first_action_id": build_action_trajectory(0, args.action_seed)["action_id"],
                "last_action_id": build_action_trajectory(len(cases) - 1, args.action_seed)["action_id"],
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
