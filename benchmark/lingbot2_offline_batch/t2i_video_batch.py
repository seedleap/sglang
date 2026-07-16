#!/usr/bin/env python3
"""Utilities for turning t2i image batches into LingBot/SGLang video cases."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_capacity_smoke_720p import (
    ACTION_KEYS,
    GENERATED_LATENT_FRAMES,
    HEIGHT,
    OUTPUT_VIDEO_FRAMES,
    WIDTH,
    quantize_actions,
)
from thirdperson_actions import ACTION_SEED, build_action_trajectory


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "item"


def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"missing required field: {key}")
    return value


def _video_config(request: dict[str, Any]) -> dict[str, Any]:
    return request.get("video") if isinstance(request.get("video"), dict) else {}


def _output_config(request: dict[str, Any]) -> dict[str, Any]:
    return request.get("output") if isinstance(request.get("output"), dict) else {}


def videos_per_image(request: dict[str, Any]) -> int:
    return int(_video_config(request).get("videos_per_image") or 5)


def action_seed(request: dict[str, Any]) -> int:
    return int(_video_config(request).get("action_seed") or ACTION_SEED)


def video_dimensions(request: dict[str, Any]) -> tuple[int, int]:
    video = _video_config(request)
    return int(video.get("width") or WIDTH), int(video.get("height") or HEIGHT)


def build_case_records(
    request: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    per_image = videos_per_image(request)
    if per_image <= 0 or per_image > 16:
        raise ValueError("videos_per_image must be between 1 and 16")
    seed = action_seed(request)
    width, height = video_dimensions(request)
    video_s3_prefix = str(_required(_output_config(request), "video_s3_prefix")).rstrip("/")
    job_id = str(request.get("generation_job_id") or "sglang-video")
    cases: list[dict[str, Any]] = []

    for image_index, row in enumerate(manifest_rows):
        item_id = str(_required(row, "item_id"))
        image_uri = str(_required(row, "image_uri"))
        video_prompt = str(_required(row, "video_prompt"))
        prompt_source = str(row.get("video_prompt_source") or "explicit")
        image_prompt = str(row.get("image_prompt") or video_prompt)
        item_stem = _safe_id(item_id)
        for case_slot in range(per_image):
            case_index = image_index * per_image + case_slot
            trajectory = build_action_trajectory(case_index, seed)
            video_actions, latent_keys = quantize_actions(trajectory)
            movement_key = trajectory["movement_key"]
            camera_key = trajectory["camera_key"]
            case_id = f"{item_stem}-action-{case_slot:02d}-{movement_key}{camera_key}"
            sample_id = f"{job_id}/TPV/{case_id}"
            video_s3_uri = f"{video_s3_prefix}/{item_stem}/{case_slot:02d}_{movement_key}{camera_key}.mp4"
            case = {
                "schema_version": 1,
                "sample_id": sample_id,
                "case_id": case_id,
                "case_index": case_index,
                "image_index": image_index,
                "image_id": item_id,
                "image_uri": image_uri,
                "video_s3_uri": video_s3_uri,
                "movement_key": movement_key,
                "camera_key": camera_key,
                "action_id": trajectory["action_id"],
                "action_seed": trajectory["action_seed"],
                "action_pattern": trajectory["action_pattern"],
                "metadata": {
                    "case_id": case_id,
                    "group": "t2i_sglang_video",
                    "image_index": image_index,
                    "image_id": item_id,
                    "source_image_uri": image_uri,
                    "image_prompt": image_prompt,
                    "video_prompt": video_prompt,
                    "video_prompt_source": prompt_source,
                    "trajectory": trajectory["action_id"],
                    "action_id": trajectory["action_id"],
                    "movement_key": movement_key,
                    "camera_key": camera_key,
                    "action_seed": trajectory["action_seed"],
                    "action_pattern": trajectory["action_pattern"],
                    "source_segments": trajectory.get("segments", []),
                    "latent_camera_actions": latent_keys,
                    "view": {"tier": "tpv", "sub": "t2i"},
                },
                "messages": [
                    {"role": "user", "type": "text", "content": video_prompt},
                    {
                        "role": "target",
                        "type": "video",
                        "uri": image_uri,
                        "reference_frame_count": 1,
                        "output": {
                            "latent_frames": GENERATED_LATENT_FRAMES + 1,
                            "height": height,
                            "width": width,
                        },
                        "controls": [
                            {
                                "type": "keyboard_direction_frame_interval",
                                "actions": video_actions,
                                "action_keys": ACTION_KEYS,
                            }
                        ],
                        "metadata": {
                            "fps": int(_video_config(request).get("fps") or 24),
                            "source_image": f"{item_stem}.png",
                            "source_video_frames": 1,
                            "output_video_frames": OUTPUT_VIDEO_FRAMES,
                            "generated_latent_frames": GENERATED_LATENT_FRAMES,
                            "negative_prompt": row.get("negative_prompt"),
                        },
                    },
                ],
            }
            cases.append(case)
    return cases


def build_callback_progress_payload(
    request: dict[str, Any],
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_item: dict[str, int] = defaultdict(int)
    case_by_id = {case["case_id"]: case for case in cases}
    videos_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        expected_by_item[case["image_id"]] += 1

    succeeded = failed = 0
    for result in results:
        case = case_by_id[str(result["case_id"])]
        status = str(result.get("status") or "")
        if status == "succeeded":
            succeeded += 1
        elif status in {"failed", "rejected"}:
            failed += 1
        videos_by_item[case["image_id"]].append(
            {
                "video_uri": result.get("video_uri") or case["video_s3_uri"],
                "movement_key": result.get("movement_key") or case["movement_key"],
                "camera_key": result.get("camera_key") or case["camera_key"],
                "action_seed": result.get("action_seed") or case["action_seed"],
                "action_pattern": result.get("action_pattern") or case["action_pattern"],
                "status": status or "running",
                "error": result.get("error") or "",
            }
        )

    items = []
    for image_id in sorted(expected_by_item):
        videos = sorted(
            videos_by_item.get(image_id, []),
            key=lambda item: (item["movement_key"], item["camera_key"], item["video_uri"]),
        )
        item_failed = any(video["status"] in {"failed", "rejected"} for video in videos)
        item_succeeded = len(videos) == expected_by_item[image_id] and all(
            video["status"] == "succeeded" for video in videos
        )
        video_status = "failed" if item_failed else "succeeded" if item_succeeded else "running"
        items.append(
            {
                "item_id": image_id,
                "status": "succeeded" if video_status == "succeeded" else "running",
                "stage": "sglang_video_generation",
                "metadata": {
                    "video_status": video_status,
                    "videos": videos,
                },
            }
        )

    total = len(cases)
    video_status = "failed" if failed else "succeeded" if succeeded == total else "running"
    return {
        "status": video_status,
        "stage": "sglang_video_generation",
        "summary": {
            "video_status": video_status,
            "video_expected_count": total,
            "video_succeeded_count": succeeded,
            "video_failed_count": failed,
            "video_output_prefix": _output_config(request).get("video_s3_prefix", ""),
        },
        "counters": {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "running": max(0, total - succeeded - failed),
        },
        "items": items,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
