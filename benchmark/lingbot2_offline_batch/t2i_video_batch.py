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
from thirdperson_actions import (
    ACTION_SEED,
    build_api_action_trajectory,
    build_action_trajectory,
    validate_action_trajectories,
)


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
    return int(_video_config(request).get("videos_per_image") or 1)


def action_seed(request: dict[str, Any]) -> int:
    return int(_video_config(request).get("action_seed") or ACTION_SEED)


def _present_action(value: Any) -> bool:
    return value not in (None, "", [], {})


def _row_action_override(row: dict[str, Any], case_slot: int) -> tuple[Any | None, str]:
    for key in ("action", "action_trajectory", "video_action", "sglang_action"):
        value = row.get(key)
        if not _present_action(value):
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ):
            if case_slot >= len(value):
                raise ValueError(
                    f"{row.get('item_id')}: action list has {len(value)} entries "
                    f"but case slot {case_slot} was requested"
                )
            return value[case_slot], str(row.get("action_source") or "api")
        return value, str(row.get("action_source") or "api")
    return None, ""


def video_dimensions(request: dict[str, Any]) -> tuple[int, int]:
    video = _video_config(request)
    return int(video.get("width") or WIDTH), int(video.get("height") or HEIGHT)


def manifest_requires_action_trajectories(
    request: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> bool:
    per_image = videos_per_image(request)
    for row in manifest_rows:
        for case_slot in range(per_image):
            action_override, _source = _row_action_override(row, case_slot)
            if action_override is None:
                return True
    return False


def build_case_records(
    request: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    *,
    action_trajectories: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> list[dict[str, Any]]:
    per_image = videos_per_image(request)
    seed = action_seed(request)
    if per_image <= 0:
        raise ValueError("videos_per_image must be positive")
    needs_random_actions = manifest_requires_action_trajectories(request, manifest_rows)
    trajectory_pool: tuple[dict[str, Any], ...] = ()
    if needs_random_actions:
        if action_trajectories is None:
            raise ValueError("action_trajectories is required when manifest rows omit action")
        trajectory_pool = validate_action_trajectories(action_trajectories)
    if trajectory_pool and per_image > len(trajectory_pool):
        raise ValueError(
            f"videos_per_image must be between 1 and {len(trajectory_pool)}"
        )
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
            action_override, action_override_source = _row_action_override(row, case_slot)
            if action_override is not None:
                trajectory = build_api_action_trajectory(
                    action_override,
                    case_index=case_index,
                    seed=seed,
                    source=action_override_source,
                )
            else:
                trajectory = build_action_trajectory(
                    case_index,
                    seed,
                    trajectories=trajectory_pool,
                    validate=False,
                )
            video_actions, latent_keys = quantize_actions(trajectory)
            movement_key = trajectory["movement_key"]
            ending_movement_key = trajectory["ending_movement_key"]
            movement_pair = trajectory["movement_pair"]
            camera_key = trajectory["camera_key"]
            traj_id = str(trajectory["traj_id"])
            traj_type = str(trajectory.get("traj_type") or "")
            action_source = str(trajectory.get("action_source") or "")
            action_index = int(trajectory["action_index"])
            action_suffix = _safe_id(traj_id)
            case_id = f"{item_stem}-action-{case_slot:02d}-{action_suffix}"
            sample_id = f"{job_id}/TPV/{case_id}"
            video_s3_uri = f"{video_s3_prefix}/{item_stem}/{case_slot:02d}_{action_suffix}.mp4"
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
                "ending_movement_key": ending_movement_key,
                "movement_pair": movement_pair,
                "camera_key": camera_key,
                "traj_id": traj_id,
                "traj_type": traj_type,
                "action_source": action_source,
                "action_index": action_index,
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
                    "ending_movement_key": ending_movement_key,
                    "movement_pair": movement_pair,
                    "camera_key": camera_key,
                    "traj_id": traj_id,
                    "traj_type": traj_type,
                    "action_source": action_source,
                    "action_index": action_index,
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
                "ending_movement_key": result.get("ending_movement_key")
                or case["ending_movement_key"],
                "movement_pair": result.get("movement_pair") or case["movement_pair"],
                "camera_key": result.get("camera_key") or case["camera_key"],
                "traj_id": result.get("traj_id") or case["traj_id"],
                "traj_type": result.get("traj_type") or case["traj_type"],
                "action_source": result.get("action_source") or case["action_source"],
                "action_index": result.get("action_index", case["action_index"]),
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
            key=lambda item: (
                item["movement_key"],
                item["ending_movement_key"],
                item["camera_key"],
                item["video_uri"],
            ),
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
