#!/usr/bin/env python3
"""Seeded random trajectory selection for third-person video batches."""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Sequence


ACTION_SEED = 20260715
MOVEMENT_KEYS = ("w", "a", "s", "d")
ACTION_SOURCE = "trajs.jsonl"
ACTION_PATTERN = "trajs.jsonl"
ACTION_FRAME_COUNT = 129
FPS = 24


@lru_cache(maxsize=None)
def combo_schedule(seed: int = ACTION_SEED) -> tuple[tuple[str, str], ...]:
    """Return a fixed seeded permutation of ordered two-movement pairs."""
    pairs = [
        (first_key, second_key)
        for first_key, second_key in product(MOVEMENT_KEYS, MOVEMENT_KEYS)
        if first_key != second_key
    ]
    random.Random(seed).shuffle(pairs)
    return tuple(pairs)


def combo_for_case(case_index: int, seed: int = ACTION_SEED) -> tuple[str, str]:
    if case_index < 0:
        raise ValueError("case_index must be non-negative")
    schedule = combo_schedule(seed)
    return schedule[case_index % len(schedule)]


def _generated_action_trajectory(case_index: int, seed: int = ACTION_SEED) -> dict:
    movement_key, ending_movement_key = combo_for_case(case_index, seed)
    action_id = f"generated_action_seed{seed}_{case_index:05d}"
    camera_actions = (
        [[movement_key] for _ in range(57)]
        + [[] for _ in range(15)]
        + [[ending_movement_key] for _ in range(57)]
    )
    return {
        "action_id": action_id,
        # Keep traj_id as a compatibility alias for quantization/error messages.
        "traj_id": action_id,
        "fps": 24,
        "num_frames": ACTION_FRAME_COUNT,
        "condition_inputs": {"camera_actions": camera_actions},
        "segments": [
            {
                "kind": "movement",
                "key": movement_key,
                "start_frame": 0,
                "end_frame": 56,
                "num_frames": 57,
            },
            {
                "kind": "noop",
                "key": None,
                "start_frame": 57,
                "end_frame": 71,
                "num_frames": 15,
            },
            {
                "kind": "movement",
                "key": ending_movement_key,
                "start_frame": 72,
                "end_frame": 128,
                "num_frames": 57,
            },
        ],
        "movement_key": movement_key,
        "ending_movement_key": ending_movement_key,
        "movement_pair": f"{movement_key}+{ending_movement_key}",
        "camera_key": "",
        "traj_type": "generated_wasd_pair",
        "action_source": "generated",
        "action_index": case_index,
        "action_seed": seed,
        "action_pattern": "generated:57 movement + 15 noop + 57 movement",
    }


def load_action_trajectories(path: str | Path) -> tuple[dict[str, Any], ...]:
    with Path(path).open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return validate_action_trajectories(rows)


def validate_action_trajectories(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not rows:
        raise ValueError("action trajectory pool is empty")
    validated: list[dict[str, Any]] = []
    for row in rows:
        traj_id = row.get("traj_id") or row.get("action_id") or "<unknown>"
        if row.get("fps") != FPS or row.get("num_frames") != ACTION_FRAME_COUNT:
            raise ValueError(f"{traj_id}: expected {FPS} FPS and {ACTION_FRAME_COUNT} frames")
        camera_actions = row.get("condition_inputs", {}).get("camera_actions", [])
        if len(camera_actions) != ACTION_FRAME_COUNT:
            raise ValueError(f"{traj_id}: camera action length mismatch")
        for frame, keys in enumerate(camera_actions):
            if not isinstance(keys, list):
                raise ValueError(f"{traj_id}: invalid action at frame {frame}")
            if len(keys) > 1 or any(key not in MOVEMENT_KEYS for key in keys):
                raise ValueError(f"{traj_id}: invalid action at frame {frame}")
        validated.append(copy.deepcopy(row))
    return tuple(validated)


@lru_cache(maxsize=None)
def _sample_order(pool_size: int, seed: int, cycle: int) -> tuple[int, ...]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    indices = list(range(pool_size))
    random.Random(seed + cycle).shuffle(indices)
    return tuple(indices)


def sampled_trajectory_index(case_index: int, pool_size: int, seed: int = ACTION_SEED) -> int:
    if case_index < 0:
        raise ValueError("case_index must be non-negative")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    cycle, offset = divmod(case_index, pool_size)
    return _sample_order(pool_size, seed, cycle)[offset]


def _first_action_key(camera_actions: Sequence[Sequence[str]]) -> str:
    for keys in camera_actions:
        if keys:
            return str(keys[0])
    return ""


def _last_action_key(camera_actions: Sequence[Sequence[str]]) -> str:
    for keys in reversed(camera_actions):
        if keys:
            return str(keys[0])
    return ""


def _action_pattern(row: dict[str, Any]) -> str:
    traj_type = str(row.get("traj_type") or "trajectory").strip() or "trajectory"
    return f"{ACTION_SOURCE}:{traj_type}"


def _source_action_pattern(row: dict[str, Any], source: str) -> str:
    existing = str(row.get("action_pattern") or "").strip()
    if existing:
        return existing
    traj_type = str(row.get("traj_type") or "custom").strip() or "custom"
    return f"{source}:{traj_type}"


def build_action_trajectory(
    case_index: int,
    seed: int = ACTION_SEED,
    trajectories: Sequence[dict[str, Any]] | None = None,
    *,
    validate: bool = True,
) -> dict:
    """Return one seeded-random trajectory from trajs.jsonl-compatible rows."""
    if trajectories is None:
        return _generated_action_trajectory(case_index, seed)

    pool = validate_action_trajectories(trajectories) if validate else tuple(trajectories)
    action_index = sampled_trajectory_index(case_index, len(pool), seed)
    selected = copy.deepcopy(pool[action_index])
    traj_id = str(selected.get("traj_id") or f"trajs_jsonl_{action_index:05d}")
    camera_actions = selected["condition_inputs"]["camera_actions"]
    movement_key = _first_action_key(camera_actions)
    ending_movement_key = _last_action_key(camera_actions)
    selected.update(
        {
            "action_id": traj_id,
            "traj_id": traj_id,
            "movement_key": movement_key,
            "ending_movement_key": ending_movement_key,
            "movement_pair": (
                f"{movement_key}+{ending_movement_key}"
                if movement_key and ending_movement_key
                else movement_key or ending_movement_key
            ),
            "camera_key": "",
            "traj_type": str(selected.get("traj_type") or ""),
            "action_source": ACTION_SOURCE,
            "action_index": action_index,
            "action_seed": seed,
            "action_pattern": _action_pattern(selected),
        }
    )
    return selected


def build_api_action_trajectory(
    payload: dict[str, Any] | list[Any],
    *,
    case_index: int,
    seed: int = ACTION_SEED,
    source: str = "api",
) -> dict:
    """Normalize one API-supplied action into a trajectory record."""
    if isinstance(payload, list):
        selected: dict[str, Any] = {
            "traj_id": f"api_action_{case_index:05d}",
            "fps": FPS,
            "num_frames": len(payload),
            "traj_type": "api_custom",
            "condition_inputs": {"camera_actions": copy.deepcopy(payload)},
        }
    elif isinstance(payload, dict):
        selected = copy.deepcopy(payload)
        if "condition_inputs" not in selected and "camera_actions" in selected:
            selected["condition_inputs"] = {
                "camera_actions": selected.pop("camera_actions")
            }
        selected.setdefault("traj_id", selected.get("action_id") or f"api_action_{case_index:05d}")
        selected.setdefault("fps", FPS)
        camera_actions = selected.get("condition_inputs", {}).get("camera_actions", [])
        selected.setdefault("num_frames", len(camera_actions))
        selected.setdefault("traj_type", "api_custom")
    else:
        raise ValueError("api action must be an object or frame action list")

    selected = copy.deepcopy(validate_action_trajectories([selected])[0])
    traj_id = str(selected.get("traj_id") or selected.get("action_id") or f"api_action_{case_index:05d}")
    camera_actions = selected["condition_inputs"]["camera_actions"]
    movement_key = _first_action_key(camera_actions)
    ending_movement_key = _last_action_key(camera_actions)
    selected.update(
        {
            "action_id": str(selected.get("action_id") or traj_id),
            "traj_id": traj_id,
            "movement_key": movement_key,
            "ending_movement_key": ending_movement_key,
            "movement_pair": (
                f"{movement_key}+{ending_movement_key}"
                if movement_key and ending_movement_key
                else movement_key or ending_movement_key
            ),
            "camera_key": str(selected.get("camera_key") or ""),
            "traj_type": str(selected.get("traj_type") or ""),
            "action_source": source,
            "action_index": int(selected.get("action_index", case_index)),
            "action_seed": int(selected.get("action_seed", seed)),
            "action_pattern": _source_action_pattern(selected, source),
        }
    )
    return selected


def validate_assignment(
    image_count: int,
    cases_per_image: int = 5,
    seed: int = ACTION_SEED,
) -> dict[str, int | dict[str, int]]:
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if not 0 < cases_per_image <= len(combo_schedule(seed)):
        raise ValueError(
            f"cases_per_image must be between 1 and {len(combo_schedule(seed))}"
        )

    pair_counts: Counter[str] = Counter()
    movement_key_counts: Counter[str] = Counter()
    for image_index in range(image_count):
        pairs = [
            combo_for_case(image_index * cases_per_image + slot, seed)
            for slot in range(cases_per_image)
        ]
        if len(set(pairs)) != cases_per_image:
            raise AssertionError(f"image {image_index} has repeated action pairs")
        pair_counts.update(f"{first}+{second}" for first, second in pairs)
        for first, second in pairs:
            movement_key_counts.update((first, second))

    counts = list(pair_counts.values())
    if (
        len(pair_counts) != len(combo_schedule(seed))
        or max(counts) - min(counts) > 1
    ):
        raise AssertionError(f"movement pairs are not globally balanced: {pair_counts}")
    movement_counts = list(movement_key_counts.values())
    if (
        set(movement_key_counts) != set(MOVEMENT_KEYS)
        or max(movement_counts) - min(movement_counts) > 2
    ):
        raise AssertionError(
            f"movement keys are not globally balanced: {movement_key_counts}"
        )
    return {
        "pair_case_counts": dict(sorted(pair_counts.items())),
        "pair_count_min": min(counts),
        "pair_count_max": max(counts),
        "movement_key_counts": dict(sorted(movement_key_counts.items())),
        "movement_key_count_min": min(movement_counts),
        "movement_key_count_max": max(movement_counts),
    }
