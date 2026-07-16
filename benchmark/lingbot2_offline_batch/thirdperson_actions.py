#!/usr/bin/env python3
"""Deterministic, globally balanced actions for third-person video batches."""

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache
from itertools import product


ACTION_SEED = 20260715
MOVEMENT_KEYS = ("w", "a", "s", "d")
CAMERA_KEYS = ("i", "j", "k", "l")
ACTION_PATTERN = "57 movement + 15 noop + 57 camera"
ACTION_FRAME_COUNT = 129


@lru_cache(maxsize=None)
def combo_schedule(seed: int = ACTION_SEED) -> tuple[tuple[str, str], ...]:
    """Return a fixed seeded permutation of all 16 movement/camera pairs."""
    pairs = list(product(MOVEMENT_KEYS, CAMERA_KEYS))
    random.Random(seed).shuffle(pairs)
    return tuple(pairs)


def combo_for_case(case_index: int, seed: int = ACTION_SEED) -> tuple[str, str]:
    if case_index < 0:
        raise ValueError("case_index must be non-negative")
    schedule = combo_schedule(seed)
    return schedule[case_index % len(schedule)]


def build_action_trajectory(case_index: int, seed: int = ACTION_SEED) -> dict:
    movement_key, camera_key = combo_for_case(case_index, seed)
    action_id = f"generated_action_seed{seed}_{case_index:05d}"
    camera_actions = (
        [[movement_key] for _ in range(57)]
        + [[] for _ in range(15)]
        + [[camera_key] for _ in range(57)]
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
                "key": movement_key,
                "start_frame": 0,
                "end_frame": 56,
                "num_frames": 57,
            },
            {
                "key": None,
                "start_frame": 57,
                "end_frame": 71,
                "num_frames": 15,
            },
            {
                "key": camera_key,
                "start_frame": 72,
                "end_frame": 128,
                "num_frames": 57,
            },
        ],
        "movement_key": movement_key,
        "camera_key": camera_key,
        "action_seed": seed,
        "action_pattern": ACTION_PATTERN,
    }


def validate_assignment(
    image_count: int,
    cases_per_image: int = 5,
    seed: int = ACTION_SEED,
) -> dict[str, int | dict[str, int]]:
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if not 0 < cases_per_image <= len(combo_schedule(seed)):
        raise ValueError("cases_per_image must be between 1 and 16")

    pair_counts: Counter[str] = Counter()
    for image_index in range(image_count):
        pairs = [
            combo_for_case(image_index * cases_per_image + slot, seed)
            for slot in range(cases_per_image)
        ]
        if len(set(pairs)) != cases_per_image:
            raise AssertionError(f"image {image_index} has repeated action pairs")
        pair_counts.update(f"{movement}+{camera}" for movement, camera in pairs)

    counts = list(pair_counts.values())
    if len(pair_counts) != 16 or max(counts) - min(counts) > 1:
        raise AssertionError(f"action pairs are not globally balanced: {pair_counts}")
    return {
        "pair_case_counts": dict(sorted(pair_counts.items())),
        "pair_count_min": min(counts),
        "pair_count_max": max(counts),
    }
