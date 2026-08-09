"""Deterministic LingBot camera-action generation without trajs.jsonl."""

from __future__ import annotations

import random
from dataclasses import dataclass


MOVEMENT_KEYS = ("w", "a", "s", "d")
CAMERA_KEYS = ("i", "j", "k", "l")
ALL_ACTION_PAIRS = tuple(
    (movement, camera) for movement in MOVEMENT_KEYS for camera in CAMERA_KEYS
)

VIDEO_FRAMES = 129
FPS = 24
MOVEMENT_VIDEO_FRAMES = 57
NOOP_VIDEO_FRAMES = 15
CAMERA_VIDEO_FRAMES = 57

GENERATED_LATENT_FRAMES = 32
MOVEMENT_LATENT_FRAMES = 14
NOOP_LATENT_FRAMES = 4
CAMERA_LATENT_FRAMES = 14
REFERENCE_LATENT_FRAMES = 1
LATENTS_PER_CHUNK = 3
MAX_CHUNKS = 11


@dataclass(frozen=True)
class ActionPair:
    movement_key: str
    camera_key: str

    def as_dict(self) -> dict[str, str]:
        return {
            "movement_key": self.movement_key,
            "camera_key": self.camera_key,
        }


def _pair_permutation(action_seed: int) -> list[tuple[str, str]]:
    pairs = list(ALL_ACTION_PAIRS)
    random.Random(action_seed).shuffle(pairs)
    return pairs


def select_action_pairs(
    *, image_index: int, variants: int = 5, action_seed: int = 20260715
) -> list[ActionPair]:
    """Return distinct pairs for one image with near-perfect global balance.

    For sequential image indexes, all requests walk one fixed shuffled cycle of
    the 16 pairs. Any per-image window of at most 16 pairs is therefore unique.
    Across N total variants, pair counts differ by at most one.
    """

    if image_index < 0:
        raise ValueError("image_index must be non-negative")
    if not 1 <= variants <= len(ALL_ACTION_PAIRS):
        raise ValueError("variants must be between 1 and 16")
    permutation = _pair_permutation(action_seed)
    start = image_index * variants
    return [
        ActionPair(*permutation[(start + slot) % len(permutation)])
        for slot in range(variants)
    ]


def validate_action_pair(movement_key: str, camera_key: str) -> ActionPair:
    if movement_key not in MOVEMENT_KEYS:
        raise ValueError("movement_key must be one of wasd")
    if camera_key not in CAMERA_KEYS:
        raise ValueError("camera_key must be one of ijkl")
    return ActionPair(movement_key, camera_key)


def video_frame_actions(pair: ActionPair) -> list[list[str]]:
    """Return the original 129-frame action representation."""

    return (
        [[pair.movement_key] for _ in range(MOVEMENT_VIDEO_FRAMES)]
        + [[] for _ in range(NOOP_VIDEO_FRAMES)]
        + [[pair.camera_key] for _ in range(CAMERA_VIDEO_FRAMES)]
    )


def realtime_latent_actions(pair: ActionPair) -> list[list[str]]:
    """Return the 33 controls consumed by an 11-chunk realtime request.

    The first noop belongs to the reference image. The remaining 32 controls
    are the 4x temporally-compressed representation of the 128 generated video
    frames: 14 movement, 4 noop, then 14 camera controls.
    """

    actions = (
        [[] for _ in range(REFERENCE_LATENT_FRAMES)]
        + [[pair.movement_key] for _ in range(MOVEMENT_LATENT_FRAMES)]
        + [[] for _ in range(NOOP_LATENT_FRAMES)]
        + [[pair.camera_key] for _ in range(CAMERA_LATENT_FRAMES)]
    )
    if len(actions) != MAX_CHUNKS * LATENTS_PER_CHUNK:
        raise AssertionError("realtime action schedule must fill exactly 11 chunks")
    return actions
