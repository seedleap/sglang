# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM Wan21/wan_utils/camera_trajectory.py

"""Camera trajectory primitives for the MinWM realtime world model.

All poses are w2c (world-to-camera) OpenCV convention. The step sizes are the
minWM training-distribution constants and must stay verbatim-aligned with
``camera_trajectory.py`` in the minWM repo — do not "fix" them independently:

  translation: 0.08 units per latent frame (w/s/a/d/u/dn)
  rotation:    3.0 degrees per latent frame (i/k/j/l)
"""

from __future__ import annotations

import numpy as np
import torch

TRANSLATION_STEP = 0.08
ROTATION_STEP_RAD = np.radians(3.0)  # 3.0 degrees per latent frame

# Key -> per-frame motion dict (identical to minWM MOTION_PRIMITIVES).
MOTION_PRIMITIVES: dict[str, dict[str, float]] = {
    "w": {"forward": TRANSLATION_STEP},
    "s": {"forward": -TRANSLATION_STEP},
    "d": {"right": TRANSLATION_STEP},
    "a": {"right": -TRANSLATION_STEP},
    "u": {"up": TRANSLATION_STEP},
    "dn": {"up": -TRANSLATION_STEP},
    "j": {"yaw": -ROTATION_STEP_RAD},  # yaw left
    "l": {"yaw": ROTATION_STEP_RAD},  # yaw right
    "i": {"pitch": ROTATION_STEP_RAD},  # pitch up
    "k": {"pitch": -ROTATION_STEP_RAD},  # pitch down
}

MINWM_DEFAULT_INTRINSICS = (0.5, 0.5, 0.5, 0.5)  # fx, fy, cx, cy (normalized)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def step_c2w(current_T: np.ndarray, motions: list[dict[str, float]]) -> tuple:
    """Advance a running camera-to-world pose by a sequence of motion dicts.

    Verbatim port of minWM ``step_c2w``: yaw before pitch, translations in the
    post-rotation local frame, "up" mapped to -Y (OpenCV Y-down).

    Returns ``(new_T, poses_after_each_motion)``; poses does NOT include
    ``current_T`` itself.
    """
    T = current_T.copy()
    poses = []
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        up = move.get("up", 0.0)
        if up:
            # up in camera frame = -Y (OpenCV Y-down)
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return T, poses


def keys_to_motion(frame_keys: list[str]) -> dict[str, float]:
    """Compose the held keys of one latent frame into a single motion dict.

    Opposing keys cancel; unknown keys are ignored (matching the serving-layer
    behavior in the private minWM stack).
    """
    motion: dict[str, float] = {}
    for key in frame_keys:
        primitive = MOTION_PRIMITIVES.get(key)
        if primitive is None:
            continue
        for field_name, delta in primitive.items():
            motion[field_name] = motion.get(field_name, 0.0) + delta
    return motion


def advance_camera_chunk(
    current_c2w: np.ndarray,
    frame_keys_per_frame: list[list[str]],
    *,
    intrinsics: tuple[float, float, float, float],
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Integrate one chunk of per-frame key states into camera tensors.

    Frame ``i`` of the chunk uses the pose *before* motion ``i`` applies (the
    chunk starts at ``current_c2w``), matching the private minWM serving stack:
    ``frame_poses = [current, after_m0, ..., after_m(N-2)]``.

    Returns ``(new_c2w, viewmats, Ks)`` where ``viewmats`` is ``(1, N, 4, 4)``
    w2c and ``Ks`` is ``(1, N, 3, 3)``.
    """
    motions = [keys_to_motion(frame_keys) for frame_keys in frame_keys_per_frame]
    new_c2w, poses_after_each = step_c2w(current_c2w, motions)
    frame_poses = [current_c2w] + poses_after_each[:-1]

    viewmats_np = np.stack([np.linalg.inv(c2w) for c2w in frame_poses]).astype(
        np.float32
    )
    fx, fy, cx, cy = intrinsics
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    ks_np = np.tile(k, (len(frame_poses), 1, 1))

    viewmats = torch.from_numpy(viewmats_np)[None].to(device=device, dtype=dtype)
    ks = torch.from_numpy(ks_np)[None].to(device=device, dtype=dtype)
    return new_c2w, viewmats, ks
