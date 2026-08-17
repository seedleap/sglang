# SPDX-License-Identifier: Apache-2.0

"""Typed media profiles for realtime VAE output.

The profile is negotiated once, when a remote VAE session opens.  Keeping the
choice out of per-chunk sampling parameters prevents a client from believing a
post-process is active when the selected VAE worker cannot actually provide it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
)


class RealtimeMediaProfile(str, Enum):
    """Versioned, wire-stable realtime media behavior."""

    NATIVE_V1 = "native_v1"
    RIFE2X_V1 = "rife2x_v1"

    @property
    def output_timeline_fps_multiplier(self) -> int:
        return 2 if self is RealtimeMediaProfile.RIFE2X_V1 else 1


def parse_media_profile(value: object) -> RealtimeMediaProfile:
    """Parse a wire value without silently downgrading an unknown profile."""

    if value in (None, ""):
        return RealtimeMediaProfile.NATIVE_V1
    if isinstance(value, RealtimeMediaProfile):
        return value
    try:
        return RealtimeMediaProfile(str(value))
    except ValueError as exc:
        raise ProtocolViolation(f"unsupported realtime media profile: {value}") from exc


def validate_source_timeline_fps(value: object) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolViolation("realtime source timeline fps must be numeric") from exc
    if not 1 <= fps <= 60:
        raise ProtocolViolation("realtime source timeline fps must be between 1 and 60")
    return fps


def resolve_remote_media_profile(
    media_profile: object,
    *,
    legacy_enabled: bool,
    legacy_exp: object = 1,
    legacy_scale: object = 1.0,
    legacy_model_path: object = None,
) -> RealtimeMediaProfile:
    """Resolve an explicit remote profile without upgrading legacy clients."""

    requested = parse_media_profile(media_profile)
    if legacy_enabled:
        # Legacy browsers do not understand the session_ready receipt and
        # legacy H.264 bridges keep the source timebase.  Silently mapping the
        # old flag would therefore desynchronize either framing or playback.
        raise ProtocolViolation(
            "remote enable_frame_interpolation is no longer supported; "
            "upgrade the client and request realtime_media_profile=rife2x_v1"
        )
    if requested is RealtimeMediaProfile.RIFE2X_V1:
        if legacy_model_path not in (None, ""):
            raise ProtocolViolation(
                "remote RIFE weights are configured only by the VAE worker"
            )
        if legacy_exp is not None:
            try:
                exp = float(legacy_exp)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProtocolViolation("remote RIFE exp must be numeric") from exc
            if not math.isfinite(exp) or exp != 1.0:
                raise ProtocolViolation(
                    "remote RIFE supports only 2x interpolation (exp=1)"
                )
        if legacy_scale is not None:
            try:
                scale = float(legacy_scale)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProtocolViolation("remote RIFE scale must be numeric") from exc
            if not math.isfinite(scale) or abs(scale - 1.0) > 1e-9:
                raise ProtocolViolation("remote RIFE supports only scale=1.0")
    return requested


@dataclass(frozen=True, slots=True)
class MediaProfileAcceptance:
    """Authoritative result returned by the VAE worker."""

    requested: RealtimeMediaProfile
    effective: RealtimeMediaProfile
    source_timeline_fps: float
    output_timeline_fps: float
    weights_sha256: str | None = None

    def __post_init__(self) -> None:
        # Profiles are currently exact capabilities.  A future fallback policy
        # must be explicit instead of silently changing this invariant.
        if self.requested is not self.effective:
            raise ValueError("realtime media profile cannot be silently downgraded")

    @property
    def interpolation_enabled(self) -> bool:
        return self.effective is RealtimeMediaProfile.RIFE2X_V1
