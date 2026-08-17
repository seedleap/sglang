# SPDX-License-Identifier: Apache-2.0

"""Shared realtime request-mode inference for Gateway and media bridges."""

from __future__ import annotations

from typing import Any, Mapping


def infer_generation_mode(message: Mapping[str, Any]) -> str:
    """Mirror adapter inference when older clients omit generation_mode."""

    explicit = message.get("generation_mode") or message.get("mode")
    if explicit:
        return str(explicit).strip().lower()
    return "i2v" if message.get("first_frame") is not None else "t2v"


def init_requests_finite_output(message: Mapping[str, Any]) -> bool:
    """Return whether an init promises one finite, complete media timeline.

    ``num_frames`` is a per-chunk shape for I2V, so only ``max_chunks`` makes
    I2V finite. T2V derives its terminal chunk count from ``num_frames`` in the
    adapter and is therefore finite even when the client omits ``max_chunks``.
    """

    if not isinstance(message, Mapping) or message.get("type") != "init":
        return False
    max_chunks = _positive_int(message.get("max_chunks"))
    num_frames = _positive_int(message.get("num_frames"))
    return max_chunks > 0 or (
        infer_generation_mode(message) == "t2v" and num_frames > 0
    )


def expected_final_chunk_from_init(message: Mapping[str, Any]) -> int | None:
    """Return a client-declared final chunk index when it is explicit."""

    max_chunks = _positive_int(message.get("max_chunks"))
    return max_chunks - 1 if max_chunks > 0 else None


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
