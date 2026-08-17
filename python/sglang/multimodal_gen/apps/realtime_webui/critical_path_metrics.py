# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Prometheus metrics for the standalone realtime WebUI image.

The platform-owned WebUI Dockerfile intentionally copies only this application
directory.  Keep the collector self-contained while preserving the shared
critical-path metric name, labels, buckets, and validation contract.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CRITICAL_PATH_BUCKETS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
CRITICAL_PATH_LABELS = (
    "stage",
    "service",
    "model",
    "lane",
    "result",
    "codec",
    "scope",
)
VALID_RESULTS = frozenset(("success", "error", "timeout", "cancelled"))
VALID_SCOPES = frozenset(("request", "chunk", "frame"))
VALID_STAGES = frozenset(
    (
        "client_serialize_queue",
        "client_uplink",
        "gateway_route_admit",
        "chunk_prepare",
        "scheduler_queue",
        "vae_encode",
        "denoiser_compute",
        "latent_transfer",
        "vae_queue",
        "vae_decode",
        "post_decode",
        "h264_pre_encode_queue",
        "frame_encode",
        "ffmpeg_mux_write",
        "output_pacing_queue",
        "websocket_build_write",
        "client_downlink",
        "client_receive_queue",
        "client_video_decode",
        "client_render_wait",
    )
)

_LABEL_VALUE_RE = re.compile(r"[^A-Za-z0-9_.:/@+-]+")
_METRIC_NAME = "world_model_critical_path_stage_duration_seconds"
_HELP = "World-model critical path stage duration."


def _clean_label_value(value: Any, *, default: str, max_length: int = 120) -> str:
    label = str(value or "").strip()
    if not label:
        return default
    label = _LABEL_VALUE_RE.sub("_", label)
    return label[:max_length] or default


def _infer_model_label(value: Any = None) -> str:
    raw = str(value or "").lower()
    if "lingbot" in raw:
        return "lingbot2"
    if "minwm" in raw or "zing" in raw:
        return "minwm"
    return _clean_label_value(value, default="unknown", max_length=80)


def _service_label(default: str) -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_SERVICE")
        or os.environ.get("SERVICE_NAME")
        or default,
        default=default,
        max_length=80,
    )


def _model_label(default: Any) -> str:
    configured = os.environ.get("WORLD_MODEL_METRIC_MODEL")
    return (
        _clean_label_value(configured, default="unknown", max_length=80)
        if configured
        else _infer_model_label(default)
    )


def _lane_label(default: str) -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_LANE")
        or os.environ.get("REALTIME_LANE")
        or default,
        default="default",
        max_length=80,
    )


def _codec_label(value: Any = None) -> str:
    codec = _clean_label_value(value, default="none", max_length=32).lower()
    return codec if codec not in {"", "raw", "rgb", "rgb24"} else "none"


def result_from_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "success"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "error"


def _duration_seconds(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


@dataclass
class _HistogramValue:
    buckets: list[int] = field(
        default_factory=lambda: [0] * (len(CRITICAL_PATH_BUCKETS) + 1)
    )
    count: int = 0
    total: float = 0.0


_LOCK = threading.Lock()
_VALUES: dict[tuple[str, ...], _HistogramValue] = {}


def observe_stage_seconds(
    stage: str,
    duration_s: Any,
    *,
    service: str | None = None,
    model: str | None = None,
    lane: str | None = None,
    result: str = "success",
    codec: str | None = None,
    scope: str = "request",
) -> bool:
    stage = str(stage or "")
    result = str(result or "")
    scope = str(scope or "")
    if (
        stage not in VALID_STAGES
        or result not in VALID_RESULTS
        or scope not in VALID_SCOPES
    ):
        return False
    duration = _duration_seconds(duration_s)
    if duration is None:
        return False
    labels = (
        stage,
        _service_label(service or "unknown"),
        _model_label(model or "unknown"),
        _lane_label(lane or "default"),
        result,
        _codec_label(codec),
        scope,
    )
    with _LOCK:
        value = _VALUES.setdefault(labels, _HistogramValue())
        for index, upper_bound in enumerate(CRITICAL_PATH_BUCKETS):
            if duration <= upper_bound:
                value.buckets[index] += 1
        value.buckets[-1] += 1
        value.count += 1
        value.total += duration
    return True


def observe_client_metric_event(
    event: dict[str, Any],
    *,
    service: str,
    model: str | None = None,
    lane: str | None = None,
) -> bool:
    if not isinstance(event, dict):
        return False
    duration_s = event.get("duration_s")
    if duration_s is None and event.get("duration_ms") is not None:
        duration_ms = _duration_seconds(event.get("duration_ms"))
        duration_s = None if duration_ms is None else duration_ms / 1000.0
    return observe_stage_seconds(
        str(event.get("stage") or ""),
        duration_s,
        service=service,
        model=model or event.get("model"),
        lane=lane or event.get("lane"),
        result=str(event.get("result") or "success"),
        codec=event.get("codec"),
        scope=str(event.get("scope") or "request"),
    )


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(values: Iterable[str], *, le: str | None = None) -> str:
    pairs = [
        f'{name}="{_escape_label(value)}"'
        for name, value in zip(CRITICAL_PATH_LABELS, values, strict=True)
    ]
    if le is not None:
        pairs.append(f'le="{le}"')
    return "{" + ",".join(pairs) + "}"


def prometheus_latest() -> bytes:
    lines = [f"# HELP {_METRIC_NAME} {_HELP}", f"# TYPE {_METRIC_NAME} histogram"]
    with _LOCK:
        snapshot = [
            (labels, list(value.buckets), value.count, value.total)
            for labels, value in sorted(_VALUES.items())
        ]
    for labels, buckets, count, total in snapshot:
        for upper_bound, bucket_count in zip(
            (*CRITICAL_PATH_BUCKETS, math.inf), buckets, strict=True
        ):
            le = "+Inf" if math.isinf(upper_bound) else f"{upper_bound:g}"
            lines.append(
                f"{_METRIC_NAME}_bucket{_format_labels(labels, le=le)} {bucket_count}"
            )
        label_text = _format_labels(labels)
        lines.append(f"{_METRIC_NAME}_sum{label_text} {total:.17g}")
        lines.append(f"{_METRIC_NAME}_count{label_text} {count}")
    return ("\n".join(lines) + "\n").encode()


def prometheus_content_type() -> str:
    return "text/plain; version=0.0.4; charset=utf-8"
