# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics for the realtime world-model critical path."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest

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
    1,
    2.5,
    5,
    10,
    30,
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
        "vae_actor_wait",
        "vae_decode",
        "post_decode",
        "frame_interpolation",
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

CRITICAL_PATH_STAGE_SECONDS = Histogram(
    "world_model_critical_path_stage_duration_seconds",
    "World-model critical path stage duration.",
    CRITICAL_PATH_LABELS,
    buckets=CRITICAL_PATH_BUCKETS,
)
COORDINATOR_WORKER_SELECTION_SECONDS = Histogram(
    "world_model_coordinator_worker_selection_duration_seconds",
    "Realtime Coordinator worker selection latency.",
    ("service", "model", "lane", "result"),
    buckets=CRITICAL_PATH_BUCKETS,
)
COORDINATOR_CAPACITY_RESERVATION_SECONDS = Histogram(
    "world_model_coordinator_capacity_reservation_duration_seconds",
    "Realtime Coordinator worker capacity reservation latency.",
    ("service", "model", "lane", "result", "worker_role"),
    buckets=CRITICAL_PATH_BUCKETS,
)

_LABEL_VALUE_RE = re.compile(r"[^A-Za-z0-9_.:/@+-]+")


def _clean_label_value(value: Any, *, default: str, max_length: int = 120) -> str:
    label = str(value or "").strip()
    if not label:
        return default
    label = _LABEL_VALUE_RE.sub("_", label)
    return label[:max_length] or default


def infer_model_label(value: Any = None) -> str:
    raw = str(value or "").lower()
    if "lingbot" in raw:
        return "lingbot2"
    if "minwm" in raw or "zing" in raw:
        return "minwm"
    return _clean_label_value(value, default="unknown", max_length=80)


def service_label(default: str = "unknown") -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_SERVICE")
        or os.environ.get("SERVICE_NAME")
        or default,
        default=default,
        max_length=80,
    )


def model_label(default: Any = "unknown") -> str:
    configured = os.environ.get("WORLD_MODEL_METRIC_MODEL")
    if configured:
        return _clean_label_value(configured, default="unknown", max_length=80)
    return infer_model_label(default)


def lane_label(default: str = "default") -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_LANE")
        or os.environ.get("REALTIME_LANE")
        or default,
        default="default",
        max_length=80,
    )


def codec_label(value: Any = None) -> str:
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


def _coerce_duration_seconds(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


@lru_cache(maxsize=4096)
def _critical_path_child(
    stage: str,
    service: str,
    model: str,
    lane: str,
    result: str,
    codec: str,
    scope: str,
):
    return CRITICAL_PATH_STAGE_SECONDS.labels(
        stage=stage,
        service=service,
        model=model,
        lane=lane,
        result=result,
        codec=codec,
        scope=scope,
    )


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
    """Observe one critical-path stage without allowing metrics to break traffic."""

    stage = str(stage or "")
    result = str(result or "")
    scope = str(scope or "")
    if (
        stage not in VALID_STAGES
        or result not in VALID_RESULTS
        or scope not in VALID_SCOPES
    ):
        return False
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return False
    try:
        _critical_path_child(
            stage,
            service_label(service or "unknown"),
            model_label(model or "unknown"),
            lane_label(lane or "default"),
            result,
            codec_label(codec),
            scope,
        ).observe(duration)
    except Exception:
        return False
    return True


def observe_stage_ms(stage: str, duration_ms: Any, **labels: Any) -> bool:
    duration = _coerce_duration_seconds(duration_ms)
    if duration is None:
        return False
    return observe_stage_seconds(stage, duration / 1000.0, **labels)


@contextmanager
def stage_timer(stage: str, **labels: Any) -> Iterator[None]:
    started = time.perf_counter()
    result = "success"
    try:
        yield
    except BaseException as exc:
        result = result_from_exception(exc)
        raise
    finally:
        observe_stage_seconds(
            stage,
            time.perf_counter() - started,
            result=result,
            **labels,
        )


def observe_client_metric_event(
    event: dict[str, Any],
    *,
    service: str,
    model: str | None = None,
    lane: str | None = None,
) -> bool:
    """Validate and observe a browser-originated metric event."""

    if not isinstance(event, dict):
        return False
    duration_s = event.get("duration_s")
    if duration_s is None and event.get("duration_ms") is not None:
        duration_ms = _coerce_duration_seconds(event.get("duration_ms"))
        duration_s = None if duration_ms is None else duration_ms / 1000.0
    return observe_stage_seconds(
        str(event.get("stage") or ""),
        duration_s,
        service=service,
        model=model or str(event.get("model") or "unknown"),
        lane=lane or str(event.get("lane") or "default"),
        result=str(event.get("result") or "success"),
        codec=str(event.get("codec") or "none"),
        scope=str(event.get("scope") or "request"),
    )


@lru_cache(maxsize=128)
def _coordinator_selection_child(
    service: str,
    model: str,
    lane: str,
    result: str,
):
    return COORDINATOR_WORKER_SELECTION_SECONDS.labels(
        service=service,
        model=model,
        lane=lane,
        result=result,
    )


@lru_cache(maxsize=256)
def _coordinator_reservation_child(
    service: str,
    model: str,
    lane: str,
    result: str,
    worker_role: str,
):
    return COORDINATOR_CAPACITY_RESERVATION_SECONDS.labels(
        service=service,
        model=model,
        lane=lane,
        result=result,
        worker_role=worker_role,
    )


def observe_coordinator_worker_selection_seconds(
    duration_s: Any,
    *,
    model: str | None = None,
    lane: str | None = None,
    result: str = "success",
) -> bool:
    if result not in VALID_RESULTS:
        return False
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return False
    try:
        _coordinator_selection_child(
            service_label("coordinator"),
            model_label(model or "unknown"),
            lane_label(lane or "default"),
            result,
        ).observe(duration)
    except Exception:
        return False
    return True


def observe_coordinator_capacity_reservation_seconds(
    duration_s: Any,
    *,
    model: str | None = None,
    lane: str | None = None,
    result: str = "success",
    worker_role: str = "unknown",
) -> bool:
    if result not in VALID_RESULTS:
        return False
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return False
    try:
        _coordinator_reservation_child(
            service_label("coordinator"),
            model_label(model or "unknown"),
            lane_label(lane or "default"),
            result,
            _clean_label_value(worker_role, default="unknown", max_length=32),
        ).observe(duration)
    except Exception:
        return False
    return True


def prometheus_latest() -> bytes:
    return generate_latest()


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST
