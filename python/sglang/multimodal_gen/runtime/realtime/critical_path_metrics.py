# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics for the realtime world-model critical path."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

CRITICAL_PATH_BUCKETS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1,
    1.25,
    1.5,
    2,
    2.5,
    3,
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
VALID_MODELS = frozenset(("lingbot2", "wan"))
VALID_CODECS = frozenset(("none", "webp", "h264", "jpeg"))
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
METRIC_OBSERVATIONS_DROPPED_TOTAL = Counter(
    "world_model_metric_observations_dropped_total",
    "Rejected world-model metric observations by low-cardinality reason.",
    ("reason",),
)

_LABEL_VALUE_RE = re.compile(r"[^A-Za-z0-9_.:/@+-]+")


def _clean_label_value(value: Any, *, default: str, max_length: int = 120) -> str:
    label = str(value or "").strip()
    if not label:
        return default
    label = _LABEL_VALUE_RE.sub("_", label)
    return label[:max_length] or default


def infer_model_label(value: Any = None) -> str | None:
    raw = str(value or "").lower()
    if "lingbot" in raw:
        return "lingbot2"
    if "minwm" in raw or "zing" in raw or "wan" in raw:
        return "wan"
    return None


def service_label(default: str = "unknown") -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_SERVICE")
        or os.environ.get("SERVICE_NAME")
        or default,
        default=default,
        max_length=80,
    )


def model_label(default: Any = None) -> str | None:
    for candidate in (
        os.environ.get("WORLD_MODEL_METRIC_MODEL"),
        default,
        os.environ.get("WORLD_MODEL_METRIC_SERVICE"),
        os.environ.get("SERVICE_NAME"),
        os.environ.get("HOSTNAME"),
    ):
        label = infer_model_label(candidate)
        if label in VALID_MODELS:
            return label
    return None


def lane_label(default: str = "default") -> str:
    return _clean_label_value(
        os.environ.get("WORLD_MODEL_METRIC_LANE")
        or os.environ.get("REALTIME_LANE")
        or default,
        default="default",
        max_length=80,
    )


def codec_label(value: Any = None) -> str | None:
    codec = _clean_label_value(value, default="none", max_length=32).lower()
    codec = {
        "": "none",
        "raw": "none",
        "rgb": "none",
        "rgb24": "none",
        "avc": "h264",
        "image/webp": "webp",
        "image/jpeg": "jpeg",
    }.get(codec, codec)
    return codec if codec in VALID_CODECS else None


def _drop(reason: str) -> bool:
    METRIC_OBSERVATIONS_DROPPED_TOTAL.labels(reason=reason).inc()
    return False


def _structured_metric_logs_enabled() -> bool:
    value = os.environ.get("WORLD_MODEL_METRIC_STRUCTURED_LOGS", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _emit_structured_metric_event(metric: str, duration: float, **fields: str) -> None:
    """Emit one bounded JSON line that the existing Vector agent can ingest."""

    if not _structured_metric_logs_enabled():
        return
    payload = {
        "event": "world_model_metric",
        "schema_version": 1,
        "level": "info",
        "metric": metric,
        "observed_at_epoch_ms": int(time.time() * 1000),
        "duration_ms": round(duration * 1000, 6),
        **fields,
    }
    try:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        os.write(sys.stdout.fileno(), line.encode("utf-8"))
    except (OSError, TypeError, ValueError):
        # Metrics and telemetry must never break a realtime request.
        pass


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
        return _drop("invalid_dimension")
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return _drop("invalid_duration")
    service_value = service_label(service or "unknown")
    model_value = model_label(model)
    codec_value = codec_label(codec)
    if model_value is None:
        return _drop("invalid_model")
    if codec_value is None:
        return _drop("invalid_codec")
    lane_value = lane_label(lane or "default")
    try:
        _critical_path_child(
            stage,
            service_value,
            model_value,
            lane_value,
            result,
            codec_value,
            scope,
        ).observe(duration)
        _emit_structured_metric_event(
            "world_model_critical_path_stage_duration_seconds",
            duration,
            stage=stage,
            component=service_value,
            model=model_value,
            lane=lane_value,
            result=result,
            codec=codec_value,
            scope=scope,
        )
    except Exception:
        return _drop("internal_error")
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
        model=model or event.get("model"),
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
        return _drop("invalid_dimension")
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return _drop("invalid_duration")
    service_value = service_label("coordinator")
    model_value = model_label(model)
    lane_value = lane_label(lane or "default")
    if model_value is None:
        return _drop("invalid_model")
    try:
        _coordinator_selection_child(
            service_value,
            model_value,
            lane_value,
            result,
        ).observe(duration)
        _emit_structured_metric_event(
            "world_model_coordinator_worker_selection_duration_seconds",
            duration,
            component=service_value,
            model=model_value,
            lane=lane_value,
            result=result,
        )
    except Exception:
        return _drop("internal_error")
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
        return _drop("invalid_dimension")
    duration = _coerce_duration_seconds(duration_s)
    if duration is None:
        return _drop("invalid_duration")
    service_value = service_label("coordinator")
    model_value = model_label(model)
    lane_value = lane_label(lane or "default")
    worker_role_value = _clean_label_value(
        worker_role, default="unknown", max_length=32
    )
    if model_value is None:
        return _drop("invalid_model")
    try:
        _coordinator_reservation_child(
            service_value,
            model_value,
            lane_value,
            result,
            worker_role_value,
        ).observe(duration)
        _emit_structured_metric_event(
            "world_model_coordinator_capacity_reservation_duration_seconds",
            duration,
            component=service_value,
            model=model_value,
            lane=lane_value,
            result=result,
            worker_role=worker_role_value,
        )
    except Exception:
        return _drop("internal_error")
    return True


def prometheus_latest() -> bytes:
    return generate_latest()


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST
