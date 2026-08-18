# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Counter, Gauge, Histogram

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    codec_label,
    observe_stage_ms,
)

STAGE_SECONDS = Histogram(
    "sglang_realtime_vae_stage_seconds",
    "Realtime remote VAE stage latency.",
    ("stage", "worker_role"),
)
ACTIVE_SESSIONS = Gauge(
    "sglang_realtime_vae_active_sessions",
    "Active stateful realtime VAE sessions.",
    ("worker_role",),
)
QUEUED_CHUNKS = Gauge(
    "sglang_realtime_vae_queued_chunks",
    "Queued realtime VAE chunks.",
    ("worker_role",),
)
FREE_SLOTS = Gauge(
    "sglang_realtime_vae_free_session_slots",
    "Free realtime VAE session slots.",
    ("worker_role",),
)
BACKPRESSURE_TOTAL = Counter(
    "sglang_realtime_vae_backpressure_total",
    "Rejected realtime VAE chunks due to a full bounded queue.",
    ("worker_role",),
)

_UNIFIED_STAGE_BY_LEGACY_STAGE = {
    "queue_wait": "vae_queue",
    "decode": "vae_decode",
    "post_decode": "post_decode",
    "frame_encode": "frame_encode",
}


def observe_stage(
    stage: str,
    duration_ms: float,
    *,
    result: str = "success",
    codec: str = "none",
    scope: str = "chunk",
) -> None:
    STAGE_SECONDS.labels(stage=stage, worker_role="vae").observe(
        max(0.0, duration_ms) / 1000.0
    )
    unified_stage = _UNIFIED_STAGE_BY_LEGACY_STAGE.get(stage)
    if unified_stage is None:
        return
    if unified_stage == "frame_encode" and codec_label(codec) == "none":
        return
    observe_stage_ms(
        unified_stage,
        duration_ms,
        service="vae",
        result=result,
        codec=codec,
        scope=scope,
    )


def update_capacity(*, active: int, queued: int, maximum: int) -> None:
    ACTIVE_SESSIONS.labels(worker_role="vae").set(active)
    QUEUED_CHUNKS.labels(worker_role="vae").set(queued)
    FREE_SLOTS.labels(worker_role="vae").set(max(0, maximum - active))


def record_backpressure() -> None:
    BACKPRESSURE_TOTAL.labels(worker_role="vae").inc()
