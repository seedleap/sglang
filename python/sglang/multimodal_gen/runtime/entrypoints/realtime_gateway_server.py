# SPDX-License-Identifier: Apache-2.0

"""Public realtime Gateway for Coordinator-routed Denoiser/VAE sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
import msgspec.msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    SessionAssignment,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    infer_model_label,
    observe_client_metric_event,
    observe_stage_seconds,
    prometheus_content_type,
    prometheus_latest,
    result_from_exception,
)
from sglang.multimodal_gen.runtime.realtime.gateway import (
    AdmissionQueueFull,
    BoundedAdmissionWaiterGate,
    BrowserPlaybackAckWindow,
    GatewayOutputRegistry,
    OutputBackpressureError,
    OutputProtocolError,
    OutputRouteClosed,
    build_denoiser_url,
    worker_message_allowed,
    worker_message_type,
)
from sglang.multimodal_gen.runtime.realtime.world_directions import (
    DirectionCoordinator,
    parse_init_directions,
)
from sglang.multimodal_gen.runtime.realtime.world_platform import (
    Principal,
    SessionPayloadSealer,
    TokenError,
    TokenReplayGuard,
    WorldCallbacks,
    WorldPlatformConfig,
    verify_session_token,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    compact_client_trace_event,
    emit_realtime_trace_payload,
    normalize_trace_id,
)

WEBUI_ROOT = Path(__file__).resolve().parents[2] / "apps" / "realtime_webui"
logger = logging.getLogger(__name__)
_IDEMPOTENT_COORDINATOR_RELEASE_REASONS = frozenset(
    {
        "LEASE_LOST",
        "WORKER_LOST",
    }
)
_WORKER_SLOT_FIELDS = frozenset(field.name for field in fields(WorkerSlot))


def _parse_ui_config(raw: str) -> dict[str, Any]:
    try:
        config = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("UI config must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("UI config must be a JSON object")
    return config


def _log_gateway_trace(trace_id: str, event: str, **fields: Any) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "trace_id": trace_id,
        "event": event,
        "server_epoch_ms": now_ms,
        "trace_seq": now_ms * 1000 + time.perf_counter_ns() % 1000,
        **fields,
    }
    emit_realtime_trace_payload(logger, payload)


def _browser_send_trace_fields(wire: bytes) -> dict[str, Any]:
    fields: dict[str, Any] = {"wire_bytes": len(wire)}
    try:
        message = decode_message(wire)
    except ProtocolViolation:
        fields["message_type"] = "invalid"
        return fields

    fields["message_type"] = message.get("type")
    for name in (
        "request_id",
        "chunk_index",
        "frame_batch_index",
        "num_frame_batches",
        "is_final_frame_batch",
        "event_id",
        "action_version",
        "prompt_version",
        "num_frames",
        "content_type",
        "encoding",
        "width",
        "height",
        "source_width",
        "source_height",
        "preview_width",
        "preview_height",
    ):
        value = message.get(name)
        if value is not None:
            fields[name] = value

    payload = message.get("payload")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        fields["payload_bytes"] = len(payload)
    payload_lengths = message.get("payload_lengths")
    if isinstance(payload_lengths, list):
        lengths: list[int] = []
        for item in payload_lengths:
            try:
                lengths.append(int(item))
            except (TypeError, ValueError):
                continue
        if lengths:
            fields["payload_bytes"] = sum(lengths)
            fields["payload_count"] = len(lengths)
    return fields


def _codec_from_message(message: dict[str, Any]) -> str:
    codec = str(message.get("codec") or message.get("encoding") or "").lower()
    content_type = str(message.get("content_type") or "").lower()
    if "h264" in codec or "h264" in content_type or "avc" in content_type:
        return "h264"
    if "webp" in codec or "webp" in content_type:
        return "webp"
    if "jpeg" in codec or "jpg" in codec or "jpeg" in content_type:
        return "jpeg"
    return codec or "none"


def _metric_scope_for_message(message: dict[str, Any]) -> str:
    message_type = message.get("type")
    if message_type in {"frame_batch", "media_batch", "media_payload"}:
        return "frame"
    if message_type in {"chunk_telemetry", "media_chunk_complete"}:
        return "chunk"
    return "request"


def _metric_labels_from_wire(wire: bytes) -> tuple[str, str]:
    try:
        message = decode_message(wire)
    except ProtocolViolation:
        return "none", "request"
    return _codec_from_message(message), _metric_scope_for_message(message)


def _observe_gateway_client_metric(
    message: dict[str, Any],
    *,
    model: str,
) -> int:
    if message.get("type") == "client_metric":
        return int(observe_client_metric_event(message, service="gateway", model=model))
    if message.get("type") != "client_metric_batch":
        return 0
    events = message.get("events")
    if not isinstance(events, list):
        return 0
    accepted = 0
    for event in events[:64]:
        accepted += int(
            observe_client_metric_event(event, service="gateway", model=model)
        )
    return accepted


def _worker_slot_from_payload(payload: dict[str, Any]) -> WorkerSlot:
    return WorkerSlot(
        **{key: value for key, value in payload.items() if key in _WORKER_SLOT_FIELDS}
    )


class CoordinatorClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def admit(self, **request: Any) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...


def _assignment(payload: dict[str, Any]) -> SessionAssignment:
    return SessionAssignment(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        generation_id=payload["generation_id"],
        token=payload["token"],
        expires_at=float(payload["expires_at"]),
        denoiser=_worker_slot_from_payload(payload["denoiser"]),
        vae=_worker_slot_from_payload(payload["vae"]),
    )


class HTTPCoordinatorClient:
    def __init__(self, base_url: str, *, timeout_s: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        )

    @staticmethod
    def _raise_rejection(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        reason = detail.get("reason") or f"COORDINATOR_HTTP_{response.status_code}"
        raise CoordinatorRejected(
            reason,
            retry_after_s=detail.get("retry_after_s"),
        )

    async def admit(self, **request: Any) -> SessionAssignment:
        response = await self._client.post("/v1/sessions/admit", json=request)
        self._raise_rejection(response)
        return _assignment(response.json())

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        response = await self._client.post(
            "/v1/sessions/renew", json=asdict(assignment)
        )
        self._raise_rejection(response)
        return _assignment(response.json())

    async def release(self, assignment: SessionAssignment) -> None:
        response = await self._client.request(
            "DELETE", "/v1/sessions/release", json=asdict(assignment)
        )
        if response.status_code == 404:
            return
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", {})
            except (ValueError, AttributeError):
                detail = {}
            if detail.get("reason") in _IDEMPOTENT_COORDINATOR_RELEASE_REASONS:
                return
        self._raise_rejection(response)

    async def close(self) -> None:
        await self._client.aclose()


class _BrowserSender:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._lock = asyncio.Lock()

    async def send(self, payload: bytes | str) -> None:
        async with self._lock:
            if isinstance(payload, bytes):
                await self.websocket.send_bytes(payload)
            else:
                await self.websocket.send_text(payload)

    async def error(self, content: str, **fields: Any) -> None:
        await self.send(encode_message("error", content=content, **fields))


def _user_id(websocket: WebSocket) -> str:
    query = websocket.query_params.get("user_id")
    if query:
        return f"query:{query[:240]}"
    header = websocket.headers.get("x-user-id")
    if header:
        return f"header:{header[:240]}"
    client = websocket.client.host if websocket.client else "unknown"
    return f"client:{client}"


async def _receive_browser(websocket: WebSocket) -> bytes | str:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if message.get("bytes") is not None:
        return message["bytes"]
    if message.get("text") is not None:
        return message["text"]
    raise WebSocketDisconnect(1002)


# Inbound allowlist for authorized sessions. This is an allowlist rather than a
# denylist so future engine control fields do not become browser-injectable just
# because they were not explicitly blocked.
_WORLD_PASSTHROUGH_TYPES = frozenset({"client_metric", "client_metric_batch", "ack"})
# Only kinds the MinWM adapter actually ingests. "camera", "move", and "action"
# used to sit here: MinWM.ingest_event rejects all three (it accepts
# action_labels, action_weights, camera_actions, prompt, scene_cut, seed, and
# chunk_seeds), so forwarding them produced an engine-side "unsupported MinWM
# event kind" error instead of doing anything. They were dead slots that made
# the contract look wider than it is.
_WORLD_ALLOWED_EVENT_KINDS = frozenset({"camera_actions", "playback_ack", "heartbeat"})
# Free-form input no longer passes raw prompt/scene_cut through. Full scene
# descriptions are server-side assets; the browser sends only the user's raw
# instruction as kind:"direction", then the gateway asks world-service to rewrite
# and dispatch the full prompt. The direction length cap matches world-service.
_WORLD_DIRECTION_MAX_CHARS = 2000
# Cap on in-flight direction/skill tasks per session. The coordinator already
# collapses a burst into "one in flight plus one queued", so normal play stays
# in the single digits; anything above this is scripted frame spam and is
# dropped. Without a cap the receive loop creates one task per frame and the
# frame rate is chosen by the attacker.
_WORLD_MAX_DIRECTION_TASKS = 8

# Init-message extension block visible only to the gateway. This matches the
# world-service zingproto.WorldExtKey value.
WORLD_EXT_KEY = "_world"


async def _cancel_tasks(tasks: set[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app(
    coordinator: CoordinatorClient,
    *,
    model_revision: str,
    vae_fingerprint: str,
    internal_output_url: str,
    lingbot2_upstream_url: str | None = None,
    lingbot2_model_revision: str = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
    lingbot2_vae_fingerprint: str | None = None,
    output_queue_depth: int = 64,
    output_enqueue_timeout_s: float = 0.0,
    output_queue_max_bytes: int = 16 * 1024 * 1024,
    output_queue_max_messages: int = 256,
    output_drain_timeout_s: float = 5.0,
    lease_renew_interval_s: float = 10.0,
    release_grace_s: float = 0.5,
    max_admission_waiters: int = 64,
    readiness_coordinator_timeout_s: float = 1.0,
    readiness_coordinator_grace_s: float = 30.0,
    readiness_clock=time.monotonic,
    connect_factory=connect,
    ui_config: dict[str, Any] | None = None,
    trace_query=None,
    world_platform: WorldPlatformConfig | None = None,
    browser_send_timeout_s: float = 15.0,
    # Unauthenticated showcase routes, static pages, and trace queries. These are
    # registered by default.
    #
    # These routes must not be public. They can create sessions without lifetime,
    # queueing, or auth limits, bypassing all authorized_generate constraints.
    # The boundary is intentionally the network layer, not this process: public
    # browsers should see only authorized_generate through ingress, while the
    # internal web UI still needs these legacy routes through the cluster Service.
    # A process-level switch cannot distinguish those two traffic classes.
    #
    # Correct production isolation: expose only authorized_generate on public
    # ingress. This switch is for deployments dedicated to platform traffic with
    # no showcase traffic at all.
    disable_legacy_routes: bool = False,
) -> FastAPI:
    if release_grace_s < 0:
        raise ValueError("release_grace_s must be non-negative")
    if browser_send_timeout_s <= 0:
        raise ValueError("browser_send_timeout_s must be positive")
    # World-platform callback client. Without it, the authorized path is disabled.
    enable_legacy_routes = not disable_legacy_routes
    world_callbacks = WorldCallbacks(world_platform) if world_platform else None
    # Session-payload unsealer. It uses the callback HMAC shared secret, with
    # HKDF deriving an independent sub-key.
    payload_sealer = (
        SessionPayloadSealer(world_platform.callback_secret) if world_platform else None
    )
    if output_drain_timeout_s <= 0:
        raise ValueError("output_drain_timeout_s must be positive")
    if readiness_coordinator_timeout_s <= 0:
        raise ValueError("readiness_coordinator_timeout_s must be positive")
    if readiness_coordinator_grace_s < 0:
        raise ValueError("readiness_coordinator_grace_s must be non-negative")
    registry = GatewayOutputRegistry(
        queue_depth=output_queue_depth,
        enqueue_timeout_s=output_enqueue_timeout_s,
        max_queue_bytes=output_queue_max_bytes,
        max_queue_messages=output_queue_max_messages,
    )
    admission_gate = BoundedAdmissionWaiterGate(max_waiters=max_admission_waiters)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        closer = getattr(coordinator, "close", None)
        if closer is not None:
            await closer()

    app = FastAPI(title="SGLang Realtime Gateway", lifespan=lifespan)
    app.state.output_registry = registry
    app.state.admission_gate = admission_gate
    last_coordinator_ready_at: float | None = None

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(prometheus_latest(), media_type=prometheus_content_type())

    @app.get("/readyz")
    async def readyz():
        nonlocal last_coordinator_ready_at
        try:
            await asyncio.wait_for(
                coordinator.health(),
                timeout=readiness_coordinator_timeout_s,
            )
        except Exception as exc:
            now = readiness_clock()
            if (
                last_coordinator_ready_at is not None
                and now - last_coordinator_ready_at <= readiness_coordinator_grace_s
            ):
                return {
                    "status": "ready",
                    "coordinator": "degraded",
                    "last_success_age_s": round(now - last_coordinator_ready_at, 3),
                }
            raise HTTPException(
                status_code=503, detail="coordinator unavailable"
            ) from exc
        last_coordinator_ready_at = readiness_clock()
        return {"status": "ready", "coordinator": "ready"}

    @app.get("/backends/minwm/v1/models")
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_revision}]}

    @app.get("/backends/lingbot2/v1/models")
    async def lingbot2_models():
        return {"object": "list", "data": [{"id": lingbot2_model_revision}]}

    # Internal diagnostics: returns worker id, engine hostname, adapter class, and
    # other implementation details. It shares the showcase-route switch; public
    # access must be blocked by ingress along with /metrics and /v1/models.
    if enable_legacy_routes:

        @app.get("/v1/realtime_video/traces/{trace_id}")
        async def get_trace(trace_id: str, after: int = 0, limit: int = 220):
            if trace_query is None:
                raise HTTPException(
                    status_code=503, detail="Trace query is not configured"
                )
            try:
                return await trace_query.query(trace_id, after=after, limit=limit)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Trace query failed for trace_id=%s", trace_id)
                raise HTTPException(
                    status_code=503, detail="trace query unavailable"
                ) from exc

    @app.post("/v1/realtime_video/traces/{trace_id}/client-events")
    async def post_client_trace(trace_id: str, payload: dict):
        normalized = normalize_trace_id(trace_id, fallback="")
        if not normalized or normalized != trace_id:
            raise HTTPException(status_code=400, detail="invalid trace_id")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > 64:
            raise HTTPException(
                status_code=400, detail="events must contain at most 64 items"
            )
        accepted = 0
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event = compact_client_trace_event(raw_event)
            event["trace_id"] = trace_id
            event["event"] = str(event.pop("name", "client.metric"))[:128]
            event["server_epoch_ms"] = int(time.time() * 1000)
            event["trace_seq"] = (
                event["server_epoch_ms"] * 1000 + int(event.get("seq") or 0) % 1000
            )
            emit_realtime_trace_payload(logger, event)
            accepted += 1
        return {"accepted": accepted}

    @app.get("/runtime-config.js")
    async def runtime_config():
        config = ui_config or {}
        body = f"globalThis.SGLANG_REALTIME_UI_CONFIG = {json.dumps(config)};\n"
        return Response(
            body,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/v1/internal/realtime_output")
    async def realtime_output(websocket: WebSocket):
        await websocket.accept()
        route = None
        session_id = ""
        generation_id = ""
        output_token = ""
        try:
            opened = decode_message(await websocket.receive_bytes())
            if opened.get("type") != "session_output_open":
                raise OutputProtocolError("session_output_open is required")
            session_id = str(opened.get("session_id") or "")
            generation_id = str(opened.get("generation_id") or "")
            output_token = str(opened.get("token") or "")
            route = await registry.bind(
                session_id,
                generation_id,
                token=output_token,
            )
            await websocket.send_bytes(
                encode_message(
                    "session_output_accepted",
                    session_id=session_id,
                    generation_id=generation_id,
                )
            )
            while True:
                wire = await websocket.receive_bytes()
                message = decode_message(wire)
                enqueue_started = time.perf_counter()
                await route.put(wire)
                _log_gateway_trace(
                    route.trace_id,
                    "gateway.output_enqueued",
                    session_id=session_id,
                    generation_id=generation_id,
                    enqueue_ms=round((time.perf_counter() - enqueue_started) * 1000, 3),
                    **route.queue_metrics(),
                    **_browser_send_trace_fields(wire),
                )
                if message.get("type") == "media_chunk_complete":
                    await websocket.send_bytes(
                        encode_message(
                            "media_chunk_complete_accepted",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=message.get("request_id"),
                            chunk_index=message.get("chunk_index"),
                        )
                    )
        except (WebSocketDisconnect, OutputRouteClosed):
            pass
        except OutputBackpressureError as exc:
            await websocket.close(code=1013, reason=str(exc))
        except (OutputProtocolError, ProtocolViolation) as exc:
            await websocket.close(code=1008, reason=str(exc))
        finally:
            if route is not None:
                await registry.unbind(session_id, generation_id, token=output_token)

    async def _generate_coordinator_session(
        websocket: WebSocket,
        *,
        selected_model_revision: str,
        selected_vae_fingerprint: str,
        principal: Principal | None = None,
    ) -> None:
        # principal set means an authorized world-platform session; D1-D4 apply
        # only to this branch. principal unset means the legacy showcase route,
        # whose behavior stays byte-for-byte compatible.
        await websocket.accept()
        sender = _BrowserSender(websocket)
        session_id = uuid4().hex
        generation_id = uuid4().hex
        trace_id = normalize_trace_id(
            websocket.query_params.get("trace_id"), fallback=session_id
        )
        output_token = secrets.token_urlsafe(32)
        metric_model = infer_model_label(selected_model_revision)
        assignment = None
        route = None
        upstream = None
        tasks: set[asyncio.Task] = set()
        expected_last_chunk: int | None = None
        playback_ack_window = BrowserPlaybackAckWindow()
        # Branch-local state for world sessions: init verification, deadline, and
        # rejection reason.
        world_session: dict[str, Any] = {"init_verified": False}
        # Player-direction orchestrator created after init validation. Dynamic
        # tasks are tracked separately and canceled together in finally.
        directions: DirectionCoordinator | None = None
        direction_tasks: set[asyncio.Task] = set()
        upstream_send_lock = asyncio.Lock()
        try:
            admitted_at = time.perf_counter()
            _log_gateway_trace(trace_id, "gateway.ws_accepted", session_id=session_id)
            try:
                async with admission_gate.waiter():
                    assignment = await coordinator.admit(
                        # D1: authorized sessions use the internal user id from
                        # the token. The wsvc: prefix isolates them from showcase.
                        # Key the lease on the account-level pseudonym rather
                        # than the per-run one: the coordinator's one-lease-per-
                        # user rule is meant to bound a person, and the per-run
                        # pseudonym changes with every run, so clearing the
                        # business-side ledger once lets the same person take a
                        # second seat while the old one still holds a GPU.
                        # Tokens without that claim fall back to the per-run
                        # pseudonym, so behaviour is unchanged before the
                        # world-service rollout.
                        user_id=(
                            f"wsvc:{principal.lease_key}"
                            if principal is not None
                            else _user_id(websocket)
                        ),
                        session_id=session_id,
                        generation_id=generation_id,
                        model_revision=selected_model_revision,
                        vae_fingerprint=selected_vae_fingerprint,
                        wait_for_capacity=True,
                        trace_id=trace_id,
                    )
            except BaseException as exc:
                observe_stage_seconds(
                    "gateway_route_admit",
                    time.perf_counter() - admitted_at,
                    service="gateway",
                    model=metric_model,
                    result=result_from_exception(exc),
                    scope="request",
                )
                raise
            observe_stage_seconds(
                "gateway_route_admit",
                time.perf_counter() - admitted_at,
                service="gateway",
                model=metric_model,
                result="success",
                scope="request",
            )
            _log_gateway_trace(
                trace_id,
                "gateway.coordinator_admit_complete",
                session_id=session_id,
                coordinator_admit_ms=round(
                    (time.perf_counter() - admitted_at) * 1000, 3
                ),
                denoiser_worker_id=assignment.denoiser.worker_id,
                vae_worker_id=assignment.vae.worker_id,
            )
            route = await registry.register(
                session_id,
                generation_id,
                token=output_token,
                trace_id=trace_id,
            )
            upstream_url = build_denoiser_url(
                assignment.denoiser.endpoint,
                session_id=session_id,
                generation_id=generation_id,
                coordinator_token=assignment.token,
                worker_epoch=assignment.denoiser.worker_epoch,
                vae_url=assignment.vae.endpoint,
                vae_worker_epoch=assignment.vae.worker_epoch,
                output_url=internal_output_url,
                output_token=output_token,
                trace_id=trace_id,
            )
            upstream = await connect_factory(
                upstream_url,
                max_size=None,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=20,
            )
            _log_gateway_trace(
                trace_id,
                "gateway.denoiser_connected",
                session_id=session_id,
            )

            async def _upstream_send(data) -> None:
                # Direction/skill tasks and browser forwarding can write the same
                # upstream connection concurrently. websockets does not permit
                # concurrent send calls, so serialize them with a lock.
                async with upstream_send_lock:
                    await upstream.send(data)

            async def _direction_rewrite(text: str, baseline: str) -> tuple[str, str]:
                # Rewrite is delegated to world-service, which owns credentials
                # and prompt assets. It uses the same HMAC-signed internal path
                # as lifecycle callbacks.
                try:
                    data = await world_callbacks.rewrite(
                        principal.run_id, text, baseline
                    )
                except Exception:
                    logger.warning(
                        "direction rewrite failed run_id=%s",
                        principal.run_id,
                        exc_info=True,
                    )
                    raise
                # Default change_type to one_time. That self-heals after the
                # revert if guessed wrong, while an accidental persistent mark
                # would burn a transient effect into the baseline.
                return str(data["prompt"]), str(data.get("change_type") or "one_time")

            async def _direction_dispatch(
                prompt: str, event_id, kind: str = "prompt"
            ) -> None:
                # The engine reads RealtimeEvent.payload. Passing event_id through
                # lets control_ack, frame_batch, and chunk_telemetry correlate
                # back to this player input. kind picks the transition and is
                # decided server side (compiled world), never by the browser.
                fields: dict[str, Any] = {"kind": kind, "payload": prompt}
                if event_id is not None:
                    fields["event_id"] = event_id
                await _upstream_send(encode_message("event", **fields))

            async def _direction_notify(event_id, status: str) -> None:
                # rewriting/failed/superseded are gateway-produced browser events,
                # not engine events, so they bypass the engine allowlist.
                await asyncio.wait_for(
                    sender.send(
                        encode_message(
                            "direction_status", event_id=event_id, status=status
                        )
                    ),
                    timeout=browser_send_timeout_s,
                )

            def _spawn_direction(coro, label: str) -> None:
                # Rewrite takes roughly one to two seconds, so run it in an
                # independent task and do not block the browser_to_worker receive
                # loop. Keep strong references in direction_tasks and cancel all
                # in finally. Exceptions are logged only; the main task group owns
                # session lifetime and one rewrite failure should not tear down
                # the session.
                if len(direction_tasks) >= _WORLD_MAX_DIRECTION_TASKS:
                    # Dropped silently, like every other out-of-whitelist frame:
                    # spam should stop working, not help the sender break their
                    # own session.
                    coro.close()
                    _log_gateway_trace(
                        trace_id,
                        "gateway.direction_dropped",
                        session_id=session_id,
                        reason="too_many_inflight",
                    )
                    return
                task = asyncio.create_task(coro, name=f"gateway-direction-{label}")
                direction_tasks.add(task)

                def _done(t: asyncio.Task) -> None:
                    direction_tasks.discard(t)
                    if not t.cancelled() and t.exception() is not None:
                        logger.warning(
                            "direction task failed run_id=%s",
                            principal.run_id if principal else "-",
                            exc_info=t.exception(),
                        )

                task.add_done_callback(_done)

            async def browser_to_worker():
                nonlocal expected_last_chunk
                try:
                    while True:
                        payload = await _receive_browser(websocket)
                        if principal is not None:
                            forwarded = await _world_browser_frame(payload)
                            if forwarded is None:
                                continue
                            payload = forwarded
                        control = None
                        if isinstance(payload, bytes):
                            try:
                                control = decode_message(payload)
                            except ProtocolViolation:
                                pass
                        if isinstance(control, dict) and control.get("type") in {
                            "client_metric",
                            "client_metric_batch",
                        }:
                            _observe_gateway_client_metric(control, model=metric_model)
                            continue
                        if isinstance(control, dict) and expected_last_chunk is None:
                            if (
                                isinstance(control, dict)
                                and control.get("type") == "init"
                            ):
                                max_chunks = int(control.get("max_chunks") or 0)
                                if max_chunks > 0:
                                    expected_last_chunk = max_chunks - 1
                        if isinstance(payload, bytes):
                            await playback_ack_window.observe_browser_message(payload)
                        await _upstream_send(payload)
                except ConnectionClosedOK:
                    return

            async def _world_browser_frame(payload):
                """Handle inbound frames for authorized sessions.

                Returns bytes to forward to the engine, or None to drop. The
                browser may only forward the sealed session payload as the first
                frame, trigger a skill by id, or send allowlisted interaction
                events. Everything else is dropped; the allowlist prevents future
                engine parameters from becoming injectable by default.
                """
                if not isinstance(payload, bytes):
                    # Drop all text frames; constraints are defined over binary frames.
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="text_frame",
                    )
                    return None

                if not world_session["init_verified"]:
                    # The first frame must be the sealed payload. Unseal failures,
                    # including tampering or copying across runs, close the session.
                    init_message = payload_sealer.open(payload, principal.run_id)
                    skills = {}
                    ext = init_message.pop(WORLD_EXT_KEY, None)
                    if isinstance(ext, dict):
                        for item in ext.get("skills") or []:
                            if isinstance(item, dict) and item.get("id"):
                                skills[str(item["id"])] = item
                    world_session["skills"] = skills
                    world_session["init_verified"] = True
                    # Direction orchestrator: baseline seed comes from init
                    # prompt, while timeline entries fold into baseline as
                    # chunk_telemetry advances.
                    nonlocal directions
                    baseline, schedule = parse_init_directions(init_message)
                    directions = DirectionCoordinator(
                        baseline=baseline,
                        schedule=schedule,
                        rewrite=_direction_rewrite,
                        dispatch=_direction_dispatch,
                        notify=_direction_notify,
                    )
                    # Strip gateway-only extensions before forwarding to the
                    # engine. encode_message has signature (message_type, *,
                    # **fields) and inserts version itself, so both keys must be
                    # removed from fields to avoid missing positional argument or
                    # duplicate keyword errors.
                    init_message.pop("version", None)
                    message_type = init_message.pop("type", "init")
                    return encode_message(message_type, **init_message)

                try:
                    control = decode_message(payload)
                except ProtocolViolation:
                    # Drop undecodable frames; never allow by default just
                    # because we cannot parse the frame.
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="undecodable",
                    )
                    return None
                if not isinstance(control, dict):
                    return None

                msg_type = control.get("type")
                if msg_type in _WORLD_PASSTHROUGH_TYPES:
                    return payload
                if msg_type != "event":
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason=f"type:{msg_type}",
                    )
                    return None

                kind = control.get("kind")
                if kind == "skill":
                    # Skill activation: the browser sends only an id. The gateway
                    # looks up the prompt, so the prompt never passes through the
                    # browser.
                    skill = (world_session.get("skills") or {}).get(
                        str(control.get("id") or "")
                    )
                    if skill is None or directions is None:
                        _log_gateway_trace(
                            trace_id,
                            "gateway.world_frame_rejected",
                            session_id=session_id,
                            reason="unknown_skill",
                        )
                        return None
                    # Skill prompts are authored as full scene descriptions,
                    # equivalent to pre-rewritten directions, so use the same
                    # apply path. This also fixes two older issues: the prompt was
                    # sent under the wrong field (prompt= while the engine reads
                    # payload), and one_time skills had no revert so their effect
                    # could become permanent.
                    _spawn_direction(
                        directions.apply(
                            str(skill.get("prompt") or ""),
                            str(skill.get("change_type") or "one_time"),
                            control.get("event_id"),
                            str(skill.get("kind") or "prompt"),
                        ),
                        "skill",
                    )
                    return None
                if kind == "direction":
                    # Raw player instruction. The independent rewrite task later
                    # dispatches kind:"prompt"; forward no bytes here.
                    if not principal.allow_free_prompt:
                        # Worlds with free input disabled silently drop instead of
                        # disconnecting. Malicious injection should fail, not help
                        # the attacker end someone else's session.
                        _log_gateway_trace(
                            trace_id,
                            "gateway.free_prompt_dropped",
                            session_id=session_id,
                        )
                        return None
                    text = str(control.get("text") or "").strip()
                    if (
                        not text
                        or len(text) > _WORLD_DIRECTION_MAX_CHARS
                        or directions is None
                    ):
                        _log_gateway_trace(
                            trace_id,
                            "gateway.world_frame_rejected",
                            session_id=session_id,
                            reason="direction_invalid",
                        )
                        return None
                    _spawn_direction(
                        directions.submit(control.get("event_id"), text), "rewrite"
                    )
                    return None
                if kind in _WORLD_ALLOWED_EVENT_KINDS:
                    return payload
                _log_gateway_trace(
                    trace_id,
                    "gateway.world_frame_rejected",
                    session_id=session_id,
                    reason=f"kind:{kind}",
                )
                return None

            async def worker_to_browser():
                try:
                    while True:
                        wire = await upstream.recv()
                        if isinstance(wire, str):
                            raise ProtocolViolation(
                                "Denoiser control messages must be binary"
                            )
                        if not worker_message_allowed(wire):
                            message_type = worker_message_type(wire)
                            raise ProtocolViolation(
                                f"Denoiser emitted forbidden message: {message_type}"
                            )
                        if directions is not None:
                            # Passing telemetry also advances the baseline by
                            # folding timeline entries into it. Worker control
                            # messages are raw msgpack without version fields, so
                            # use the same raw decode path as worker_message_allowed.
                            # decode_message would treat them as protocol violations.
                            # Upstream control messages are tiny, so the extra
                            # decode is negligible.
                            telemetry = msgspec.msgpack.decode(wire)
                            if (
                                isinstance(telemetry, dict)
                                and telemetry.get("type") == "chunk_telemetry"
                                and isinstance(telemetry.get("chunk_index"), int)
                            ):
                                directions.observe_chunk(telemetry["chunk_index"])
                        await send_browser_with_trace(wire, send_source="denoiser")
                except ConnectionClosedOK:
                    return

            async def output_to_browser():
                while True:
                    output = await route.get_output()
                    wire = output.wire
                    codec, scope = _metric_labels_from_wire(wire)
                    observe_stage_seconds(
                        "output_pacing_queue",
                        time.monotonic() - output.enqueued_at,
                        service="gateway",
                        model=metric_model,
                        result="success",
                        codec=codec,
                        scope=scope,
                    )
                    try:
                        await send_browser_with_trace(
                            wire,
                            send_source="vae",
                            queue_fields=route.queue_metrics(),
                        )
                    finally:
                        route.task_done()

            async def send_browser_with_trace(
                wire: bytes,
                *,
                send_source: str,
                queue_fields: dict[str, Any] | None = None,
            ) -> None:
                send_started = time.perf_counter()
                codec, scope = _metric_labels_from_wire(wire)
                send_fields = _browser_send_trace_fields(wire)
                queue_fields = queue_fields or {}
                if not await playback_ack_window.allow_output(wire):
                    observe_stage_seconds(
                        "websocket_build_write",
                        time.perf_counter() - send_started,
                        service="gateway",
                        model=metric_model,
                        result="cancelled",
                        codec=codec,
                        scope=scope,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.browser_send_dropped",
                        session_id=session_id,
                        generation_id=generation_id,
                        send_source=send_source,
                        drop_reason="playback_ack_window",
                        last_received_chunk=playback_ack_window.last_received_chunk,
                        last_rendered_chunk=playback_ack_window.last_rendered_chunk,
                        **queue_fields,
                        **send_fields,
                    )
                    return
                try:
                    # D5 ghost cleanup: a dead peer may never drain its TCP write
                    # buffer, causing this await to hang forever because keepalive
                    # close is ineffective while the buffer is not empty. Bounding
                    # the send turns that case into a normal task completion and
                    # lets existing finally cleanup release the reservation.
                    await asyncio.wait_for(
                        sender.send(wire), timeout=browser_send_timeout_s
                    )
                except Exception as exc:
                    browser_send_ms = round(
                        (time.perf_counter() - send_started) * 1000, 3
                    )
                    observe_stage_seconds(
                        "websocket_build_write",
                        browser_send_ms / 1000.0,
                        service="gateway",
                        model=metric_model,
                        result=result_from_exception(exc),
                        codec=codec,
                        scope=scope,
                    )
                    _log_gateway_trace(
                        trace_id,
                        "gateway.browser_send_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        send_source=send_source,
                        send_ok=False,
                        send_ms=browser_send_ms,
                        browser_send_ms=browser_send_ms,
                        error_type=type(exc).__name__,
                        **queue_fields,
                        **send_fields,
                    )
                    raise
                browser_send_ms = round((time.perf_counter() - send_started) * 1000, 3)
                observe_stage_seconds(
                    "websocket_build_write",
                    browser_send_ms / 1000.0,
                    service="gateway",
                    model=metric_model,
                    result="success",
                    codec=codec,
                    scope=scope,
                )
                _log_gateway_trace(
                    trace_id,
                    "gateway.browser_send_complete",
                    session_id=session_id,
                    generation_id=generation_id,
                    send_source=send_source,
                    send_ok=True,
                    send_ms=browser_send_ms,
                    browser_send_ms=browser_send_ms,
                    **queue_fields,
                    **send_fields,
                )

            async def renew_lease():
                nonlocal assignment
                while True:
                    await asyncio.sleep(lease_renew_interval_s)
                    assignment = await coordinator.renew(assignment)

            browser_input_task = asyncio.create_task(
                browser_to_worker(), name="gateway-browser-input"
            )
            worker_control_task = asyncio.create_task(
                worker_to_browser(), name="gateway-worker-control"
            )
            output_task = asyncio.create_task(
                output_to_browser(), name="gateway-vae-output"
            )
            lease_task = asyncio.create_task(renew_lease(), name="gateway-lease-renew")
            tasks = {
                browser_input_task,
                worker_control_task,
                output_task,
                lease_task,
            }
            if principal is not None:
                # D2: server-authoritative session lifetime. When the deadline
                # task completes, FIRST_COMPLETED wakes the session and existing
                # finally cleanup cancels lease renewal and releases slots. The
                # close reason matches the SessionLifetimeGuard contract.
                async def _session_deadline() -> None:
                    await asyncio.sleep(principal.max_lifetime_s)
                    world_session["deadline_hit"] = True
                    logger.info(
                        "world session deadline reached run_id=%s", principal.run_id
                    )

                tasks.add(
                    asyncio.create_task(_session_deadline(), name="gateway-deadline")
                )
                # D4: after admit and upstream readiness, notify the backend that
                # the session has started.
                if world_callbacks is not None:
                    # Record that this session really came up. Terminal
                    # callbacks carry the flag so the business side can tell
                    # "a live session died" from "a connection attempt that
                    # never established failed".
                    world_session["started_reported"] = True
                    world_callbacks.started(
                        principal.run_id, trace_id, principal.max_lifetime_s
                    )
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            if route is not None and worker_control_task in done:
                # Engine finished normally after max_chunks, so classify the run
                # as completed rather than user_left.
                world_session["worker_finished"] = True
                try:
                    if expected_last_chunk is not None:
                        await asyncio.wait_for(
                            route.wait_until_chunk_completed(expected_last_chunk),
                            timeout=output_drain_timeout_s,
                        )
                    else:
                        await asyncio.wait_for(
                            route.wait_until_output_closed(),
                            timeout=output_drain_timeout_s,
                        )
                    await asyncio.wait_for(route.join(), timeout=output_drain_timeout_s)
                except TimeoutError:
                    logger.warning(
                        "Gateway media drain timed out for session_id=%s",
                        session_id,
                    )
        except TokenError:
            # Session payload unseal failed because of tampering, cross-run copy,
            # or a non-init payload. Externally expose only a fixed business code;
            # detailed reasons stay in server logs to avoid oracle feedback.
            world_session["init_rejected"] = True
            logger.warning(
                "world session payload rejected run_id=%s",
                principal.run_id if principal else "-",
                exc_info=True,
            )
            try:
                await asyncio.wait_for(
                    sender.error("SESSION_INVALID", code="SESSION_INVALID"),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=1008, reason="SESSION_INVALID")
            except Exception:
                pass
        except CoordinatorRejected as exc:
            world_session["admit_rejected"] = exc.reason
            try:
                await asyncio.wait_for(
                    sender.error(
                        f"realtime admission rejected: {exc.reason}",
                        reason=exc.reason,
                        retry_after_s=exc.retry_after_s,
                    ),
                    timeout=browser_send_timeout_s,
                )
                close_code = 1013 if exc.reason == "CAPACITY_EXHAUSTED" else 1008
                await websocket.close(code=close_code, reason=exc.reason)
            except Exception:
                pass
        except AdmissionQueueFull as exc:
            try:
                await asyncio.wait_for(
                    sender.error(f"realtime admission rejected: {exc.reason}"),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=1013, reason=exc.reason)
            except Exception:
                pass
        except WebSocketDisconnect:
            pass  # Browser disconnected: a real user_left.
        except OutputRouteClosed:
            # Output path closed by the system side, such as VAE or route failure;
            # this is not user behavior.
            world_session["system_error"] = "output route closed"
        except Exception as exc:
            # str() can be empty, e.g. bare TimeoutError, so guard splitlines()[0].
            detail = (str(exc).splitlines() or [type(exc).__name__])[0]
            world_session["system_error"] = detail
            try:
                # The browser might be the non-reading peer that caused the error,
                # so even the notification must be bounded. Otherwise we could
                # hang before finally and leak the reservation.
                await asyncio.wait_for(
                    sender.error(f"realtime gateway error: {detail}"),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=1011, reason="gateway session failed")
            except Exception:
                pass
        finally:
            if directions is not None:
                directions.close()  # cancel revert timer
            await _cancel_tasks(set(direction_tasks))
            await _cancel_tasks(tasks)
            if upstream is not None:
                await upstream.close()
                if release_grace_s:
                    await asyncio.sleep(release_grace_s)
            if route is not None:
                await registry.unregister(session_id, generation_id, token=output_token)
            if assignment is not None:
                try:
                    await coordinator.release(assignment)
                except Exception:
                    logger.exception(
                        "Coordinator release failed for session_id=%s",
                        assignment.session_id,
                    )
            _log_gateway_trace(
                trace_id, "gateway.session_closed", session_id=session_id
            )
            # D4: terminal session callback. It is fire-and-forget; world-service
            # has a deadline fallback for lost callbacks.
            if principal is not None and world_callbacks is not None:
                # Whether this session ever came up. Rejected or invalid-payload
                # attempts never did, and the business side uses that to refuse
                # letting them terminate a run that is already live.
                established = bool(world_session.get("started_reported"))
                if world_session.get("admit_rejected"):
                    world_callbacks.aborted(
                        principal.run_id,
                        trace_id,
                        fault="ours",
                        reason=str(world_session["admit_rejected"]),
                        established=established,
                    )
                elif world_session.get("init_rejected"):
                    world_callbacks.aborted(
                        principal.run_id,
                        trace_id,
                        fault="client",
                        reason="init payload mismatch",
                        established=established,
                    )
                elif assignment is None:
                    world_callbacks.aborted(
                        principal.run_id,
                        trace_id,
                        fault="ours",
                        reason="not admitted",
                        established=established,
                    )
                elif world_session.get("deadline_hit"):
                    world_callbacks.ended(principal.run_id, trace_id, "completed")
                elif world_session.get("system_error"):
                    # Gateway or engine-side fault. Mark fault=ours so the
                    # business side can exempt the player from run consumption.
                    world_callbacks.aborted(
                        principal.run_id,
                        trace_id,
                        fault="ours",
                        reason=str(world_session["system_error"])[:200],
                        established=established,
                    )
                elif world_session.get("worker_finished"):
                    world_callbacks.ended(principal.run_id, trace_id, "completed")
                else:
                    world_callbacks.ended(principal.run_id, trace_id, "user_left")
            # 只有"内容没跑完就被时长掐断"才算腰斩。worker 已经正常收尾（跑满
            # max_chunks）时哪怕 deadline 随后到点，也不能写这个理由 —— 业务侧
            # 靠它区分"内容播完"与"被硬顶截断"，写错会把一次完整生成误判成事故。
            close_reason = (
                "maximum session lifetime reached"
                if world_session.get("deadline_hit")
                and not world_session.get("worker_finished")
                else ""
            )
            try:
                await websocket.close(code=1000, reason=close_reason)
            except Exception:
                pass

    # Unauthenticated showcase routes used by the internal web UI. Public
    # isolation relies on ingress exposing only authorized_generate; see the
    # disable_legacy_routes notes in create_app.
    if enable_legacy_routes:

        @app.websocket("/backends/lingbot2/v1/realtime_video/generate")
        async def generate_lingbot2(websocket: WebSocket):
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=lingbot2_model_revision,
                selected_vae_fingerprint=(lingbot2_vae_fingerprint or vae_fingerprint),
            )

        @app.websocket("/backends/minwm/v1/realtime_video/generate")
        @app.websocket("/v1/realtime_video/generate")
        async def generate(websocket: WebSocket):
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=model_revision,
                selected_vae_fingerprint=vae_fingerprint,
            )

    if world_platform is not None:
        token_replay_guard = TokenReplayGuard()

        @app.websocket("/backends/minwm/v1/realtime_video/authorized_generate")
        async def authorized_generate(websocket: WebSocket):
            """Authorized world-platform route.

            It verifies the token, then reuses the shared session logic. Legacy
            unauthenticated routes continue serving showcase; this route is the
            only platform entrypoint.
            """
            token = websocket.query_params.get("token") or ""
            try:
                principal = verify_session_token(token, world_platform.public_key)
            except TokenError as exc:
                # Expose only a fixed business code. Distinguishing bad signature,
                # expiry, or aud mismatch would give token forgers a field oracle.
                # Detailed reasons stay in server logs.
                logger.warning("world token rejected: %s", exc)
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return
            if not token_replay_guard.consume(principal):
                # One-use credential by jti: reject the second session for a token.
                logger.warning("world token replayed run_id=%s", principal.run_id)
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=model_revision,
                selected_vae_fingerprint=vae_fingerprint,
                principal=principal,
            )

    # Showcase static frontend shares the legacy-route switch.
    if enable_legacy_routes:

        @app.get("/")
        async def index():
            return FileResponse(WEBUI_ROOT / "index.html")

        app.mount("/", StaticFiles(directory=WEBUI_ROOT), name="realtime-webui")
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--lingbot2-upstream-url",
        default=os.environ.get("LINGBOT2_UPSTREAM_WS"),
    )
    parser.add_argument(
        "--lingbot2-model-revision",
        default=os.environ.get(
            "LINGBOT2_MODEL_REVISION",
            "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        ),
    )
    parser.add_argument("--vae-fingerprint", default="taew2_2")
    parser.add_argument(
        "--lingbot2-vae-fingerprint",
        default=os.environ.get("LINGBOT2_VAE_FINGERPRINT"),
    )
    parser.add_argument(
        "--internal-output-url",
        default=os.environ.get("REALTIME_GATEWAY_OUTPUT_URL"),
    )
    parser.add_argument("--output-queue-depth", type=int, default=64)
    parser.add_argument("--output-enqueue-timeout-s", type=float, default=0.0)
    parser.add_argument("--output-queue-max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output-queue-max-messages", type=int, default=256)
    parser.add_argument("--output-drain-timeout-s", type=float, default=5.0)
    parser.add_argument("--lease-renew-interval-s", type=float, default=10.0)
    parser.add_argument("--release-grace-s", type=float, default=0.5)
    parser.add_argument("--max-admission-waiters", type=int, default=64)
    parser.add_argument("--readiness-coordinator-timeout-s", type=float, default=1.0)
    parser.add_argument("--readiness-coordinator-grace-s", type=float, default=30.0)
    parser.add_argument("--trace-log-group")
    # ---- World platform integration. All values are required to enable it. ----
    parser.add_argument(
        "--world-token-ed25519-pub",
        default=os.environ.get("WORLD_TOKEN_ED25519_PUB", ""),
    )
    parser.add_argument(
        "--world-callback-url", default=os.environ.get("WORLD_CALLBACK_URL", "")
    )
    parser.add_argument(
        "--world-callback-hmac-secret",
        default=os.environ.get("WORLD_CALLBACK_HMAC_SECRET", ""),
    )
    parser.add_argument(
        "--disable-legacy-routes",
        action="store_true",
        help=(
            "Do not register unauthenticated showcase routes, static pages, or "
            "trace queries. They are registered by default because the internal "
            "web UI depends on them; public ingress should expose only the "
            "authorized_generate path."
        ),
    )
    parser.add_argument("--browser-send-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--ui-config-json",
        default=os.environ.get("REALTIME_UI_CONFIG_JSON", "{}"),
    )
    return parser.parse_args()


def _build_world_platform(args) -> WorldPlatformConfig | None:
    """Enable world-platform routes only when all --world-* settings are present.

    All three or none. A partial set used to silently skip route registration:
    authorized_generate then answers 404 and the operator debugs "route not
    found" instead of reading a config error. That 404 is also underdetermined
    (a proxy in front produces the same one), so the silent path wastes hours.
    Failing startup turns a misconfiguration into a ten-minute deploy failure
    that names the missing flag.
    """
    provided = {
        "--world-token-ed25519-pub": bool(args.world_token_ed25519_pub),
        "--world-callback-url": bool(args.world_callback_url),
        "--world-callback-hmac-secret": bool(args.world_callback_hmac_secret),
    }
    if not any(provided.values()):
        return None  # world platform intentionally disabled (showcase-only deploy)
    missing = [flag for flag, ok in provided.items() if not ok]
    if missing:
        raise SystemExit(
            "world platform is partially configured; refusing to start with the "
            "authorized_generate route silently missing. Provide all of "
            f"{sorted(provided)} or none. Missing: {missing}"
        )
    from sglang.multimodal_gen.runtime.realtime.world_platform import load_public_key

    return WorldPlatformConfig(
        public_key=load_public_key(args.world_token_ed25519_pub),
        callback_url=args.world_callback_url,
        callback_app_id="zing-gateway",
        callback_key_id="k1",
        callback_secret=args.world_callback_hmac_secret,
    )


def main() -> None:
    args = _parse_args()
    if not args.internal_output_url:
        raise SystemExit("--internal-output-url is required")
    try:
        ui_config = _parse_ui_config(args.ui_config_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    coordinator = HTTPCoordinatorClient(args.coordinator_url)
    trace_query = None
    if args.trace_log_group:
        try:
            import boto3
        except ImportError as exc:
            raise SystemExit("boto3 is required for CloudWatch Trace query") from exc
        from sglang.multimodal_gen.runtime.realtime.trace_query import (
            CloudWatchTraceQuery,
        )

        trace_query = CloudWatchTraceQuery(
            boto3.client("logs"), log_group=args.trace_log_group
        )
    app = create_app(
        coordinator,
        model_revision=args.model_revision,
        vae_fingerprint=args.vae_fingerprint,
        internal_output_url=args.internal_output_url,
        lingbot2_upstream_url=args.lingbot2_upstream_url,
        lingbot2_model_revision=args.lingbot2_model_revision,
        lingbot2_vae_fingerprint=args.lingbot2_vae_fingerprint,
        output_queue_depth=args.output_queue_depth,
        output_enqueue_timeout_s=args.output_enqueue_timeout_s,
        output_queue_max_bytes=args.output_queue_max_bytes,
        output_queue_max_messages=args.output_queue_max_messages,
        output_drain_timeout_s=args.output_drain_timeout_s,
        lease_renew_interval_s=args.lease_renew_interval_s,
        release_grace_s=args.release_grace_s,
        max_admission_waiters=args.max_admission_waiters,
        readiness_coordinator_timeout_s=args.readiness_coordinator_timeout_s,
        readiness_coordinator_grace_s=args.readiness_coordinator_grace_s,
        ui_config=ui_config,
        trace_query=trace_query,
        world_platform=_build_world_platform(args),
        browser_send_timeout_s=args.browser_send_timeout_s,
        disable_legacy_routes=args.disable_legacy_routes,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        # Pin keepalive explicitly. Defaults are currently 20/20, but uvicorn is
        # not pinned in pyproject, and dependency drift should not change the
        # transport contract that ghost cleanup relies on.
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()
