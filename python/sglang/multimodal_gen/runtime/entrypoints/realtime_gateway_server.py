# SPDX-License-Identifier: Apache-2.0

"""Public realtime Gateway for Coordinator-routed Denoiser/VAE sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import hashlib
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
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
from sglang.multimodal_gen.runtime.realtime.world_platform import (
    Principal,
    SessionPayloadSealer,
    TokenError,
    TokenReplayGuard,
    WorldCallbacks,
    WorldPlatformConfig,
    verify_session_token,
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
        denoiser=WorkerSlot(**payload["denoiser"]),
        vae=WorkerSlot(**payload["vae"]),
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


# 鉴权会话的入站白名单。白名单而非黑名单：将来引擎新增控制字段，
# 不会因为「没被拉黑」而自动获得可注入性。
_WORLD_PASSTHROUGH_TYPES = frozenset({"client_metric", "client_metric_batch", "ack"})
_WORLD_ALLOWED_EVENT_KINDS = frozenset(
    {"camera", "camera_actions", "move", "action", "playback_ack", "heartbeat"}
)
# 自由输入类事件：仅当凭证 allow_free_prompt=true 时放行
_WORLD_FREE_PROMPT_KINDS = frozenset({"prompt", "scene_cut"})

# init 消息里「只给网关看」的扩展块键名（与 world-service zingproto.WorldExtKey 一致）
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
    # 无鉴权 showcase 路由、静态页与 trace 查询。默认注册。
    #
    # 这些路由不该被公网访问 —— 走它们能拿到不限时、不排队、不鉴权的会话，
    # 等于绕过 authorized_generate 的全部约束。但**边界在网络层，不在这里**：
    # 同一个进程同时服务两拨人 —— 公网浏览器经 ingress 只该看到
    # authorized_generate，内网 webui 经集群 Service 需要这些 legacy 路由。
    # 应用层的开关分不清这两者，一刀切会连内网调用一起打死。
    #
    # 正确做法：公网 ingress 只暴露 authorized_generate 这一条 path。
    # 本开关留给「这台网关专供平台、根本没有 showcase 流量」的部署形态。
    disable_legacy_routes: bool = False,
) -> FastAPI:
    if release_grace_s < 0:
        raise ValueError("release_grace_s must be non-negative")
    if browser_send_timeout_s <= 0:
        raise ValueError("browser_send_timeout_s must be positive")
    # world 平台回调客户端（未配置则整条 authorized 链路不启用）
    enable_legacy_routes = not disable_legacy_routes
    world_callbacks = WorldCallbacks(world_platform) if world_platform else None
    # 会话载荷解封器：与回调 HMAC 同一个共享密钥，HKDF 分流出独立子密钥
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

    # 内部排障接口：返回 worker id、引擎主机名、adapter 类名等实现细节。
    # 与 showcase 路由同一个开关；公网必须靠 ingress 挡住（连同 /metrics、/v1/models）。
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
        # principal 非空 = world 平台的鉴权会话（D1~D4 只作用于这条分支）；
        # 为空 = 旧 showcase 路由，行为与改造前逐字节一致。
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
        # world 会话的分支状态（init 校验 / deadline / 拒绝原因）
        world_session: dict[str, Any] = {"init_verified": False}
        try:
            admitted_at = time.perf_counter()
            _log_gateway_trace(trace_id, "gateway.ws_accepted", session_id=session_id)
            try:
                async with admission_gate.waiter():
                    assignment = await coordinator.admit(
                        # D1：鉴权会话用凭证里的内部用户 ID（wsvc: 前缀与 showcase 隔离）
                        user_id=(
                            f"wsvc:{principal.user_id}"
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
                        await upstream.send(payload)
                except ConnectionClosedOK:
                    return

            async def _world_browser_frame(payload):
                """鉴权会话的入站帧处理。返回要转发给引擎的字节，None 表示丢弃。

                浏览器在鉴权会话里只被允许做三件事：转发密封的会话载荷（第一帧）、
                释放技能（发技能 id）、发送白名单内的交互事件。其余一律丢弃 ——
                白名单而非黑名单，新增引擎参数不会自动获得可注入性。
                """
                if not isinstance(payload, bytes):
                    # 文本帧一律丢弃：所有约束都定义在二进制帧上
                    _log_gateway_trace(
                        trace_id,
                        "gateway.world_frame_rejected",
                        session_id=session_id,
                        reason="text_frame",
                    )
                    return None

                if not world_session["init_verified"]:
                    # 第一帧必须是密封载荷。解封失败（含篡改、挪用他局）即断开。
                    init_message = payload_sealer.open(payload, principal.run_id)
                    skills = {}
                    ext = init_message.pop(WORLD_EXT_KEY, None)
                    if isinstance(ext, dict):
                        for item in ext.get("skills") or []:
                            if isinstance(item, dict) and item.get("id"):
                                skills[str(item["id"])] = item
                    world_session["skills"] = skills
                    world_session["init_verified"] = True
                    # 剥掉只给网关看的扩展块后再交给引擎
                    return encode_message(**init_message)

                try:
                    control = decode_message(payload)
                except ProtocolViolation:
                    # 解不出来就丢弃 —— 绝不因为「看不懂」而放行
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
                    # 技能：浏览器只发 id，提示词由网关查表补上（全程不经过浏览器）
                    skill = (world_session.get("skills") or {}).get(
                        str(control.get("id") or "")
                    )
                    if skill is None:
                        _log_gateway_trace(
                            trace_id,
                            "gateway.world_frame_rejected",
                            session_id=session_id,
                            reason="unknown_skill",
                        )
                        return None
                    return encode_message(
                        "event", kind="prompt", prompt=str(skill.get("prompt") or "")
                    )
                if kind in _WORLD_ALLOWED_EVENT_KINDS:
                    return payload
                if kind in _WORLD_FREE_PROMPT_KINDS:
                    if principal.allow_free_prompt:
                        return payload
                    # 关闭自由输入的世界：静默丢弃而不是断开 ——
                    # 恶意注入只应失效，不应帮攻击者结束别人的会话
                    _log_gateway_trace(
                        trace_id,
                        "gateway.free_prompt_dropped",
                        session_id=session_id,
                    )
                    return None
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
                    # D5（幽灵回收）：死端的 TCP 写缓冲永不排空，这个 await 会永久
                    # 停住（keepalive 的 close 在缓冲非空时是空操作）。加界让超时
                    # 变成一次普通任务完成，走既有 finally 拆解 —— 不新增代码路径。
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
                # D2：服务端权威会话时长。到点任务完成 → FIRST_COMPLETED 唤醒
                # → 走既有 finally 拆解（取消续租、释放席位）。close reason 与
                # 前端 SessionLifetimeGuard 的既有契约逐字一致。
                async def _session_deadline() -> None:
                    await asyncio.sleep(principal.max_lifetime_s)
                    world_session["deadline_hit"] = True
                    logger.info(
                        "world session deadline reached run_id=%s", principal.run_id
                    )

                tasks.add(
                    asyncio.create_task(_session_deadline(), name="gateway-deadline")
                )
                # D4：admit 成功 + 上游就绪，通知业务后端会话已建立
                if world_callbacks is not None:
                    world_callbacks.started(
                        principal.run_id, trace_id, principal.max_lifetime_s
                    )
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            if route is not None and worker_control_task in done:
                # 引擎正常收尾（跑满 max_chunks）：终局归类为 completed 而非 user_left
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
            # 会话载荷解封失败（篡改、挪用他局、非 init）。对外只给固定业务码，
            # 具体原因只进服务端日志 —— 不给攻击者任何试探反馈。
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
            pass  # 浏览器断开：真正的 user_left
        except OutputRouteClosed:
            # 输出通道被系统侧关闭（VAE/路由异常）——不是用户行为
            world_session["system_error"] = "output route closed"
        except Exception as exc:
            # str() 可能为空（如裸 TimeoutError），splitlines()[0] 会 IndexError
            detail = (str(exc).splitlines() or [type(exc).__name__])[0]
            world_session["system_error"] = detail
            try:
                # 出错时浏览器可能正是那个不读数据的对端 —— 通知必须限时，
                # 否则会卡死在 finally 之前，席位永不释放（幽灵会话回魂）
                await asyncio.wait_for(
                    sender.error(f"realtime gateway error: {detail}"),
                    timeout=browser_send_timeout_s,
                )
                await websocket.close(code=1011, reason="gateway session failed")
            except Exception:
                pass
        finally:
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
            # D4：会话终局回调（fire-and-forget；丢失由业务后端 deadline 兜底）
            if principal is not None and world_callbacks is not None:
                if world_session.get("admit_rejected"):
                    world_callbacks.aborted(
                        principal.run_id, trace_id,
                        fault="ours", reason=str(world_session["admit_rejected"]),
                    )
                elif world_session.get("init_rejected"):
                    world_callbacks.aborted(
                        principal.run_id, trace_id,
                        fault="client", reason="init payload mismatch",
                    )
                elif assignment is None:
                    world_callbacks.aborted(
                        principal.run_id, trace_id, fault="ours", reason="not admitted"
                    )
                elif world_session.get("deadline_hit"):
                    world_callbacks.ended(principal.run_id, trace_id, "completed")
                elif world_session.get("system_error"):
                    # 网关/引擎侧故障：fault=ours —— 业务侧免责处理，不消耗玩家局数
                    world_callbacks.aborted(
                        principal.run_id, trace_id,
                        fault="ours", reason=str(world_session["system_error"])[:200],
                    )
                elif world_session.get("worker_finished"):
                    world_callbacks.ended(principal.run_id, trace_id, "completed")
                else:
                    world_callbacks.ended(principal.run_id, trace_id, "user_left")
            close_reason = (
                "maximum session lifetime reached"
                if world_session.get("deadline_hit")
                else ""
            )
            try:
                await websocket.close(code=1000, reason=close_reason)
            except Exception:
                pass

    # 无鉴权的 showcase 路由（内网 webui 在用）。公网侧的隔离靠 ingress 只
    # 暴露 authorized_generate 一条 path；见 create_app 的 disable_legacy_routes 注释。
    if enable_legacy_routes:

        @app.websocket("/backends/lingbot2/v1/realtime_video/generate")
        async def generate_lingbot2(websocket: WebSocket):
            await _generate_coordinator_session(
                websocket,
                selected_model_revision=lingbot2_model_revision,
                selected_vae_fingerprint=(
                    lingbot2_vae_fingerprint or vae_fingerprint
                ),
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
            """world 平台的鉴权路由：凭证验签 → 复用共享会话逻辑。

            老路由（无鉴权）保持原样服务 showcase；本路由是新平台的唯一入口。
            """
            token = websocket.query_params.get("token") or ""
            try:
                principal = verify_session_token(token, world_platform.public_key)
            except TokenError as exc:
                # 对外只给固定业务码：区分「签名错」「过期」「aud 不符」等于给
                # 伪造者一个逐字段试探的预言机。具体原因只进服务端日志。
                logger.warning("world token rejected: %s", exc)
                await websocket.accept()
                await websocket.close(code=1008, reason="UNAUTHORIZED")
                return
            if not token_replay_guard.consume(principal):
                # 凭证一次性（jti）：同一凭证第二次建会话直接拒绝
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

    # showcase 静态前端与老路由同一个开关。
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
    parser.add_argument("--output-drain-timeout-s", type=float, default=5.0)
    parser.add_argument("--lease-renew-interval-s", type=float, default=10.0)
    parser.add_argument("--release-grace-s", type=float, default=0.5)
    parser.add_argument("--max-admission-waiters", type=int, default=64)
    parser.add_argument("--readiness-coordinator-timeout-s", type=float, default=1.0)
    parser.add_argument("--readiness-coordinator-grace-s", type=float, default=30.0)
    parser.add_argument("--trace-log-group")
    # ---- world 平台接入（全部提供才启用 authorized_generate 路由） ----
    parser.add_argument("--world-token-ed25519-pub", default=os.environ.get("WORLD_TOKEN_ED25519_PUB", ""))
    parser.add_argument("--world-callback-url", default=os.environ.get("WORLD_CALLBACK_URL", ""))
    parser.add_argument("--world-callback-hmac-secret", default=os.environ.get("WORLD_CALLBACK_HMAC_SECRET", ""))
    parser.add_argument(
        "--disable-legacy-routes",
        action="store_true",
        help=(
            "不注册无鉴权的 showcase 路由/静态页/trace 查询。"
            "默认注册 —— 内网 webui 依赖它们；"
            "公网侧请在 ingress 上只暴露 authorized_generate 这一条 path。"
        ),
    )
    parser.add_argument("--browser-send-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--ui-config-json",
        default=os.environ.get("REALTIME_UI_CONFIG_JSON", "{}"),
    )
    return parser.parse_args()


def _build_world_platform(args) -> WorldPlatformConfig | None:
    """三项配置齐全才启用 world 平台路由；配置错误在启动期立即暴露。"""
    if not (
        args.world_token_ed25519_pub
        and args.world_callback_url
        and args.world_callback_hmac_secret
    ):
        return None
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
        # 显式钉死 keepalive：默认值一直是 20/20，但 pyproject 未锁 uvicorn 版本，
        # 依赖漂移不该改变传输层契约（幽灵回收依赖它）
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()
