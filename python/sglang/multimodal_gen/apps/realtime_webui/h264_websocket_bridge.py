# SPDX-License-Identifier: Apache-2.0
"""Low-latency H.264/fMP4 over WebSocket bridge for realtime backends.

Each browser session owns one upstream model connection and one ffmpeg encoder.
The bridge accepts both Zing and LingBot2 through a backend-aware route while
keeping the model/VAE protocol unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import msgspec
from aiohttp import WSMsgType, web
from PIL import Image

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    observe_client_metric_event,
    observe_stage_seconds,
    prometheus_content_type,
    prometheus_latest,
    result_from_exception,
)
from sglang.multimodal_gen.runtime.realtime.request_mode import (
    init_requests_finite_output,
)

LOGGER = logging.getLogger(__name__)
H264_WS_MANAGER = web.AppKey("h264_websocket_bridge_manager", object)
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
ENCODED_IMAGE_TYPES = {"image/jpeg", "image/webp", "image/png"}
ALLOWED_CONTROL_KINDS = {
    "camera_actions",
    "heartbeat",
    "playback_ack",
    "prompt",
    "scene_cut",
}


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _decode_first_frame(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith("data:"):
        return value
    try:
        _metadata, encoded = value.split(",", 1)
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("first_frame data URL is invalid") from error


def _split_payload(header: dict[str, Any], payload: bytes) -> list[bytes]:
    lengths = header.get("payload_lengths")
    if isinstance(lengths, list) and lengths:
        frames: list[bytes] = []
        offset = 0
        for raw_length in lengths:
            length = int(raw_length)
            if length <= 0 or offset + length > len(payload):
                raise ValueError("invalid encoded frame payload_lengths")
            frames.append(payload[offset : offset + length])
            offset += length
        if offset != len(payload):
            raise ValueError("encoded frame payload has trailing bytes")
        return frames
    if str(header.get("content_type") or "") == RAW_RGB_CONTENT_TYPE:
        width = int(header.get("width") or 0)
        height = int(header.get("height") or 0)
        channels = int(header.get("channels") or 3)
        frame_size = width * height * channels
        if frame_size <= 0 or len(payload) % frame_size:
            raise ValueError("raw RGB payload size does not match frame dimensions")
        return [
            payload[offset : offset + frame_size]
            for offset in range(0, len(payload), frame_size)
        ]
    return [payload]


def _encoded_image_to_rgb(frame: bytes, *, width: int, height: int) -> bytes:
    image = Image.open(BytesIO(frame)).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image.tobytes()


def _raw_channel_filter(channel_order: str) -> str:
    if channel_order == "gbr":
        return (
            "colorchannelmixer=" "rr=0:rg=0:rb=1:" "gr=1:gg=0:gb=0:" "br=0:bg=1:bb=0,"
        )
    return ""


@dataclass(frozen=True)
class _QueuedFrame:
    rgb: bytes
    width: int
    height: int
    chunk_index: int
    event_id: int
    frame_batch_index: int
    num_frame_batches: int
    is_final_frame_batch: bool
    bridge_received_epoch_ms: float
    server_sent_epoch_ms: float


@dataclass
class H264WebSocketSession:
    manager: H264WebSocketBridgeManager
    websocket: web.WebSocketResponse
    init: dict[str, Any]
    backend: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    upstream: Any = None
    ffmpeg: asyncio.subprocess.Process | None = None
    encoder_task: asyncio.Task | None = None
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    pending_header: dict[str, Any] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    width: int = 0
    height: int = 0
    frames: int = 0
    media_bytes: int = 0
    media_payload_sequence: int = 0
    minimum_event_id: int = 0
    pending_cutover_event_id: int = 0
    dropped_frames: int = 0
    latency_dropped_frames: int = 0
    repeated_frames: int = 0
    final_completion: dict[str, Any] | None = None
    finite_request: bool = field(init=False)
    startup_drop_frames: int = field(init=False)
    startup_drop_remaining: int = field(init=False)
    startup_dropped_frames: int = 0
    encoder_preset: str = field(init=False)
    encoder_profile: str = field(init=False)
    encoder_crf: int = field(init=False)
    encoder_bitrate_kbps: int = field(init=False)
    encoder_vbv_buffer_ms: int = field(init=False)
    encoder_gop_seconds: int = field(init=False)
    frame_queue: asyncio.Queue[_QueuedFrame] = field(init=False)
    encoder_start_times_s: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.frame_queue = asyncio.Queue(maxsize=self.manager.max_queued_frames)
        self.encoder_start_times_s = deque(maxlen=4096)
        self.finite_request = init_requests_finite_output(self.init)
        requested_preset = str(
            self.init.get("h264_preset") or self.manager.preset
        ).lower()
        self.encoder_preset = (
            requested_preset
            if requested_preset
            in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium"}
            else self.manager.preset
        )
        requested_profile = str(
            self.init.get("h264_profile") or self.manager.profile
        ).lower()
        self.encoder_profile = (
            requested_profile
            if requested_profile in {"baseline", "main", "high"}
            else self.manager.profile
        )
        self.encoder_crf = _bounded_int(
            self.init.get("h264_crf"),
            default=self.manager.crf,
            minimum=12,
            maximum=35,
        )
        self.encoder_bitrate_kbps = _bounded_int(
            self.init.get("h264_bitrate_kbps"),
            default=self.manager.bitrate_kbps,
            minimum=250,
            maximum=20000,
        )
        self.encoder_vbv_buffer_ms = _bounded_int(
            self.init.get("h264_vbv_buffer_ms"),
            default=self.manager.vbv_buffer_ms,
            minimum=40,
            maximum=2000,
        )
        self.encoder_gop_seconds = _bounded_int(
            self.init.get("h264_gop_seconds"),
            default=self.manager.gop_seconds,
            minimum=1,
            maximum=5,
        )
        # LingBot2's first generated chunk contains a short I2V/TAEHV
        # transition. Drop it before ffmpeg starts so the first retained sharp
        # frame becomes the stream's initial H.264 IDR. Zing stays unaffected.
        self.startup_drop_frames = (
            _bounded_int(
                self.init.get("h264_startup_drop_frames"),
                default=self.manager.startup_drop_frames(self.backend),
                minimum=0,
                maximum=120,
            )
            if self.backend == "lingbot2"
            else 0
        )
        self.startup_drop_remaining = self.startup_drop_frames

    @property
    def upstream_url(self) -> str:
        parsed = urlsplit(self.manager.upstream_url(self.backend))
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["user_id"] = f"h264ws-{self.session_id}"
        query["trace_id"] = str(self.init.get("trace_id") or self.session_id)
        return urlunsplit(parsed._replace(query=urlencode(query)))

    async def run(self) -> None:
        await self._run_upstream()

    async def _run_upstream(self) -> None:
        client = self.manager.app[self.manager.upstream_session_key]
        tasks: list[asyncio.Task] = []
        try:
            async with client.ws_connect(
                self.upstream_url,
                max_msg_size=0,
                heartbeat=20,
            ) as upstream:
                self.upstream = upstream
                await upstream.send_bytes(msgspec.msgpack.encode(self.init))
                await self._send_json(
                    {
                        "type": "status",
                        "state": "connected",
                        "session_id": self.session_id,
                        "backend": self.backend,
                        "codec": "h264",
                        "protocol": "websocket",
                        "bitrate_kbps": self.encoder_bitrate_kbps,
                        "crf": self.encoder_crf,
                        "preset": self.encoder_preset,
                        "profile": self.encoder_profile,
                        "gop_seconds": self.encoder_gop_seconds,
                        "vbv_buffer_ms": self.encoder_vbv_buffer_ms,
                        "startup_drop_frames": self.startup_drop_frames,
                    }
                )
                self.encoder_task = asyncio.create_task(
                    self._encode_frames(), name=f"h264ws-encoder-{self.session_id}"
                )
                upstream_task = asyncio.create_task(
                    self._receive_upstream(),
                    name=f"h264ws-upstream-{self.session_id}",
                )
                control_task = asyncio.create_task(
                    self._receive_controls(),
                    name=f"h264ws-control-{self.session_id}",
                )
                tasks = [upstream_task, control_task, self.encoder_task]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if self.encoder_task in done:
                    # The encoder is intentionally an open-ended consumer.  It
                    # can only finish by itself when FFmpeg or the browser send
                    # path failed, so never mistake that for graceful EOS.
                    self.encoder_task.result()
                    raise RuntimeError("H.264 encoder stopped before stream end")
                if control_task in done:
                    # The browser disconnected.  Do not spend time encoding a
                    # tail that no longer has a consumer.
                    control_task.result()
                    return

                close_code = upstream_task.result()
                if self.pending_header is not None:
                    raise RuntimeError("upstream closed before raw frame payload")
                if self.finite_request and self.final_completion is None:
                    raise RuntimeError(
                        "finite upstream closed without final media completion "
                        f"({close_code})"
                    )
                if self.final_completion is None and close_code not in {1000, 1001}:
                    raise RuntimeError(
                        f"upstream H.264 source closed unexpectedly ({close_code})"
                    )
                await self._finish_graceful_stream(abort_task=control_task)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.exception("H.264 WebSocket session %s failed", self.session_id)
            if not self.websocket.closed:
                with contextlib.suppress(Exception):
                    await self._send_json({"type": "error", "message": str(error)})
                if self.finite_request:
                    with contextlib.suppress(Exception):
                        await self.websocket.close(
                            code=1011,
                            message=b"upstream output incomplete",
                        )
        finally:
            self.upstream = None
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._stop_ffmpeg()

    async def _receive_upstream(self) -> int | None:
        async for message in self.upstream:
            if message.type == WSMsgType.BINARY:
                await self._receive_binary(bytes(message.data))
                if self.final_completion is not None:
                    # The final media marker is ordered after the last frame
                    # batch.  Start draining immediately instead of depending
                    # on a second, transport-level close handshake.
                    return 1000
            elif message.type == WSMsgType.TEXT:
                await self._send_json({"type": "upstream", "data": message.data})
            elif message.type == WSMsgType.ERROR:
                error = self.upstream.exception()
                raise RuntimeError(f"upstream WebSocket error: {error or 'unknown'}")
            elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                break
        return self.upstream.close_code

    async def _receive_controls(self) -> None:
        async for message in self.websocket:
            if message.type != WSMsgType.TEXT:
                if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
                continue
            try:
                envelope = json.loads(message.data)
            except json.JSONDecodeError:
                await self._send_json({"type": "error", "message": "invalid JSON"})
                continue
            if envelope.get("type") in {"client_metric", "client_metric_batch"}:
                self._observe_client_metric(envelope)
                continue
            if envelope.get("type") != "event":
                await self._send_json(
                    {"type": "error", "message": "control envelope type must be event"}
                )
                continue
            if envelope.get("kind") not in ALLOWED_CONTROL_KINDS:
                await self._send_json(
                    {"type": "error", "message": "unsupported control event kind"}
                )
                continue
            received_epoch_ms = time.time() * 1000
            started = time.perf_counter()
            event_id = int(envelope.get("event_id") or 0)
            if envelope.get("kind") in {"camera_actions", "prompt", "scene_cut"}:
                self.pending_cutover_event_id = max(
                    self.pending_cutover_event_id, event_id
                )
            if self.upstream is not None and not self.upstream.closed:
                await self.upstream.send_bytes(msgspec.msgpack.encode(envelope))
            await self._send_json(
                {
                    "type": "control_ack",
                    "stage": "bridge",
                    "kind": str(envelope.get("kind") or ""),
                    "event_id": event_id,
                    "client_sent_epoch_ms": envelope.get("client_sent_epoch_ms"),
                    "bridge_received_epoch_ms": round(received_epoch_ms, 3),
                    "server_received_epoch_ms": round(received_epoch_ms, 3),
                    "server_sent_epoch_ms": round(time.time() * 1000, 3),
                    "bridge_forward_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                }
            )

    async def _receive_binary(self, data: bytes) -> None:
        if self.pending_header is not None:
            header = self.pending_header
            self.pending_header = None
            await self._handle_frame_payload(header, data)
            return
        message = msgspec.msgpack.decode(data)
        if not isinstance(message, dict):
            return
        message_type = message.get("type")
        if message_type == "error":
            raise RuntimeError(
                str(
                    message.get("content")
                    or message.get("reason")
                    or message.get("message")
                    or "upstream error"
                )
            )
        if message_type in {"chunk_telemetry", "control_ack"}:
            await self._send_json(message)
            return
        if message_type == "media_chunk_complete":
            if message.get("is_final_chunk") is True:
                self.final_completion = dict(message)
            return
        if message_type not in {"frame_batch", "frame_batch_header"}:
            return
        payload = message.pop("payload", None)
        if payload is None:
            self.pending_header = message
            return
        await self._handle_frame_payload(message, bytes(payload))

    async def _handle_frame_payload(
        self, header: dict[str, Any], payload: bytes
    ) -> None:
        content_type = str(header.get("content_type") or RAW_RGB_CONTENT_TYPE)
        width = int(header.get("width") or header.get("preview_width") or 0)
        height = int(header.get("height") or header.get("preview_height") or 0)
        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            raise ValueError(f"invalid frame dimensions {width}x{height}")
        self.width = width
        self.height = height
        event_id = int(header.get("event_id") or 0)
        if self.pending_cutover_event_id and event_id >= self.pending_cutover_event_id:
            self.minimum_event_id = max(self.minimum_event_id, event_id)
            self.pending_cutover_event_id = 0
            self._discard_queued_before(self.minimum_event_id)
        if self.minimum_event_id and event_id < self.minimum_event_id:
            self.dropped_frames += max(1, int(header.get("num_frames") or 1))
            return
        frames = _split_payload(header, payload)
        chunk_index = int(header.get("chunk_index") or 0)
        if self.startup_drop_remaining:
            if chunk_index > 0:
                # Chunk 0 can be shed before it reaches the bridge. In that
                # case chunk 1 is already the first stable boundary.
                self.startup_drop_remaining = 0
            else:
                drop_count = min(self.startup_drop_remaining, len(frames))
                if drop_count:
                    frames = frames[drop_count:]
                    self.startup_drop_remaining -= drop_count
                    self.startup_dropped_frames += drop_count
                    self.dropped_frames += drop_count
                if not frames:
                    return
        received_epoch_ms = time.time() * 1000
        for frame_index, frame in enumerate(frames):
            if content_type == RAW_RGB_CONTENT_TYPE:
                rgb = frame
            elif content_type in ENCODED_IMAGE_TYPES:
                rgb = await asyncio.to_thread(
                    _encoded_image_to_rgb, frame, width=width, height=height
                )
            else:
                raise ValueError(
                    f"unsupported realtime frame content type: {content_type}"
                )
            queued = _QueuedFrame(
                rgb=rgb,
                width=width,
                height=height,
                chunk_index=chunk_index,
                event_id=event_id,
                frame_batch_index=int(header.get("frame_batch_index") or 0),
                num_frame_batches=int(header.get("num_frame_batches") or 0),
                is_final_frame_batch=(
                    bool(header.get("is_final_frame_batch"))
                    and frame_index == len(frames) - 1
                ),
                bridge_received_epoch_ms=received_epoch_ms,
                server_sent_epoch_ms=float(header.get("server_sent_epoch_ms") or 0),
            )
            if self.finite_request:
                # A finite render promises a complete timeline.  Backpressure
                # the upstream reader instead of silently replacing its tail.
                await self.frame_queue.put(queued)
            else:
                if self.frame_queue.full():
                    dropped = self.frame_queue.get_nowait()
                    self.frame_queue.task_done()
                    self.dropped_frames += 1
                    observe_stage_seconds(
                        "h264_pre_encode_queue",
                        max(
                            0.0,
                            time.time() * 1000 - dropped.bridge_received_epoch_ms,
                        )
                        / 1000.0,
                        service="world-studio-webui",
                        model=self.backend,
                        result="cancelled",
                        codec="h264",
                        scope="frame",
                    )
                self.frame_queue.put_nowait(queued)

    async def _encode_frames(self) -> None:
        # Drain decoded frames as soon as FFmpeg can accept them.  The FPS in
        # the init message defines media timestamps/GOP size only; playback
        # pacing and smoothing belong to the browser.
        while True:
            frame = await self.frame_queue.get()
            while True:
                queue_age_ms = max(
                    0.0, time.time() * 1000 - frame.bridge_received_epoch_ms
                )
                if (
                    not self.finite_request
                    and queue_age_ms > self.manager.max_frame_age_ms
                    and self.frame_queue.qsize() > self.manager.live_edge_frames
                ):
                    self.dropped_frames += 1
                    self.latency_dropped_frames += 1
                    observe_stage_seconds(
                        "h264_pre_encode_queue",
                        queue_age_ms / 1000.0,
                        service="world-studio-webui",
                        model=self.backend,
                        result="cancelled",
                        codec="h264",
                        scope="frame",
                    )
                    self.frame_queue.task_done()
                    frame = await self.frame_queue.get()
                    continue
                break
            try:
                if self.minimum_event_id and frame.event_id < self.minimum_event_id:
                    self.dropped_frames += 1
                    continue
                if self.ffmpeg is None:
                    self.width = frame.width
                    self.height = frame.height
                    await self._start_ffmpeg(frame.width, frame.height)
                elif (frame.width, frame.height) != (self.width, self.height):
                    raise RuntimeError(
                        "H.264 source dimensions changed from "
                        f"{self.width}x{self.height} to {frame.width}x{frame.height}"
                    )
                if self.ffmpeg is None or self.ffmpeg.stdin is None:
                    raise RuntimeError("H.264 encoder is unavailable")
                frame_index = self.frames
                encode_started_s = time.perf_counter()
                encode_started_epoch_ms = time.time() * 1000
                observe_stage_seconds(
                    "h264_pre_encode_queue",
                    max(0.0, encode_started_epoch_ms - frame.bridge_received_epoch_ms)
                    / 1000.0,
                    service="world-studio-webui",
                    model=self.backend,
                    result="success",
                    codec="h264",
                    scope="frame",
                )
                # Publish the frame mapping before feeding ffmpeg.  The stdout
                # pump can then drain a large keyframe immediately without
                # contending with stdin.drain() for the WebSocket send lock.
                # Holding that lock across stdin.drain() can deadlock when an
                # encoded keyframe is larger than the OS pipe buffer.
                async with self.send_lock:
                    await self.websocket.send_json(
                        {
                            "type": "media_batch",
                            "chunk_index": frame.chunk_index,
                            "event_id": frame.event_id,
                            "first_frame_index": frame_index,
                            "num_frames": 1,
                            "frame_batch_index": frame.frame_batch_index,
                            "num_frame_batches": frame.num_frame_batches,
                            "is_final_frame_batch": frame.is_final_frame_batch,
                            "server_sent_epoch_ms": frame.server_sent_epoch_ms,
                            "bridge_received_epoch_ms": frame.bridge_received_epoch_ms,
                            "bridge_encode_started_epoch_ms": encode_started_epoch_ms,
                            "bridge_encoded_epoch_ms": encode_started_epoch_ms,
                            "bridge_queue_ms": max(
                                0.0,
                                encode_started_epoch_ms
                                - frame.bridge_received_epoch_ms,
                            ),
                            "bridge_encoder_feed_ms": 0.0,
                            "dropped_frames": self.dropped_frames,
                            "latency_dropped_frames": self.latency_dropped_frames,
                            "startup_dropped_frames": self.startup_dropped_frames,
                            "repeated_frame": False,
                            "repeated_frames": self.repeated_frames,
                        }
                    )
                feed_started = time.perf_counter()
                try:
                    self.encoder_start_times_s.append(encode_started_s)
                    self.ffmpeg.stdin.write(frame.rgb)
                    await self.ffmpeg.stdin.drain()
                except BaseException as exc:
                    if (
                        self.encoder_start_times_s
                        and self.encoder_start_times_s[-1] == encode_started_s
                    ):
                        self.encoder_start_times_s.pop()
                    observe_stage_seconds(
                        "frame_encode",
                        time.perf_counter() - feed_started,
                        service="world-studio-webui",
                        model=self.backend,
                        result=result_from_exception(exc),
                        codec="h264",
                        scope="frame",
                    )
                    raise
                feed_completed_epoch_ms = time.time() * 1000
                # stdin.drain() measures how long the bridge waited for FFmpeg
                # to accept this frame.  Report it as a separate update: the
                # media metadata must be sent before feeding FFmpeg so stdout
                # can never deadlock behind the WebSocket send lock.
                await self._send_json(
                    {
                        "type": "media_encode_timing",
                        "first_frame_index": frame_index,
                        "bridge_encoded_epoch_ms": feed_completed_epoch_ms,
                        "bridge_encoder_feed_ms": max(
                            0.0,
                            feed_completed_epoch_ms - encode_started_epoch_ms,
                        ),
                    }
                )
                self.frames += 1
            finally:
                self.frame_queue.task_done()

    async def _finish_graceful_stream(
        self, *, abort_task: asyncio.Task | None = None
    ) -> None:
        """Drain every accepted frame, flush fMP4, then publish terminal EOS."""

        if self.finite_request and self.final_completion is None:
            raise RuntimeError("finite stream lacks final media completion")
        if self.encoder_task is None:
            raise RuntimeError("H.264 encoder task is unavailable")
        queue_drained = asyncio.create_task(
            self.frame_queue.join(), name=f"h264ws-drain-{self.session_id}"
        )
        waiters: set[asyncio.Task] = {queue_drained, self.encoder_task}
        if abort_task is not None:
            waiters.add(abort_task)
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=self.manager.drain_timeout_ms / 1000.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("H.264 frame queue drain timed out")
            if self.encoder_task in done:
                self.encoder_task.result()
                raise RuntimeError("H.264 encoder stopped before queue drain")
            if abort_task is not None and abort_task in done:
                abort_task.result()
                raise ConnectionError("browser disconnected during H.264 drain")
            if queue_drained not in done:
                raise RuntimeError("H.264 queue drain stopped unexpectedly")
            await queue_drained
        finally:
            if not queue_drained.done():
                queue_drained.cancel()
            await asyncio.gather(queue_drained, return_exceptions=True)

        self.encoder_task.cancel()
        await asyncio.gather(self.encoder_task, return_exceptions=True)
        await self._stop_ffmpeg(flush_output=True)
        if self.final_completion is None or self.websocket.closed:
            return
        await self._send_json(
            {
                "type": "stream_complete",
                "chunk_index": int(self.final_completion.get("chunk_index") or 0),
                "event_id": int(self.final_completion.get("event_id") or 0),
                "num_frames": int(self.final_completion.get("num_frames") or 0),
                "encoded_frames": self.frames,
                "dropped_frames": self.dropped_frames,
                "media_bytes": self.media_bytes,
            }
        )

    async def _start_ffmpeg(self, width: int, height: int) -> None:
        fps = _bounded_int(self.init.get("fps"), default=24, minimum=1, maximum=60)
        gop = max(4, fps * self.encoder_gop_seconds)
        # FFmpeg's input-side nobuffer flag sheds one raw-video boundary frame
        # in a finite pipe. Keep the existing live-edge behavior for open-ended
        # sessions, but require exact input accounting for finite renders.
        input_buffering_args = [] if self.finite_request else ["-fflags", "nobuffer"]
        command = [
            self.manager.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            *input_buffering_args,
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.encoder_preset,
            "-tune",
            "zerolatency",
            "-profile:v",
            self.encoder_profile,
            "-level:v",
            "3.1",
            "-vf",
            (
                _raw_channel_filter(self.manager.raw_channel_order(self.backend))
                + "scale=in_range=pc:out_range=tv:"
                "in_color_matrix=bt709:out_color_matrix=bt709"
            ),
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-crf",
            str(self.encoder_crf),
            "-maxrate",
            f"{self.encoder_bitrate_kbps}k",
            "-bufsize",
            f"{max(128, self.encoder_bitrate_kbps * self.encoder_vbv_buffer_ms // 1000)}k",
            "-movflags",
            "empty_moov+default_base_moof+frag_every_frame+omit_tfhd_offset",
            "-video_track_timescale",
            "90000",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "pipe:1",
        ]
        self.ffmpeg = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        process = self.ffmpeg
        self.stdout_task = asyncio.create_task(self._pump_stdout(process))
        self.stderr_task = asyncio.create_task(self._log_stderr(process))

    async def _pump_stdout(
        self, process: asyncio.subprocess.Process | None = None
    ) -> None:
        process = process or self.ffmpeg
        if process is None or process.stdout is None:
            return
        while data := await process.stdout.read(64 * 1024):
            packet_ready_s = time.perf_counter()
            self.media_bytes += len(data)
            if self.encoder_start_times_s:
                observe_stage_seconds(
                    "frame_encode",
                    packet_ready_s - self.encoder_start_times_s.popleft(),
                    service="world-studio-webui",
                    model=self.backend,
                    result="success",
                    codec="h264",
                    scope="frame",
                )
            async with self.send_lock:
                # Frame metadata is emitted before FFmpeg is fed, while this
                # header is emitted at the actual WebSocket payload boundary.
                # The browser can therefore measure wire/event-loop downlink
                # independently from encoder and playback buffering latency.
                payload_sent_epoch_ms = time.time() * 1000
                payload_sequence = self.media_payload_sequence
                self.media_payload_sequence += 1
                write_started = time.perf_counter()
                try:
                    await self.websocket.send_json(
                        {
                            "type": "media_payload",
                            "sequence": payload_sequence,
                            "num_bytes": len(data),
                            "server_sent_epoch_ms": payload_sent_epoch_ms,
                        }
                    )
                    await self.websocket.send_bytes(data)
                except BaseException as exc:
                    observe_stage_seconds(
                        "ffmpeg_mux_write",
                        time.perf_counter() - write_started,
                        service="world-studio-webui",
                        model=self.backend,
                        result=result_from_exception(exc),
                        codec="h264",
                        scope="frame",
                    )
                    raise
                observe_stage_seconds(
                    "ffmpeg_mux_write",
                    time.perf_counter() - write_started,
                    service="world-studio-webui",
                    model=self.backend,
                    result="success",
                    codec="h264",
                    scope="frame",
                )

    async def _log_stderr(
        self, process: asyncio.subprocess.Process | None = None
    ) -> None:
        process = process or self.ffmpeg
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            LOGGER.warning(
                "h264ws-ffmpeg[%s]: %s",
                self.session_id,
                line.decode("utf-8", "replace").rstrip(),
            )

    def _discard_queued_before(self, event_id: int) -> None:
        retained: list[_QueuedFrame] = []
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.frame_queue.task_done()
            if frame.event_id < event_id:
                self.dropped_frames += 1
            else:
                retained.append(frame)
        for frame in retained:
            self.frame_queue.put_nowait(frame)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    def _observe_client_metric(self, message: dict[str, Any]) -> None:
        if message.get("type") == "client_metric":
            observe_client_metric_event(
                message,
                service="world-studio-webui",
                model=self.backend,
            )
            return
        events = message.get("events")
        if not isinstance(events, list):
            return
        for event in events[:64]:
            observe_client_metric_event(
                event,
                service="world-studio-webui",
                model=self.backend,
            )

    async def _stop_ffmpeg(self, *, flush_output: bool = False) -> None:
        process = self.ffmpeg
        self.ffmpeg = None
        process_timed_out = False
        kill_wait_timed_out = False
        io_timed_out = False
        timeout_seconds = self.manager.drain_timeout_ms / 1000.0
        io_tasks = [
            task for task in (self.stdout_task, self.stderr_task) if task is not None
        ]

        async def cancel_and_settle(
            tasks: list[asyncio.Task] | set[asyncio.Task],
        ) -> set[asyncio.Task]:
            pending = {task for task in tasks if not task.done()}
            for task in pending:
                task.cancel()
            if not pending:
                return set()
            _, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
            return still_pending

        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(Exception):
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                process_timed_out = True
                # stdout can be blocked in websocket.send_bytes while FFmpeg is
                # blocked on its full output pipe. Cancel those pumps before
                # killing the process, and keep both cancellation and reap
                # waits bounded so graceful shutdown can never deadlock.
                pending_io = await cancel_and_settle(io_tasks)
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                except TimeoutError:
                    kill_wait_timed_out = True
                if pending_io:
                    _, pending_io = await asyncio.wait(
                        pending_io, timeout=timeout_seconds
                    )
                    io_timed_out = bool(pending_io)

        if not process_timed_out and io_tasks:
            _, pending_io = await asyncio.wait(io_tasks, timeout=timeout_seconds)
            if pending_io:
                io_timed_out = True
                await cancel_and_settle(pending_io)

        io_errors: list[BaseException] = []
        for task in io_tasks:
            if not task.done() or task.cancelled():
                continue
            with contextlib.suppress(asyncio.CancelledError):
                error = task.exception()
                if error is not None:
                    io_errors.append(error)
        self.stdout_task = None
        self.stderr_task = None
        # Give the stdout pump the entire graceful-flush window to match
        # accepted frames with emitted fMP4 payloads. Only work that remains
        # after the pump has settled is genuinely cancelled.
        cancelled_at = time.perf_counter()
        while self.encoder_start_times_s:
            observe_stage_seconds(
                "frame_encode",
                cancelled_at - self.encoder_start_times_s.popleft(),
                service="world-studio-webui",
                model=self.backend,
                result="cancelled",
                codec="h264",
                scope="frame",
            )
        if not flush_output:
            return
        if process_timed_out or kill_wait_timed_out or io_timed_out:
            raise TimeoutError("H.264 FFmpeg output flush timed out")
        if process is not None and process.returncode not in {0, None}:
            raise RuntimeError(f"H.264 FFmpeg exited with code {process.returncode}")
        if io_errors:
            raise io_errors[0]


class H264WebSocketBridgeManager:
    def __init__(
        self,
        app: web.Application,
        upstream_session_key: web.AppKey,
        upstream_resolver: Callable[[str], str],
    ) -> None:
        self.app = app
        self.upstream_session_key = upstream_session_key
        self.upstream_resolver = upstream_resolver
        self.ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
        self.preset = os.environ.get("H264_WS_PRESET", "veryfast")
        self.profile = os.environ.get("H264_WS_PROFILE", "main")
        self.crf = _bounded_int(
            os.environ.get("H264_WS_CRF"),
            default=16,
            minimum=12,
            maximum=35,
        )
        self.bitrate_kbps = _bounded_int(
            os.environ.get("H264_WS_BITRATE_KBPS"),
            default=10000,
            minimum=250,
            maximum=20000,
        )
        self.vbv_buffer_ms = _bounded_int(
            os.environ.get("H264_WS_VBV_BUFFER_MS"),
            default=125,
            minimum=40,
            maximum=2000,
        )
        self.gop_seconds = _bounded_int(
            os.environ.get("H264_WS_GOP_SECONDS"),
            default=1,
            minimum=1,
            maximum=5,
        )
        self.max_queued_frames = _bounded_int(
            os.environ.get("H264_WS_MAX_QUEUED_FRAMES"),
            default=12,
            minimum=4,
            maximum=120,
        )
        self.max_frame_age_ms = _bounded_int(
            os.environ.get("H264_WS_MAX_FRAME_AGE_MS"),
            default=250,
            minimum=40,
            maximum=2000,
        )
        self.live_edge_frames = _bounded_int(
            os.environ.get("H264_WS_LIVE_EDGE_FRAMES"),
            default=6,
            minimum=1,
            maximum=self.max_queued_frames,
        )
        self.drain_timeout_ms = _bounded_int(
            os.environ.get("H264_WS_DRAIN_TIMEOUT_MS"),
            default=5000,
            minimum=100,
            maximum=30000,
        )
        self.max_sessions = _bounded_int(
            os.environ.get("H264_WS_MAX_SESSIONS"),
            default=8,
            minimum=1,
            maximum=64,
        )
        self.active_sessions: set[str] = set()

    def upstream_url(self, backend: str) -> str:
        return self.upstream_resolver(backend)

    def raw_channel_order(self, backend: str) -> str:
        prefix = backend.upper().replace("-", "_")
        value = os.environ.get(
            f"{prefix}_H264_WS_RAW_CHANNEL_ORDER",
            os.environ.get("H264_WS_RAW_CHANNEL_ORDER", "rgb"),
        ).lower()
        return value if value in {"rgb", "gbr"} else "rgb"

    def startup_drop_frames(self, backend: str) -> int:
        prefix = backend.upper().replace("-", "_")
        return _bounded_int(
            os.environ.get(f"{prefix}_H264_WS_STARTUP_DROP_FRAMES"),
            default=8 if backend == "lingbot2" else 0,
            minimum=0,
            maximum=120,
        )


async def _h264_websocket(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20)
    await websocket.prepare(request)
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=15)
        if first.type != WSMsgType.TEXT:
            raise ValueError("first H.264 WebSocket message must be JSON init")
        init = json.loads(first.data)
        if not isinstance(init, dict) or init.get("type") != "init":
            raise ValueError("H.264 WebSocket init is invalid")
        init = dict(init)
        init["first_frame"] = _decode_first_frame(init.get("first_frame"))
        init["realtime_output_format"] = "raw"
        init.pop("realtime_preview_max_width", None)
        init.pop("output_compression", None)
        init["playback_ack_enabled"] = bool(init.get("playback_ack_enabled"))
        manager: H264WebSocketBridgeManager = request.app[H264_WS_MANAGER]
        backend = str(request.match_info.get("backend") or "minwm")
        try:
            manager.upstream_url(backend)
        except KeyError as error:
            raise ValueError(f"unknown realtime backend: {backend}") from error
        if len(manager.active_sessions) >= manager.max_sessions:
            raise ValueError("H.264 bridge has no free encoder slot")
        session = H264WebSocketSession(
            manager=manager,
            websocket=websocket,
            init=init,
            backend=backend,
        )
        manager.active_sessions.add(session.session_id)
        try:
            await session.run()
        finally:
            manager.active_sessions.discard(session.session_id)
    except (json.JSONDecodeError, ValueError, TimeoutError) as error:
        if not websocket.closed:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"{type(error).__name__}: {error!r}",
                }
            )
    finally:
        if not websocket.closed:
            await websocket.close()
    return websocket


async def _metrics(_request: web.Request) -> web.Response:
    return web.Response(
        body=prometheus_latest(),
        headers={"Content-Type": prometheus_content_type()},
    )


def install_h264_websocket_bridge(
    app: web.Application,
    *,
    upstream_session_key: web.AppKey,
    upstream_resolver: Callable[[str], str],
) -> None:
    """Install backend-aware H.264/fMP4 WebSocket endpoints."""
    app[H264_WS_MANAGER] = H264WebSocketBridgeManager(
        app,
        upstream_session_key,
        upstream_resolver,
    )
    app.router.add_get("/metrics", _metrics)
    app.router.add_get("/api/h264ws", _h264_websocket)
    app.router.add_get("/api/h264ws/{backend}", _h264_websocket)
