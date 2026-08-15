# SPDX-License-Identifier: Apache-2.0
"""Bridge realtime Zing frames to H.264 RTP/WebRTC through MediaMTX.

The model-facing leg stays inside the cluster and requests raw RGB frames over
the existing realtime WebSocket protocol.  FFmpeg converts those frames to a
low-latency H.264 RTSP publisher; MediaMTX exposes the stream as WHEP/WebRTC.
Only control events remain on a small WebSocket connection.
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
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import msgspec
from aiohttp import WSMsgType, web
from PIL import Image


LOGGER = logging.getLogger(__name__)
BRIDGE_MANAGER = web.AppKey("webrtc_bridge_manager", object)
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
ENCODED_IMAGE_TYPES = {"image/jpeg", "image/webp", "image/png"}
ALLOWED_CONTROL_KINDS = {
    "camera_actions",
    "heartbeat",
    "playback_ack",
    "prompt",
}


def _raw_channel_filter(channel_order: str) -> str:
    """Return the FFmpeg filter that normalizes packed source bytes to RGB."""
    # The remote VAE raw transport used by the Zing lab currently emits packed
    # GBR bytes even though the wire content type is application/x-raw-rgb.
    # Keep the correction explicit and configurable so conventional RGB
    # producers do not receive an unconditional channel swap.
    if channel_order == "gbr":
        return (
            "colorchannelmixer="
            "rr=0:rg=0:rb=1:"
            "gr=1:gg=0:gb=0:"
            "br=0:bg=1:bb=0,"
        )
    return ""


def _playback_ack_enabled(init: dict[str, Any]) -> bool:
    """Preserve ACK limiting for both Canvas and native video playback."""
    return bool(init.get("playback_ack_enabled"))


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
        raise web.HTTPBadRequest(text="first_frame data URL is invalid") from error


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

    content_type = str(header.get("content_type") or "")
    if content_type == RAW_RGB_CONTENT_TYPE:
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
class WebRTCBridgeSession:
    manager: "WebRTCBridgeManager"
    session_id: str
    init: dict[str, Any]
    codec: str
    bitrate_kbps: int
    source_only: bool = False
    created_at: float = field(default_factory=time.time)
    state: str = "connecting"
    error: str = ""
    frames: int = 0
    source_bytes: int = 0
    width: int = 0
    height: int = 0
    ffmpeg: asyncio.subprocess.Process | None = None
    upstream: Any = None
    task: asyncio.Task | None = None
    encoder_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    pending_header: dict[str, Any] | None = None
    control_clients: set[web.WebSocketResponse] = field(default_factory=set)
    media_batch_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=1024)
    )
    media_fps: int = 24
    next_frame_deadline: float | None = None
    minimum_event_id: int = 0
    pending_cutover_event_id: int = 0
    transition_cutovers: int = 0
    dropped_batches: int = 0
    dropped_frames: int = 0
    dropped_source_bytes: int = 0
    last_media_event_id: int = 0
    last_media_event_epoch_ms: float = 0.0
    queue_overflow_dropped_frames: int = 0
    control_dropped_frames: int = 0
    latency_dropped_frames: int = 0
    repeated_frames: int = 0
    comparison_frame_subscribers: set[asyncio.Queue[_QueuedFrame]] = field(
        default_factory=set
    )
    comparison_metadata_subscribers: set[asyncio.Queue[dict[str, Any]]] = field(
        default_factory=set
    )
    frame_queue: asyncio.Queue[_QueuedFrame] = field(init=False)

    def __post_init__(self) -> None:
        self.frame_queue = asyncio.Queue(
            maxsize=self.manager.bridge_max_queued_frames
        )

    @property
    def media_path(self) -> str:
        return f"zing-{self.session_id}"

    @property
    def stream_page_url(self) -> str:
        base = self.manager.media_http_base.rstrip("/")
        return (
            f"{base}/{self.media_path}"
            "?autoplay=true&muted=true&controls=false&playsInline=true"
        )

    @property
    def whep_url(self) -> str:
        return f"{self.manager.media_http_base.rstrip('/')}/{self.media_path}/whep"

    @property
    def upstream_url(self) -> str:
        parsed = urlsplit(self.manager.upstream_ws)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["user_id"] = f"webrtc-{self.session_id}"
        query["trace_id"] = str(self.init.get("trace_id") or self.session_id)
        return urlunsplit(parsed._replace(query=urlencode(query)))

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name=f"webrtc-bridge-{self.session_id}")
        try:
            await asyncio.wait_for(self.connected.wait(), timeout=15)
        except TimeoutError as error:
            await self.stop("upstream connect timeout")
            raise web.HTTPBadGateway(text="Zing upstream connection timed out") from error
        if self.state == "error":
            raise web.HTTPBadGateway(text=self.error or "Zing upstream unavailable")

    async def _run(self) -> None:
        try:
            client = self.manager.app[self.manager.upstream_session_key]
            async with client.ws_connect(
                self.upstream_url,
                max_msg_size=0,
                heartbeat=20,
            ) as upstream:
                self.upstream = upstream
                await upstream.send_bytes(msgspec.msgpack.encode(self.init))
                self.state = "generating"
                self.connected.set()
                if not self.source_only:
                    self.encoder_task = asyncio.create_task(
                        self._encode_frames(),
                        name=f"webrtc-encoder-{self.session_id}",
                    )
                async for message in upstream:
                    if self.encoder_task is not None and self.encoder_task.done():
                        await self.encoder_task
                    if message.type == WSMsgType.BINARY:
                        await self._receive_binary(bytes(message.data))
                    elif message.type == WSMsgType.TEXT:
                        await self._broadcast({"type": "upstream", "data": message.data})
                    elif message.type in {
                        WSMsgType.CLOSE,
                        WSMsgType.CLOSED,
                        WSMsgType.ERROR,
                    }:
                        break
                if self.state not in {"closing", "closed", "error"}:
                    self.state = "closed"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.exception("WebRTC bridge session %s failed", self.session_id)
            self.state = "error"
            self.error = str(error)
            self.connected.set()
            await self._broadcast({"type": "error", "message": self.error})
        finally:
            self.upstream = None
            await self._stop_encoder()
            await self._stop_ffmpeg()
            self.stopped.set()

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
            detail = (
                message.get("content")
                or message.get("reason")
                or message.get("message")
                or "upstream error"
            )
            raise RuntimeError(str(detail))
        if message_type == "media_chunk_complete":
            # Per-frame bridge metadata already carries is_final_frame_batch.
            # Forwarding this completion immediately would race ahead of the
            # independently paced encoder and could falsely mark an early
            # frame as the end of the chunk in the browser.
            return
        if message_type in {"chunk_telemetry", "control_ack"}:
            await self._broadcast(message)
            return
        if message_type not in {"frame_batch", "frame_batch_header"}:
            return
        payload = message.pop("payload", None)
        if payload is None:
            self.pending_header = message
            return
        await self._handle_frame_payload(message, bytes(payload))

    async def _handle_frame_payload(self, header: dict[str, Any], payload: bytes) -> None:
        content_type = str(header.get("content_type") or RAW_RGB_CONTENT_TYPE)
        width = int(header.get("width") or header.get("preview_width") or 0)
        height = int(header.get("height") or header.get("preview_height") or 0)
        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            raise ValueError(f"invalid frame dimensions {width}x{height}")
        self.width = width
        self.height = height
        self.source_bytes += len(payload)
        event_id = int(header.get("event_id") or 0)
        if self.pending_cutover_event_id and event_id >= self.pending_cutover_event_id:
            self.minimum_event_id = max(self.minimum_event_id, event_id)
            self.pending_cutover_event_id = 0
            dropped = self._discard_queued_before(self.minimum_event_id)
            self.transition_cutovers += 1
            LOGGER.info(
                "WebRTC bridge %s cut over on first media for event=%s "
                "and discarded %s stale queued frames",
                self.session_id,
                self.minimum_event_id,
                dropped,
            )
        if self.minimum_event_id and event_id < self.minimum_event_id:
            dropped_frames = max(1, int(header.get("num_frames") or 1))
            self.dropped_batches += 1
            self.dropped_frames += dropped_frames
            self.dropped_source_bytes += len(payload)
            LOGGER.info(
                "WebRTC bridge %s shed stale media chunk=%s event=%s before event=%s frames=%s",
                self.session_id,
                int(header.get("chunk_index") or 0),
                event_id,
                self.minimum_event_id,
                dropped_frames,
            )
            return
        frames = _split_payload(header, payload)
        bridge_received_epoch_ms = time.time() * 1000
        if event_id != self.last_media_event_id:
            self.last_media_event_id = event_id
            self.last_media_event_epoch_ms = bridge_received_epoch_ms
        chunk_index = int(header.get("chunk_index") or 0)
        frame_batch_index = int(header.get("frame_batch_index") or 0)
        num_frame_batches = int(header.get("num_frame_batches") or 0)
        is_final_frame_batch = bool(header.get("is_final_frame_batch"))
        for frame_index, frame in enumerate(frames):
            if content_type == RAW_RGB_CONTENT_TYPE:
                rgb = frame
            elif content_type in ENCODED_IMAGE_TYPES:
                rgb = await asyncio.to_thread(
                    _encoded_image_to_rgb,
                    frame,
                    width=width,
                    height=height,
                )
            else:
                raise ValueError(f"unsupported realtime frame content type: {content_type}")
            queued = _QueuedFrame(
                rgb=rgb,
                width=width,
                height=height,
                chunk_index=chunk_index,
                event_id=event_id,
                frame_batch_index=frame_batch_index,
                num_frame_batches=num_frame_batches,
                is_final_frame_batch=(
                    is_final_frame_batch and frame_index == len(frames) - 1
                ),
                bridge_received_epoch_ms=bridge_received_epoch_ms,
                server_sent_epoch_ms=float(header.get("server_sent_epoch_ms") or 0),
            )
            self._enqueue_frame(queued)

    def _enqueue_frame(self, frame: _QueuedFrame) -> None:
        if self.minimum_event_id and frame.event_id < self.minimum_event_id:
            self.dropped_frames += 1
            self.dropped_source_bytes += len(frame.rgb)
            self.control_dropped_frames += 1
            return
        closed_subscribers: list[asyncio.Queue[_QueuedFrame]] = []
        for subscriber in self.comparison_frame_subscribers:
            try:
                if subscriber.full():
                    subscriber.get_nowait()
                    subscriber.task_done()
                subscriber.put_nowait(frame)
            except Exception:
                closed_subscribers.append(subscriber)
        for subscriber in closed_subscribers:
            self.comparison_frame_subscribers.discard(subscriber)
        if self.source_only:
            # The bitrate A/B lab needs one authoritative model/control
            # session, but its raw frames are encoded only by the two H.264
            # WebSocket subscribers. Avoid starting a third, invisible
            # H.264/RTP publisher that would consume CPU and browser bandwidth
            # and make the transport measurements impossible to interpret.
            self.frames += 1
            self.state = "streaming"
            return
        if self.frame_queue.full():
            dropped = self.frame_queue.get_nowait()
            self.frame_queue.task_done()
            self.dropped_frames += 1
            self.dropped_source_bytes += len(dropped.rgb)
            self.queue_overflow_dropped_frames += 1
        self.frame_queue.put_nowait(frame)

    async def _encode_frames(self) -> None:
        last_frame: _QueuedFrame | None = None
        while True:
            await self._pace_frame()
            from_queue = False
            while True:
                try:
                    frame = self.frame_queue.get_nowait()
                    from_queue = True
                except asyncio.QueueEmpty:
                    if last_frame is None:
                        frame = await self.frame_queue.get()
                        from_queue = True
                    else:
                        frame = last_frame
                        self.repeated_frames += 1
                    break
                queue_age_ms = max(
                    0.0,
                    time.time() * 1000 - frame.bridge_received_epoch_ms,
                )
                if (
                    queue_age_ms > self.manager.bridge_max_frame_age_ms
                    and self.frame_queue.qsize() > self.manager.bridge_live_edge_frames
                ):
                    self.dropped_frames += 1
                    self.dropped_source_bytes += len(frame.rgb)
                    self.latency_dropped_frames += 1
                    self.frame_queue.task_done()
                    from_queue = False
                    continue
                break
            try:
                if self.minimum_event_id and frame.event_id < self.minimum_event_id:
                    self.dropped_frames += 1
                    self.dropped_source_bytes += len(frame.rgb)
                    self.control_dropped_frames += 1
                    continue
                if self.ffmpeg is None:
                    await self._start_ffmpeg(self.width, self.height)
                if self.ffmpeg is None or self.ffmpeg.stdin is None:
                    raise RuntimeError("H.264 encoder is unavailable")
                encode_started_epoch_ms = time.time() * 1000
                self.ffmpeg.stdin.write(frame.rgb)
                await self.ffmpeg.stdin.drain()
                encoded_epoch_ms = time.time() * 1000
                repeated_frame = not from_queue
                media_batch = {
                    "type": "media_batch",
                    "chunk_index": frame.chunk_index,
                    "event_id": frame.event_id,
                    "first_frame_index": self.frames,
                    "num_frames": 1,
                    "frame_batch_index": frame.frame_batch_index,
                    "num_frame_batches": frame.num_frame_batches,
                    "is_final_frame_batch": frame.is_final_frame_batch,
                    "server_sent_epoch_ms": (
                        encode_started_epoch_ms
                        if repeated_frame
                        else frame.server_sent_epoch_ms
                    ),
                    "bridge_received_epoch_ms": (
                        encode_started_epoch_ms
                        if repeated_frame
                        else frame.bridge_received_epoch_ms
                    ),
                    "bridge_encode_started_epoch_ms": encode_started_epoch_ms,
                    "bridge_encoded_epoch_ms": encoded_epoch_ms,
                    "bridge_queue_ms": max(
                        0.0,
                        0.0
                        if repeated_frame
                        else encode_started_epoch_ms
                        - frame.bridge_received_epoch_ms,
                    ),
                    "bridge_encoder_feed_ms": max(
                        0.0, encoded_epoch_ms - encode_started_epoch_ms
                    ),
                    "repeated_frame": repeated_frame,
                }
                self.media_batch_history.append(media_batch)
                await self._broadcast(media_batch)
                self.frames += 1
                self.state = "streaming"
                last_frame = frame
            finally:
                if from_queue:
                    self.frame_queue.task_done()

    def _discard_queued_before(self, event_id: int) -> int:
        retained: list[_QueuedFrame] = []
        dropped = 0
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.frame_queue.task_done()
            if frame.event_id < event_id:
                dropped += 1
                self.dropped_frames += 1
                self.dropped_source_bytes += len(frame.rgb)
                self.control_dropped_frames += 1
            else:
                retained.append(frame)
        for frame in retained:
            self.frame_queue.put_nowait(frame)
        return dropped

    async def _pace_frame(self) -> None:
        interval = 1 / max(1, self.media_fps)
        now = asyncio.get_running_loop().time()
        if self.next_frame_deadline is None or now - self.next_frame_deadline > interval * 4:
            self.next_frame_deadline = now
        delay_s = self.next_frame_deadline - now
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        self.next_frame_deadline += interval

    async def _start_ffmpeg(self, width: int, height: int) -> None:
        fps = _bounded_int(self.init.get("fps"), default=16, minimum=1, maximum=60)
        self.media_fps = fps
        self.next_frame_deadline = None
        # A one-second GOP retains fast decoder recovery while avoiding the
        # excessive IDR overhead of the previous half-second GOP on detailed
        # 720p scenes.
        gop = max(4, fps * self.manager.h264_gop_seconds)
        rtsp_url = f"{self.manager.media_rtsp_base.rstrip('/')}/{self.media_path}"
        command = [
            self.manager.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-avioflags",
            "direct",
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
            self.manager.h264_preset,
            "-tune",
            "zerolatency",
            "-profile:v",
            self.manager.h264_profile,
            "-level:v",
            "3.1",
            "-vf",
            (
                _raw_channel_filter(self.manager.raw_channel_order)
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
            str(self.manager.h264_crf),
            "-maxrate",
            f"{self.bitrate_kbps}k",
            "-bufsize",
            f"{max(128, self.bitrate_kbps * self.manager.h264_vbv_buffer_ms // 1000)}k",
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            rtsp_url,
        ]
        self.ffmpeg = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self.stderr_task = asyncio.create_task(self._log_ffmpeg_stderr())

    async def _log_ffmpeg_stderr(self) -> None:
        process = self.ffmpeg
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            LOGGER.warning(
                "ffmpeg[%s]: %s",
                self.session_id,
                line.decode("utf-8", "replace").rstrip(),
            )

    async def send_control(self, envelope: dict[str, Any]) -> None:
        if envelope.get("type") != "event":
            raise web.HTTPBadRequest(text="control envelope type must be event")
        if envelope.get("kind") not in ALLOWED_CONTROL_KINDS:
            raise web.HTTPBadRequest(text="unsupported control event kind")
        upstream = self.upstream
        if upstream is None or upstream.closed:
            raise web.HTTPConflict(text="Zing upstream session is not connected")
        event_id = int(envelope.get("event_id") or 0)
        if envelope.get("kind") in {"camera_actions", "prompt", "scene_cut"}:
            # Keep presenting already-generated media until the first frame for
            # the new control state actually arrives. Purging here creates an
            # avoidable one-chunk underflow: the old queue is empty while the
            # model is still generating the new latent chunk. The receive path
            # performs an atomic stale-frame cutover when new-event media is
            # available, preserving responsiveness without a visible freeze.
            self.pending_cutover_event_id = max(
                self.pending_cutover_event_id,
                event_id,
            )
        await upstream.send_bytes(msgspec.msgpack.encode(envelope))

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if payload.get("type") in {"chunk_telemetry", "control_ack"}:
            closed_subscribers: list[asyncio.Queue[dict[str, Any]]] = []
            for subscriber in self.comparison_metadata_subscribers:
                try:
                    if subscriber.full():
                        subscriber.get_nowait()
                        subscriber.task_done()
                    subscriber.put_nowait(dict(payload))
                except Exception:
                    closed_subscribers.append(subscriber)
            for subscriber in closed_subscribers:
                self.comparison_metadata_subscribers.discard(subscriber)
        closed: list[web.WebSocketResponse] = []
        for client in self.control_clients:
            try:
                await client.send_json(payload)
            except Exception:
                closed.append(client)
        for client in closed:
            self.control_clients.discard(client)

    async def _stop_ffmpeg(self) -> None:
        process = self.ffmpeg
        self.ffmpeg = None
        if process is None:
            return
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self.stderr_task is not None:
            self.stderr_task.cancel()
            await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.stderr_task = None

    async def _stop_encoder(self) -> None:
        task = self.encoder_task
        self.encoder_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def stop(self, reason: str = "client stopped") -> None:
        if self.state in {"closing", "closed"} and self.stopped.is_set():
            return
        self.state = "closing"
        await self._broadcast({"type": "closing", "reason": reason})
        if self.upstream is not None and not self.upstream.closed:
            await self.upstream.close(code=1000, message=reason.encode()[:120])
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._stop_encoder()
        await self._stop_ffmpeg()
        self.state = "closed"
        self.stopped.set()
        for client in list(self.control_clients):
            await client.close(code=1000, message=b"session stopped")
        self.control_clients.clear()
        self.comparison_frame_subscribers.clear()
        self.comparison_metadata_subscribers.clear()

    def status(self) -> dict[str, Any]:
        elapsed = max(0.001, time.time() - self.created_at)
        return {
            "id": self.session_id,
            "state": self.state,
            "error": self.error,
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
            "source_only": self.source_only,
            "h264_preset": self.manager.h264_preset,
            "h264_profile": self.manager.h264_profile,
            "h264_crf": self.manager.h264_crf,
            "h264_vbv_buffer_ms": self.manager.h264_vbv_buffer_ms,
            "h264_gop_seconds": self.manager.h264_gop_seconds,
            "raw_channel_order": self.manager.raw_channel_order,
            "frames": self.frames,
            "source_bytes": self.source_bytes,
            "average_source_mbps": round(self.source_bytes * 8 / elapsed / 1_000_000, 3),
            "minimum_event_id": self.minimum_event_id,
            "pending_cutover_event_id": self.pending_cutover_event_id,
            "transition_cutovers": self.transition_cutovers,
            "dropped_batches": self.dropped_batches,
            "dropped_frames": self.dropped_frames,
            "dropped_source_bytes": self.dropped_source_bytes,
            "queued_frames": self.frame_queue.qsize(),
            "max_queued_frames": self.frame_queue.maxsize,
            "queue_overflow_dropped_frames": self.queue_overflow_dropped_frames,
            "control_dropped_frames": self.control_dropped_frames,
            "latency_dropped_frames": self.latency_dropped_frames,
            "repeated_frames": self.repeated_frames,
            "max_frame_age_ms": self.manager.bridge_max_frame_age_ms,
            "live_edge_frames": self.manager.bridge_live_edge_frames,
            "last_media_event_id": self.last_media_event_id,
            "last_media_event_epoch_ms": self.last_media_event_epoch_ms,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
            "stream_page_url": "" if self.source_only else self.stream_page_url,
            "whep_url": "" if self.source_only else self.whep_url,
        }


class WebRTCBridgeManager:
    def __init__(self, app: web.Application, upstream_session_key: web.AppKey) -> None:
        self.app = app
        self.upstream_session_key = upstream_session_key
        self.upstream_ws = os.environ.get(
            "ZING_WEBRTC_UPSTREAM_WS",
            "ws://127.0.0.1:30000/v1/realtime_video/generate",
        )
        self.media_rtsp_base = os.environ.get(
            "WEBRTC_MEDIA_RTSP_BASE", "rtsp://127.0.0.1:8554"
        )
        self.media_http_base = os.environ.get(
            "WEBRTC_MEDIA_HTTP_BASE", "http://127.0.0.1:8889"
        )
        self.ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
        requested_h264_preset = os.environ.get(
            "WEBRTC_H264_PRESET", "superfast"
        ).lower()
        self.h264_preset = (
            requested_h264_preset
            if requested_h264_preset
            in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium"}
            else "superfast"
        )
        requested_h264_profile = os.environ.get(
            "WEBRTC_H264_PROFILE", "main"
        ).lower()
        self.h264_profile = (
            requested_h264_profile
            if requested_h264_profile in {"baseline", "main", "high"}
            else "main"
        )
        self.h264_crf = _bounded_int(
            os.environ.get("WEBRTC_H264_CRF"),
            default=20,
            minimum=12,
            maximum=35,
        )
        self.h264_vbv_buffer_ms = _bounded_int(
            os.environ.get("WEBRTC_H264_VBV_BUFFER_MS"),
            default=250,
            minimum=100,
            maximum=2000,
        )
        self.h264_gop_seconds = _bounded_int(
            os.environ.get("WEBRTC_H264_GOP_SECONDS"),
            default=1,
            minimum=1,
            maximum=5,
        )
        requested_raw_channel_order = os.environ.get(
            "WEBRTC_RAW_CHANNEL_ORDER", "rgb"
        ).lower()
        self.raw_channel_order = (
            requested_raw_channel_order
            if requested_raw_channel_order in {"rgb", "gbr"}
            else "rgb"
        )
        self.max_sessions = _bounded_int(
            os.environ.get("WEBRTC_BRIDGE_MAX_SESSIONS"),
            default=8,
            minimum=1,
            maximum=64,
        )
        self.bridge_max_queued_frames = _bounded_int(
            os.environ.get("WEBRTC_BRIDGE_MAX_QUEUED_FRAMES"),
            default=24,
            minimum=4,
            maximum=120,
        )
        self.bridge_max_frame_age_ms = _bounded_int(
            os.environ.get("WEBRTC_BRIDGE_MAX_FRAME_AGE_MS"),
            default=250,
            minimum=40,
            maximum=2000,
        )
        self.bridge_live_edge_frames = _bounded_int(
            os.environ.get("WEBRTC_BRIDGE_LIVE_EDGE_FRAMES"),
            default=6,
            minimum=1,
            maximum=self.bridge_max_queued_frames,
        )
        self.ttl_s = _bounded_int(
            os.environ.get("WEBRTC_BRIDGE_SESSION_TTL_S"),
            default=600,
            minimum=30,
            maximum=3600,
        )
        self.sessions: dict[str, WebRTCBridgeSession] = {}
        self.reaper: asyncio.Task | None = None

    async def start(self) -> None:
        self.reaper = asyncio.create_task(self._reap_loop(), name="webrtc-bridge-reaper")

    async def close(self) -> None:
        if self.reaper is not None:
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)
        await asyncio.gather(
            *(session.stop("bridge shutdown") for session in self.sessions.values()),
            return_exceptions=True,
        )
        self.sessions.clear()

    async def create(self, body: dict[str, Any]) -> WebRTCBridgeSession:
        active = [
            session
            for session in self.sessions.values()
            if session.state not in {"closed", "error"}
        ]
        if len(active) >= self.max_sessions:
            raise web.HTTPTooManyRequests(text="WebRTC bridge capacity exhausted")
        init = body.get("init")
        if not isinstance(init, dict):
            raise web.HTTPBadRequest(text="request must include an init object")
        init = dict(init)
        init["first_frame"] = _decode_first_frame(init.get("first_frame"))
        init["realtime_output_format"] = "raw"
        init.pop("realtime_preview_max_width", None)
        init.pop("output_compression", None)
        # ACK-aware output limiting is independent from client-side managed
        # playback. Native <video> presentation callbacks provide truthful
        # rendered-chunk ACKs without routing frames through the Canvas queue.
        init["playback_ack_enabled"] = _playback_ack_enabled(init)
        init.setdefault("trace_id", f"webrtc-{uuid.uuid4()}")
        codec = str(body.get("codec") or "h264").lower()
        if codec != "h264":
            raise web.HTTPBadRequest(text="only H.264 is enabled for the WebRTC lab")
        bitrate_kbps = _bounded_int(
            body.get("bitrate_kbps"), default=3500, minimum=250, maximum=20000
        )
        session_id = uuid.uuid4().hex
        session = WebRTCBridgeSession(
            manager=self,
            session_id=session_id,
            init=init,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            source_only=bool(body.get("source_only")),
        )
        self.sessions[session_id] = session
        try:
            await session.start()
        except Exception:
            self.sessions.pop(session_id, None)
            raise
        return session

    async def remove(self, session_id: str, reason: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        await session.stop(reason)
        return True

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            cutoff = time.time() - self.ttl_s
            expired = [
                session_id
                for session_id, session in self.sessions.items()
                if session.created_at < cutoff
            ]
            for session_id in expired:
                await self.remove(session_id, "session TTL reached")


def _manager(request: web.Request) -> WebRTCBridgeManager:
    return request.app[BRIDGE_MANAGER]


async def _create_session(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(text="request body must be valid JSON") from error
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    session = await _manager(request).create(body)
    return web.json_response(session.status(), status=201)


async def _get_session(request: web.Request) -> web.Response:
    session = _manager(request).sessions.get(request.match_info["session_id"])
    if session is None:
        raise web.HTTPNotFound(text="unknown WebRTC session")
    return web.json_response(session.status(), headers={"Cache-Control": "no-store"})


async def _delete_session(request: web.Request) -> web.Response:
    removed = await _manager(request).remove(
        request.match_info["session_id"], "client requested stop"
    )
    if not removed:
        raise web.HTTPNotFound(text="unknown WebRTC session")
    return web.json_response({"stopped": True})


async def _control_session(request: web.Request) -> web.WebSocketResponse:
    session = _manager(request).sessions.get(request.match_info["session_id"])
    if session is None:
        raise web.HTTPNotFound(text="unknown WebRTC session")
    websocket = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024, heartbeat=20)
    await websocket.prepare(request)
    session.control_clients.add(websocket)
    await websocket.send_json({"type": "status", **session.status()})
    for media_batch in session.media_batch_history:
        await websocket.send_json(media_batch)
    try:
        async for message in websocket:
            if message.type == WSMsgType.TEXT:
                try:
                    envelope = json.loads(message.data)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "invalid JSON"})
                    continue
            elif message.type == WSMsgType.BINARY:
                try:
                    envelope = msgspec.msgpack.decode(message.data)
                except Exception:
                    await websocket.send_json(
                        {"type": "error", "message": "invalid MessagePack"}
                    )
                    continue
            else:
                break
            if not isinstance(envelope, dict):
                await websocket.send_json(
                    {"type": "error", "message": "control event must be an object"}
                )
                continue
            try:
                received_epoch_ms = time.time() * 1000
                forward_started = time.perf_counter()
                await session.send_control(envelope)
                await websocket.send_json(
                    {
                        "type": "control_ack",
                        "stage": "bridge",
                        "kind": str(envelope.get("kind") or ""),
                        "event_id": int(envelope.get("event_id") or 0),
                        "client_sent_epoch_ms": envelope.get("client_sent_epoch_ms"),
                        "bridge_received_epoch_ms": round(received_epoch_ms, 3),
                        "server_received_epoch_ms": round(received_epoch_ms, 3),
                        "server_sent_epoch_ms": round(time.time() * 1000, 3),
                        "bridge_forward_ms": round(
                            (time.perf_counter() - forward_started) * 1000, 3
                        ),
                        "minimum_event_id": session.minimum_event_id,
                        "pending_cutover_event_id": session.pending_cutover_event_id,
                    }
                )
            except web.HTTPException as error:
                await websocket.send_json({"type": "error", "message": error.text})
    finally:
        session.control_clients.discard(websocket)
    return websocket


async def _bridge_context(app: web.Application):
    manager = app[BRIDGE_MANAGER]
    await manager.start()
    yield
    await manager.close()


def install_webrtc_bridge(
    app: web.Application, *, upstream_session_key: web.AppKey
) -> None:
    """Install isolated WebRTC bridge routes into the existing WebUI server."""
    manager = WebRTCBridgeManager(app, upstream_session_key)
    app[BRIDGE_MANAGER] = manager
    app.cleanup_ctx.append(_bridge_context)
    app.router.add_post("/api/webrtc/sessions", _create_session)
    app.router.add_get("/api/webrtc/sessions/{session_id}", _get_session)
    app.router.add_delete("/api/webrtc/sessions/{session_id}", _delete_session)
    app.router.add_get(
        "/api/webrtc/sessions/{session_id}/control", _control_session
    )
