# SPDX-License-Identifier: Apache-2.0
"""Low-latency H.264 over WebSocket bridge for protocol A/B testing.

This deliberately shares the same raw Zing source and x264 settings as the
WebRTC bridge.  The only material difference is the delivery leg: fragmented
MP4 is sent over one WebSocket and appended to MSE in the browser.  That makes
the three-way lab comparison isolate transport/playout behaviour rather than
model or VAE differences.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import msgspec
from aiohttp import WSMsgType, web

from webrtc_bridge import (
    ALLOWED_CONTROL_KINDS,
    BRIDGE_MANAGER,
    ENCODED_IMAGE_TYPES,
    RAW_RGB_CONTENT_TYPE,
    _bounded_int,
    _decode_first_frame,
    _encoded_image_to_rgb,
    _raw_channel_filter,
    _split_payload,
)


LOGGER = logging.getLogger(__name__)
H264_WS_MANAGER = web.AppKey("h264_websocket_bridge_manager", object)


@dataclass(frozen=True)
class _QueuedFrame:
    rgb: bytes
    chunk_index: int
    event_id: int
    frame_batch_index: int
    num_frame_batches: int
    is_final_frame_batch: bool
    bridge_received_epoch_ms: float
    server_sent_epoch_ms: float


@dataclass
class H264WebSocketSession:
    manager: "H264WebSocketBridgeManager"
    websocket: web.WebSocketResponse
    init: dict[str, Any]
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
    media_fps: int = 24
    next_frame_deadline: float | None = None
    minimum_event_id: int = 0
    pending_cutover_event_id: int = 0
    dropped_frames: int = 0
    latency_dropped_frames: int = 0
    shared_metadata_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=128)
    )
    frame_queue: asyncio.Queue[_QueuedFrame] = field(init=False)

    def __post_init__(self) -> None:
        self.frame_queue = asyncio.Queue(maxsize=self.manager.max_queued_frames)

    @property
    def upstream_url(self) -> str:
        parsed = urlsplit(self.manager.upstream_ws)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["user_id"] = f"h264ws-{self.session_id}"
        query["trace_id"] = str(self.init.get("trace_id") or self.session_id)
        return urlunsplit(parsed._replace(query=urlencode(query)))

    async def run(self) -> None:
        shared_session_id = str(self.init.get("shared_webrtc_session_id") or "")
        if shared_session_id:
            manager = self.manager.app.get(BRIDGE_MANAGER)
            shared_session = manager.sessions.get(shared_session_id) if manager else None
            if shared_session is None:
                raise RuntimeError("shared WebRTC source session is unavailable")
            await self._run_shared(shared_session)
            return
        await self._run_upstream()

    async def _run_shared(self, shared_session: Any) -> None:
        tasks: list[asyncio.Task] = []
        shared_session.comparison_frame_subscribers.add(self.frame_queue)
        shared_session.comparison_metadata_subscribers.add(self.shared_metadata_queue)
        try:
            await self._send_json(
                {
                    "type": "status",
                    "state": "connected",
                    "session_id": self.session_id,
                    "shared_webrtc_session_id": shared_session.session_id,
                    "codec": "h264",
                    "protocol": "websocket",
                }
            )
            self.encoder_task = asyncio.create_task(
                self._encode_frames(), name=f"h264ws-encoder-{self.session_id}"
            )
            tasks = [
                self.encoder_task,
                asyncio.create_task(
                    self._receive_controls(), name=f"h264ws-control-{self.session_id}"
                ),
                asyncio.create_task(
                    self._forward_shared_metadata(),
                    name=f"h264ws-metadata-{self.session_id}",
                ),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.exception("Shared H.264 WebSocket session %s failed", self.session_id)
            if not self.websocket.closed:
                await self._send_json({"type": "error", "message": str(error)})
        finally:
            shared_session.comparison_frame_subscribers.discard(self.frame_queue)
            shared_session.comparison_metadata_subscribers.discard(
                self.shared_metadata_queue
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._stop_ffmpeg()

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
                        "codec": "h264",
                        "protocol": "websocket",
                    }
                )
                self.encoder_task = asyncio.create_task(
                    self._encode_frames(), name=f"h264ws-encoder-{self.session_id}"
                )
                tasks = [
                    asyncio.create_task(
                        self._receive_upstream(), name=f"h264ws-upstream-{self.session_id}"
                    ),
                    asyncio.create_task(
                        self._receive_controls(), name=f"h264ws-control-{self.session_id}"
                    ),
                    self.encoder_task,
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.exception("H.264 WebSocket session %s failed", self.session_id)
            if not self.websocket.closed:
                await self._send_json({"type": "error", "message": str(error)})
        finally:
            self.upstream = None
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._stop_ffmpeg()

    async def _forward_shared_metadata(self) -> None:
        while True:
            payload = await self.shared_metadata_queue.get()
            try:
                await self._send_json(payload)
            finally:
                self.shared_metadata_queue.task_done()

    async def _receive_upstream(self) -> None:
        async for message in self.upstream:
            if message.type == WSMsgType.BINARY:
                await self._receive_binary(bytes(message.data))
            elif message.type == WSMsgType.TEXT:
                await self._send_json({"type": "upstream", "data": message.data})
            elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break

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
        received_epoch_ms = time.time() * 1000
        for frame_index, frame in enumerate(frames):
            if content_type == RAW_RGB_CONTENT_TYPE:
                rgb = frame
            elif content_type in ENCODED_IMAGE_TYPES:
                rgb = await asyncio.to_thread(
                    _encoded_image_to_rgb, frame, width=width, height=height
                )
            else:
                raise ValueError(f"unsupported realtime frame content type: {content_type}")
            queued = _QueuedFrame(
                rgb=rgb,
                chunk_index=int(header.get("chunk_index") or 0),
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
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
                self.dropped_frames += 1
            self.frame_queue.put_nowait(queued)

    async def _encode_frames(self) -> None:
        while True:
            frame = await self.frame_queue.get()
            try:
                queue_age_ms = max(
                    0.0, time.time() * 1000 - frame.bridge_received_epoch_ms
                )
                if (
                    queue_age_ms > self.manager.max_frame_age_ms
                    and self.frame_queue.qsize() > self.manager.live_edge_frames
                ):
                    self.dropped_frames += 1
                    self.latency_dropped_frames += 1
                    continue
                await self._pace_frame()
                if self.minimum_event_id and frame.event_id < self.minimum_event_id:
                    self.dropped_frames += 1
                    continue
                if self.ffmpeg is None:
                    await self._start_ffmpeg(self.width, self.height)
                if self.ffmpeg is None or self.ffmpeg.stdin is None:
                    raise RuntimeError("H.264 encoder is unavailable")
                encode_started_epoch_ms = time.time() * 1000
                # Publish metadata before making the frame available to FFmpeg.
                # The stdout pump may otherwise win the scheduling race and the
                # browser can present a frame before its event/chunk identity is
                # known, shifting every latency sample by one frame.
                await self._send_json(
                    {
                        "type": "media_batch",
                        "chunk_index": frame.chunk_index,
                        "event_id": frame.event_id,
                        "first_frame_index": self.frames,
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
                            encode_started_epoch_ms - frame.bridge_received_epoch_ms,
                        ),
                        "bridge_encoder_feed_ms": 0.0,
                        "dropped_frames": self.dropped_frames,
                        "latency_dropped_frames": self.latency_dropped_frames,
                    }
                )
                self.ffmpeg.stdin.write(frame.rgb)
                await self.ffmpeg.stdin.drain()
                self.frames += 1
            finally:
                self.frame_queue.task_done()

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
        fps = _bounded_int(self.init.get("fps"), default=24, minimum=1, maximum=60)
        self.media_fps = fps
        self.next_frame_deadline = None
        gop = max(4, fps * self.manager.gop_seconds)
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
            self.manager.preset,
            "-tune",
            "zerolatency",
            "-profile:v",
            self.manager.profile,
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
            str(self.manager.crf),
            "-maxrate",
            f"{self.manager.bitrate_kbps}k",
            "-bufsize",
            f"{max(128, self.manager.bitrate_kbps * self.manager.vbv_buffer_ms // 1000)}k",
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
        self.stdout_task = asyncio.create_task(self._pump_stdout())
        self.stderr_task = asyncio.create_task(self._log_stderr())

    async def _pump_stdout(self) -> None:
        process = self.ffmpeg
        if process is None or process.stdout is None:
            return
        while data := await process.stdout.read(64 * 1024):
            self.media_bytes += len(data)
            async with self.send_lock:
                await self.websocket.send_bytes(data)

    async def _log_stderr(self) -> None:
        process = self.ffmpeg
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

    async def _stop_ffmpeg(self) -> None:
        process = self.ffmpeg
        self.ffmpeg = None
        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(Exception):
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()
        for task in (self.stdout_task, self.stderr_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self.stdout_task, self.stderr_task) if task is not None),
            return_exceptions=True,
        )
        self.stdout_task = None
        self.stderr_task = None


class H264WebSocketBridgeManager:
    def __init__(self, app: web.Application, upstream_session_key: web.AppKey) -> None:
        self.app = app
        self.upstream_session_key = upstream_session_key
        self.upstream_ws = os.environ.get(
            "ZING_WEBRTC_UPSTREAM_WS",
            "ws://127.0.0.1:30000/v1/realtime_video/generate",
        )
        self.ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
        self.preset = os.environ.get("H264_WS_PRESET", os.environ.get("WEBRTC_H264_PRESET", "veryfast"))
        self.profile = os.environ.get("H264_WS_PROFILE", os.environ.get("WEBRTC_H264_PROFILE", "main"))
        self.crf = _bounded_int(
            os.environ.get("H264_WS_CRF", os.environ.get("WEBRTC_H264_CRF")),
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
            os.environ.get("H264_WS_VBV_BUFFER_MS", os.environ.get("WEBRTC_H264_VBV_BUFFER_MS")),
            default=125,
            minimum=40,
            maximum=2000,
        )
        self.gop_seconds = _bounded_int(
            os.environ.get("H264_WS_GOP_SECONDS", os.environ.get("WEBRTC_H264_GOP_SECONDS")),
            default=1,
            minimum=1,
            maximum=5,
        )
        self.raw_channel_order = os.environ.get(
            "H264_WS_RAW_CHANNEL_ORDER",
            os.environ.get("WEBRTC_RAW_CHANNEL_ORDER", "rgb"),
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
        session = H264WebSocketSession(
            manager=request.app[H264_WS_MANAGER],
            websocket=websocket,
            init=init,
        )
        await session.run()
    except (json.JSONDecodeError, ValueError, TimeoutError) as error:
        if not websocket.closed:
            await websocket.send_json({"type": "error", "message": str(error)})
    finally:
        if not websocket.closed:
            await websocket.close()
    return websocket


def install_h264_websocket_bridge(
    app: web.Application, *, upstream_session_key: web.AppKey
) -> None:
    """Install the isolated H.264/fMP4 WebSocket comparison endpoint."""
    app[H264_WS_MANAGER] = H264WebSocketBridgeManager(app, upstream_session_key)
    app.router.add_get("/api/h264ws", _h264_websocket)
