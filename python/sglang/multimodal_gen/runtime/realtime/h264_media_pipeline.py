# SPDX-License-Identifier: Apache-2.0

"""Per-session low-latency H.264/fMP4 output for realtime VAE workers.

The VAE callback only copies decoded RGB8 frames into a bounded in-process
queue.  A separate asyncio task feeds FFmpeg and sends muxed fMP4 payloads to
the Gateway, so browser/network backpressure never stalls VAE decode.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
    ProtocolViolation,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    EncodedFrameBatch,
)
from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    observe_stage_seconds,
    result_from_exception,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
)

logger = logging.getLogger(__name__)

MediaSender = Callable[[bytes], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class H264PipelineConfig:
    enabled: bool = False
    trigger_output_format: str = "jpeg"
    ffmpeg_bin: str = "ffmpeg"
    fps: int = 24
    threads: int = 2
    preset: str = "fast"
    profile: str = "main"
    crf: int = 20
    bitrate_kbps: int = 3000
    vbv_buffer_ms: int = 250
    gop_seconds: int = 2
    max_queued_frames: int = 24
    max_frame_age_ms: int = 250
    live_edge_frames: int = 6
    startup_drop_frames: int = 0
    raw_channel_order: str = "rgb"
    service: str = "vae"
    model: str = "unknown"

    def validate(self) -> None:
        if self.trigger_output_format not in {"jpeg", "webp"}:
            raise ValueError("H.264 trigger output format must be jpeg or webp")
        if not 1 <= self.fps <= 60:
            raise ValueError("H.264 fps must be between 1 and 60")
        if not 1 <= self.threads <= 16:
            raise ValueError("H.264 encoder threads must be between 1 and 16")
        if self.preset not in {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
        }:
            raise ValueError("unsupported H.264 preset")
        if self.profile not in {"baseline", "main", "high"}:
            raise ValueError("unsupported H.264 profile")
        if not 12 <= self.crf <= 35:
            raise ValueError("H.264 CRF must be between 12 and 35")
        if not 250 <= self.bitrate_kbps <= 20000:
            raise ValueError("H.264 bitrate must be between 250 and 20000 kbps")
        if not 40 <= self.vbv_buffer_ms <= 2000:
            raise ValueError("H.264 VBV buffer must be between 40 and 2000 ms")
        if not 1 <= self.gop_seconds <= 5:
            raise ValueError("H.264 GOP must be between 1 and 5 seconds")
        if self.max_queued_frames < 1:
            raise ValueError("H.264 queue must contain at least one frame")
        if self.max_frame_age_ms < 0:
            raise ValueError("H.264 maximum frame age must be non-negative")
        if not 0 <= self.live_edge_frames <= self.max_queued_frames:
            raise ValueError("H.264 live edge must fit in the frame queue")
        if not 0 <= self.startup_drop_frames <= 120:
            raise ValueError("H.264 startup drop must be between 0 and 120 frames")
        if self.raw_channel_order not in {"rgb", "gbr"}:
            raise ValueError("H.264 raw channel order must be rgb or gbr")


@dataclass(frozen=True, slots=True)
class _QueuedFrame:
    rgb: bytes
    width: int
    height: int
    request_id: str
    chunk_index: int
    event_id: int
    frame_batch_index: int
    is_final_frame_batch: bool
    server_sent_epoch_ms: float
    queued_epoch_ms: float


@dataclass(frozen=True, slots=True)
class _ChunkCompletion:
    request_id: str
    chunk_index: int
    event_id: int
    num_frames: int
    is_final_chunk: bool


class H264MediaPipeline:
    """Bounded RGB-to-H.264 pipeline owned by one realtime session."""

    def __init__(
        self,
        *,
        session_id: str,
        generation_id: str,
        send: MediaSender,
        config: H264PipelineConfig,
    ) -> None:
        config.validate()
        self.session_id = session_id
        self.generation_id = generation_id
        self.send = send
        self.config = config
        self._queue: deque[_QueuedFrame | _ChunkCompletion] = deque()
        self._queue_ready = asyncio.Event()
        self._queued_frames = 0
        self._closed = False
        self._runner = asyncio.create_task(
            self._run(), name=f"vae-h264-{session_id[:8]}"
        )
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._width = 0
        self._height = 0
        self._sequence = 0
        self._frame_index = 0
        self._encoder_start_times: deque[float] = deque(maxlen=4096)
        self._minimum_event_id = 0
        self._startup_drop_remaining = config.startup_drop_frames
        self.dropped_frames = 0
        self.latency_dropped_frames = 0
        self.startup_dropped_frames = 0

    def enqueue(self, header: LatentChunkHeader, batch: EncodedFrameBatch) -> None:
        if self._closed:
            return
        self._raise_if_failed()
        if batch.content_type != RAW_RGB_CONTENT_TYPE:
            raise ProtocolViolation(
                "VAE-side H.264 requires application/x-raw-rgb input"
            )
        if header.event_id is not None and header.event_id > self._minimum_event_id:
            self._minimum_event_id = int(header.event_id)
            self._discard_queued_before(self._minimum_event_id)

        frames = list(batch.payloads)
        if self._startup_drop_remaining:
            if header.chunk_index > 0:
                self._startup_drop_remaining = 0
            else:
                drop_count = min(self._startup_drop_remaining, len(frames))
                frames = frames[drop_count:]
                self._startup_drop_remaining -= drop_count
                self.startup_dropped_frames += drop_count
                self.dropped_frames += drop_count
        queued_epoch_ms = time.time() * 1000
        for offset, rgb in enumerate(frames):
            if len(rgb) != batch.width * batch.height * 3:
                raise ProtocolViolation(
                    "raw RGB payload size does not match dimensions"
                )
            while self._queued_frames >= self.config.max_queued_frames:
                if not self._drop_oldest_frame(result="cancelled"):
                    break
            self._queue.append(
                _QueuedFrame(
                    rgb=rgb,
                    width=batch.width,
                    height=batch.height,
                    request_id=header.request_id,
                    chunk_index=header.chunk_index,
                    event_id=int(header.event_id or 0),
                    frame_batch_index=batch.frame_batch_index,
                    is_final_frame_batch=batch.is_final and offset == len(frames) - 1,
                    server_sent_epoch_ms=queued_epoch_ms,
                    queued_epoch_ms=queued_epoch_ms,
                )
            )
            self._queued_frames += 1
        if frames:
            self._queue_ready.set()

    def enqueue_completion(
        self,
        header: LatentChunkHeader,
        *,
        num_frames: int,
    ) -> None:
        if self._closed:
            return
        self._raise_if_failed()
        self._queue.append(
            _ChunkCompletion(
                request_id=header.request_id,
                chunk_index=header.chunk_index,
                event_id=int(header.event_id or 0),
                num_frames=num_frames,
                is_final_chunk=header.is_final_chunk,
            )
        )
        self._queue_ready.set()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue_ready.set()
        self._runner.cancel()
        await asyncio.gather(self._runner, return_exceptions=True)
        await self._stop_ffmpeg()
        self._queue.clear()
        self._queued_frames = 0

    def _raise_if_failed(self) -> None:
        if not self._runner.done() or self._runner.cancelled():
            return
        failure = self._runner.exception()
        if failure is not None:
            raise RuntimeError("VAE H.264 pipeline is unavailable") from failure

    async def _next(self) -> _QueuedFrame | _ChunkCompletion | None:
        while not self._closed:
            if self._queue:
                item = self._queue.popleft()
                if isinstance(item, _QueuedFrame):
                    self._queued_frames -= 1
                if not self._queue:
                    self._queue_ready.clear()
                return item
            self._queue_ready.clear()
            if self._queue:
                continue
            await self._queue_ready.wait()
        return None

    async def _run(self) -> None:
        try:
            while (item := await self._next()) is not None:
                if isinstance(item, _ChunkCompletion):
                    await self._send_completion(item)
                    continue
                frame = item
                if (
                    time.time() * 1000 - frame.queued_epoch_ms
                    > self.config.max_frame_age_ms
                    and self._queued_frames > self.config.live_edge_frames
                ):
                    self.dropped_frames += 1
                    self.latency_dropped_frames += 1
                    self._observe_queue(frame, result="cancelled")
                    continue
                await self._encode_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "VAE H.264 pipeline failed: session_id=%s generation_id=%s",
                self.session_id,
                self.generation_id,
            )
            raise

    async def _encode_frame(self, frame: _QueuedFrame) -> None:
        if self._minimum_event_id and frame.event_id < self._minimum_event_id:
            self.dropped_frames += 1
            self._observe_queue(frame, result="cancelled")
            return
        if self._ffmpeg is None:
            await self._start_ffmpeg(frame.width, frame.height)
        elif (frame.width, frame.height) != (self._width, self._height):
            raise RuntimeError(
                "H.264 dimensions changed from "
                f"{self._width}x{self._height} to {frame.width}x{frame.height}"
            )
        process = self._ffmpeg
        if process is None or process.stdin is None:
            raise RuntimeError("H.264 encoder is unavailable")

        encode_started_s = time.perf_counter()
        encode_started_epoch_ms = time.time() * 1000
        queue_ms = max(0.0, encode_started_epoch_ms - frame.queued_epoch_ms)
        self._observe_queue(frame, result="success")
        await self.send(
            encode_message(
                "media_batch",
                session_id=self.session_id,
                generation_id=self.generation_id,
                request_id=frame.request_id,
                chunk_index=frame.chunk_index,
                event_id=frame.event_id,
                first_frame_index=self._frame_index,
                num_frames=1,
                frame_batch_index=frame.frame_batch_index,
                is_final_frame_batch=frame.is_final_frame_batch,
                server_sent_epoch_ms=frame.server_sent_epoch_ms,
                h264_encode_started_epoch_ms=encode_started_epoch_ms,
                h264_queue_ms=queue_ms,
                h264_encoder_feed_ms=0.0,
                dropped_frames=self.dropped_frames,
                latency_dropped_frames=self.latency_dropped_frames,
                startup_dropped_frames=self.startup_dropped_frames,
            )
        )
        feed_started = time.perf_counter()
        self._encoder_start_times.append(encode_started_s)
        try:
            process.stdin.write(frame.rgb)
            await process.stdin.drain()
        except BaseException as exc:
            if self._encoder_start_times:
                self._encoder_start_times.pop()
            observe_stage_seconds(
                "frame_encode",
                time.perf_counter() - feed_started,
                service=self.config.service,
                model=self.config.model,
                result=result_from_exception(exc),
                codec="h264",
                scope="frame",
            )
            raise
        feed_completed_epoch_ms = time.time() * 1000
        await self.send(
            encode_message(
                "media_encode_timing",
                session_id=self.session_id,
                generation_id=self.generation_id,
                request_id=frame.request_id,
                chunk_index=frame.chunk_index,
                first_frame_index=self._frame_index,
                h264_encoded_epoch_ms=feed_completed_epoch_ms,
                h264_encoder_feed_ms=max(
                    0.0, feed_completed_epoch_ms - encode_started_epoch_ms
                ),
            )
        )
        self._frame_index += 1

    async def _send_completion(self, completion: _ChunkCompletion) -> None:
        await self.send(
            encode_message(
                "media_chunk_complete",
                session_id=self.session_id,
                generation_id=self.generation_id,
                request_id=completion.request_id,
                chunk_index=completion.chunk_index,
                event_id=completion.event_id,
                num_frames=completion.num_frames,
                is_final_chunk=completion.is_final_chunk,
                media_transport="h264",
            )
        )

    async def _start_ffmpeg(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        gop = max(4, self.config.fps * self.config.gop_seconds)
        channel_filter = ""
        if self.config.raw_channel_order == "gbr":
            channel_filter = (
                "colorchannelmixer=" "rr=0:rg=0:rb=1:gr=1:gg=0:gb=0:br=0:bg=1:bb=0,"
            )
        command = [
            self.config.ffmpeg_bin,
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
            str(self.config.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            str(self.config.threads),
            "-preset",
            self.config.preset,
            "-tune",
            "zerolatency",
            "-profile:v",
            self.config.profile,
            "-level:v",
            "3.1",
            "-vf",
            channel_filter + "scale=in_range=pc:out_range=tv:"
            "in_color_matrix=bt709:out_color_matrix=bt709",
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
            str(self.config.crf),
            "-maxrate",
            f"{self.config.bitrate_kbps}k",
            "-bufsize",
            f"{max(128, self.config.bitrate_kbps * self.config.vbv_buffer_ms // 1000)}k",
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
        self._ffmpeg = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self.send(
            encode_message(
                "media_init",
                session_id=self.session_id,
                generation_id=self.generation_id,
                codec="h264",
                container="fmp4",
                mime_type='video/mp4; codecs="avc1.4D401F"',
                width=width,
                height=height,
                fps=self.config.fps,
                bitrate_kbps=self.config.bitrate_kbps,
                crf=self.config.crf,
                preset=self.config.preset,
                profile=self.config.profile,
                gop_seconds=self.config.gop_seconds,
                vbv_buffer_ms=self.config.vbv_buffer_ms,
            )
        )
        self._stdout_task = asyncio.create_task(
            self._pump_stdout(), name=f"vae-h264-out-{self.session_id[:8]}"
        )
        self._stderr_task = asyncio.create_task(
            self._log_stderr(), name=f"vae-h264-err-{self.session_id[:8]}"
        )

    async def _pump_stdout(self) -> None:
        process = self._ffmpeg
        if process is None or process.stdout is None:
            return
        while data := await process.stdout.read(64 * 1024):
            ready_s = time.perf_counter()
            if self._encoder_start_times:
                observe_stage_seconds(
                    "frame_encode",
                    ready_s - self._encoder_start_times.popleft(),
                    service=self.config.service,
                    model=self.config.model,
                    result="success",
                    codec="h264",
                    scope="frame",
                )
            sent_epoch_ms = time.time() * 1000
            sequence = self._sequence
            self._sequence += 1
            write_started = time.perf_counter()
            try:
                await self.send(
                    encode_message(
                        "media_payload",
                        session_id=self.session_id,
                        generation_id=self.generation_id,
                        sequence=sequence,
                        codec="h264",
                        container="fmp4",
                        num_bytes=len(data),
                        payload=data,
                        server_sent_epoch_ms=sent_epoch_ms,
                    )
                )
            except BaseException as exc:
                observe_stage_seconds(
                    "ffmpeg_mux_write",
                    time.perf_counter() - write_started,
                    service=self.config.service,
                    model=self.config.model,
                    result=result_from_exception(exc),
                    codec="h264",
                    scope="frame",
                )
                raise
            observe_stage_seconds(
                "ffmpeg_mux_write",
                time.perf_counter() - write_started,
                service=self.config.service,
                model=self.config.model,
                result="success",
                codec="h264",
                scope="frame",
            )

    async def _log_stderr(self) -> None:
        process = self._ffmpeg
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            logger.warning(
                "vae-h264-ffmpeg[%s]: %s",
                self.session_id,
                line.decode("utf-8", "replace").rstrip(),
            )

    def _discard_queued_before(self, event_id: int) -> None:
        retained: deque[_QueuedFrame | _ChunkCompletion] = deque()
        while self._queue:
            item = self._queue.popleft()
            if isinstance(item, _QueuedFrame) and item.event_id < event_id:
                self._queued_frames -= 1
                self.dropped_frames += 1
                self._observe_queue(item, result="cancelled")
            else:
                retained.append(item)
        self._queue = retained

    def _drop_oldest_frame(self, *, result: str) -> bool:
        for index, item in enumerate(self._queue):
            if not isinstance(item, _QueuedFrame):
                continue
            del self._queue[index]
            self._queued_frames -= 1
            self.dropped_frames += 1
            self._observe_queue(item, result=result)
            return True
        return False

    def _observe_queue(self, frame: _QueuedFrame, *, result: str) -> None:
        observe_stage_seconds(
            "h264_pre_encode_queue",
            max(0.0, time.time() * 1000 - frame.queued_epoch_ms) / 1000.0,
            service=self.config.service,
            model=self.config.model,
            result=result,
            codec="h264",
            scope="frame",
        )

    async def _stop_ffmpeg(self) -> None:
        process = self._ffmpeg
        self._ffmpeg = None
        cancelled_at = time.perf_counter()
        while self._encoder_start_times:
            observe_stage_seconds(
                "frame_encode",
                cancelled_at - self._encoder_start_times.popleft(),
                service=self.config.service,
                model=self.config.model,
                result="cancelled",
                codec="h264",
                scope="frame",
            )
        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(Exception):
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()
        tasks = [
            task for task in (self._stdout_task, self._stderr_task) if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
