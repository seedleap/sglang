# SPDX-License-Identifier: Apache-2.0

"""Bounded stateful TAEHV worker for realtime MinWM decoding."""

from __future__ import annotations

import asyncio
import inspect
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.realtime.async_vae_metrics import (
    observe_stage,
    record_backpressure,
    update_capacity,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    AcceptDisposition,
    ChunkSequenceTracker,
    LatentChunkHeader,
    ProtocolViolation,
)
from sglang.multimodal_gen.runtime.realtime.media_profile import (
    MediaProfileAcceptance,
    RealtimeMediaProfile,
    parse_media_profile,
    validate_source_timeline_fps,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
)


class VAEBackpressureError(RuntimeError):
    pass


class VAESessionCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SessionOpen:
    session_id: str
    generation_id: str
    decoder_backend: str | None = None
    response_transport: str = "websocket"
    trace_id: str | None = None
    output_format: str = "raw"
    quality: int = 90
    preview_max_width: int | None = None
    output_url: str | None = None
    output_token: str | None = None
    media_profile: RealtimeMediaProfile = RealtimeMediaProfile.NATIVE_V1
    source_timeline_fps: float = 24.0


@dataclass(frozen=True, slots=True)
class EncodedFrameBatch:
    payloads: tuple[bytes, ...]
    content_type: str
    width: int
    height: int
    frame_batch_index: int
    is_final: bool
    encode_ms: float
    source_width: int
    source_height: int
    preview_width: int
    preview_height: int
    media_profile: RealtimeMediaProfile
    source_timeline_fps: float
    output_timeline_fps: float

    @property
    def num_frames(self) -> int:
        return len(self.payloads)


@dataclass(frozen=True, slots=True)
class DecodeResult:
    disposition: AcceptDisposition
    num_frames: int
    source_num_frames: int
    output_num_frames: int
    media_profile: RealtimeMediaProfile
    source_timeline_fps: float
    output_timeline_fps: float
    frame_batches: tuple[EncodedFrameBatch, ...]
    queue_wait_ms: float
    decode_ms: float
    encode_ms: float
    post_decode_ms: float = 0.0
    rife_interpolation_ms: float = 0.0


FrameBatchCallback = Callable[[EncodedFrameBatch], Awaitable[None]]


def _codec_from_content_type(content_type: str) -> str:
    normalized = content_type.lower()
    if "webp" in normalized:
        return "webp"
    if "jpeg" in normalized or "jpg" in normalized:
        return "jpeg"
    return "none"


def _next_item(iterator) -> tuple[bool, Any]:
    """Advance a synchronous decoder iterator without leaking StopIteration."""

    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


@dataclass(slots=True)
class _DecodeJob:
    header: LatentChunkHeader
    latents: torch.Tensor
    submitted_at: float
    future: asyncio.Future[DecodeResult]
    on_frame_batch: FrameBatchCallback | None = None
    on_decode_started: Callable[[], Any] | None = None


@dataclass(slots=True)
class _WorkerSession:
    opened: SessionOpen
    decoder: Any
    tracker: ChunkSequenceTracker
    queue: asyncio.Queue[_DecodeJob]
    media_acceptance: MediaProfileAcceptance
    runner: asyncio.Task[None] | None = None
    first_t2v_latent: torch.Tensor | None = None
    previous_source_frame: torch.Tensor | None = None
    media_boundary_initialized: bool = False
    last_media_event_id: int | None = None
    last_media_prompt_version: int = 0
    last_activity_at: float = field(default_factory=time.monotonic)
    processing: bool = False


class AsyncVAEWorker:
    """Owns shared weights and isolated streaming state for each generation."""

    def __init__(
        self,
        engine: Any,
        *,
        max_sessions: int,
        queue_depth_per_session: int = 1,
        encoded_frames_per_batch: int = 1,
        encode_workers: int = 4,
        rife_processor: Any | None = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if queue_depth_per_session < 1:
            raise ValueError("queue_depth_per_session must be positive")
        if encode_workers < 1:
            raise ValueError("encode_workers must be positive")
        engine_max_sessions = getattr(engine, "max_sessions", None)
        if engine_max_sessions is not None and max_sessions > engine_max_sessions:
            raise ValueError(
                f"{getattr(engine, 'backend', 'VAE')} backend supports at most "
                f"{engine_max_sessions} active session(s)"
            )
        self.engine = engine
        self.decoder_backend = getattr(engine, "backend", None)
        self.max_sessions = max_sessions
        self.queue_depth_per_session = queue_depth_per_session
        self.encoded_frames_per_batch = max(1, encoded_frames_per_batch)
        self.encode_workers = encode_workers
        self.rife_processor = rife_processor
        self._encode_executor = ThreadPoolExecutor(
            max_workers=encode_workers,
            thread_name_prefix="realtime-webp",
        )
        self._encode_executor_shutdown = False
        self._sessions: dict[tuple[str, str], _WorkerSession] = {}
        self._session_lock = asyncio.Lock()
        self._actor_lock = asyncio.Lock()
        self._service_time_ms = 0.0

    @property
    def supported_media_profiles(self) -> tuple[RealtimeMediaProfile, ...]:
        profiles = [RealtimeMediaProfile.NATIVE_V1]
        if self.rife_processor is not None and bool(
            getattr(self.rife_processor, "ready", False)
        ):
            profiles.append(RealtimeMediaProfile.RIFE2X_V1)
        return tuple(profiles)

    @property
    def media_capability_fingerprint(self) -> str | None:
        """Fingerprint used to verify a homogeneous RIFE worker pool."""

        if RealtimeMediaProfile.RIFE2X_V1 not in self.supported_media_profiles:
            return None
        return f"rife2x_v1:{self.rife_processor.weights_sha256}"

    def _accept_media_profile(self, request: SessionOpen) -> MediaProfileAcceptance:
        requested = parse_media_profile(request.media_profile)
        source_timeline_fps = validate_source_timeline_fps(request.source_timeline_fps)
        if requested not in self.supported_media_profiles:
            raise ProtocolViolation(
                f"realtime media profile is unavailable: {requested.value}"
            )
        weights_sha256 = None
        if requested is RealtimeMediaProfile.RIFE2X_V1:
            weights_sha256 = str(
                getattr(self.rife_processor, "weights_sha256", "") or ""
            ).lower()
            if len(weights_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in weights_sha256
            ):
                raise ProtocolViolation("RIFE media capability has no weights digest")
        return MediaProfileAcceptance(
            requested=requested,
            effective=requested,
            source_timeline_fps=source_timeline_fps,
            output_timeline_fps=source_timeline_fps
            * requested.output_timeline_fps_multiplier,
            weights_sha256=weights_sha256,
        )

    async def open(self, request: SessionOpen) -> MediaProfileAcceptance:
        if not request.session_id or not request.generation_id:
            raise ProtocolViolation("session generation identity is required")
        if (
            request.decoder_backend is not None
            and request.decoder_backend != self.decoder_backend
        ):
            raise ProtocolViolation(
                "requested decoder backend does not match this VAE worker"
            )
        if request.response_transport not in {"websocket", "shared_memory"}:
            raise ProtocolViolation("unsupported VAE response transport")
        media_acceptance = self._accept_media_profile(request)
        request = replace(
            request,
            media_profile=media_acceptance.effective,
            source_timeline_fps=media_acceptance.source_timeline_fps,
        )
        identity = (request.session_id, request.generation_id)
        async with self._session_lock:
            if identity in self._sessions:
                raise ProtocolViolation("VAE session generation is already active")
            if len(self._sessions) >= self.max_sessions:
                raise VAESessionCapacityError(
                    f"VAE session capacity exhausted: {self.max_sessions}"
                )
            decoder = self.engine.create_decoder(identity)
            state = _WorkerSession(
                opened=request,
                decoder=decoder,
                tracker=ChunkSequenceTracker(*identity),
                queue=asyncio.Queue(maxsize=self.queue_depth_per_session),
                media_acceptance=media_acceptance,
            )
            self._sessions[identity] = state
            state.runner = asyncio.create_task(
                self._run_session(identity, state),
                name=f"realtime-vae-{request.session_id[:8]}",
            )
            self._update_capacity_metrics()
        return media_acceptance

    async def submit(
        self,
        header: LatentChunkHeader,
        latents: torch.Tensor,
        *,
        on_frame_batch: FrameBatchCallback | None = None,
        on_decode_started: Callable[[], Any] | None = None,
    ) -> asyncio.Future[DecodeResult]:
        identity = (header.session_id, header.generation_id)
        state = self._sessions.get(identity)
        if state is None:
            raise ProtocolViolation("unknown VAE session generation")
        if state.queue.full():
            record_backpressure()
            raise VAEBackpressureError("VAE session decode queue is full")

        disposition = state.tracker.accept(header)
        loop = asyncio.get_running_loop()
        if disposition is AcceptDisposition.DUPLICATE:
            future: asyncio.Future[DecodeResult] = loop.create_future()
            future.set_result(
                DecodeResult(
                    disposition=disposition,
                    num_frames=0,
                    source_num_frames=0,
                    output_num_frames=0,
                    media_profile=state.media_acceptance.effective,
                    source_timeline_fps=state.media_acceptance.source_timeline_fps,
                    output_timeline_fps=state.media_acceptance.output_timeline_fps,
                    frame_batches=(),
                    queue_wait_ms=0.0,
                    decode_ms=0.0,
                    encode_ms=0.0,
                    rife_interpolation_ms=0.0,
                )
            )
            return future

        expected_shape = tuple(int(value) for value in header.shape)
        if tuple(latents.shape) != expected_shape:
            raise ProtocolViolation(
                f"latent shape mismatch: expected {expected_shape}, got {tuple(latents.shape)}"
            )
        if str(latents.dtype).removeprefix("torch.") != header.dtype:
            raise ProtocolViolation(
                f"latent dtype mismatch: expected {header.dtype}, got {latents.dtype}"
            )

        future = loop.create_future()
        job = _DecodeJob(
            header=header,
            latents=latents.detach().contiguous(),
            submitted_at=time.perf_counter(),
            future=future,
            on_frame_batch=on_frame_batch,
            on_decode_started=on_decode_started,
        )
        try:
            state.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            record_backpressure()
            raise VAEBackpressureError("VAE session decode queue is full") from exc
        state.last_activity_at = time.monotonic()
        self._update_capacity_metrics()
        return future

    async def decode(
        self,
        header: LatentChunkHeader,
        latents: torch.Tensor,
        *,
        on_frame_batch: FrameBatchCallback | None = None,
        on_decode_started: Callable[[], Any] | None = None,
    ) -> DecodeResult:
        future = await self.submit(
            header,
            latents,
            on_frame_batch=on_frame_batch,
            on_decode_started=on_decode_started,
        )
        return await future

    async def _run_session(
        self,
        identity: tuple[str, str],
        state: _WorkerSession,
    ) -> None:
        fatal_error: Exception | None = None
        try:
            while True:
                job = await state.queue.get()
                service_started = time.perf_counter()
                state.processing = True
                final_chunk_complete = False
                try:
                    result = await self._decode_job(state, job)
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    raise
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                    fatal_error = exc
                else:
                    if not job.future.done():
                        job.future.set_result(result)
                    final_chunk_complete = job.header.is_final_chunk
                finally:
                    service_time_ms = (time.perf_counter() - service_started) * 1000.0
                    self._service_time_ms = (
                        service_time_ms
                        if self._service_time_ms == 0
                        else 0.8 * self._service_time_ms + 0.2 * service_time_ms
                    )
                    state.processing = False
                    state.queue.task_done()
                    state.last_activity_at = time.monotonic()
                    self._update_capacity_metrics()
                if final_chunk_complete:
                    return
                if fatal_error is not None:
                    return
        finally:
            current = self._sessions.get(identity)
            if current is state:
                self._sessions.pop(identity, None)
            while not state.queue.empty():
                try:
                    pending_job = state.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not pending_job.future.done():
                    if fatal_error is None:
                        pending_job.future.cancel()
                    else:
                        pending_job.future.set_exception(
                            RuntimeError("VAE session terminated after decoder failure")
                        )
                state.queue.task_done()
            reset = getattr(state.decoder, "reset", None)
            if callable(reset):
                reset()
            self._update_capacity_metrics()

    async def _decode_job(
        self,
        state: _WorkerSession,
        job: _DecodeJob,
    ) -> DecodeResult:
        queue_wait_ms = (time.perf_counter() - job.submitted_at) * 1000.0
        header = job.header
        source = job.latents
        first_chunk = header.chunk_index == 0
        drop_leading_frames = 0

        if header.chunk_index == 0 and not header.has_reference:
            state.first_t2v_latent = source.detach().clone()
        elif header.chunk_index == 1 and not header.has_reference:
            first_latent = state.first_t2v_latent
            state.first_t2v_latent = None
            if first_latent is not None:
                source = torch.cat([first_latent, source], dim=2).contiguous()
                first_chunk = True
                drop_leading_frames = 1

        decode_started = time.perf_counter()
        decode_resume_at = decode_started
        decode_active_s = 0.0
        decode_ms = 0.0
        post_decode_ms = 0.0
        encode_tasks: list[asyncio.Task[tuple[EncodedFrameBatch, ...]]] = []
        callback_tail: asyncio.Task | None = None
        frame_batch_index = 0
        remaining_drop = drop_leading_frames
        pending_frames: list[torch.Tensor] = []
        pending_frame_count = 0
        source_num_frames = 0
        rife_interpolation_ms = 0.0
        reset_media_boundary = (
            not state.media_boundary_initialized
            or state.last_media_event_id != header.event_id
            or state.last_media_prompt_version != header.prompt_version
        )
        state.media_boundary_initialized = True
        state.last_media_event_id = header.event_id
        state.last_media_prompt_version = header.prompt_version
        if reset_media_boundary:
            state.previous_source_frame = None

        def schedule_encode(frames: torch.Tensor) -> None:
            nonlocal callback_tail, frame_batch_index
            task = asyncio.create_task(
                self._encode_and_emit(
                    frames,
                    state.opened,
                    frame_batch_index=frame_batch_index,
                    previous_callback=callback_tail,
                    on_frame_batch=job.on_frame_batch,
                )
            )
            encode_tasks.append(task)
            callback_tail = task
            frame_batch_index += (
                int(frames.shape[2]) + self.encoded_frames_per_batch - 1
            ) // self.encoded_frames_per_batch

        try:
            async with self._actor_lock:
                if job.on_decode_started is not None:
                    started = job.on_decode_started()
                    if inspect.isawaitable(started):
                        await started
                    decode_resume_at = time.perf_counter()
                async for raw_frames in self._iter_decoded_frames(
                    state.decoder,
                    source,
                    first_chunk=first_chunk,
                ):
                    post_started = time.perf_counter()
                    decode_active_s += max(0.0, post_started - decode_resume_at)
                    try:
                        frames = self._normalize_frames(
                            raw_frames,
                            keep_on_device=(
                                state.media_acceptance.interpolation_enabled
                            ),
                        )
                    finally:
                        post_decode_ms += (time.perf_counter() - post_started) * 1000.0
                    if remaining_drop:
                        drop_count = min(remaining_drop, int(frames.shape[2]))
                        frames = frames[:, :, drop_count:]
                        remaining_drop -= drop_count
                    if frames.shape[2] == 0:
                        decode_resume_at = time.perf_counter()
                        continue

                    source_num_frames += int(frames.shape[2])
                    if state.media_acceptance.interpolation_enabled:
                        interpolation_started = time.perf_counter()
                        try:
                            frames = await self._interpolate_rife2x(state, frames)
                        finally:
                            rife_interpolation_ms += (
                                time.perf_counter() - interpolation_started
                            ) * 1000.0

                    pending_frames.append(frames)
                    pending_frame_count += int(frames.shape[2])
                    if pending_frame_count < self.encoded_frames_per_batch:
                        decode_resume_at = time.perf_counter()
                        continue

                    merged = (
                        pending_frames[0]
                        if len(pending_frames) == 1
                        else torch.cat(pending_frames, dim=2)
                    )
                    while int(merged.shape[2]) >= self.encoded_frames_per_batch:
                        schedule_encode(
                            merged[:, :, : self.encoded_frames_per_batch].contiguous()
                        )
                        merged = merged[:, :, self.encoded_frames_per_batch :]
                    pending_frames = [merged.contiguous()] if merged.shape[2] else []
                    pending_frame_count = int(merged.shape[2])
                    decode_resume_at = time.perf_counter()

                decode_active_s += max(0.0, time.perf_counter() - decode_resume_at)
                if pending_frames:
                    schedule_encode(
                        pending_frames[0]
                        if len(pending_frames) == 1
                        else torch.cat(pending_frames, dim=2).contiguous()
                    )

            decode_ms = decode_active_s * 1000.0
            encoded_groups = await asyncio.gather(*encode_tasks) if encode_tasks else []
            encoded = tuple(batch for group in encoded_groups for batch in group)
            encode_ms = sum(batch.encode_ms for batch in encoded)
        except BaseException as exc:
            result = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            observe_stage("queue_wait", queue_wait_ms, result=result)
            elapsed_decode_ms = max(
                0.0,
                (time.perf_counter() - decode_started) * 1000.0
                - post_decode_ms
                - rife_interpolation_ms,
            )
            if elapsed_decode_ms > 0:
                observe_stage("decode", elapsed_decode_ms, result=result)
            if post_decode_ms > 0:
                observe_stage("post_decode", post_decode_ms, result=result)
            if rife_interpolation_ms > 0:
                observe_stage(
                    "rife_interpolation",
                    rife_interpolation_ms,
                    result=result,
                )
            raise

        codec = _codec_from_content_type(encoded[0].content_type) if encoded else "none"
        observe_stage("queue_wait", queue_wait_ms)
        observe_stage("decode", decode_ms)
        if post_decode_ms > 0:
            observe_stage("post_decode", post_decode_ms)
        if encode_ms > 0:
            observe_stage("frame_encode", encode_ms, codec=codec, scope="frame")
        if state.media_acceptance.interpolation_enabled:
            observe_stage("rife_interpolation", rife_interpolation_ms)

        return DecodeResult(
            disposition=AcceptDisposition.ACCEPT,
            num_frames=sum(batch.num_frames for batch in encoded),
            source_num_frames=source_num_frames,
            output_num_frames=sum(batch.num_frames for batch in encoded),
            media_profile=state.media_acceptance.effective,
            source_timeline_fps=state.media_acceptance.source_timeline_fps,
            output_timeline_fps=state.media_acceptance.output_timeline_fps,
            frame_batches=encoded,
            queue_wait_ms=queue_wait_ms,
            decode_ms=decode_ms,
            encode_ms=encode_ms,
            post_decode_ms=post_decode_ms,
            rife_interpolation_ms=rife_interpolation_ms,
        )

    async def _interpolate_rife2x(
        self,
        state: _WorkerSession,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Interleave RIFE midpoints while preserving the cross-chunk seam."""

        processor = self.rife_processor
        if processor is None:
            raise RuntimeError("RIFE media processor is unavailable")
        # Preserve the decoder tensor on its current device until RIFE has read
        # it.  A taehv CUDA frame would otherwise make a needless D2H -> H2D
        # round trip before interpolation.
        current = frames[:, :3].contiguous()
        previous = state.previous_source_frame
        if previous is not None and (
            previous.shape[0] != current.shape[0]
            or previous.shape[1] != current.shape[1]
            or previous.shape[-2:] != current.shape[-2:]
        ):
            previous = None
        source = current if previous is None else torch.cat((previous, current), dim=2)
        if source.shape[2] < 2:
            state.previous_source_frame = current[:, :, -1:].clone()
            return current.to(device="cpu", dtype=torch.float32)

        midpoints = await asyncio.to_thread(processor.interpolate_midpoints, source)
        expected_midpoints = int(source.shape[2]) - 1
        expected_shape = (
            int(current.shape[0]),
            3,
            expected_midpoints,
            int(current.shape[-2]),
            int(current.shape[-1]),
        )
        if tuple(midpoints.shape) != expected_shape:
            raise RuntimeError(
                "RIFE midpoint shape mismatch: "
                f"expected {expected_shape}, got {tuple(midpoints.shape)}"
            )
        # Commit the seam only after interpolation succeeds.  A failed job
        # must not poison state if its error is inspected or retried.
        state.previous_source_frame = current[:, :, -1:].clone()
        current = current.to(device="cpu", dtype=torch.float32)
        midpoints = midpoints.to(device="cpu", dtype=torch.float32).contiguous()
        if previous is None:
            output = torch.empty(
                (
                    current.shape[0],
                    3,
                    current.shape[2] * 2 - 1,
                    current.shape[-2],
                    current.shape[-1],
                ),
                dtype=torch.float32,
            )
            output[:, :, 0::2] = current
            output[:, :, 1::2] = midpoints
            return output

        output = torch.empty(
            (
                current.shape[0],
                3,
                current.shape[2] * 2,
                current.shape[-2],
                current.shape[-1],
            ),
            dtype=torch.float32,
        )
        output[:, :, 0::2] = midpoints
        output[:, :, 1::2] = current
        return output

    async def _iter_decoded_frames(self, decoder, source, *, first_chunk):
        iterator_factory = getattr(self.engine, "iter_decode", None)
        if iterator_factory is None:
            decode = self.engine.decode
            if inspect.iscoroutinefunction(decode):
                frames = await decode(
                    decoder,
                    source,
                    first_chunk=first_chunk,
                )
            else:
                # Native VAE decode is synchronous. Keep it off the protocol event
                # loop so the next latent can be admitted while CUDA is busy.
                decode_task = asyncio.create_task(
                    asyncio.to_thread(
                        decode,
                        decoder,
                        source,
                        first_chunk=first_chunk,
                    )
                )
                try:
                    frames = await asyncio.shield(decode_task)
                except asyncio.CancelledError:
                    # A native CUDA call keeps running after to_thread is
                    # cancelled. Do not reset a model-global causal cache while
                    # that decode (or its distributed collectives) is in flight.
                    try:
                        await decode_task
                    except Exception:
                        pass
                    raise
                if inspect.isawaitable(frames):
                    frames = await frames
            yield frames
            return

        frames = iterator_factory(decoder, source, first_chunk=first_chunk)
        if inspect.isawaitable(frames):
            frames = await frames
        if hasattr(frames, "__aiter__"):
            async for frame_batch in frames:
                yield frame_batch
            return
        iterator = iter(frames)
        while True:
            has_value, frame_batch = await asyncio.to_thread(_next_item, iterator)
            if not has_value:
                return
            yield frame_batch

    async def _encode_and_emit(
        self,
        frames: torch.Tensor,
        opened: SessionOpen,
        *,
        frame_batch_index: int,
        previous_callback: asyncio.Task | None,
        on_frame_batch: FrameBatchCallback | None,
    ) -> tuple[EncodedFrameBatch, ...]:
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            self._encode_executor,
            self._encode_frames,
            frames,
            opened,
        )
        indexed = tuple(
            EncodedFrameBatch(
                payloads=batch.payloads,
                content_type=batch.content_type,
                width=batch.width,
                height=batch.height,
                frame_batch_index=frame_batch_index + offset,
                # chunk_complete is the authoritative end marker for streamed output.
                is_final=False,
                encode_ms=batch.encode_ms,
                source_width=batch.source_width,
                source_height=batch.source_height,
                preview_width=batch.preview_width,
                preview_height=batch.preview_height,
                media_profile=opened.media_profile,
                source_timeline_fps=opened.source_timeline_fps,
                output_timeline_fps=(
                    opened.source_timeline_fps
                    * parse_media_profile(
                        opened.media_profile
                    ).output_timeline_fps_multiplier
                ),
            )
            for offset, batch in enumerate(encoded)
        )
        if previous_callback is not None:
            await previous_callback
        if on_frame_batch is not None:
            for batch in indexed:
                await on_frame_batch(batch)
        return indexed

    def _normalize_frames(
        self,
        frames: torch.Tensor,
        *,
        keep_on_device: bool = False,
    ) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
            raise RuntimeError("VAE engine must return a five-dimensional frame tensor")
        if frames.shape[1] not in (1, 3, 4) and frames.shape[2] in (1, 3, 4):
            frames = frames.permute(0, 2, 1, 3, 4)
        if frames.shape[1] not in (1, 3, 4):
            raise RuntimeError("VAE frame tensor must be BCTHW or BTCHW")
        frames = frames.detach()
        if keep_on_device:
            return frames.clamp(0, 1).contiguous()
        if getattr(self.engine, "gpu_rgb8_d2h", False):
            # Keep native exact output on-device. The raw encoder below clamps
            # and quantizes it once, avoiding an extra full-size FP32 copy.
            return frames
        return frames.clamp(0, 1).contiguous().cpu()

    def _encode_frames(
        self,
        frames: torch.Tensor,
        opened: SessionOpen,
    ) -> list[EncodedFrameBatch]:
        encode_started = time.perf_counter()
        if frames.shape[0] != 1:
            raise RuntimeError("Realtime VAE supports one sample per session")
        # mul() owns this temporary, so the following in-place operations avoid
        # another 1248x704x16 FP32 allocation without mutating decoder output.
        rgb_values = frames[0, :3].mul(255)
        rgb_values.clamp_(0, 255)
        rgb_quantization = getattr(self.engine, "rgb_quantization", "round")
        if rgb_quantization == "round":
            rgb_values.round_()
        elif rgb_quantization != "truncate":
            raise RuntimeError(f"unsupported VAE RGB quantization: {rgb_quantization}")
        rgb8 = rgb_values.to(torch.uint8).permute(1, 2, 3, 0).contiguous()
        if rgb8.device.type != "cpu":
            # Exact 720p output is ~161 MiB as float32 but ~40 MiB as RGB8.
            # Quantize before the synchronous D2H copy to keep the wire contract
            # while removing three quarters of the host-transfer volume.
            rgb8 = rgb8.cpu()
        array = rgb8.numpy()
        raw_frames = [frame.tobytes() for frame in array]
        source_height = int(array.shape[1]) if len(array) else int(frames.shape[-2])
        source_width = int(array.shape[2]) if len(array) else int(frames.shape[-1])
        height = source_height
        width = source_width
        output_format = opened.output_format.lower()
        content_type = RAW_RGB_CONTENT_TYPE
        encoded_frames = raw_frames

        if output_format in {"webp", "jpeg"}:
            encoded_frames = [
                self._encode_image(
                    frame,
                    width=width,
                    height=height,
                    output_format=output_format,
                    quality=opened.quality,
                    preview_max_width=opened.preview_max_width,
                )
                for frame in raw_frames
            ]
            content_type = (
                WEBP_FRAME_CONTENT_TYPE
                if output_format == "webp"
                else JPEG_FRAME_CONTENT_TYPE
            )
            if opened.preview_max_width and width > opened.preview_max_width:
                height = max(1, round(height * opened.preview_max_width / width))
                width = opened.preview_max_width

        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        chunks = [
            encoded_frames[index : index + self.encoded_frames_per_batch]
            for index in range(0, len(encoded_frames), self.encoded_frames_per_batch)
        ]
        return [
            EncodedFrameBatch(
                payloads=tuple(payloads),
                content_type=content_type,
                width=width,
                height=height,
                frame_batch_index=index,
                is_final=index == len(chunks) - 1,
                encode_ms=encode_ms if index == len(chunks) - 1 else 0.0,
                source_width=source_width,
                source_height=source_height,
                preview_width=width,
                preview_height=height,
                media_profile=parse_media_profile(opened.media_profile),
                source_timeline_fps=validate_source_timeline_fps(
                    opened.source_timeline_fps
                ),
                output_timeline_fps=(
                    validate_source_timeline_fps(opened.source_timeline_fps)
                    * parse_media_profile(
                        opened.media_profile
                    ).output_timeline_fps_multiplier
                ),
            )
            for index, payloads in enumerate(chunks)
        ]

    @staticmethod
    def _encode_image(
        frame: bytes,
        *,
        width: int,
        height: int,
        output_format: str,
        quality: int,
        preview_max_width: int | None,
    ) -> bytes:
        image = Image.frombytes("RGB", (width, height), frame)
        if preview_max_width and width > preview_max_width:
            preview_height = max(1, round(height * preview_max_width / width))
            image = image.resize(
                (preview_max_width, preview_height), Image.Resampling.BICUBIC
            )
        buffer = io.BytesIO()
        if output_format == "webp":
            image.save(buffer, format="WEBP", quality=quality, method=0)
        else:
            image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        return buffer.getvalue()

    async def close(self, session_id: str, generation_id: str) -> None:
        identity = (session_id, generation_id)
        async with self._session_lock:
            state = self._sessions.pop(identity, None)
        if state is None:
            return
        if state.runner is not None:
            state.runner.cancel()
        while not state.queue.empty():
            try:
                job = state.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.cancel()
            state.queue.task_done()
        if state.runner is not None:
            await asyncio.gather(state.runner, return_exceptions=True)
        self._update_capacity_metrics()

    async def close_all(self) -> None:
        for session_id, generation_id in list(self._sessions):
            await self.close(session_id, generation_id)
        if not self._encode_executor_shutdown:
            self._encode_executor_shutdown = True
            await asyncio.to_thread(
                self._encode_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def media_acceptance(
        self,
        session_id: str,
        generation_id: str,
    ) -> MediaProfileAcceptance:
        state = self._sessions.get((session_id, generation_id))
        if state is None:
            raise ProtocolViolation("unknown VAE session generation")
        return state.media_acceptance

    def runtime_state(self) -> dict[str, int | float]:
        return {
            "runnable_sessions": sum(
                state.processing or not state.queue.empty()
                for state in self._sessions.values()
            ),
            "blocked_sessions": sum(
                state.queue.full() for state in self._sessions.values()
            ),
            "queue_depth": sum(
                state.queue.qsize() for state in self._sessions.values()
            ),
            "service_time_ms": self._service_time_ms,
            "encode_workers": self.encode_workers,
        }

    def _update_capacity_metrics(self) -> None:
        update_capacity(
            active=len(self._sessions),
            queued=sum(state.queue.qsize() for state in self._sessions.values()),
            maximum=self.max_sessions,
        )


class TAEHVEngine:
    """Shared immutable TAEHV weights with per-generation decoder objects."""

    backend = "taehv"
    rgb_quantization = "round"
    _DEFAULT_WARMUP_SPATIAL_SHAPE = (1, 30, 52)

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        try:
            from taehv import TAEHV
        except ImportError as exc:
            raise RuntimeError(
                "The taehv package is required by the VAE worker"
            ) from exc
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = (
            TAEHV(checkpoint_path=checkpoint_path)
            .eval()
            .to(device=self.device, dtype=dtype)
            .requires_grad_(False)
        )

    def create_decoder(self, identity):
        del identity
        from taehv import StreamingTAEHV

        return StreamingTAEHV(self.model).eval()

    @torch.no_grad()
    def warmup(
        self,
        latent_shape: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        """Pay decoder/CUDA initialization before the worker becomes ready."""
        if latent_shape is None:
            latent_shape = (
                1,
                int(self.model.latent_channels),
                *self._DEFAULT_WARMUP_SPATIAL_SHAPE,
            )
        decoder = self.create_decoder(("warmup", "warmup"))
        latents = torch.zeros(latent_shape, dtype=self.dtype)
        for _ in self.iter_decode(decoder, latents, first_chunk=True):
            pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def iter_decode(self, decoder, latents: torch.Tensor, *, first_chunk: bool):
        if first_chunk:
            decoder.reset()
        source = latents.to(device=self.device, dtype=self.dtype, non_blocking=True)
        source = source.permute(0, 2, 1, 3, 4).contiguous()
        frame = decoder.decode(source)
        while frame is not None:
            yield frame.permute(0, 2, 1, 3, 4).contiguous().clamp(0, 1)
            frame = decoder.decode()

    @torch.no_grad()
    def decode(self, decoder, latents: torch.Tensor, *, first_chunk: bool):
        decoded_frames = list(
            self.iter_decode(decoder, latents, first_chunk=first_chunk)
        )
        if decoded_frames:
            return torch.cat(decoded_frames, dim=2)
        height = int(latents.shape[-2]) * 16
        width = int(latents.shape[-1]) * 16
        return torch.empty((1, 3, 0, height, width), dtype=self.dtype)
