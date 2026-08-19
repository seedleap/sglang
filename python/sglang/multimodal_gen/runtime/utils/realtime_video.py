# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    cancel_async_shared_memory_payload,
    discard_async_shared_memory_payload,
    materialize_async_payload_from_shared_memory,
    publish_async_shared_memory_error,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )

logger = init_logger(__name__)

RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip"
RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip"
WEBP_FRAME_CONTENT_TYPE = "image/webp"
JPEG_FRAME_CONTENT_TYPE = "image/jpeg"
RAW_RGB_CHANNELS = 3
RAW_RGBA_CHANNELS = 4
_RAW_RGB_DELTA_GZIP_LEVEL = 0
_ASYNC_RAW_RGB_MAX_IN_FLIGHT = 2
_ASYNC_RAW_RGB_SLOT_WAIT_S = 0.05


class AsyncRawRGBFrameMaterializer:
    """Move CUDA RGB frames off the scheduler critical path with bounded staging."""

    def __init__(
        self,
        *,
        max_in_flight: int = _ASYNC_RAW_RGB_MAX_IN_FLIGHT,
        shared_memory_dir: str | None = None,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self._shared_memory_dir = shared_memory_dir
        self._max_cached_buffers = max_in_flight
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._pool_lock = threading.Lock()
        self._host_pool: list[torch.Tensor] = []
        self._outstanding_lock = threading.Lock()
        self._outstanding_refs: dict[str, dict[str, Any]] = {}
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight,
            thread_name_prefix="sglang-raw-rgb",
        )
        logger.info(
            "Async raw RGB frame materializer initialized: "
            "max_in_flight=%d shared_memory_dir=%s",
            max_in_flight,
            shared_memory_dir,
        )

    def enqueue(
        self,
        output: torch.Tensor,
        *,
        request_id: str,
        chunk_index: int,
        session_id: str,
        generation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Enqueue exact RGB24 conversion and D2H; return a serializable handle."""
        if output.device.type != "cuda" or output.dim() != 5:
            raise ValueError("async raw RGB output must be a 5D CUDA tensor")
        if int(output.shape[1]) != RAW_RGB_CHANNELS:
            raise ValueError("async raw RGB output must contain exactly three channels")
        if self._closed:
            raise RuntimeError("async raw RGB materializer is closed")

        if not self._slots.acquire(timeout=_ASYNC_RAW_RGB_SLOT_WAIT_S):
            raise RuntimeError("async raw RGB staging slots are full")
        host_frames: torch.Tensor | None = None
        reference: dict[str, Any] | None = None
        copy_done: torch.cuda.Event | None = None
        try:
            # Preserve the existing truncating conversion and B/C/T/H/W -> B/T/H/W/C.
            device_frames = _tensor_batch_to_rgb24_tensor(output)
            shape = tuple(int(dim) for dim in device_frames.shape)
            batch_size, num_frames, height, width, channels = shape
            num_bytes = device_frames.numel() * device_frames.element_size()
            host_frames = self._acquire_host_buffer(shape)
            reference = reserve_async_shared_memory_payload(
                num_bytes,
                root=self._shared_memory_dir,
            )
            reference.update(
                {
                    "kind": "raw_rgb24_frames",
                    "version": 1,
                    "request_id": request_id,
                    "chunk_index": int(chunk_index),
                    "session_id": session_id,
                    "generation_id": generation_id,
                    "batch_num_frames": [num_frames] * batch_size,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

            device_index = output.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            copy_stream = self._copy_streams.get(device_index)
            if copy_stream is None:
                copy_stream = torch.cuda.Stream(device=output.device)
                self._copy_streams[device_index] = copy_stream
            producer_stream = torch.cuda.current_stream(output.device)
            copy_done = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(producer_stream)
                host_frames.copy_(device_frames, non_blocking=True)
                copy_done.record(copy_stream)

            with self._outstanding_lock:
                self._outstanding_refs[reference["path"]] = reference
            self._executor.submit(
                self._finish_copy,
                copy_done,
                device_frames,
                host_frames,
                reference,
            )
            return reference, {
                "format": "rgb24",
                "width": width,
                "height": height,
                "channels": channels,
                "bytes_per_frame": width * height * channels,
            }
        except Exception:
            if reference is not None:
                with self._outstanding_lock:
                    self._outstanding_refs.pop(reference["path"], None)
            if copy_done is not None:
                copy_done.synchronize()
            if reference is not None:
                discard_async_shared_memory_payload(
                    reference,
                    root=self._shared_memory_dir,
                )
            if host_frames is not None:
                self._release_host_buffer(host_frames)
            self._slots.release()
            raise

    def _acquire_host_buffer(self, shape: tuple[int, ...]) -> torch.Tensor:
        with self._pool_lock:
            for index, buffer in enumerate(self._host_pool):
                if tuple(int(dim) for dim in buffer.shape) == shape:
                    return self._host_pool.pop(index)
            # Drop one stale shape before allocating so the pinned cache remains
            # bounded across sessions that use different resolutions/chunk sizes.
            if self._host_pool:
                self._host_pool.pop()
        return torch.empty(shape, dtype=torch.uint8, device="cpu", pin_memory=True)

    def _release_host_buffer(self, buffer: torch.Tensor) -> None:
        with self._pool_lock:
            if len(self._host_pool) < self._max_cached_buffers:
                self._host_pool.append(buffer)

    def _finish_copy(
        self,
        copy_done: torch.cuda.Event,
        device_frames: torch.Tensor,
        host_frames: torch.Tensor,
        reference: dict[str, Any],
    ) -> None:
        try:
            copy_done.synchronize()
            publish_async_shared_memory_payload(
                reference,
                host_frames.numpy(),
                root=self._shared_memory_dir,
            )
        except Exception as error:
            logger.exception("asynchronous raw RGB materialization failed")
            try:
                publish_async_shared_memory_error(
                    reference,
                    error,
                    root=self._shared_memory_dir,
                )
            except Exception:
                logger.exception("failed to publish raw RGB materialization error")
        finally:
            # Holding device_frames until the event completes prevents the caching
            # allocator from reusing its storage while the copy stream is reading it.
            del device_frames
            self._release_host_buffer(host_frames)
            try:
                terminal = wait_for_async_shared_memory_terminal(
                    reference,
                    root=self._shared_memory_dir,
                )
                if terminal == "owner_timeout":
                    logger.warning(
                        "async raw RGB reference expired without consumer ACK: "
                        "request_id=%s chunk_idx=%s",
                        reference.get("request_id"),
                        reference.get("chunk_index"),
                    )
            except Exception:
                logger.exception("failed to reclaim async raw RGB reference")
                discard_async_shared_memory_payload(
                    reference,
                    root=self._shared_memory_dir,
                )
            finally:
                with self._outstanding_lock:
                    self._outstanding_refs.pop(reference["path"], None)
                self._slots.release()

    def cancel_session(
        self,
        session_id: str,
        generation_id: str | None = None,
    ) -> None:
        """Cancel producer-owned refs for a released realtime session."""
        with self._outstanding_lock:
            references = [
                reference
                for reference in self._outstanding_refs.values()
                if reference.get("session_id") == session_id
                and (
                    generation_id is None
                    or reference.get("generation_id") == generation_id
                )
            ]
        for reference in references:
            cancel_async_shared_memory_payload(
                reference,
                root=self._shared_memory_dir,
            )

    def close(self) -> None:
        """Drain background copies and release cached pinned host buffers."""
        self._closed = True
        with self._outstanding_lock:
            references = list(self._outstanding_refs.values())
        for reference in references:
            cancel_async_shared_memory_payload(
                reference,
                root=self._shared_memory_dir,
            )
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._pool_lock:
            self._host_pool.clear()
        self._copy_streams.clear()
        with self._outstanding_lock:
            self._outstanding_refs.clear()


def materialize_async_raw_rgb_frame_batches(
    reference: dict[str, Any],
    *,
    shared_memory_dir: str | None = None,
    timeout_s: float | None = None,
) -> list[list[bytes]]:
    """Resolve an async RGB24 handle into the established list-of-frame-bytes API."""
    try:
        if reference.get("kind") != "raw_rgb24_frames" or reference.get("version") != 1:
            raise ValueError("unsupported asynchronous raw RGB frame reference")
        width = int(reference.get("width", 0))
        height = int(reference.get("height", 0))
        channels = int(reference.get("channels", 0))
        batch_num_frames = [
            int(value) for value in reference.get("batch_num_frames", [])
        ]
        if (
            width <= 0
            or height <= 0
            or channels != RAW_RGB_CHANNELS
            or not batch_num_frames
            or any(value < 0 for value in batch_num_frames)
        ):
            raise ValueError("invalid asynchronous raw RGB frame metadata")

        bytes_per_frame = width * height * channels
        expected_bytes = sum(batch_num_frames) * bytes_per_frame
        if int(reference.get("num_bytes", -1)) != expected_bytes:
            raise ValueError("asynchronous raw RGB frame byte length is inconsistent")
        materialize_kwargs: dict[str, Any] = {"root": shared_memory_dir}
        if timeout_s is not None:
            materialize_kwargs["timeout_s"] = timeout_s
        payload = materialize_async_payload_from_shared_memory(
            reference,
            **materialize_kwargs,
        )

        frame_batches: list[list[bytes]] = []
        offset = 0
        for num_frames in batch_num_frames:
            frame_batch = []
            for _ in range(num_frames):
                next_offset = offset + bytes_per_frame
                frame_batch.append(payload[offset:next_offset])
                offset = next_offset
            frame_batches.append(frame_batch)
        if offset != len(payload):
            raise ValueError("asynchronous raw RGB payload has trailing bytes")
        return frame_batches
    except Exception:
        cancel_async_shared_memory_payload(reference, root=shared_memory_dir)
        raise


def cancel_async_raw_rgb_frame_reference(
    reference: dict[str, Any] | None,
    *,
    shared_memory_dir: str | None = None,
) -> bool:
    """Cancel an unconsumed raw-frame reference and return producer credit."""
    return cancel_async_shared_memory_payload(reference, root=shared_memory_dir)


def build_delta_gzip_raw_rgb_payload(
    frames: list[bytes],
    *,
    reference_frame: bytes | None = None,
) -> bytes:
    if not frames:
        return b""

    frame_size = len(frames[0])
    if reference_frame is not None and len(reference_frame) != frame_size:
        raise ValueError("raw RGB delta gzip reference frame size mismatch")

    previous = (
        np.frombuffer(reference_frame, dtype=np.uint8)
        if reference_frame is not None
        else None
    )
    # keep gzip framing for lossless transport without spending realtime budget on compression
    compressor = zlib.compressobj(
        level=_RAW_RGB_DELTA_GZIP_LEVEL, method=zlib.DEFLATED, wbits=31
    )
    compressed_chunks = []
    for frame in frames:
        if len(frame) != frame_size:
            raise ValueError("raw RGB delta gzip requires fixed-size frames")
        current = np.frombuffer(frame, dtype=np.uint8)
        if previous is None:
            delta_frame = frame
        else:
            delta_frame = np.bitwise_xor(current, previous).tobytes()
        compressed_chunks.append(compressor.compress(delta_frame))
        previous = current

    compressed_chunks.append(compressor.flush())
    return b"".join(compressed_chunks)


def restore_delta_gzip_raw_rgb_payload(
    payload: bytes,
    *,
    bytes_per_frame: int,
    num_frames: int,
    reference_frame: bytes | None = None,
) -> bytes:
    if reference_frame is not None and len(reference_frame) != bytes_per_frame:
        raise ValueError("delta gzip reference frame size mismatch")

    delta_payload = zlib.decompress(payload, wbits=31)
    expected_size = bytes_per_frame * num_frames
    if len(delta_payload) != expected_size:
        raise ValueError(
            "delta gzip payload size mismatch: "
            f"expected {expected_size}, got {len(delta_payload)}"
        )

    restored = bytearray(delta_payload)
    previous = (
        np.frombuffer(reference_frame, dtype=np.uint8)
        if reference_frame is not None
        else None
    )
    for frame_idx in range(num_frames):
        offset = frame_idx * bytes_per_frame
        current = np.frombuffer(
            restored, dtype=np.uint8, count=bytes_per_frame, offset=offset
        )
        if previous is not None:
            current ^= previous
        previous = current
    return bytes(restored)


def build_raw_rgb_frame_batches(
    output: Any,
    req: Req,
    output_batch: OutputBatch,
    post_process_sample_fn: Callable[..., Any],
) -> tuple[list[list[bytes]], dict[str, Any]]:
    """post-process for realtime responses, returns only the batched frames and metadata"""
    start = time.monotonic()
    sample_to_frames_ms = 0.0
    frames_to_bytes_ms = 0.0
    raw_bytes = 0
    num_frames = 0
    frame_shape = None
    frame_batches = []
    if isinstance(output, torch.Tensor):
        outputs = list(output)
    else:
        outputs = output if isinstance(output, Sequence) else [output]

    for sample in outputs:
        stage_start = time.monotonic()
        if (
            isinstance(sample, torch.Tensor)
            and not req.enable_frame_interpolation
            and not req.enable_upscaling
        ):
            frames = _tensor_sample_to_rgb24_array(sample)
        else:
            frames = post_process_sample_fn(
                sample,
                req.data_type,
                req.fps,
                False,
                None,
                audio_sample_rate=output_batch.audio_sample_rate,
                output_compression=req.output_compression,
                enable_frame_interpolation=req.enable_frame_interpolation,
                frame_interpolation_exp=req.frame_interpolation_exp,
                frame_interpolation_scale=req.frame_interpolation_scale,
                frame_interpolation_model_path=req.frame_interpolation_model_path,
                enable_upscaling=False,
                upscaling_model_path=req.upscaling_model_path,
                upscaling_scale=req.upscaling_scale,
            )
            if req.enable_upscaling and frames:
                from sglang.multimodal_gen.runtime.postprocess import (
                    batch_upscale_frames,
                )

                frames = batch_upscale_frames(
                    frames,
                    model_path=req.upscaling_model_path,
                    scale=req.upscaling_scale,
                )
        sample_to_frames_ms += (time.monotonic() - stage_start) * 1000.0

        stage_start = time.monotonic()

        # numpy frames to RGB24 bytes
        raw_frames = []
        for frame in frames:
            if frame.ndim == 2:
                frame = frame[:, :, None]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] > RAW_RGB_CHANNELS:
                frame = frame[:, :, :RAW_RGB_CHANNELS]
            frame = np.ascontiguousarray(frame)
            frame_shape = tuple(int(dim) for dim in frame.shape)
            frame_bytes = frame.tobytes()
            raw_bytes += len(frame_bytes)
            num_frames += 1
            raw_frames.append(frame_bytes)
        frames_to_bytes_ms += (time.monotonic() - stage_start) * 1000.0
        frame_batches.append(raw_frames)

    total_ms = (time.monotonic() - start) * 1000.0
    logger.info(
        "realtime raw RGB frame batch timing: request_id=%s "
        "chunk_idx=%s sample_to_frames=%.2fms frames_to_bytes=%.2fms "
        "total=%.2fms batches=%d frames=%d frame_shape=%s "
        "raw_bytes=%d content_type=%s",
        req.request_id,
        req.block_idx,
        sample_to_frames_ms,
        frames_to_bytes_ms,
        total_ms,
        len(frame_batches),
        num_frames,
        frame_shape,
        raw_bytes,
        RAW_RGB_CONTENT_TYPE,
    )
    frame_metadata: dict[str, Any] = {}
    if frame_shape is not None and len(frame_shape) == 3:
        frame_height, frame_width, channels = frame_shape
        frame_metadata = {
            "format": "rgb24",
            "width": frame_width,
            "height": frame_height,
            "channels": channels,
            "bytes_per_frame": frame_width * frame_height * channels,
        }
    return frame_batches, frame_metadata


def _tensor_sample_to_rgb24_array(sample: torch.Tensor) -> np.ndarray:
    if sample.dim() == 3:
        sample = sample.unsqueeze(1)
    return _tensor_batch_to_rgb24_tensor(sample.unsqueeze(0))[0].cpu().numpy()


def _tensor_batch_to_rgb24_tensor(output: torch.Tensor) -> torch.Tensor:
    """Apply the established truncating quantization and return B/T/H/W/C."""
    if output.dim() != 5:
        raise ValueError("raw RGB tensor batch must have B/C/T/H/W layout")
    return (
        (output * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()
    )
