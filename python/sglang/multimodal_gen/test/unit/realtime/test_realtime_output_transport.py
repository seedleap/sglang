# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import msgspec.msgpack
import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_output_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
    _cancel_unconsumed_output_ref,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    SHARED_MEMORY_DIR_ENV,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
    build_delta_gzip_raw_rgb_payload,
    build_raw_rgb_frame_batches,
    materialize_async_raw_rgb_frame_batches,
    restore_delta_gzip_raw_rgb_payload,
)


def _unpack_frame_batch_messages(payloads):
    messages = []
    payload_iter = iter(payloads)
    for payload in payload_iter:
        message = msgspec.msgpack.decode(payload)
        message_type = message.pop("type")
        if message_type == "frame_batch":
            frame_payload = message.pop("payload")
        else:
            assert message_type == "frame_batch_header"
            frame_payload = next(payload_iter)
        messages.append((message, frame_payload))
    return messages


def test_raw_rgb_frame_batches_preserve_frame_bytes_and_metadata():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    grayscale = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rgba = np.array(
        [
            [[5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20]],
        ],
        dtype=np.uint8,
    )

    def post_process_sample(*_args, **_kwargs):
        return [grayscale, rgba]

    frame_batches, metadata = build_raw_rgb_frame_batches(
        object(),
        req,
        output_batch,
        post_process_sample,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert frame_batches[0][0] == bytes([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    assert frame_batches[0][1] == bytes([5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
    assert RAW_RGB_CONTENT_TYPE == "application/x-raw-rgb"


def test_raw_rgb_frame_batches_use_tensor_fast_path_without_postprocess():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    output = torch.tensor(
        [[[[[0.0]], [[0.25]]], [[[0.5]], [[0.75]]], [[[1.0]], [[1.0]]]]]
    )

    def post_process_sample(*_args, **_kwargs):
        raise AssertionError("tensor realtime output should not use postprocess")

    frame_batches, metadata = build_raw_rgb_frame_batches(
        output,
        req,
        output_batch,
        post_process_sample,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 1,
        "height": 1,
        "channels": 3,
        "bytes_per_frame": 3,
    }
    assert frame_batches == [[bytes([0, 127, 255]), bytes([63, 191, 255])]]


def test_output_batch_uses_raw_frame_transport_names():
    output_batch = OutputBatch(
        raw_frame_batches=[[b"rgb"]],
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
        raw_frame_metadata={"format": "rgb24"},
    )

    assert output_batch.raw_frame_batches == [[b"rgb"]]
    assert output_batch.raw_frame_content_type == RAW_RGB_CONTENT_TYPE
    assert output_batch.raw_frame_metadata == {"format": "rgb24"}


def _published_raw_rgb_reference(
    payload: bytes,
    *,
    root,
    chunk_index: int,
    width: int,
    height: int,
    batch_num_frames: list[int],
):
    reference = reserve_async_shared_memory_payload(len(payload), root=root)
    reference.update(
        {
            "kind": "raw_rgb24_frames",
            "version": 1,
            "request_id": f"req-{chunk_index}",
            "chunk_index": chunk_index,
            "session_id": "session-1",
            "generation_id": "generation-1",
            "batch_num_frames": batch_num_frames,
            "width": width,
            "height": height,
            "channels": 3,
        }
    )

    def publish_and_reclaim():
        publish_async_shared_memory_payload(reference, payload, root=root)
        wait_for_async_shared_memory_terminal(reference, root=root, timeout_s=1)

    producer = threading.Thread(target=publish_and_reclaim)
    producer.start()
    return reference, producer


def test_async_raw_rgb_references_preserve_two_chunk_order_and_cleanup(tmp_path):
    frame0 = bytes([1, 2, 3, 4, 5, 6])
    frame1 = bytes([7, 8, 9, 10, 11, 12])
    frame2 = bytes([13, 14, 15, 16, 17, 18])
    first, first_producer = _published_raw_rgb_reference(
        frame0 + frame1,
        root=tmp_path,
        chunk_index=0,
        width=2,
        height=1,
        batch_num_frames=[2],
    )
    second, second_producer = _published_raw_rgb_reference(
        frame2,
        root=tmp_path,
        chunk_index=1,
        width=2,
        height=1,
        batch_num_frames=[1],
    )

    first_frames = materialize_async_raw_rgb_frame_batches(
        first,
        shared_memory_dir=str(tmp_path),
    )
    second_frames = materialize_async_raw_rgb_frame_batches(
        second,
        shared_memory_dir=str(tmp_path),
    )
    first_producer.join()
    second_producer.join()

    assert first_frames == [[frame0, frame1]]
    assert second_frames == [[frame2]]
    assert not Path(first["path"]).exists()
    assert not Path(first["ready_path"]).exists()
    assert not Path(second["path"]).exists()
    assert not Path(second["ready_path"]).exists()


def test_720p_rgb24_chunk_byte_length_contract():
    assert 16 * 704 * 1248 * 3 == 42_172_416


def test_explicit_large_raw_payload_build_is_offloaded():
    assert realtime_output_adapter._should_build_payload_off_loop(
        content_type=RAW_RGB_CONTENT_TYPE,
        output_format="raw",
        transport_frames=[
            bytes(realtime_output_adapter.FRAME_BATCH_PACK_OFFLOAD_BYTES)
        ],
    )
    assert not realtime_output_adapter._should_build_payload_off_loop(
        content_type=RAW_RGB_CONTENT_TYPE,
        output_format="raw",
        transport_frames=[b"small"],
    )


def test_realtime_output_adapter_materializes_async_ref_before_webp_preview(
    monkeypatch,
    tmp_path,
):
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    monkeypatch.setenv(SHARED_MEMORY_DIR_ENV, str(tmp_path))
    frame = bytes([255, 0, 0, 0, 255, 0])
    reference, producer = _published_raw_rgb_reference(
        frame,
        root=tmp_path,
        chunk_index=0,
        width=2,
        height=1,
        batch_num_frames=[1],
    )
    result = OutputBatch(
        raw_frame_shared_memory_ref=reference,
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
        raw_frame_metadata={
            "format": "rgb24",
            "width": 2,
            "height": 1,
            "channels": 3,
            "bytes_per_frame": 6,
        },
    )
    batch = SimpleNamespace(
        block_idx=0,
        request_id="req-0",
        width=2,
        height=1,
        enable_upscaling=False,
        realtime_event_id=0,
        realtime_trace_id="trace-async",
        realtime_output_format="webp",
        realtime_preview_max_width=480,
        output_compression=90,
    )

    async def run():
        ws = _WebSocket()
        stats = await RawRGBRealtimeOutputAdapter().send(
            ws,
            SimpleNamespace(
                id="session-1",
                generation_id="generation-1",
                trace_id="trace-async",
            ),
            result,
            batch,
        )
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())
    producer.join()

    [(header, encoded_frame)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["source_width"] == 2
    assert header["source_height"] == 1
    assert encoded_frame.startswith(b"RIFF")
    assert stats["num_frames"] == 1
    assert result.raw_frame_batches == [[frame]]
    assert result.raw_frame_shared_memory_ref is None
    assert not Path(reference["path"]).exists()
    assert not Path(reference["ready_path"]).exists()


def test_realtime_output_adapter_cancellation_keeps_ref_cleanup_running(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    def materialize(_reference):
        started.set()
        try:
            release.wait(timeout=1)
            return [[bytes([1, 2, 3])]]
        finally:
            cleaned.set()

    monkeypatch.setattr(
        realtime_output_adapter,
        "materialize_async_raw_rgb_frame_batches",
        materialize,
    )
    monkeypatch.setattr(
        realtime_output_adapter,
        "cancel_async_raw_rgb_frame_reference",
        lambda _reference: True,
    )
    monkeypatch.setattr(
        realtime_output_adapter,
        "_validate_async_raw_frame_reference",
        lambda *_args: None,
    )
    result = OutputBatch(
        raw_frame_shared_memory_ref={"kind": "raw_rgb24_frames"},
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
    )

    async def run():
        send_task = asyncio.create_task(
            RawRGBRealtimeOutputAdapter().send(
                SimpleNamespace(),
                SimpleNamespace(),
                result,
                SimpleNamespace(block_idx=0),
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        release.set()
        assert await asyncio.to_thread(cleaned.wait, 1)

    asyncio.run(run())

    assert result.raw_frame_shared_memory_ref is None


def test_pre_adapter_result_cancellation_returns_producer_credit(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(SHARED_MEMORY_DIR_ENV, str(tmp_path))
    reference = reserve_async_shared_memory_payload(4, root=tmp_path)
    terminal = []

    def reclaim_after_cancel():
        terminal.append(
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
        )

    producer = threading.Thread(target=reclaim_after_cancel)
    producer.start()
    result = OutputBatch(raw_frame_shared_memory_ref=reference)

    _cancel_unconsumed_output_ref(result=result)
    producer.join()

    assert terminal == ["cancel"]
    assert result.raw_frame_shared_memory_ref is None
    assert list(tmp_path.iterdir()) == []


def test_async_raw_frame_reference_identity_mismatch_is_cancelled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(SHARED_MEMORY_DIR_ENV, str(tmp_path))
    reference, producer = _published_raw_rgb_reference(
        bytes([1, 2, 3]),
        root=tmp_path,
        chunk_index=0,
        width=1,
        height=1,
        batch_num_frames=[1],
    )
    result = OutputBatch(
        raw_frame_shared_memory_ref=reference,
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
    )

    async def run():
        with pytest.raises(ValueError, match="request_id"):
            await RawRGBRealtimeOutputAdapter().send(
                SimpleNamespace(),
                SimpleNamespace(id="session-1", generation_id="generation-1"),
                result,
                SimpleNamespace(block_idx=0, request_id="wrong-request"),
            )

    asyncio.run(run())
    producer.join()

    assert result.raw_frame_shared_memory_ref is None
    assert list(tmp_path.iterdir()) == []


def test_delta_gzip_raw_rgb_payload_roundtrips_exactly():
    frames = [
        bytes([1, 2, 3, 4, 5, 6]),
        bytes([1, 2, 4, 4, 6, 6]),
        bytes([2, 2, 4, 5, 6, 7]),
    ]

    payload = build_delta_gzip_raw_rgb_payload(frames)
    restored = restore_delta_gzip_raw_rgb_payload(
        payload,
        bytes_per_frame=6,
        num_frames=3,
    )

    assert restored == b"".join(frames)


def test_raw_rgb_realtime_output_adapter_can_send_explicit_lossless_raw_payload():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_trace_id="trace-raw",
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, frame0 + frame1

    payloads, stats, expected_frames = asyncio.run(run())

    [(first_header, first_payload)] = _unpack_frame_batch_messages(payloads)
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert first_header["trace_id"] == "trace-raw"
    assert first_header["encoding"] == "raw"
    assert first_header["event_id"] == 3
    assert first_header["format"] == "rgb24"
    assert first_header["channels"] == 3
    assert first_header["bytes_per_frame"] == 3000
    assert first_header["raw_size"] == 6000
    assert first_header["total_size"] == len(first_payload)
    assert first_header["num_frames"] == 2
    assert first_header["num_frame_batches"] == 1
    assert first_header["frame_batch_index"] == 0
    assert "delta_reference" not in first_header
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 2
    assert first_payload == expected_frames


def test_raw_rgb_realtime_output_adapter_defaults_to_webp_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-offload-raw",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads

    payloads = asyncio.run(run())

    frame_batches = _unpack_frame_batch_messages(payloads)
    assert len(frame_batches) == 2
    first_header, first_payload = frame_batches[0]
    assert first_header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert first_header["encoding"] == "webp"
    assert first_header["source_width"] == 1000
    assert first_header["preview_width"] == 480
    assert first_header["width"] == 480
    assert first_header["num_frames"] == 1
    assert frame_batches[1][0]["num_frames"] == 1
    assert first_payload.startswith(b"RIFF")
    assert frame_batches[1][1].startswith(b"RIFF")


def test_raw_rgb_realtime_output_adapter_can_send_uncompressed_raw_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-raw",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, frame0 + frame1

    payloads, stats, expected_frames = asyncio.run(run())

    [(first_header, first_payload)] = _unpack_frame_batch_messages(payloads)
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert first_header["encoding"] == "raw"
    assert first_header["raw_size"] == 6000
    assert first_header["total_size"] == 6000
    assert first_header["num_frames"] == 2
    assert first_header["num_frame_batches"] == 1
    assert first_header["frame_batch_index"] == 0
    assert first_payload == expected_frames
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 2
    assert stats["ws_payload_bytes"] == sum(len(payload) for payload in payloads)


def test_raw_rgb_realtime_output_adapter_does_not_require_previous_frame_reference():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        base_batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
            realtime_output_format="raw",
        )
        next_batch = SimpleNamespace(
            block_idx=1,
            request_id="req-2",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
            realtime_output_format="raw",
        )
        metadata = {
            "format": "rgb24",
            "width": 2,
            "height": 1,
            "channels": 3,
            "bytes_per_frame": 6,
        }
        first = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3, 4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )
        second = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 4, 4, 6, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )

        await adapter.send(ws, SimpleNamespace(), first, base_batch)
        await adapter.send(ws, SimpleNamespace(), second, next_batch)
        return ws.payloads

    payloads = asyncio.run(run())

    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert second_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert "delta_reference" not in first_header
    assert "delta_reference" not in second_header
    assert first_payload == bytes([1, 2, 3, 4, 5, 6])
    assert second_payload == bytes([1, 2, 4, 4, 6, 6])


def test_raw_rgb_realtime_output_adapter_splits_large_frame_batches():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frames = [bytes([idx, idx + 1, idx + 2]) for idx in range(17)]
        batch = SimpleNamespace(
            block_idx=4,
            request_id="req-split",
            width=1,
            height=1,
            enable_upscaling=False,
            realtime_event_id=12,
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[frames],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    headers = [header for header, _ in _unpack_frame_batch_messages(payloads)]
    assert len(headers) == 2
    assert [header["chunk_index"] for header in headers] == [4, 4]
    assert [header["frame_batch_index"] for header in headers] == [0, 1]
    assert [header["num_frame_batches"] for header in headers] == [2, 2]
    assert [header["num_frames"] for header in headers] == [16, 1]
    assert [header["is_final_frame_batch"] for header in headers] == [False, True]
    assert "delta_reference" not in headers[0]
    assert "delta_reference" not in headers[1]
    assert stats["num_batches"] == 2
    assert stats["num_frames"] == 17


def test_raw_rgb_realtime_output_adapter_sends_large_payload_separately():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame = bytes([7]) * (72 * 1024)
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-large",
            width=len(frame) // 3,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[[frame]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": len(frame) // 3,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": len(frame),
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    assert len(payloads) == 2
    header = msgspec.msgpack.decode(payloads[0])
    assert header["type"] == "frame_batch_header"
    assert "payload" not in header
    assert header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert header["total_size"] == len(payloads[1])
    assert payloads[1] == bytes([7]) * (72 * 1024)
    assert stats["raw_bytes"] == len(payloads[1])
    assert stats["ws_payload_bytes"] == len(payloads[0]) + len(payloads[1])
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_can_send_webp_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["format"] == "webp"
    assert header["encoding"] == "webp"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"RIFF")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_offloads_preview_encoding(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )

    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame_count = realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE + 1
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp-offload",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[
                [
                    bytes([idx % 256, 0, 0, 0, 255, idx % 256])
                    for idx in range(frame_count)
                ]
            ],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads

    payloads = asyncio.run(run())

    assert [call[0] for call in calls] == [
        realtime_output_adapter._encode_rgb_frame_to_webp
    ] * (realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE + 1)
    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert first_header["encoding"] == "webp"
    assert first_header["num_frames"] == (
        realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE
    )
    assert first_header["frame_batch_index"] == 0
    assert first_header["num_frame_batches"] == 2
    assert first_header["is_final_frame_batch"] is False
    assert len(first_header["payload_lengths"]) == (
        realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE
    )
    assert second_header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert second_header["encoding"] == "webp"
    assert second_header["num_frames"] == 1
    assert second_header["frame_batch_index"] == 1
    assert second_header["num_frame_batches"] == 2
    assert second_header["is_final_frame_batch"] is True
    assert len(second_header["payload_lengths"]) == 1
    assert first_payload.startswith(b"RIFF")
    assert second_payload.startswith(b"RIFF")


def test_encoded_preview_frames_are_sent_as_soon_as_each_split_is_ready(monkeypatch):
    encode_count = 0
    encode_counts_at_send = []

    async def fake_to_thread(fn, *args, **kwargs):
        nonlocal encode_count
        encode_count += 1
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )
    monkeypatch.setattr(
        realtime_output_adapter,
        "ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE",
        1,
    )

    class _WebSocket:
        async def send_bytes(self, payload):
            encode_counts_at_send.append(encode_count)

    async def run():
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp-streaming",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[
                [
                    bytes([255, 0, 0, 0, 255, 0]),
                    bytes([0, 0, 255, 255, 255, 0]),
                ]
            ],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        await adapter.send(_WebSocket(), SimpleNamespace(), result, batch)

    asyncio.run(run())

    assert encode_counts_at_send[0] == 1


def test_raw_rgb_realtime_output_adapter_can_send_jpeg_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-jpeg",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="jpeg",
            output_compression=85,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == JPEG_FRAME_CONTENT_TYPE
    assert header["format"] == "jpeg"
    assert header["encoding"] == "jpeg"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"\xff\xd8")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1
