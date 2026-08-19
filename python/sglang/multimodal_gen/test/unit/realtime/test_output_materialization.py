# SPDX-License-Identifier: Apache-2.0

import threading
import time

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    save_outputs,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    SHARED_MEMORY_DIR_ENV,
    materialize_async_payload_from_shared_memory,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    AsyncRawRGBFrameMaterializer,
    _tensor_batch_to_rgb24_tensor,
    build_raw_rgb_frame_batches,
)


def test_async_materializer_logs_initialization_once(monkeypatch, tmp_path):
    from sglang.multimodal_gen.runtime.utils import realtime_video

    log_calls = []
    monkeypatch.setattr(
        realtime_video,
        "logger",
        type("Logger", (), {"info": lambda _self, *args: log_calls.append(args)})(),
    )

    materializer = AsyncRawRGBFrameMaterializer(
        max_in_flight=2,
        shared_memory_dir=str(tmp_path),
    )
    materializer.close()

    assert len(log_calls) == 1
    assert log_calls[0][0].startswith("Async raw RGB frame materializer initialized")
    assert log_calls[0][1:] == (2, str(tmp_path))


def test_materialize_output_sample_converts_tensor_to_uint8_frames():
    sample = torch.zeros(3, 1, 2, 2)
    sample[0] = 1.0
    sample[1] = 0.5

    materialized = materialize_output_sample(sample, DataType.VIDEO, fps=24)

    assert materialized.fps == 24
    assert materialized.audio is None
    assert len(materialized.frames) == 1
    frame = materialized.frames[0]
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[..., 0] == 255)
    assert np.all(frame[..., 1] == 127)
    assert np.all(frame[..., 2] == 0)


def test_save_outputs_can_materialize_without_saving(tmp_path):
    sample = np.full((2, 2, 3), 0.25, dtype=np.float32)
    output_path = tmp_path / "image.png"
    samples_out = []
    frames_out = []

    paths = save_outputs(
        [sample],
        DataType.IMAGE,
        fps=1,
        save_output=False,
        build_output_path=lambda _idx: str(output_path),
        samples_out=samples_out,
        frames_out=frames_out,
    )

    assert paths == [str(output_path)]
    assert not output_path.exists()
    assert samples_out[0] is sample
    assert len(frames_out) == 1
    assert len(frames_out[0]) == 1
    assert frames_out[0][0].dtype == np.uint8
    assert np.all(frames_out[0][0] == 63)


def test_file_path_transport_clears_in_memory_outputs():
    worker = GPUWorker.__new__(GPUWorker)
    worker.rank = 0
    output_batch = OutputBatch(
        output=[object()],
        audio=torch.zeros(1),
        audio_sample_rate=16000,
    )

    def save_output_paths(batch):
        batch.output_file_paths = ["/tmp/output.png"]

    worker._materialize_file_path_transport(output_batch, save_output_paths)

    assert output_batch.output_file_paths == ["/tmp/output.png"]
    assert output_batch.output is None
    assert output_batch.audio is None
    assert output_batch.audio_sample_rate is None


def test_raw_rgb_frame_batches_convert_batched_video_tensor_to_thwc_bytes():
    output = torch.zeros(1, 3, 2, 2, 2)
    output[0, 0] = 1.0
    output[0, 1] = 0.5
    req = type(
        "Req",
        (),
        {
            "enable_frame_interpolation": False,
            "enable_upscaling": False,
            "request_id": "req",
            "block_idx": 0,
        },
    )()
    output_batch = OutputBatch(audio_sample_rate=None)

    frame_batches, metadata = build_raw_rgb_frame_batches(
        output,
        req,
        output_batch,
        post_process_sample_fn=lambda *args, **kwargs: None,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert len(frame_batches[0]) == 2
    first = np.frombuffer(frame_batches[0][0], dtype=np.uint8).reshape(2, 2, 3)
    assert np.all(first[..., 0] == 255)
    assert np.all(first[..., 1] == 127)
    assert np.all(first[..., 2] == 0)


def test_raw_rgb_frame_batches_apply_realtime_upscaling(monkeypatch):
    calls = []

    def fake_batch_upscale_frames(frames, *, model_path, scale):
        calls.append((model_path, scale, [frame.shape for frame in frames]))
        return [
            np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
            for frame in frames
        ]

    from sglang.multimodal_gen.runtime import postprocess

    monkeypatch.setattr(postprocess, "batch_upscale_frames", fake_batch_upscale_frames)

    req = type(
        "Req",
        (),
        {
            "data_type": DataType.VIDEO,
            "fps": 24,
            "output_compression": None,
            "enable_frame_interpolation": False,
            "frame_interpolation_exp": 1,
            "frame_interpolation_scale": 1.0,
            "frame_interpolation_model_path": None,
            "enable_upscaling": True,
            "upscaling_model_path": "mock-sr",
            "upscaling_scale": 2,
            "request_id": "req",
            "block_idx": 0,
        },
    )()
    output_batch = OutputBatch(audio_sample_rate=None)

    def post_process_sample(_sample, *_args, **kwargs):
        assert kwargs["enable_upscaling"] is False
        return [np.array([[[1, 2, 3]]], dtype=np.uint8)]

    frame_batches, metadata = build_raw_rgb_frame_batches(
        torch.zeros(1, 3, 1, 1, 1),
        req,
        output_batch,
        post_process_sample,
    )

    assert calls == [("mock-sr", 2, [(1, 1, 3)])]
    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert frame_batches[0][0] == bytes([1, 2, 3] * 4)


def test_async_quantization_helper_matches_existing_truncation_and_layout():
    output = torch.tensor(
        [
            [
                [[[-0.1, 0.0]], [[0.25, 0.5]]],
                [[[0.5, 0.75]], [[1.0, 1.1]]],
                [[[1.0, 0.999]], [[0.003, 0.501]]],
            ]
        ],
        dtype=torch.float32,
    )
    expected = (
        (output * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()
    )

    actual = _tensor_batch_to_rgb24_tensor(output)

    assert actual.shape == (1, 2, 1, 2, 3)
    assert torch.equal(actual, expected)


def test_local_taehv_cpu_output_uses_synchronous_raw_frame_fallback():
    worker = GPUWorker.__new__(GPUWorker)
    worker.rank = 0
    worker.server_args = type(
        "ServerArgs",
        (),
        {
            "num_gpus": 1,
            "realtime_vae_backend": "local",
            "pipeline_config": type(
                "PipelineConfig",
                (),
                {
                    "vae_config": type(
                        "VAEConfig",
                        (),
                        {"taehv_checkpoint_path": "/model/taehv.pt"},
                    )()
                },
            )(),
        },
    )()
    worker._async_raw_frame_materializer = None
    req = type(
        "Req",
        (),
        {
            "is_warmup": False,
            "enable_frame_interpolation": False,
            "enable_upscaling": False,
            "request_id": "req",
            "block_idx": 0,
            "return_raw_frames": True,
        },
    )()
    output_batch = OutputBatch(output=torch.ones(1, 3, 1, 2, 2))

    worker._materialize_raw_frame_transport(output_batch, req)

    assert output_batch.output is None
    assert output_batch.raw_frame_shared_memory_ref is None
    assert output_batch.raw_frame_content_type == RAW_RGB_CONTENT_TYPE
    assert output_batch.raw_frame_batches == [[bytes([255, 255, 255] * 4)]]


def test_async_materializer_credit_is_held_until_consumer_ack(tmp_path):
    class _CompletedEvent:
        @staticmethod
        def synchronize():
            pass

    materializer = AsyncRawRGBFrameMaterializer(
        max_in_flight=2,
        shared_memory_dir=str(tmp_path),
    )
    references = [
        reserve_async_shared_memory_payload(4, root=tmp_path) for _ in range(2)
    ]
    materializer._slots.acquire()
    materializer._slots.acquire()
    producers = [
        threading.Thread(
            target=materializer._finish_copy,
            args=(
                _CompletedEvent(),
                torch.zeros(1),
                torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                reference,
            ),
        )
        for reference in references
    ]
    for producer in producers:
        producer.start()
    deadline = time.monotonic() + 1
    while not all(
        (tmp_path / f"{reference['path'].split('/')[-1]}.ready").exists()
        for reference in references
    ):
        assert time.monotonic() < deadline
        time.sleep(0.001)

    third_acquired = threading.Event()

    def acquire_third_credit():
        materializer._slots.acquire()
        third_acquired.set()
        materializer._slots.release()

    third = threading.Thread(target=acquire_third_credit)
    third.start()
    assert not third_acquired.wait(timeout=0.02)

    assert materialize_async_payload_from_shared_memory(
        references[0], root=tmp_path, timeout_s=1
    ) == bytes([1, 2, 3, 4])
    assert third_acquired.wait(timeout=1)
    assert materialize_async_payload_from_shared_memory(
        references[1], root=tmp_path, timeout_s=1
    ) == bytes([1, 2, 3, 4])

    for producer in producers:
        producer.join()
    third.join()
    materializer.close()
    assert list(tmp_path.iterdir()) == []


def test_scheduler_send_failure_cancels_async_raw_frame_reference(
    monkeypatch,
    tmp_path,
):
    class _FailingReceiver:
        @staticmethod
        def send_multipart(_payload):
            raise RuntimeError("send failed")

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
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.receiver = _FailingReceiver()
    scheduler.server_args = type(
        "ServerArgs",
        (),
        {"scheduler_endpoint": "tcp://127.0.0.1:5555"},
    )()
    output_batch = OutputBatch(raw_frame_shared_memory_ref=reference)

    with pytest.raises(RuntimeError, match="send failed"):
        scheduler.return_result(output_batch, identity=b"client")
    producer.join()

    assert terminal == ["cancel"]
    assert output_batch.raw_frame_shared_memory_ref is None
    assert list(tmp_path.iterdir()) == []


def test_session_release_cancels_materializer_owned_reference(tmp_path):
    class _CompletedEvent:
        @staticmethod
        def synchronize():
            pass

    materializer = AsyncRawRGBFrameMaterializer(
        max_in_flight=1,
        shared_memory_dir=str(tmp_path),
    )
    reference = reserve_async_shared_memory_payload(4, root=tmp_path)
    reference.update(
        {
            "session_id": "session-1",
            "generation_id": "generation-1",
            "request_id": "request-1",
            "chunk_index": 0,
        }
    )
    with materializer._outstanding_lock:
        materializer._outstanding_refs[reference["path"]] = reference
    materializer._slots.acquire()
    producer = threading.Thread(
        target=materializer._finish_copy,
        args=(
            _CompletedEvent(),
            torch.zeros(1),
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            reference,
        ),
    )
    producer.start()
    deadline = time.monotonic() + 1
    while not (tmp_path / f"{reference['path'].split('/')[-1]}.ready").exists():
        assert time.monotonic() < deadline
        time.sleep(0.001)

    materializer.cancel_session("session-1", "generation-1")
    producer.join()
    materializer.close()

    assert list(tmp_path.iterdir()) == []
