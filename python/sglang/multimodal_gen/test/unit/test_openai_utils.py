# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from contextlib import nullcontext
from types import SimpleNamespace

from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_process_generation_batch_accepts_async_raw_frame_reference(monkeypatch):
    expected = OutputBatch(
        raw_frame_shared_memory_ref={"kind": "raw_rgb24_frames", "version": 1}
    )

    class _SchedulerClient:
        async def forward(self, _batch):
            return expected

    monkeypatch.setattr(openai_utils, "trace_req", lambda _trace_ctx: nullcontext())
    monkeypatch.setattr(
        openai_utils,
        "log_generation_timer",
        lambda _logger, _prompt: nullcontext(),
    )
    monkeypatch.setattr(
        openai_utils,
        "get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )

    paths, result = asyncio.run(
        process_generation_batch(
            _SchedulerClient(),
            SimpleNamespace(trace_ctx=None, prompt="test"),
        )
    )

    assert paths == []
    assert result is expected
