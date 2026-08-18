# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.utils.realtime_trace import _should_measure_cuda


def test_realtime_trace_cuda_timing_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", raising=False)

    assert not _should_measure_cuda(None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_realtime_trace_cuda_timing_requires_explicit_truthy_env(
    monkeypatch, value, expected
):
    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", value)

    assert _should_measure_cuda(None) is expected


def test_realtime_trace_cuda_timing_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", "1")
    assert not _should_measure_cuda(False)

    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", "0")
    assert _should_measure_cuda(True)
