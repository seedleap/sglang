# SPDX-License-Identifier: Apache-2.0

import sys
import time
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.utils import realtime_trace
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        metrics=RequestMetrics("request-1"),
        request_id="request-1",
        block_idx=1,
        realtime_trace_id="trace-1",
        realtime_trace_started_at=time.perf_counter(),
        realtime_session_id="session-1",
    )


def test_realtime_trace_cuda_timing_defaults_to_wall_only(monkeypatch):
    monkeypatch.delenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", raising=False)
    enabled_values = []
    emitted = []
    monkeypatch.setattr(
        realtime_trace,
        "_new_cuda_events",
        lambda enabled: enabled_values.append(enabled) or (None, None),
    )
    monkeypatch.setattr(
        realtime_trace,
        "log_realtime_trace_for_batch",
        lambda _logger, _batch, event, **fields: emitted.append((event, fields)),
    )

    batch = _batch()
    with realtime_trace.realtime_trace_span(
        None,
        batch,
        "server.model_denoise_complete",
        component="minwm_denoising",
        chunk_index=1,
    ):
        pass

    assert enabled_values == [False]
    fields = emitted[0][1]
    assert fields["duration_ms"] >= 0
    assert fields["wall_timing_source"] == "perf_counter"
    assert fields["cuda_timing_status"] == "disabled"
    assert "cuda_ms" not in fields
    assert batch.metrics.realtime_component_timings == [
        {
            "event": "server.model_denoise_complete",
            "component": "minwm_denoising",
            "duration_ms": fields["duration_ms"],
            "wall_timing_source": "perf_counter",
            "cuda_timing_status": "disabled",
            "chunk_index": 1,
            "request_id": "request-1",
        }
    ]


def test_realtime_trace_cuda_timing_is_explicit_opt_in(monkeypatch):
    class FakeEvent:
        def __init__(self):
            self.record_count = 0
            self.synchronize_count = 0

        def record(self):
            self.record_count += 1

        def synchronize(self):
            self.synchronize_count += 1

        def elapsed_time(self, _other):
            return 12.5

    events = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def Event(*, enable_timing):
            assert enable_timing is True
            event = FakeEvent()
            events.append(event)
            return event

    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", "1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    emitted = []
    monkeypatch.setattr(
        realtime_trace,
        "log_realtime_trace_for_batch",
        lambda _logger, _batch, event, **fields: emitted.append((event, fields)),
    )

    with realtime_trace.realtime_trace_span(
        None,
        _batch(),
        "server.vae_decode_complete",
        component="vae_decoder",
        chunk_index=1,
    ):
        pass

    assert len(events) == 2
    assert events[0].record_count == 1
    assert events[1].record_count == 1
    assert events[1].synchronize_count == 1
    fields = emitted[0][1]
    assert fields["cuda_ms"] == 12.5
    assert fields["cuda_timing_status"] == "available"
    assert fields["wall_timing_source"] == "perf_counter"


def test_realtime_trace_cuda_timing_unsupported_falls_back_to_wall(monkeypatch):
    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", "true")
    monkeypatch.setattr(
        realtime_trace, "_new_cuda_events", lambda _enabled: (None, None)
    )
    emitted = []
    monkeypatch.setattr(
        realtime_trace,
        "log_realtime_trace_for_batch",
        lambda _logger, _batch, event, **fields: emitted.append((event, fields)),
    )

    with realtime_trace.realtime_trace_span(
        None,
        _batch(),
        "server.vae_decode_complete",
        component="vae_decoder",
        chunk_index=1,
    ):
        pass

    fields = emitted[0][1]
    assert fields["duration_ms"] >= 0
    assert fields["cuda_timing_status"] == "unavailable"
    assert "cuda_ms" not in fields


def test_realtime_trace_cuda_timing_unknown_env_value_fails_closed(monkeypatch):
    monkeypatch.setenv("SGLANG_REALTIME_TRACE_SYNC_CUDA", "unexpected")
    assert realtime_trace._should_measure_cuda(None) is False
    assert realtime_trace._should_measure_cuda(True) is True
