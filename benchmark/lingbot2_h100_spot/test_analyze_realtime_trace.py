#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from analyze_realtime_trace import parse_realtime_trace_lines, summarize_async_potential


def test_summarize_async_potential_uses_steady_chunks_and_transfer_budget():
    lines = [
        'prefix realtime_trace {"trace_id":"t1","event":"server.model_denoise_complete","chunk_index":0,"duration_ms":100}',
        'prefix realtime_trace {"trace_id":"t1","event":"server.vae_decode_complete","chunk_index":0,"duration_ms":50}',
        'prefix realtime_trace {"trace_id":"t1","event":"server.model_denoise_complete","chunk_index":1,"duration_ms":80}',
        'prefix realtime_trace {"trace_id":"t1","event":"server.vae_decode_complete","chunk_index":1,"duration_ms":40}',
        'prefix realtime_trace {"trace_id":"t1","event":"server.model_denoise_complete","chunk_index":2,"duration_ms":90}',
        'prefix realtime_trace {"trace_id":"t1","event":"server.vae_decode_complete","chunk_index":2,"duration_ms":60}',
    ]

    events = parse_realtime_trace_lines(lines)
    summary = summarize_async_potential(events, warmup_chunks=1, transfer_ms=10)

    assert summary["measured_chunks"] == 2
    assert summary["sync_compute_ms"]["mean"] == 135
    assert summary["async_critical_ms"]["mean"] == 85
    assert summary["saved_ms"]["mean"] == 50
    assert round(summary["speedup"]["mean"], 3) == 1.588


def test_summarize_async_potential_falls_back_to_denoising_stage_trace():
    lines = [
        'prefix realtime_trace {"trace_id":"t2","event":"server.pipeline_stage_complete","stage":"MinWMCausalDenoisingStage","chunk_index":1,"duration_ms":70}',
        'prefix realtime_trace {"trace_id":"t2","event":"server.vae_decode_complete","chunk_index":1,"duration_ms":30}',
    ]

    events = parse_realtime_trace_lines(lines)
    summary = summarize_async_potential(events, warmup_chunks=1)

    assert summary["measured_chunks"] == 1
    assert summary["denoise_ms"]["mean"] == 70
    assert summary["sync_compute_ms"]["mean"] == 100


if __name__ == "__main__":
    test_summarize_async_potential_uses_steady_chunks_and_transfer_budget()
    test_summarize_async_potential_falls_back_to_denoising_stage_trace()
    print("analyze_realtime_trace tests ok")
