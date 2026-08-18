import asyncio
from argparse import Namespace

import msgspec.msgpack
import pytest
from load_test import (
    aggregate_measurement_seconds,
    chunk_stats_from_trace,
    collect_trace_events,
    completion_chunk,
    derive_trace_http_url,
    final_frame_batch_chunk,
    frame_batch_contract_metadata,
    init_request,
    measurement_window_start,
    merged_stage_values,
    record_action_latency,
    record_frame_batch,
    run_session,
    server_action_latencies,
    stage_values,
    stage_values_from_chunk_messages,
    trace_contract_summary,
    validate_media_profile_contract,
)

RIFE_SHA256 = "8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692"


def _single_chunk_args(ws_url: str) -> Namespace:
    return Namespace(
        ws_url=ws_url,
        warmup_chunks=0,
        measured_chunks=1,
        realtime_media_profile="native_v1",
        model="/work/model",
        prompt="test",
        generation_mode="t2v",
        size="832x480",
        fps=24,
        sink=9,
        window=18,
        timeout_s=2.0,
        completion_signal="final-frame",
        skip_trace_query=True,
        trace_http_url=None,
        trace_timeout_s=1.0,
        expected_media_weights_sha256=None,
    )


@pytest.mark.parametrize(
    ("code", "reason", "should_pass"),
    [
        (1000, "generation complete", True),
        (1000, "normal", True),
        (1000, "silent truncation", False),
        (1011, "backend failed after final frame", False),
    ],
)
def test_run_session_requires_a_normal_finite_websocket_close(
    code, reason, should_pass
):
    async def exercise():
        import websockets

        async def handler(websocket):
            await websocket.recv()
            await websocket.send(
                msgspec.msgpack.encode(
                    {
                        "type": "frame_batch",
                        "chunk_index": 0,
                        "num_frames": 1,
                        "is_final_frame_batch": True,
                        "source_width": 832,
                        "source_height": 480,
                        "preview_width": 560,
                        "preview_height": 323,
                        "width": 560,
                        "height": 323,
                    }
                )
            )
            await websocket.close(code=code, reason=reason)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await run_session(
                _single_chunk_args(f"ws://127.0.0.1:{port}"), 1, 0
            )

    if should_pass:
        result = asyncio.run(exercise())
        assert result["websocket_close"] == {"code": code, "reason": reason}
    else:
        with pytest.raises(RuntimeError, match="did not close normally"):
            asyncio.run(exercise())


def test_init_request_keeps_t2v_frame_count_aligned_with_chunk_count():
    args = Namespace(
        model="/work/model",
        prompt="test prompt",
        size="832x480",
        fps=24,
    )

    request = init_request(args, total_chunks=5, trace_id="trace-1")

    assert request["num_frames"] == 65
    assert request["max_chunks"] == 5


def test_init_request_supports_i2v_reference_bytes():
    args = Namespace(
        model="lingbot-world-v2-14b-causal-fast-diffusers",
        prompt="test prompt",
        size="1280x704",
        fps=24,
        generation_mode="i2v",
        first_frame_bytes=b"reference-image",
        sink=9,
        window=18,
    )

    request = init_request(args, total_chunks=3, trace_id="trace-i2v")

    assert request["generation_mode"] == "i2v"
    assert request["first_frame"] == b"reference-image"
    assert request["max_chunks"] == 3
    assert request["realtime_causal_sink_size"] == 9
    assert request["realtime_causal_kv_cache_num_frames"] == 18
    assert "num_frames" not in request


def test_init_request_only_adds_an_explicit_non_native_media_profile():
    native = Namespace(
        model="/work/model",
        prompt="test prompt",
        size="1280x704",
        fps=24,
        realtime_media_profile="native_v1",
    )
    rife_values = vars(native).copy()
    rife_values["realtime_media_profile"] = "rife2x_v1"
    rife = Namespace(**rife_values)
    rife3_values = vars(native).copy()
    rife3_values["realtime_media_profile"] = "rife3x_v1"
    rife3 = Namespace(**rife3_values)

    assert "realtime_media_profile" not in init_request(
        native, total_chunks=2, trace_id="native"
    )
    assert (
        init_request(rife, total_chunks=2, trace_id="rife")["realtime_media_profile"]
        == "rife2x_v1"
    )
    assert (
        init_request(rife3, total_chunks=2, trace_id="rife3")["realtime_media_profile"]
        == "rife3x_v1"
    )


def test_rife_contract_requires_exact_wire_counts_and_timeline():
    result = validate_media_profile_contract(
        media_profile="rife2x_v1",
        requested_fps=24,
        expected_media_weights_sha256=RIFE_SHA256,
        session_ready={
            "requested_media_profile": "rife2x_v1",
            "effective_media_profile": "rife2x_v1",
            "source_timeline_fps": 24,
            "output_timeline_fps": 48,
            "media_weights_sha256": RIFE_SHA256,
        },
        completions={
            0: {
                "media_profile": "rife2x_v1",
                "source_num_frames": 4,
                "output_num_frames": 7,
                "num_frames": 7,
                "source_timeline_fps": 24,
                "output_timeline_fps": 48,
            },
            1: {
                "media_profile": "rife2x_v1",
                "source_num_frames": 4,
                "output_num_frames": 8,
                "num_frames": 8,
                "source_timeline_fps": 24,
                "output_timeline_fps": 48,
            },
        },
        frame_counts={0: 7, 1: 8},
        frame_messages={
            0: [
                {
                    "media_profile": "rife2x_v1",
                    "source_timeline_fps": 24,
                    "output_timeline_fps": 48,
                }
            ],
            1: [
                {
                    "media_profile": "rife2x_v1",
                    "source_timeline_fps": 24,
                    "output_timeline_fps": 48,
                }
            ],
        },
        expected_chunks={0, 1},
    )

    assert result["source_frames"] == 8
    assert result["output_frames"] == 15
    assert result["acceptance"]["output_timeline_fps"] == 48


def test_rife3x_contract_keeps_timeline_72_separate_from_exact_wire_counts():
    result = validate_media_profile_contract(
        media_profile="rife3x_v1",
        requested_fps=24,
        expected_media_weights_sha256=RIFE_SHA256,
        session_ready={
            "requested_media_profile": "rife3x_v1",
            "effective_media_profile": "rife3x_v1",
            "source_timeline_fps": 24,
            "output_timeline_fps": 72,
            "media_weights_sha256": RIFE_SHA256,
        },
        completions={
            0: {
                "media_profile": "rife3x_v1",
                "source_num_frames": 0,
                "output_num_frames": 0,
                "num_frames": 0,
                "source_timeline_fps": 24,
                "output_timeline_fps": 72,
            },
            1: {
                "media_profile": "rife3x_v1",
                "source_num_frames": 4,
                "output_num_frames": 10,
                "num_frames": 10,
                "source_timeline_fps": 24,
                "output_timeline_fps": 72,
            },
            2: {
                "media_profile": "rife3x_v1",
                "source_num_frames": 4,
                "output_num_frames": 12,
                "num_frames": 12,
                "source_timeline_fps": 24,
                "output_timeline_fps": 72,
            },
        },
        frame_counts={1: 10, 2: 12},
        frame_messages={
            1: [
                {
                    "media_profile": "rife3x_v1",
                    "source_timeline_fps": 24,
                    "output_timeline_fps": 72,
                }
            ],
            2: [
                {
                    "media_profile": "rife3x_v1",
                    "source_timeline_fps": 24,
                    "output_timeline_fps": 72,
                }
            ],
        },
        expected_chunks={0, 1, 2},
    )

    assert result["source_frames"] == 8
    assert result["output_frames"] == 22
    assert result["acceptance"]["output_timeline_fps"] == 72


def test_rife_contract_rejects_silent_frame_loss():
    try:
        validate_media_profile_contract(
            media_profile="rife2x_v1",
            requested_fps=24,
            expected_media_weights_sha256=RIFE_SHA256,
            session_ready={
                "requested_media_profile": "rife2x_v1",
                "effective_media_profile": "rife2x_v1",
                "source_timeline_fps": 24,
                "output_timeline_fps": 48,
                "media_weights_sha256": RIFE_SHA256,
            },
            completions={
                0: {
                    "media_profile": "rife2x_v1",
                    "source_num_frames": 4,
                    "output_num_frames": 7,
                    "num_frames": 7,
                    "source_timeline_fps": 24,
                    "output_timeline_fps": 48,
                }
            },
            frame_counts={0: 6},
            frame_messages={
                0: [
                    {
                        "media_profile": "rife2x_v1",
                        "source_timeline_fps": 24,
                        "output_timeline_fps": 48,
                    }
                ]
            },
            expected_chunks={0},
        )
    except RuntimeError as exc:
        assert "frame count mismatch" in str(exc)
    else:
        raise AssertionError("RIFE frame loss must fail the load-test contract")


@pytest.mark.parametrize(
    ("receipt_updates", "error"),
    [
        ({"media_weights_sha256": "bogus"}, "invalid weights digest"),
        ({"media_weights_sha256": "0" * 64}, "weights digest mismatch"),
        ({"source_timeline_fps": float("nan")}, "non-finite timeline"),
        ({"output_timeline_fps": float("nan")}, "non-finite timeline"),
    ],
)
def test_rife_contract_rejects_untrusted_acceptance(receipt_updates, error):
    receipt = {
        "requested_media_profile": "rife2x_v1",
        "effective_media_profile": "rife2x_v1",
        "source_timeline_fps": 24,
        "output_timeline_fps": 48,
        "media_weights_sha256": RIFE_SHA256,
        **receipt_updates,
    }

    with pytest.raises(RuntimeError, match=error):
        validate_media_profile_contract(
            media_profile="rife2x_v1",
            requested_fps=24,
            expected_media_weights_sha256=RIFE_SHA256,
            session_ready=receipt,
            completions={},
            frame_counts={},
            frame_messages={},
            expected_chunks=set(),
        )


@pytest.mark.parametrize(
    ("completion_updates", "error"),
    [
        ({"num_frames": 999}, "frame count mismatch"),
        ({"source_timeline_fps": 12}, "completion timeline drifted"),
        ({"output_timeline_fps": 24}, "completion timeline drifted"),
    ],
)
def test_rife_contract_rejects_conflicting_completion_metadata(
    completion_updates, error
):
    completion = {
        "media_profile": "rife2x_v1",
        "source_num_frames": 4,
        "output_num_frames": 7,
        "num_frames": 7,
        "source_timeline_fps": 24,
        "output_timeline_fps": 48,
        **completion_updates,
    }

    with pytest.raises(RuntimeError, match=error):
        validate_media_profile_contract(
            media_profile="rife2x_v1",
            requested_fps=24,
            expected_media_weights_sha256=RIFE_SHA256,
            session_ready={
                "requested_media_profile": "rife2x_v1",
                "effective_media_profile": "rife2x_v1",
                "source_timeline_fps": 24,
                "output_timeline_fps": 48,
                "media_weights_sha256": RIFE_SHA256,
            },
            completions={0: completion},
            frame_counts={0: 7},
            frame_messages={
                0: [
                    {
                        "media_profile": "rife2x_v1",
                        "source_timeline_fps": 24,
                        "output_timeline_fps": 48,
                    }
                ]
            },
            expected_chunks={0},
        )


def test_record_frame_batch_counts_all_batches_in_the_same_chunk():
    frame_counts = {}

    record_frame_batch({"chunk_index": 3, "num_frames": 8}, frame_counts=frame_counts)
    record_frame_batch({"chunk_index": 3, "num_frames": 8}, frame_counts=frame_counts)

    assert frame_counts == {3: 16}


def test_frame_batch_contract_metadata_drops_encoded_payload():
    assert frame_batch_contract_metadata(
        {
            "media_profile": "rife2x_v1",
            "source_timeline_fps": 24,
            "output_timeline_fps": 48,
            "payload": b"encoded-frame" * 100,
            "unrelated": "value",
        }
    ) == {
        "media_profile": "rife2x_v1",
        "source_timeline_fps": 24,
        "output_timeline_fps": 48,
    }


def test_final_frame_batch_is_the_media_websocket_completion_signal():
    assert (
        final_frame_batch_chunk(
            {
                "type": "frame_batch",
                "chunk_index": 3,
                "frame_batch_index": 15,
                "is_final_frame_batch": True,
            }
        )
        == 3
    )


def test_monolithic_ordered_output_can_complete_on_chunk_stats_after_frames():
    stats = {"type": "chunk_stats", "chunk_index": 4}

    assert (
        completion_chunk(
            stats,
            completion_signal="chunk-stats",
            frame_counts={4: 9},
        )
        == 4
    )
    assert (
        completion_chunk(
            stats,
            completion_signal="chunk-stats",
            frame_counts={},
        )
        is None
    )
    assert (
        completion_chunk(
            stats,
            completion_signal="final-frame",
            frame_counts={4: 9},
        )
        is None
    )
    assert (
        final_frame_batch_chunk(
            {
                "type": "frame_batch",
                "chunk_index": 3,
                "frame_batch_index": 14,
                "is_final_frame_batch": False,
            }
        )
        is None
    )
    assert (
        final_frame_batch_chunk(
            {
                "type": "media_chunk_complete",
                "chunk_index": 3,
                "num_frames": 16,
            }
        )
        == 3
    )


def test_chunk_stats_are_read_from_the_separate_trace_transport():
    events = [
        {
            "event": "server.chunk_complete",
            "chunk_index": 0,
            "chunk_total_ms": 712.5,
            "scheduler_forward_ms": 630.0,
            "trace_seq": 10,
        },
        {
            "event": "server.vae_decode_complete",
            "chunk_index": 0,
            "duration_ms": 52.0,
            "trace_seq": 11,
        },
        {
            "event": "server.chunk_complete",
            "chunk_index": 1,
            "chunk_total_ms": 481.25,
            "scheduler_forward_ms": 401.0,
            "trace_seq": 20,
        },
    ]

    assert chunk_stats_from_trace(events) == {
        0: {
            "event": "server.chunk_complete",
            "chunk_index": 0,
            "chunk_total_ms": 712.5,
            "scheduler_forward_ms": 630.0,
            "trace_seq": 10,
        },
        1: {
            "event": "server.chunk_complete",
            "chunk_index": 1,
            "chunk_total_ms": 481.25,
            "scheduler_forward_ms": 401.0,
            "trace_seq": 20,
        },
    }


def test_stage_values_excludes_warmup_and_records_local_vae():
    events = [
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 0,
            "cuda_ms": 900.0,
        },
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 2,
            "cuda_ms": 310.0,
        },
        {
            "event": "server.vae_decode_complete",
            "chunk_index": 2,
            "cuda_ms": 16.0,
        },
    ]

    assert stage_values(events, min_chunk_index=2) == {
        "denoise_ms": [310.0],
        "vae_decode_ms": [16.0],
    }


def test_stage_values_records_remote_rife_costs_separately():
    result = stage_values(
        [
            {
                "event": "server.remote_vae_complete",
                "chunk_index": 3,
                "vae_decode_ms": 12.0,
                "vae_post_decode_ms": 3.0,
                "actor_wait_ms": 4.0,
                "rife_interpolation_ms": 21.0,
            }
        ]
    )

    assert result == {
        "vae_decode_ms": [12.0],
        "vae_post_decode_ms": [3.0],
        "actor_wait_ms": [4.0],
        "rife_interpolation_ms": [21.0],
    }


def test_chunk_telemetry_provides_rife_stages_without_trace_api():
    result = stage_values_from_chunk_messages(
        {
            0: {"chunk_total_ms": 900, "rife_interpolation_ms": 18},
            1: {
                "chunk_total_ms": 1100,
                "actor_wait_ms": 3,
                "rife_interpolation_ms": 20,
                "source_realtime_factor": 0.62,
                "output_realtime_factor": 0.62,
            },
        },
        min_chunk_index=1,
    )

    assert result == {
        "chunk_total_ms": [1100.0],
        "actor_wait_ms": [3.0],
        "rife_interpolation_ms": [20.0],
        "source_realtime_factor": [0.62],
        "output_realtime_factor": [0.62],
    }


def test_trace_stages_override_shared_fields_without_dropping_realtime_factor():
    result = merged_stage_values(
        [
            {
                "event": "server.remote_vae_complete",
                "chunk_index": 2,
                "vae_decode_ms": 11,
            }
        ],
        {
            2: {
                "type": "chunk_telemetry",
                "vae_decode_ms": 99,
                "source_realtime_factor": 0.62,
                "output_realtime_factor": 0.62,
            }
        },
        min_chunk_index=2,
    )

    assert result == {
        "vae_decode_ms": [11.0],
        "source_realtime_factor": [0.62],
        "output_realtime_factor": [0.62],
    }


def test_partial_trace_column_does_not_replace_complete_chunk_telemetry():
    result = merged_stage_values(
        [
            {
                "event": "server.remote_vae_complete",
                "chunk_index": 2,
                "vae_decode_ms": 11,
            }
        ],
        {
            2: {"vae_decode_ms": 21},
            3: {"vae_decode_ms": 22},
        },
        min_chunk_index=2,
    )

    assert result == {"vae_decode_ms": [21.0, 22.0]}


def test_trace_http_url_is_derived_from_the_public_websocket_origin():
    assert (
        derive_trace_http_url(
            "wss://realtime.example.com/v1/realtime_video/generate?mode=t2v"
        )
        == "https://realtime.example.com"
    )
    assert (
        derive_trace_http_url("ws://127.0.0.1:18080/v1/realtime_video/generate")
        == "http://127.0.0.1:18080"
    )


def test_collect_trace_events_polls_full_snapshots_and_deduplicates():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, *, params):
            self.calls.append((url, dict(params)))
            if len(self.calls) == 1:
                return Response(
                    {
                        "events": [
                            {"event": "gateway.ws_accepted", "trace_seq": 1},
                            {"event": "server.chunk_complete", "trace_seq": 2},
                        ],
                        "next_cursor": 2,
                    }
                )
            return Response(
                {
                    "events": [
                        {"event": "server.chunk_complete", "trace_seq": 2},
                        {"event": "server.vae_decode_complete", "trace_seq": 3},
                    ],
                    "next_cursor": 3,
                }
            )

    async def run():
        client = Client()
        events = await collect_trace_events(
            "http://gateway",
            "trace-a",
            client=client,
            timeout_s=0.1,
            poll_interval_s=0,
            stable_polls=1,
        )
        assert [event["trace_seq"] for event in events] == [1, 2, 3]
        assert client.calls[0][1]["after"] == 0
        assert client.calls[1][1]["after"] == 0

    asyncio.run(run())


def test_collect_trace_events_waits_for_all_chunks_and_terminal_event():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    snapshots = [
        {
            "events": [
                {"event": "gateway.ws_accepted", "trace_seq": 10},
                {
                    "event": "server.chunk_complete",
                    "chunk_index": 0,
                    "trace_seq": 20,
                },
            ],
            "next_cursor": 20,
        },
        {
            "events": [
                {"event": "gateway.ws_accepted", "trace_seq": 10},
                {
                    "event": "server.chunk_complete",
                    "chunk_index": 0,
                    "trace_seq": 20,
                },
            ],
            "next_cursor": 20,
        },
        {
            "events": [
                {
                    "event": "server.vae_decode_complete",
                    "chunk_index": 1,
                    "trace_seq": 15,
                },
                {"event": "gateway.ws_accepted", "trace_seq": 10},
                {
                    "event": "server.chunk_complete",
                    "chunk_index": 0,
                    "trace_seq": 20,
                },
                {
                    "event": "server.chunk_complete",
                    "chunk_index": 1,
                    "trace_seq": 30,
                },
                {"event": "gateway.session_closed", "trace_seq": 40},
            ],
            "next_cursor": 40,
        },
    ]

    class Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, *, params):
            self.calls.append((url, dict(params)))
            index = min(len(self.calls) - 1, len(snapshots) - 1)
            return Response(snapshots[index])

    async def run():
        client = Client()
        events = await collect_trace_events(
            "http://gateway",
            "trace-eventual",
            client=client,
            timeout_s=0.1,
            poll_interval_s=0,
            stable_polls=1,
            expected_chunks=2,
        )

        assert [event["trace_seq"] for event in events] == [10, 15, 20, 30, 40]
        assert len(client.calls) == 4
        assert all(call[1]["after"] == 0 for call in client.calls)

    asyncio.run(run())


def test_collect_trace_events_retries_a_transient_transport_timeout():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "events": [{"event": "server.chunk_complete", "trace_seq": 1}],
                "next_cursor": 1,
            }

    class Client:
        def __init__(self):
            self.calls = 0

        async def get(self, _url, *, params):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient trace read timeout")
            return Response()

    async def run():
        client = Client()
        events = await collect_trace_events(
            "http://gateway",
            "trace-retry",
            client=client,
            timeout_s=0.1,
            poll_interval_s=0,
            stable_polls=1,
        )

        assert [event["trace_seq"] for event in events] == [1]
        assert client.calls == 3

    asyncio.run(run())


def test_stage_values_backfills_overlap_after_next_denoise_completes():
    events = [
        {
            "event": "server.remote_vae_complete",
            "chunk_index": 3,
            "overlap_with_next_denoise_ms": 0,
            "overlap_ratio": 0,
        },
        {
            "event": "server.vae_denoise_overlap_complete",
            "chunk_index": 3,
            "next_chunk_index": 4,
            "overlap_with_next_denoise_ms": 71.5,
            "overlap_ratio": 0.82,
        },
    ]

    result = stage_values(events)

    assert result["overlap_with_next_denoise_ms"] == [71.5]
    assert result["overlap_ratio"] == [0.82]


def test_action_latency_uses_chunk_stats_sampled_event_id():
    first_frame_at = {3: 12.5}
    action_sent_at = {1: 11.8, 2: 12.0}
    action_latencies = []

    record_action_latency(
        {"chunk_index": 3, "event_id": 2},
        first_frame_at=first_frame_at,
        action_sent_at=action_sent_at,
        action_latencies=action_latencies,
        min_chunk_index=2,
    )

    assert action_latencies == [500.0]
    assert action_sent_at == {}


def test_action_latency_discards_warmup_samples_without_recording_them():
    action_sent_at = {1: 1.0}
    action_latencies = []

    record_action_latency(
        {"chunk_index": 1, "event_id": 1},
        first_frame_at={1: 1.2},
        action_sent_at=action_sent_at,
        action_latencies=action_latencies,
        min_chunk_index=2,
    )

    assert action_latencies == []
    assert action_sent_at == {}


def test_aggregate_measurement_seconds_uses_real_overlapping_wall_window():
    sessions = [
        {"measured_started_at": 10.0, "measured_completed_at": 12.0},
        {"measured_started_at": 10.5, "measured_completed_at": 13.0},
    ]

    assert aggregate_measurement_seconds(sessions) == 3.0


def test_measurement_window_starts_after_the_last_warmup_chunk():
    assert (
        measurement_window_start(
            chunk_index=1,
            observed_at=10.0,
            warmup_chunks=2,
            current=None,
        )
        == 10.0
    )
    assert (
        measurement_window_start(
            chunk_index=2,
            observed_at=10.5,
            warmup_chunks=2,
            current=10.0,
        )
        == 10.0
    )
    assert (
        measurement_window_start(
            chunk_index=0,
            observed_at=9.0,
            warmup_chunks=2,
            current=None,
        )
        is None
    )


def test_server_action_latencies_use_sampled_event_and_first_frame_marker():
    events = [
        {
            "event": "server.event_received",
            "event_id": 7,
            "client_sent_epoch_ms": 10_000.0,
            "server_epoch_ms": 10_020.0,
            "server_elapsed_ms": 20.0,
        },
        {
            "event": "server.remote_first_frame_received",
            "chunk_index": 3,
            "event_id": 7,
            "server_epoch_ms": 10_540.0,
            "server_elapsed_ms": 540.0,
        },
    ]

    assert server_action_latencies(events, min_chunk_index=2) == {
        "action_to_server_first_frame_ms": [540.0],
        "action_ingress_to_server_first_frame_ms": [520.0],
    }


def test_server_action_latencies_support_sync_marker_and_latest_prior_event():
    events = [
        {
            "event": "server.event_received",
            "event_id": 3,
            "client_sent_epoch_ms": 20_000.0,
            "server_epoch_ms": 20_010.0,
            "server_elapsed_ms": 10.0,
        },
        {
            "event": "server.output_send_start",
            "chunk_index": 4,
            "event_id": 4,
            "server_epoch_ms": 20_430.0,
            "server_elapsed_ms": 430.0,
        },
    ]

    assert server_action_latencies(events) == {
        "action_to_server_first_frame_ms": [430.0],
        "action_ingress_to_server_first_frame_ms": [420.0],
    }


def test_trace_contract_summary_proves_direct_vae_media_route():
    summary = trace_contract_summary(
        [
            {"event": "gateway.ws_accepted"},
            {"event": "coordinator.admit_complete"},
            {"event": "server.model_denoise_complete"},
            {"event": "server.vae_decode_complete"},
            {
                "event": "server.vae_frame_batch_sent",
                "output_direct": True,
            },
        ]
    )

    assert summary["event_names"] == [
        "coordinator.admit_complete",
        "gateway.ws_accepted",
        "server.model_denoise_complete",
        "server.vae_decode_complete",
        "server.vae_frame_batch_sent",
    ]
    assert summary["direct_vae_frame_batches"] == 1
