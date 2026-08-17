import pytest

from summarize import (
    build_report,
    latency_summary,
    render_markdown,
    run_meets_output_slo,
    summarize_runs,
)


def _run(concurrency, p95, fps, error_rate=0.0):
    return {
        "concurrency": concurrency,
        "action_to_first_frame_ms": {
            "p50": p95 * 0.8,
            "p95": p95,
            "p99": p95 * 1.1,
        },
        "chunk_total_ms": {"p95": p95},
        "aggregate_fps": fps,
        "min_session_fps": fps,
        "error_rate": error_rate,
    }


def test_report_selects_highest_concurrency_that_meets_slo():
    report = summarize_runs(
        [
            _run(1, p95=700, fps=18),
            _run(2, p95=920, fps=16.5),
            _run(4, p95=1400, fps=12),
        ]
    )
    assert report["max_supported_concurrency"] == 2


def test_report_rejects_high_aggregate_fps_when_one_session_is_slow():
    run = _run(4, p95=800, fps=64)
    run["min_session_fps"] = 12
    report = summarize_runs([run])
    assert report["max_supported_concurrency"] == 0


def test_rife_slo_uses_source_fps_instead_of_interpolated_output_fps():
    run = _run(2, p95=800, fps=29.4)
    run["aggregate_source_fps"] = 29.4
    run["min_session_source_fps"] = 14.7

    report = summarize_runs([run])

    assert report["max_supported_concurrency"] == 0


def test_output_ux_gate_uses_measured_wall_fps_and_monotone_ladder():
    slow_delivery = _run(2, p95=800, fps=23.9)
    slow_delivery["min_session_source_fps"] = 12.0
    slow_delivery["min_session_output_wall_fps"] = 23.9
    slow_delivery["media_profile_acceptance"] = [
        {
            "source_timeline_fps": 24,
            "output_timeline_fps": 72,
        }
    ]
    assert run_meets_output_slo(slow_delivery) is False

    sustained_delivery = _run(4, p95=1200, fps=24.1)
    sustained_delivery["min_session_source_fps"] = 8.1
    sustained_delivery["min_session_output_wall_fps"] = 24.1
    report = summarize_runs([slow_delivery, sustained_delivery])

    # Model/source and output UX gates are independent, but both capacity
    # claims must stop at the first failed level in the test ladder.
    assert report["max_supported_concurrency"] == 0
    assert report["max_supported_output_concurrency"] == 0
    assert report["min_output_wall_fps"] == 24.0


def test_capacity_ladder_fails_closed_when_declared_level_is_missing():
    one = _run(1, p95=700, fps=25)
    four = _run(4, p95=700, fps=25)

    report = summarize_runs(
        [one, four],
        expected_concurrencies=[1, 2, 4],
    )

    assert report["max_supported_concurrency"] == 1
    assert report["max_supported_output_concurrency"] == 1
    assert report["missing_concurrency_levels"] == [2]


def test_capacity_ladder_rejects_duplicate_trials_instead_of_cherry_picking():
    first = _run(2, p95=700, fps=25)
    second = _run(2, p95=700, fps=23)

    with pytest.raises(ValueError, match="duplicate concurrency result: 2"):
        summarize_runs([first, second])


def test_output_ux_gate_fails_closed_without_per_session_wall_evidence():
    aggregate_only = {
        "aggregate_output_wall_fps": 96.0,
        "output_timeline_fps": 72.0,
        "error_rate": 0.0,
    }

    assert run_meets_output_slo(aggregate_only) is False


def test_report_calculates_async_improvement_at_common_concurrency():
    report = build_report(
        {"runs": [_run(1, p95=800, fps=16)]},
        {"runs": [_run(1, p95=600, fps=18)]},
    )
    assert report["comparison"]["async_improvement_pct"] == 25.0
    assert report["comparison"]["by_concurrency"] == [
        {
            "concurrency": 1,
            "baseline_action_p95_ms": 800.0,
            "async_action_p95_ms": 600.0,
            "action_improvement_pct": 25.0,
            "baseline_chunk_p95_ms": 800.0,
            "async_chunk_p95_ms": 600.0,
            "chunk_improvement_pct": 25.0,
            "baseline_fps": 16.0,
            "async_fps": 18.0,
            "throughput_improvement_pct": 12.5,
        }
    ]


def test_markdown_contains_hardware_stages_and_per_level_improvement():
    baseline = {"runs": [_run(1, p95=800, fps=16)], "hardware": {}}
    asynchronous = {"runs": [_run(1, p95=600, fps=18)], "hardware": {}}
    asynchronous["runs"][0]["stage_ms"] = {
        "denoise_ms": {"p95": 300},
        "vae_decode_ms": {"p95": 30},
        "frame_encode_ms": {"p95": 120},
        "latent_send_ms": {"p95": 0.3},
        "vae_queue_wait_ms": {"p95": 0.1},
        "overlap_with_next_denoise_ms": {"p95": 35},
    }

    markdown = render_markdown(build_report(baseline, asynchronous))

    assert "## 硬件与部署" in markdown
    assert "## 异步收益" in markdown
    assert "远端 TAEHV decode：30.0 ms" in markdown
    assert "| 1 | 25.00% | 25.00% | 12.50% |" in markdown
    assert "output realtime factor = output/wall ÷ 72" in markdown
    assert "presented FPS rolling p5" in markdown


def test_latency_summary_uses_nearest_rank_percentiles():
    summary = latency_summary(range(1, 101))
    assert summary["p50"] == 50.5
    assert summary["p95"] == 95
    assert summary["p99"] == 99
