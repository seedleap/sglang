from __future__ import annotations

import copy
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement import (  # noqa: E402
    MeasurementValidationError,
    available,
    build_measurement,
    coefficient_of_variation,
    stage_trace_values,
    validate_measurement,
)
from measurement_tool import aggregate  # noqa: E402
from nsys_metrics import merge_nsys_metrics  # noqa: E402


def _latency(value: float) -> dict:
    return available(
        {
            "count": 10,
            "mean": value,
            "p50": value,
            "p95": value,
            "p99": value,
            "max": value,
        },
        "ms_per_chunk",
        "test fixture",
    )


def _record(mode: str = "profiler_off", run_id: str = "run-1") -> dict:
    off_metrics = {
        "client_fps": available(16.0, "frames_per_second", "test fixture"),
        "scheduler_fps": available(16.1, "frames_per_second", "test fixture"),
        "scheduler_chunk_wall_ms": _latency(1000.0),
        "dit_wall_ms": _latency(600.0),
        "vae_wall_ms": _latency(250.0),
    }
    return build_measurement(
        mode=mode,
        run_id=run_id,
        profile_name="bf16-fast-sp2",
        timestamp_utc="2026-08-07T00:00:00+00:00",
        sglang_commit="a" * 40,
        minwm_commit=available("b" * 40, "git_commit", "fixture"),
        container_image=available("image@sha256:123", "image_reference", "fixture"),
        gpu_model=available("NVIDIA H200", "model_name", "fixture"),
        gpu_count=2,
        sp_degree=2,
        checkpoint_id="global_step_003200/ema_student/model.pt",
        checkpoint_step=3200,
        width=1248,
        height=704,
        warmup_chunks=20,
        measured_chunks=200 if mode == "profiler_off" else 10,
        precondition_warmup_chunks=0 if mode == "profiler_off" else 20,
        precision="bf16",
        fast_lane=True,
        comparison_contract={"case": "00_forward_pottery"},
        profiler_off_metrics=off_metrics,
        profiler_on_cuda_metrics={
            "dit_cuda_ms": _latency(590.0),
            "vae_cuda_ms": _latency(240.0),
        },
        artifacts={"client_result": "/results/client.json"},
    )


def test_profiler_off_schema_keeps_timing_domains_separate() -> None:
    record = _record()
    validate_measurement(record)
    assert record["measurement_contract"]["headline_eligible"] is True
    assert record["metrics"]["profiler_on"] is None
    assert record["workload"]["dmd_forwards_per_chunk"] == 4
    assert record["workload"]["clean_cache_forwards_per_chunk"] == 1
    assert set(record["measurement_contract"]["timing_domains"]) == {
        "client",
        "scheduler",
        "stage_wall",
        "cuda",
    }


def test_schema_rejects_missing_metric_and_implicit_unavailable_value() -> None:
    record = _record()
    del record["metrics"]["profiler_off"]["dit_wall_ms"]
    with pytest.raises(MeasurementValidationError, match="dit_wall_ms"):
        validate_measurement(record)

    record = _record()
    record["provenance"]["gpu"]["model"] = {
        "status": "unavailable",
        "reason": "permission_denied",
    }
    with pytest.raises(MeasurementValidationError, match="evidence"):
        validate_measurement(record)


def test_profiler_on_requires_ten_stable_chunks() -> None:
    record = _record("profiler_on")
    record["workload"]["measured_chunks"] = 9
    with pytest.raises(MeasurementValidationError, match="at least 10"):
        validate_measurement(record)


def test_stage_trace_values_selects_source_and_chunk_window() -> None:
    events = [
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 20,
            "duration_ms": 601,
            "source": "scheduler_result_metrics",
        },
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 20,
            "duration_ms": 590,
            "cuda_ms": 589,
            "component": "minwm_denoising",
        },
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 19,
            "duration_ms": 999,
            "source": "scheduler_result_metrics",
        },
    ]
    assert stage_trace_values(
        events,
        event="server.model_denoise_complete",
        field="duration_ms",
        measured_indices={20},
        source="scheduler_result_metrics",
    ) == [601.0]
    assert stage_trace_values(
        events,
        event="server.model_denoise_complete",
        field="cuda_ms",
        measured_indices={20},
        component="minwm_denoising",
    ) == [589.0]


def _create_nsys_fixture(path: Path, include_gpu_metrics: bool) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER, start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, start INTEGER, end INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES (2, 'cudaMemcpyAsync_v3020');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1, 0, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2, 2, 3);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 0, 5000);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000, 30000);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 25000, 100000);
        """
    )
    if include_gpu_metrics:
        connection.executescript(
            """
            CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS (typeId INTEGER, metricId INTEGER, value REAL);
            INSERT INTO TARGET_INFO_GPU_METRICS VALUES (3, 'SM Active');
            INSERT INTO TARGET_INFO_GPU_METRICS VALUES (5, 'Tensor Active');
            INSERT INTO TARGET_INFO_GPU_METRICS VALUES (18, 'DRAM Throughput');
            INSERT INTO GPU_METRICS VALUES (0, 3, 80.0);
            INSERT INTO GPU_METRICS VALUES (0, 3, 82.0);
            INSERT INTO GPU_METRICS VALUES (0, 5, 55.0);
            INSERT INTO GPU_METRICS VALUES (0, 18, 40.0);
            """
        )
    connection.commit()
    connection.close()


def test_nsys_merge_extracts_counts_buckets_busy_and_gpu_metrics(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(sqlite_path, include_gpu_metrics=True)
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    on = record["metrics"]["profiler_on"]
    assert on["kernel_count"]["value"] == 3
    assert on["cuda_api_count"]["value"] == 2
    assert on["kernel_launch_api_count"]["value"] == 1
    assert on["short_kernel_buckets"]["value"] == {
        "lt_10_us": 1,
        "10_to_lt_50_us": 1,
        "50_to_lt_100_us": 1,
        "gte_100_us": 0,
    }
    assert on["gpu_kernel_busy"]["value"]["mean_pct"] == pytest.approx(95.0)
    assert on["gpu_metrics"]["sm_active"]["value"]["mean"] == 81.0
    assert on["gpu_metrics"]["tensor_active"]["status"] == "available"
    assert on["gpu_metrics"]["dram"]["status"] == "available"


def test_nsys_gpu_permission_degradation_is_explicit(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(sqlite_path, include_gpu_metrics=False)
    record = merge_nsys_metrics(
        _record("profiler_on"),
        sqlite_path,
        "GPU Metrics: permission denied while opening perf_event",
    )
    for metric in record["metrics"]["profiler_on"]["gpu_metrics"].values():
        assert metric["status"] == "unavailable"
        assert metric["reason"] == "permission_denied"
        assert "permission denied" in metric["evidence"].lower()


def test_repeat_summary_uses_sample_cv_and_flags_variance() -> None:
    first = _record(run_id="run-1")
    second = copy.deepcopy(first)
    second["run_id"] = "run-2"
    second["metrics"]["profiler_off"]["client_fps"]["value"] = 17.0
    summary = aggregate([first, second], explanation="shared-node power noise")
    assert summary["metrics"]["client_fps"]["cv"] == pytest.approx(
        coefficient_of_variation([16.0, 17.0])
    )
    assert summary["metrics"]["client_fps"]["passes"] is False
    assert "scheduler_chunk_wall_ms" not in summary["acceptance"]["required_metrics"]
    assert summary["acceptance"]["environment_noise_explanation"]
