from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_realtime_throughput import (  # noqa: E402
    incomplete_measurement_diagnostic,
    missing_required_stage_trace,
    record_required_stage_trace,
    required_stage_trace_chunks,
    required_stage_trace_is_complete,
)
from measurement import (  # noqa: E402
    MeasurementValidationError,
    available,
    build_measurement,
    coefficient_of_variation,
    stage_trace_values,
    unavailable,
    validate_measurement,
)
from measurement_tool import (  # noqa: E402
    _is_invalid_result,
    aggregate,
    build_invalid_marker,
    load_aggregate_records,
    require_complete_stable_nsys,
)
from nsys_metrics import _is_sm_active_metric, merge_nsys_metrics  # noqa: E402


def _latency(value: float, count: int) -> dict:
    return available(
        {
            "count": count,
            "mean": value,
            "p50": value,
            "p95": value,
            "p99": value,
            "max": value,
        },
        "ms_per_chunk",
        "test fixture",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SM Active", True),
        ("SMs Active [Throughput %]", True),
        ("Tensor Active [Throughput %]", False),
        ("SM Issue [Throughput %]", False),
        ("SM Clock Frequency [MHz]", False),
        ("Unallocated Warps in Active SMs [Throughput %]", False),
        ("Pixel Warps in Flight [Throughput %]", False),
    ],
)
def test_sm_active_metric_aliases_are_controlled(name: str, expected: bool) -> None:
    assert _is_sm_active_metric(name) is expected


def _record(
    mode: str = "profiler_off", run_id: str = "run-1", gpu_count: int = 2
) -> dict:
    measured_chunks = 200 if mode == "profiler_off" else 10
    off_metrics = {
        "client_fps": available(16.0, "frames_per_second", "test fixture"),
        "scheduler_fps": available(16.1, "frames_per_second", "test fixture"),
        "scheduler_chunk_wall_ms": _latency(1000.0, measured_chunks),
        "dit_wall_ms": _latency(600.0, measured_chunks),
        "vae_wall_ms": _latency(250.0, measured_chunks),
    }
    return build_measurement(
        mode=mode,
        run_id=run_id,
        profile_name=f"bf16-fast-sp{gpu_count}",
        timestamp_utc="2026-08-07T00:00:00+00:00",
        sglang_commit="a" * 40,
        minwm_commit=available("b" * 40, "git_commit", "fixture"),
        container_image=available("image@sha256:123", "image_reference", "fixture"),
        gpu_model=available("NVIDIA H200", "model_name", "fixture"),
        gpu_count=gpu_count,
        allocated_gpu_count=8,
        sp_degree=gpu_count,
        checkpoint_id="global_step_003200/ema_student/model.pt",
        checkpoint_step=3200,
        width=1248,
        height=704,
        warmup_chunks=20 if mode == "profiler_off" else 1,
        measured_chunks=measured_chunks,
        precondition_warmup_chunks=0 if mode == "profiler_off" else 20,
        precision="bf16",
        fast_lane=True,
        comparison_contract={"case": "00_forward_pottery"},
        profiler_off_metrics=off_metrics,
        profiler_on_cuda_metrics={
            "dit_cuda_ms": _latency(590.0, measured_chunks),
            "vae_cuda_ms": _latency(240.0, measured_chunks),
        },
        artifacts={"client_result": "/results/client.json"},
    )


def _machine_schema_validator() -> Draft202012Validator:
    schema_path = Path(__file__).with_name("measurement_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_machine_schema_accepts_profiler_off_and_pending_profiler_on() -> None:
    validator = _machine_schema_validator()
    validator.validate(_record("profiler_off"))
    validator.validate(_record("profiler_on"))


def test_profiler_off_schema_keeps_timing_domains_separate() -> None:
    record = _record()
    validate_measurement(record)
    assert record["measurement_contract"]["headline_eligible"] is True
    assert record["metrics"]["profiler_on"] is None
    assert record["workload"]["dmd_forwards_per_chunk"] == 4
    assert record["workload"]["clean_cache_forwards_per_chunk"] == 1
    assert record["provenance"]["gpu"]["count"] == 2
    assert record["provenance"]["gpu"]["allocated_count"] == 8
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


@pytest.mark.parametrize("mode", ["profiler_off", "profiler_on"])
def test_latency_metric_requires_count_equal_to_measured_chunks(mode: str) -> None:
    record = _record(mode)
    if mode == "profiler_off":
        metric = record["metrics"]["profiler_off"]["dit_wall_ms"]
    else:
        metric = record["metrics"]["profiler_on"]["dit_cuda_ms"]

    del metric["value"]["count"]
    with pytest.raises(MeasurementValidationError, match=r"value\.count"):
        validate_measurement(record)
    assert list(_machine_schema_validator().iter_errors(record))

    record = _record(mode)
    if mode == "profiler_off":
        metric = record["metrics"]["profiler_off"]["dit_wall_ms"]
    else:
        metric = record["metrics"]["profiler_on"]["dit_cuda_ms"]
    metric["value"]["count"] = record["workload"]["measured_chunks"] - 1
    with pytest.raises(
        MeasurementValidationError, match=r"must equal workload\.measured_chunks"
    ):
        validate_measurement(record)


def test_profiler_on_requires_ten_stable_chunks() -> None:
    record = _record("profiler_on")
    record["workload"]["measured_chunks"] = 9
    with pytest.raises(MeasurementValidationError, match="at least 10"):
        validate_measurement(record)


def test_schema_rejects_allocation_smaller_than_active_gpu_count() -> None:
    record = _record()
    record["provenance"]["gpu"]["allocated_count"] = 1
    with pytest.raises(MeasurementValidationError, match="allocated_count"):
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
            "source": "scheduler_result_component_timing",
        },
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 20,
            "cuda_ms": 999,
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
        source="scheduler_result_component_timing",
        component="minwm_denoising",
    ) == [589.0]


def test_required_stage_trace_waits_for_wall_and_profiler_on_cuda() -> None:
    required = required_stage_trace_chunks("profiler_on")
    assert len(required) == 4
    traces = [
        {
            "event": "server.model_denoise_complete",
            "source": "scheduler_result_metrics",
            "chunk_index": 9,
        },
        {
            "event": "server.model_denoise_complete",
            "component": "minwm_denoising",
            "chunk_index": 9,
        },
        {
            "event": "server.vae_decode_complete",
            "source": "scheduler_result_metrics",
            "chunk_index": 9,
        },
        {
            "event": "server.vae_decode_complete",
            "component": "vae_decoder",
            "chunk_index": 9,
        },
    ]
    for trace in traces:
        record_required_stage_trace(required, trace)
    assert all(indices == {9} for indices in required.values())

    profiler_off = required_stage_trace_chunks("profiler_off")
    assert len(profiler_off) == 2


def test_required_stage_trace_rejects_equal_length_with_out_of_range_index() -> None:
    required = required_stage_trace_chunks("profiler_off")
    expected = {0, 1}
    for observed in required.values():
        observed.update({0, 2})
    assert required_stage_trace_is_complete(required, expected) is False
    diagnostic = missing_required_stage_trace(required, expected)
    assert len(diagnostic) == 2
    for detail in diagnostic.values():
        assert detail == {"missing": [1], "unexpected": [2]}


def test_normal_close_diagnostic_lists_every_missing_selector() -> None:
    required = required_stage_trace_chunks("profiler_on")
    expected = {0, 1}
    for observed in required.values():
        observed.add(0)
    diagnostic = incomplete_measurement_diagnostic(
        required,
        expected,
        stats_by_chunk={0: {}},
        payload_complete_ns={0: 1, 1: 2},
    )
    assert diagnostic["missing_stats"] == [1]
    assert diagnostic["missing_payloads"] == []
    assert set(diagnostic["stage_trace"]) == {
        "server.model_denoise_complete/source/scheduler_result_metrics",
        "server.vae_decode_complete/source/scheduler_result_metrics",
        "server.model_denoise_complete/component/minwm_denoising",
        "server.vae_decode_complete/component/vae_decoder",
    }
    assert all(
        detail == {"missing": [1], "unexpected": []}
        for detail in diagnostic["stage_trace"].values()
    )


def _create_nsys_fixture(
    path: Path,
    include_gpu_metrics: bool,
    include_process_ids: bool = True,
    marker_issue: str | None = None,
    include_boundary_event: bool = False,
    missing_kernel_device: bool = False,
    missing_gpu_type_chunk: bool = False,
    active_cuda_devices: tuple[int, ...] = (0, 1),
    pw_to_cuda: dict[int, int] | None = None,
    collected_pw_gpu_ids: tuple[int, ...] | None = None,
    all_zero_sm: bool = False,
) -> None:
    connection = sqlite3.connect(path)
    runtime_process_column = ", globalTid INTEGER" if include_process_ids else ""
    kernel_process_column = ", globalPid INTEGER" if include_process_ids else ""
    connection.executescript(f"""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER, start INTEGER, end INTEGER{runtime_process_column});
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, start INTEGER, end INTEGER{kernel_process_column});
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES (2, 'cudaMemcpyAsync_v3020');
        """)
    if include_process_ids:
        connection.execute(
            "CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT)"
        )
        connection.executemany(
            "INSERT INTO PROCESSES VALUES (?, ?, ?)",
            [
                (pid << 24, pid, f"rank-{rank}")
                for rank, pid in enumerate(
                    100 * (device + 1) for device in active_cuda_devices
                )
            ],
        )

    def marker(
        trace_id: str,
        chunk: int,
        role: str,
        start: int,
        end: int,
        request_id: str | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
            (
                start,
                end,
                "sglang.realtime.chunk|"
                f"trace_id={trace_id}|request_id={request_id or f'request-{chunk}'}|"
                f"chunk_index={chunk}|role={role}",
            ),
        )

    marker("run-1", 0, "discard", 10_000, 20_000)
    marker("sibling-run", 1, "measured", 30_000, 40_000)
    for chunk in range(1, 11):
        if marker_issue == "missing" and chunk == 5:
            continue
        marker_chunk = 6 if marker_issue == "out_of_order" and chunk == 5 else chunk
        marker_chunk = (
            5 if marker_issue == "out_of_order" and chunk == 6 else marker_chunk
        )
        start = marker_chunk * 100_000
        request_id = (
            "request-4" if marker_issue == "duplicate_request" and chunk == 5 else None
        )
        marker(
            "run-1",
            chunk,
            "measured",
            start,
            start + 50_000,
            request_id=request_id,
        )
        if marker_issue == "duplicate" and chunk == 5:
            marker("run-1", chunk, "measured", start + 1, start + 49_999)
    marker("run-1", 11, "outside", 1_100_000, 1_150_000)

    runtime_rows = []
    kernel_rows = []
    for chunk in range(1, 11):
        start = chunk * 100_000
        for device in active_cuda_devices:
            pid = 100 * (device + 1)
            runtime_rows.append(
                (
                    1 if device % 2 == 0 else 2,
                    start + 1_000 + device * 100,
                    start + 2_000 + device * 100,
                    pid,
                )
            )
        kernel_devices = (
            active_cuda_devices[:1] if missing_kernel_device else active_cuda_devices
        )
        for device in kernel_devices:
            kernel_rows.append(
                (device, start + 5_000, start + 15_000, 100 * (device + 1))
            )
    for start in (12_000, 32_000, 1_102_000):
        runtime_rows.extend(
            [
                (
                    1 if device % 2 == 0 else 2,
                    start,
                    start + 500,
                    100 * (device + 1),
                )
                for device in active_cuda_devices
            ]
        )
        kernel_rows.extend(
            [
                (device, start, start + 500, 100 * (device + 1))
                for device in active_cuda_devices
            ]
        )
    if include_boundary_event:
        runtime_rows.append((1, 149_000, 151_000, 100))
        kernel_rows.append((0, 149_000, 151_000, 100))
    if include_process_ids:
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            [
                (name_id, start, end, (pid << 24) + pid)
                for name_id, start, end, pid in runtime_rows
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            [
                (device, start, end, pid << 24)
                for device, start, end, pid in kernel_rows
            ],
        )
    else:
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
            [row[:3] for row in runtime_rows],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
            [row[:3] for row in kernel_rows],
        )
    if include_gpu_metrics:
        connection.executescript("""
            CREATE TABLE TARGET_INFO_GPU (
                pwGpuId INTEGER,
                cuDevice INTEGER,
                busLocation TEXT,
                uuid TEXT,
                name TEXT
            );
            CREATE TABLE TARGET_INFO_GPU_METRICS (
                typeId INTEGER,
                sourceId INTEGER,
                typeName TEXT,
                metricId INTEGER,
                metricName TEXT
            );
            CREATE TABLE GPU_METRICS (typeId INTEGER, metricId INTEGER, value REAL, timestamp INTEGER);
            """)
        gpu_metric_type_base = 0x1000100000000
        pw_to_cuda = pw_to_cuda or {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
        }
        collected_pw_gpu_ids = collected_pw_gpu_ids or tuple(pw_to_cuda)
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU VALUES (?, ?, ?, ?, ?)",
            [
                (
                    pw_gpu_id,
                    cuda_device_id,
                    f"0000:{pw_gpu_id:02x}:00.0",
                    f"GPU-{pw_gpu_id}",
                    "NVIDIA H200",
                )
                for pw_gpu_id, cuda_device_id in pw_to_cuda.items()
            ],
        )
        metric_names = {
            3: "SMs Active [Throughput %]",
            5: "Tensor Active [Throughput %]",
            7: "SM Issue [Throughput %]",
            8: "Unallocated Warps in Active SMs [Throughput %]",
            18: "DRAM Throughput",
        }
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, '', ?, ?)",
            [
                (
                    gpu_metric_type_base + pw_gpu_id,
                    gpu_metric_type_base,
                    metric_id,
                    name,
                )
                for pw_gpu_id in collected_pw_gpu_ids
                for metric_id, name in metric_names.items()
            ],
        )
        gpu_rows = []
        for chunk in range(1, 11):
            timestamp = chunk * 100_000 + 25_000
            rows = []
            for pw_gpu_id in collected_pw_gpu_ids:
                cuda_device_id = pw_to_cuda[pw_gpu_id]
                type_id = gpu_metric_type_base + pw_gpu_id
                active = cuda_device_id in active_cuda_devices
                rows.extend(
                    [
                        (
                            type_id,
                            3,
                            (
                                0.0
                                if all_zero_sm and active
                                else 80.0 + cuda_device_id if active else 99.0
                            ),
                            timestamp,
                        ),
                        (type_id, 5, 55.0 if active else 88.0, timestamp),
                        (type_id, 18, 40.0 if active else 77.0, timestamp),
                        (type_id, 7, 999.0, timestamp),
                        (type_id, 8, 998.0, timestamp),
                    ]
                )
            if missing_gpu_type_chunk and chunk == 5:
                rows.remove((gpu_metric_type_base + 3, 3, 81.0, timestamp))
            gpu_rows.extend(rows)
        gpu_rows.extend(
            [
                (gpu_metric_type_base + pw_gpu_id, metric_id, 999.0, 15_000)
                for pw_gpu_id in collected_pw_gpu_ids
                for metric_id in (3, 5, 18)
            ]
        )
        connection.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", gpu_rows)
    connection.commit()
    connection.close()


def test_nsys_merge_extracts_counts_buckets_busy_and_gpu_metrics(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(sqlite_path, include_gpu_metrics=True)
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    _machine_schema_validator().validate(record)
    on = record["metrics"]["profiler_on"]
    window = on["stable_window_coverage"]["value"]
    assert window["expected_stable_chunk_indices"] == list(range(1, 11))
    assert window["observed_stable_chunk_indices"] == list(range(1, 11))
    assert window["observed_discard_chunk_indices"] == [0]
    assert window["excluded_precondition_chunks"] == 20
    assert window["normalization_denominator"] == 10
    assert on["kernel_count"]["value"] == {
        "raw_total": 20,
        "captured_raw_total": 26,
        "excluded_raw_total": 6,
        "boundary_overlap_count": 0,
        "per_stable_chunk": 2.0,
        "per_device": {
            "0": {"raw_total": 10, "per_stable_chunk": 1.0},
            "1": {"raw_total": 10, "per_stable_chunk": 1.0},
        },
        "stable_chunk_denominator": 10,
        "capture_scope": "union of exact measured outer chunk NVTX ranges",
    }
    assert on["cuda_api_count"]["value"]["raw_total"] == 20
    assert on["cuda_api_count"]["value"]["captured_raw_total"] == 26
    assert on["cuda_api_count"]["value"]["excluded_raw_total"] == 6
    assert on["cuda_api_count"]["value"]["total_per_chunk"] == 2.0
    assert on["cuda_api_count"]["value"]["per_rank_per_chunk"]["value"] == (
        pytest.approx(1.0)
    )
    assert on["kernel_launch_api_count"]["value"]["raw_total"] == 10
    buckets = on["short_kernel_buckets"]["value"]
    assert buckets["raw_total"] == {
        "lt_10_us": 0,
        "10_to_lt_50_us": 20,
        "50_to_lt_100_us": 0,
        "gte_100_us": 0,
    }
    assert buckets["per_device"]["0"]["raw_total"]["10_to_lt_50_us"] == 10
    assert on["capture_coverage"]["status"] == "available"
    coverage = on["capture_coverage"]["value"]
    assert coverage["process_id_source"].endswith(
        "globalTid masked to PROCESSES.globalPid"
    )
    assert coverage["observed_process_ids"] == [100 << 24, 200 << 24]
    assert coverage["kernel_process_ids"] == coverage["observed_process_ids"]
    assert coverage["kernel_processes_by_device"] == {
        "0": [100 << 24],
        "1": [200 << 24],
    }
    assert on["gpu_kernel_busy"]["value"]["mean_pct"] == pytest.approx(20.0)
    assert on["gpu_metrics"]["sm_active"]["value"]["mean"] == 80.5
    assert on["gpu_metrics"]["sm_active"]["value"]["raw_metric_name"] == (
        "SMs Active [Throughput %]"
    )
    assert on["gpu_metrics"]["sm_active"]["value"]["sample_count"] == 20
    assert on["gpu_metrics"]["sm_active"]["value"]["captured_sample_count"] == 22
    assert on["gpu_metrics"]["sm_active"]["value"]["excluded_sample_count"] == 2
    assert on["gpu_metrics"]["sm_active"]["value"]["all_collected_sample_count"] == 88
    assert (
        on["gpu_metrics"]["sm_active"]["value"]["excluded_inactive_target_sample_count"]
        == 66
    )
    assert on["gpu_metrics"]["sm_active"]["value"]["active_cuda_device_ids"] == [
        0,
        1,
    ]
    assert on["gpu_metrics"]["sm_active"]["value"]["active_pw_gpu_ids"] == [2, 3]
    assert on["gpu_metrics"]["sm_active"]["value"]["collected_target_count"] == 8
    assert on["gpu_metrics"]["sm_active"]["value"]["active_target_count"] == 2
    assert on["gpu_metrics"]["sm_active"]["value"]["allocated_target_count"] == 8
    assert set(
        on["gpu_metrics"]["sm_active"]["value"]["per_chunk_sample_count"].values()
    ) == {2}
    assert (
        "Tensor Active [Throughput %]"
        in on["gpu_metrics"]["sm_active"]["value"]["exposed_metric_names"]
    )
    assert on["gpu_metrics"]["tensor_active"]["status"] == "available"
    assert on["gpu_metrics"]["dram"]["status"] == "available"
    require_complete_stable_nsys(record)

    forged_zero_sm = copy.deepcopy(record)
    forged_zero_value = forged_zero_sm["metrics"]["profiler_on"]["gpu_metrics"][
        "sm_active"
    ]["value"]
    forged_zero_value["nonzero_sample_count"] = 0
    for field in ("mean", "min", "p50", "max"):
        forged_zero_value[field] = 0.0
    with pytest.raises(MeasurementValidationError, match="cannot be all zero"):
        validate_measurement(forged_zero_sm)

    no_dram = copy.deepcopy(record)
    no_dram["metrics"]["profiler_on"]["gpu_metrics"]["dram"] = unavailable(
        "metric_not_exposed",
        "Nsight exposed GPU metric names: SMs Active [Throughput %], Tensor Active [Throughput %]",
    )
    require_complete_stable_nsys(no_dram)

    bad_dram_degradation = copy.deepcopy(record)
    bad_dram_degradation["metrics"]["profiler_on"]["gpu_metrics"]["dram"] = unavailable(
        "permission_denied", "GPU metrics start failed"
    )
    with pytest.raises(MeasurementValidationError, match="gpu_metrics.dram"):
        require_complete_stable_nsys(bad_dram_degradation)

    no_sm = copy.deepcopy(record)
    no_sm["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"] = unavailable(
        "permission_denied", "GPU metrics start failed"
    )
    with pytest.raises(MeasurementValidationError, match="gpu_metrics.sm_active"):
        require_complete_stable_nsys(no_sm)

    incomplete_type_matrix = copy.deepcopy(record)
    missing_type_id = str(
        incomplete_type_matrix["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"][
            "value"
        ]["observed_type_ids"][1]
    )
    del incomplete_type_matrix["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"][
        "value"
    ]["per_type_per_chunk_sample_count"][missing_type_id]["5"]
    with pytest.raises(
        MeasurementValidationError, match="must cover every stable chunk"
    ):
        validate_measurement(incomplete_type_matrix)
    assert list(_machine_schema_validator().iter_errors(incomplete_type_matrix))


@pytest.mark.parametrize(
    "marker_issue", ["missing", "duplicate", "out_of_order", "duplicate_request"]
)
def test_nsys_merge_refuses_incomplete_stable_marker_window(
    tmp_path: Path, marker_issue: str
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path, include_gpu_metrics=True, marker_issue=marker_issue
    )
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    on = record["metrics"]["profiler_on"]
    assert on["stable_window_coverage"]["status"] == "unavailable"
    assert on["stable_window_coverage"]["reason"] == ("stable_window_marker_incomplete")
    for name in (
        "kernel_count",
        "cuda_api_count",
        "kernel_launch_api_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
    ):
        assert on[name]["status"] == "unavailable"
        assert on[name]["reason"] == "stable_window_unproven"
    with pytest.raises(MeasurementValidationError, match="lacks complete"):
        require_complete_stable_nsys(record)


def test_nsys_merge_refuses_events_crossing_measured_range_boundary(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path, include_gpu_metrics=True, include_boundary_event=True
    )
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    on = record["metrics"]["profiler_on"]
    assert on["stable_window_coverage"]["status"] == "available"
    for name in (
        "kernel_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
        "cuda_api_count",
        "kernel_launch_api_count",
    ):
        assert on[name]["status"] == "unavailable"
        assert on[name]["reason"] == "event_crosses_stable_window_boundary"
        assert "boundary_overlap_count=1" in on[name]["evidence"]


def test_nsys_kernel_metrics_require_every_active_device(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path, include_gpu_metrics=True, missing_kernel_device=True
    )
    on = merge_nsys_metrics(_record("profiler_on"), sqlite_path)["metrics"][
        "profiler_on"
    ]
    for name in ("kernel_count", "short_kernel_buckets", "gpu_kernel_busy"):
        assert on[name]["status"] == "unavailable"
        assert on[name]["reason"] == "device_capture_coverage_incomplete"
        assert "device_ids=[0]" in on[name]["evidence"]


def test_nsys_gpu_metric_requires_each_type_in_each_stable_chunk(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path, include_gpu_metrics=True, missing_gpu_type_chunk=True
    )
    gpu = merge_nsys_metrics(_record("profiler_on"), sqlite_path)["metrics"][
        "profiler_on"
    ]["gpu_metrics"]
    assert gpu["sm_active"]["status"] == "unavailable"
    assert gpu["sm_active"]["reason"] == "gpu_metric_window_coverage_incomplete"
    assert "'5'" in gpu["sm_active"]["evidence"]
    assert gpu["tensor_active"]["status"] == "available"
    assert gpu["dram"]["status"] == "available"


def test_nsys_gpu_metrics_map_active_cuda_devices_to_perfworks_targets(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(sqlite_path, include_gpu_metrics=True)
    gpu = merge_nsys_metrics(_record("profiler_on"), sqlite_path)["metrics"][
        "profiler_on"
    ]["gpu_metrics"]
    value = gpu["sm_active"]["value"]
    assert value["active_cuda_device_ids"] == [0, 1]
    assert value["active_pw_gpu_ids"] == [2, 3]
    assert value["collected_target_count"] == 8
    assert value["active_target_count"] == 2
    assert value["allocated_target_count"] == 8
    assert {
        (row["cuda_device_id"], row["pw_gpu_id"])
        for row in value["target_mapping"]
        if row["active"]
    } == {(0, 2), (1, 3)}


def test_nsys_gpu_metrics_support_non_contiguous_sp4_perfworks_mapping(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    pw_to_cuda = {2: 0, 5: 1, 1: 2, 7: 3, 0: 4, 3: 5, 4: 6, 6: 7}
    _create_nsys_fixture(
        sqlite_path,
        include_gpu_metrics=True,
        active_cuda_devices=(0, 1, 2, 3),
        pw_to_cuda=pw_to_cuda,
    )
    record = merge_nsys_metrics(_record("profiler_on", gpu_count=4), sqlite_path)
    value = record["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"]["value"]
    assert value["active_cuda_device_ids"] == [0, 1, 2, 3]
    assert value["active_pw_gpu_ids"] == [1, 2, 5, 7]
    assert value["active_target_count"] == 4
    assert set(value["per_device_per_chunk_sample_count"]) == {"0", "1", "2", "3"}
    require_complete_stable_nsys(record)


def test_nsys_gpu_metrics_reject_wrong_collected_physical_targets(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path,
        include_gpu_metrics=True,
        collected_pw_gpu_ids=(0, 1),
    )
    gpu = merge_nsys_metrics(_record("profiler_on"), sqlite_path)["metrics"][
        "profiler_on"
    ]["gpu_metrics"]
    for metric in gpu.values():
        assert metric["status"] == "unavailable"
        assert metric["reason"] == "gpu_metric_target_coverage_incomplete"
        assert '"active_pw_gpu_ids": [2, 3]' in metric["evidence"]
        assert '"collected_pw_gpu_ids": [0, 1]' in metric["evidence"]


def test_nsys_sm_active_all_zero_under_kernel_load_fails_closed(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path,
        include_gpu_metrics=True,
        all_zero_sm=True,
    )
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    sm = record["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"]
    assert sm["status"] == "unavailable"
    assert sm["reason"] == "gpu_metric_all_zero_under_kernel_load"
    assert "stable_sample_count=20" in sm["evidence"]
    assert "nonzero_sample_count=0" in sm["evidence"]
    assert (
        record["metrics"]["profiler_on"]["gpu_metrics"]["tensor_active"]["status"]
        == "available"
    )
    with pytest.raises(MeasurementValidationError, match="gpu_metrics.sm_active"):
        require_complete_stable_nsys(record)


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


def test_nsys_rank_normalization_degrades_without_process_coverage(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _create_nsys_fixture(
        sqlite_path, include_gpu_metrics=False, include_process_ids=False
    )
    record = merge_nsys_metrics(_record("profiler_on"), sqlite_path)
    on = record["metrics"]["profiler_on"]
    assert on["capture_coverage"]["status"] == "unavailable"
    assert on["capture_coverage"]["reason"] == "rank_capture_coverage_unconfirmed"
    per_rank = on["cuda_api_count"]["value"]["per_rank_per_chunk"]
    assert per_rank["status"] == "unavailable"
    assert "columns" in per_rank["evidence"]


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


def test_invalid_attempt_marker_inventories_files_and_aggregate_excludes_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt" / "s0-measurement"
    root.mkdir(parents=True)
    artifact = root / "partial.json"
    artifact.write_bytes(b"partial evidence\n")
    marker_path = root / "invalid-marker.json"
    preserved_root = tmp_path / "attempt" / "invalid" / "retry" / root.name
    marker = build_invalid_marker(
        root,
        reason="old contract missing latency count",
        marker_path=marker_path,
        preserved_root=preserved_root,
        timestamp_utc="2026-08-07T00:00:00+00:00",
    )
    assert marker["recoverability"] == "moved_to_attempt_invalid"
    assert marker["reason"] == "old contract missing latency count"
    assert marker["files"] == [
        {
            "original_path": str(artifact),
            "preserved_path": str(preserved_root / artifact.name),
            "size_bytes": len(b"partial evidence\n"),
            "sha256": hashlib.sha256(b"partial evidence\n").hexdigest(),
            "recoverable": True,
        }
    ]

    valid_one = tmp_path / "valid-one.json"
    valid_two = tmp_path / "valid-two.json"
    invalid = tmp_path / "attempt" / "invalid" / "old.json"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    for path, record in (
        (valid_one, _record(run_id="valid-one")),
        (valid_two, _record(run_id="valid-two")),
        (invalid, _record(run_id="invalid")),
    ):
        path.write_text(json.dumps(record), encoding="utf-8")
    records, excluded = load_aggregate_records([valid_one, invalid, valid_two])
    assert [record["run_id"] for record in records] == ["valid-one", "valid-two"]
    assert excluded == [invalid]


def test_aggregate_excludes_in_place_result_with_sibling_invalid_marker(
    tmp_path: Path,
) -> None:
    measurement_root = tmp_path / "attempt-one" / "s0-measurement"
    invalid_result = measurement_root / "sp2" / "profiler-off-repeat1.json"
    invalid_result.parent.mkdir(parents=True)
    invalid_result.write_text(json.dumps(_record(run_id="partial")), encoding="utf-8")
    (measurement_root / "invalid-marker-20260807T000000Z.json").write_text(
        "{}", encoding="utf-8"
    )

    valid_result = (
        tmp_path
        / "attempt-two"
        / "s0-measurement"
        / "sp2"
        / "profiler-off-repeat1.json"
    )
    valid_result.parent.mkdir(parents=True)
    valid_result.write_text(json.dumps(_record(run_id="valid")), encoding="utf-8")

    assert _is_invalid_result(invalid_result) is True
    assert _is_invalid_result(valid_result) is False
    records, excluded = load_aggregate_records([invalid_result, valid_result])
    assert [record["run_id"] for record in records] == ["valid"]
    assert excluded == [invalid_result]


def test_lane_marker_does_not_invalidate_sibling_lane(tmp_path: Path) -> None:
    root = tmp_path / "attempt" / "s0-measurement" / "sp2"
    profiler_on = root / "profiler-on"
    profiler_on.mkdir(parents=True)
    invalid_profile = profiler_on / "client.json"
    invalid_profile.write_text(json.dumps(_record("profiler_on")), encoding="utf-8")
    (profiler_on / "invalid-marker-20260807T000000Z.json").write_text(
        "{}", encoding="utf-8"
    )

    valid_off = root / "profiler-off-repeat1.json"
    valid_off.write_text(json.dumps(_record()), encoding="utf-8")
    assert _is_invalid_result(invalid_profile) is True
    assert _is_invalid_result(valid_off) is False

    (root.parent / "invalid-marker-root.json").write_text("{}", encoding="utf-8")
    assert _is_invalid_result(valid_off) is True


def test_runner_exports_sqlite_before_nsys_stats() -> None:
    runner = (
        Path(__file__).with_name("run_s0_measurement.sh").read_text(encoding="utf-8")
    )
    export_offset = runner.index("  nsys export ")
    stats_offset = runner.index("  nsys stats ")
    assert export_offset < stats_offset
    assert "--force-overwrite" not in runner


def test_runner_collects_all_allocated_gpu_metric_targets() -> None:
    runner = (
        Path(__file__).with_name("run_s0_measurement.sh").read_text(encoding="utf-8")
    )
    assert "--gpu-metrics-devices=all" in runner
    assert '--gpu-metrics-devices="${gpu_devices}"' not in runner
    assert "nvidia-smi --query-gpu=index" not in runner
