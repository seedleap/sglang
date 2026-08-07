import sqlite3
from pathlib import Path

from compare_temb_hoist_nsys import (
    INDEX_KERNEL_RE,
    _assert_contract,
    _assert_historical_runner_flag_guard,
    _assert_off_resume_contract,
    _kernel_name_counts,
    _metric_summary,
)
from measurement import API_BOUNDARY_ATTRIBUTION_POLICY


def _off_record(sglang_commit: str, variant: str) -> dict[str, object]:
    return {
        "mode": "profiler_off",
        "provenance": {
            "sglang_commit": sglang_commit,
            "gpu": {"count": 2, "allocated_count": 8},
        },
        "workload": {
            "sp_degree": 2,
            "precision": "bf16",
            "fast_lane": True,
            "warmup_chunks": 20,
            "measured_chunks": 200,
            "dmd_forwards_per_chunk": 4,
            "clean_cache_forwards_per_chunk": 1,
        },
        "comparison_contract": {"kv_cache_num_frames": 45},
        "run_id": f"test-temb-hoist-{variant}-sp2-off-r1",
        "profile_name": "bf16-fast-sp2",
        "metrics": {
            "profiler_off": {
                name: {"status": "available", "value": {"count": 200}}
                for name in (
                    "scheduler_chunk_wall_ms",
                    "dit_wall_ms",
                    "vae_wall_ms",
                )
            }
        },
    }


def test_off_resume_contract_uses_its_historical_source_ref() -> None:
    record = _off_record("historical-off-ref", "legacy")
    _assert_off_resume_contract(record, "historical-off-ref", "legacy")
    try:
        _assert_off_resume_contract(record, "current-nsys-ref", "legacy")
    except ValueError as exc:
        assert "unexpected sglang commit" in str(exc)
    else:
        raise AssertionError("resume validation accepted the current Nsight ref")


def test_off_resume_contract_rejects_wrong_variant_or_incomplete_count() -> None:
    record = _off_record("historical-off-ref", "legacy")
    try:
        _assert_off_resume_contract(record, "historical-off-ref", "candidate")
    except ValueError as exc:
        assert "not labeled candidate" in str(exc)
    else:
        raise AssertionError("resume validation accepted the wrong variant label")

    record = _off_record("historical-off-ref", "legacy")
    record["metrics"]["profiler_off"]["dit_wall_ms"]["value"]["count"] = 199
    try:
        _assert_off_resume_contract(record, "historical-off-ref", "legacy")
    except ValueError as exc:
        assert "expected 200 complete" in str(exc)
    else:
        raise AssertionError("resume validation accepted an incomplete wall count")


def test_historical_runner_flag_guard_rejects_swapped_flag() -> None:
    legacy = "\n".join(
        (
            'if [[ "${lane}" == "legacy" ]]; then',
            "export MINWM_HOIST_TIMESTEP_MODULATION=0",
            'export MINWM_S0_RUN_LABEL="temb-hoist-${lane}"',
            'bash "${SCRIPT_DIR}/run_s0_measurement.sh"',
        )
    )
    _assert_historical_runner_flag_guard(legacy, "legacy")
    try:
        _assert_historical_runner_flag_guard(
            legacy.replace("MODULATION=0", "MODULATION=1"), "legacy"
        )
    except ValueError as exc:
        assert "lost its flag guard" in str(exc)
    else:
        raise AssertionError("historical runner accepted a swapped implementation flag")


def test_kernel_name_counts_use_exact_ranges_and_active_devices(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "capture.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES (1, 'index_elementwise_kernel');
            INSERT INTO StringIds VALUES (2, 'gemm_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, start INTEGER, end INTEGER, shortName INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 110, 120, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 130, 140, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 150, 190, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7, 150, 190, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 210, 220, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 90, 95, 1);
            """
        )
    intervals = [
        {"start_ns": 100, "end_ns": 200},
        {"start_ns": 200, "end_ns": 300},
    ]
    counts = _kernel_name_counts(sqlite_path, intervals, {0, 1})
    assert counts == {"index_elementwise_kernel": 3, "gemm_kernel": 1}


def test_kernel_name_counts_reject_boundary_overlap(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "capture.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES (1, 'index_elementwise_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, start INTEGER, end INTEGER, shortName INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 190, 210, 1);
            """
        )
    intervals = [
        {"start_ns": 100, "end_ns": 200},
        {"start_ns": 220, "end_ns": 300},
    ]
    try:
        _kernel_name_counts(sqlite_path, intervals, {0})
    except ValueError as exc:
        assert "cross stable boundaries" in str(exc)
    else:
        raise AssertionError("boundary-overlap kernel should fail closed")


def test_index_kernel_matcher_is_narrow() -> None:
    assert INDEX_KERNEL_RE.search("void index_elementwise_kernel<float>")
    assert INDEX_KERNEL_RE.search("at::native::gather_kernel")
    assert not INDEX_KERNEL_RE.search("flash_attention_forward")


def test_metric_summary_preserves_api_boundary_evidence() -> None:
    def available(value: object) -> dict[str, object]:
        return {"status": "available", "value": value}

    cuda_api = {
        "total_per_chunk": 123.4,
        "boundary_attribution_policy": API_BOUNDARY_ATTRIBUTION_POLICY,
        "boundary_spanning_count": 1,
        "boundary_included_by_start_count": 1,
        "boundary_excluded_by_start_count": 0,
        "boundary_event_examples": [{"raw_api_name": "cudaEventQuery"}],
    }
    launch_api = {
        "total_per_chunk": 45.6,
        "boundary_attribution_policy": API_BOUNDARY_ATTRIBUTION_POLICY,
        "boundary_spanning_count": 0,
        "boundary_included_by_start_count": 0,
        "boundary_excluded_by_start_count": 0,
        "boundary_event_examples": [],
    }
    record = {
        "metrics": {
            "profiler_on": {
                "observed_wall_with_profiler_overhead": {
                    "dit_wall_ms": available({"count": 10, "mean": 0.9}),
                    "vae_wall_ms": available({"count": 10, "mean": 1.9}),
                },
                "dit_cuda_ms": available({"mean": 1.0}),
                "vae_cuda_ms": available({"mean": 2.0}),
                "kernel_count": available({"per_stable_chunk": 3.0}),
                "cuda_api_count": available(cuda_api),
                "kernel_launch_api_count": available(launch_api),
                "short_kernel_buckets": available({"per_stable_chunk": {}}),
                "gpu_kernel_busy": available(4.0),
                "gpu_metrics": {
                    "sm_active": available({"mean": 5.0}),
                    "tensor_active": available({"mean": 6.0}),
                    "dram": {"status": "unavailable", "reason": "metric_not_exposed"},
                },
            }
        }
    }

    summary = _metric_summary(record)
    assert summary["cuda_api_per_chunk"] == 123.4
    assert summary["launch_api_per_chunk"] == 45.6
    assert summary["cuda_api_boundary_evidence"] is cuda_api
    assert summary["launch_api_boundary_evidence"] is launch_api


def test_formal_contract_reads_wall_from_observed_profiler_container() -> None:
    def available(value: object) -> dict[str, object]:
        return {"status": "available", "value": value}

    stage = available({"count": 10})
    gpu_metric = available(
        {
            "collected_target_count": 8,
            "allocated_target_count": 8,
            "active_target_count": 2,
            "active_cuda_device_ids": [0, 1],
            "active_pw_gpu_ids": [2, 3],
            "target_mapping": [{} for _ in range(8)],
            "per_device_per_chunk_sample_count": {
                "0": [{} for _ in range(10)],
                "1": [{} for _ in range(10)],
            },
        }
    )
    api = available(
        {"boundary_attribution_policy": API_BOUNDARY_ATTRIBUTION_POLICY}
    )
    record = {
        "mode": "profiler_on",
        "provenance": {"gpu": {"count": 2, "allocated_count": 8}},
        "workload": {
            "sp_degree": 2,
            "warmup_chunks": 1,
            "precondition_warmup_chunks": 20,
            "measured_chunks": 10,
            "precision": "bf16",
            "dmd_forwards_per_chunk": 4,
            "clean_cache_forwards_per_chunk": 1,
        },
        "comparison_contract": {"kv_cache_num_frames": 45},
        "metrics": {
            "profiler_on": {
                "observed_wall_with_profiler_overhead": {
                    "dit_wall_ms": stage,
                    "vae_wall_ms": stage,
                },
                "dit_cuda_ms": stage,
                "vae_cuda_ms": stage,
                "kernel_count": available({}),
                "cuda_api_count": api,
                "kernel_launch_api_count": api,
                "short_kernel_buckets": available({}),
                "gpu_kernel_busy": available({}),
                "stable_window_coverage": available(
                    {
                        "expected_stable_chunk_indices": list(range(1, 11)),
                        "observed_stable_chunk_indices": list(range(1, 11)),
                        "normalization_denominator": 10,
                        "intervals": [{} for _ in range(10)],
                    }
                ),
                "capture_coverage": available({}),
                "gpu_metrics": {
                    "sm_active": gpu_metric,
                    "tensor_active": gpu_metric,
                },
            }
        },
    }
    _assert_contract(record)
    del record["metrics"]["profiler_on"]["observed_wall_with_profiler_overhead"]
    try:
        _assert_contract(record)
    except ValueError as exc:
        assert "observed wall metrics are missing" in str(exc)
    else:
        raise AssertionError("formal contract accepted misplaced wall metrics")
