import sqlite3
from pathlib import Path

import pytest

from compare_s3_post_nsys import (
    FUSED_POST_RE,
    NCCL_A2A_RE,
    _event_group_summary,
    _kernel_events,
    _nccl_a2a_summary,
    _off_metric_summary,
    _scalar_comparison,
)


def _create_kernel_fixture(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES (1, '_fused_rope_cache_update_kernel');
            INSERT INTO StringIds VALUES (2, 'ncclDevKernel_SendRecv');
            INSERT INTO StringIds VALUES (3, 'gemm_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, start INTEGER, end INTEGER, shortName INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 110000, 120000, 3);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 130000, 150000, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 160000, 175000, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 110000, 125000, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7, 130000, 150000, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 210000, 220000, 3);
            """
        )


def test_kernel_events_use_exact_ranges_and_active_devices(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "capture.sqlite"
    _create_kernel_fixture(sqlite_path)
    intervals = [
        {"chunk_index": 1, "start_ns": 100000, "end_ns": 200000},
        {"chunk_index": 2, "start_ns": 200000, "end_ns": 300000},
    ]
    events = _kernel_events(sqlite_path, intervals, {0, 1})
    assert [
        (event["device"], event["chunk_index"], event["name"]) for event in events
    ] == [
        (0, 1, "gemm_kernel"),
        (0, 1, "ncclDevKernel_SendRecv"),
        (0, 1, "_fused_rope_cache_update_kernel"),
        (1, 1, "ncclDevKernel_SendRecv"),
        (0, 2, "gemm_kernel"),
    ]


def test_kernel_events_reject_boundary_overlap(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "capture.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES (1, 'gemm_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, start INTEGER, end INTEGER, shortName INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 190, 210, 1);
            """
        )
    intervals = [{"chunk_index": 1, "start_ns": 100, "end_ns": 200}]
    with pytest.raises(ValueError, match="cross stable boundaries"):
        _kernel_events(sqlite_path, intervals, {0})


def test_post_and_nccl_groups_report_duration_and_device_visible_gaps() -> None:
    events = [
        {
            "device": 0,
            "chunk_index": 1,
            "chunk_start_ns": 0,
            "chunk_end_ns": 100000,
            "start_ns": 10000,
            "end_ns": 20000,
            "duration_ns": 10000,
            "name": "gemm_kernel",
        },
        {
            "device": 0,
            "chunk_index": 1,
            "chunk_start_ns": 0,
            "chunk_end_ns": 100000,
            "start_ns": 30000,
            "end_ns": 50000,
            "duration_ns": 20000,
            "name": "ncclDevKernel_SendRecv",
        },
        {
            "device": 0,
            "chunk_index": 1,
            "chunk_start_ns": 0,
            "chunk_end_ns": 100000,
            "start_ns": 65000,
            "end_ns": 80000,
            "duration_ns": 15000,
            "name": "_fused_rope_cache_update_kernel",
        },
    ]
    fused = _event_group_summary(
        events, lambda name: bool(FUSED_POST_RE.search(name)), 1
    )
    assert fused["raw_total"] == 1
    assert fused["duration_ms_raw_total"] == 0.015
    nccl = _nccl_a2a_summary(events, 1)
    assert nccl["raw_total"] == 1
    assert nccl["duration_ms_raw_total"] == 0.02
    assert nccl["device_visible_predecessor_gap"]["mean_us"] == 10.0
    assert nccl["device_visible_successor_gap"]["mean_us"] == 15.0


def test_matchers_are_narrow_to_fused_post_and_nccl_a2a() -> None:
    assert FUSED_POST_RE.search("_fused_rope_cache_update_kernel")
    assert not FUSED_POST_RE.search("flash_attention_forward")
    assert NCCL_A2A_RE.search("ncclDevKernel_SendRecv")
    assert NCCL_A2A_RE.search("ncclKernel_AllToAll")
    assert not NCCL_A2A_RE.search("ncclDevKernel_AllReduce")


def test_off_summary_computes_scheduler_unclassified_interval() -> None:
    summary = {
        "metrics": {
            "client_fps": {"mean": 12.0},
            "scheduler_fps": {"mean": 12.1},
            "scheduler_chunk_wall_ms": {"mean": 1300.0},
            "dit_wall_ms": {"mean": 720.0},
            "vae_wall_ms": {"mean": 420.0},
        }
    }
    values = _off_metric_summary(summary)
    assert values["scheduler_unclassified_ms"] == 160.0
    comparison = _scalar_comparison(
        {"scheduler_unclassified_ms": 150.0},
        {"scheduler_unclassified_ms": 160.0},
    )
    assert comparison["scheduler_unclassified_ms"]["candidate_minus_baseline"] == 10.0
