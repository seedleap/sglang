from __future__ import annotations

import pytest

from async_a2a_nsys_metrics import (
    Kernel,
    _chunk_membership,
    _intersection_duration,
    _merged_duration,
    _output_transport,
    _summary,
    compare,
)


def test_interval_union_and_intersection_do_not_double_count_overlap() -> None:
    assert _merged_duration([(0, 10), (5, 12), (20, 25)]) == 17
    assert _intersection_duration([(0, 10), (20, 30)], [(5, 25)]) == 10


def test_chunk_membership_requires_full_containment() -> None:
    chunks = [
        {"start_ns": 100, "end_ns": 200},
        {"start_ns": 220, "end_ns": 300},
    ]
    assert _chunk_membership(110, 190, chunks) == 0
    assert _chunk_membership(230, 290, chunks) == 1
    assert _chunk_membership(190, 230, chunks) is None


def test_summary_reports_interpolated_p10_p90() -> None:
    value = _summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert value["median"] == 3.0
    assert value["p10"] == pytest.approx(1.4)
    assert value["p90"] == pytest.approx(4.6)


def _kernel(name: str, start: int) -> Kernel:
    return Kernel(start, start + 10, 0, 7, start, 100, name)


def test_output_transport_recognizes_sp2_ipc_protocol_without_layout_kernels() -> None:
    kernels = [
        _kernel("layout_kernel", 0),
        _kernel("elementwise_kernel", 10),
        _kernel("bump_signal_kernel", 20),
        _kernel("elementwise_kernel", 30),
        _kernel("spin_wait_kernel", 40),
    ]
    assert [kernel.name for kernel in _output_transport(kernels)] == [
        "elementwise_kernel",
        "bump_signal_kernel",
        "elementwise_kernel",
        "spin_wait_kernel",
    ]
    assert _output_transport(kernels[:3]) == []


def test_output_transport_prefers_nccl_when_present() -> None:
    kernels = [
        _kernel("elementwise_kernel", 0),
        _kernel("ncclDevKernel_SendRecv", 10),
        _kernel("bump_signal_kernel", 20),
        _kernel("spin_wait_kernel", 30),
    ]
    assert [kernel.name for kernel in _output_transport(kernels)] == [
        "ncclDevKernel_SendRecv"
    ]


def _comparison_fixture(exposed: float, stage: float, overlaps: int, sync: int):
    chunks = []
    for index in range(10):
        chunks.append(
            {
                "critical_rank_max": {
                    "input_compute_overlap_ms": 1.0 if index < overlaps else 0.0
                }
            }
        )
    return {
        "trace_id": f"trace-{exposed}",
        "aggregate": {
            "metrics": {
                "critical_rank_max": {
                    "input_exposed_kernel_ms": {"median": exposed},
                    "input_exposed_stage_ms": {"median": stage},
                }
            }
        },
        "chunks": chunks,
        "synchronization_apis": {"cudaDeviceSynchronize": {"count": sync}},
    }


def test_mechanism_comparison_requires_repeatable_overlap_and_no_new_sync() -> None:
    baseline = _comparison_fixture(100.0, 120.0, 0, 0)
    candidate = _comparison_fixture(79.0, 110.0, 8, 0)
    result = compare(baseline, candidate)
    assert result["input_exposed_kernel_ms"]["reduction_percent"] == 21.0
    assert result["repeatable_compute_communication_overlap"]["passes"] is True
    assert result["mechanism_acceptance"]["passes"] is True

    candidate["synchronization_apis"]["cudaDeviceSynchronize"]["count"] = 1
    assert compare(baseline, candidate)["mechanism_acceptance"]["passes"] is False
