from __future__ import annotations

import pytest

from assert_latency_counts import assert_latency_counts


def _record(count: int | None) -> dict:
    value = {"mean": 1.0}
    if count is not None:
        value["count"] = count
    return {
        "workload": {"measured_chunks": 200},
        "metrics": {
            "profiler_off": {
                "dit_wall_ms": {
                    "status": "available",
                    "unit": "ms_per_chunk",
                    "value": value,
                }
            }
        },
    }


def test_accepts_exact_latency_count() -> None:
    assert assert_latency_counts(_record(200)) == ["metrics.profiler_off.dit_wall_ms"]


@pytest.mark.parametrize("count", [None, 199])
def test_rejects_missing_or_incomplete_latency_count(count: int | None) -> None:
    with pytest.raises(ValueError, match=r"value\.count"):
        assert_latency_counts(_record(count))
