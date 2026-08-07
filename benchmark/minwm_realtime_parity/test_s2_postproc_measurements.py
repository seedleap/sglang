from __future__ import annotations

import json
from pathlib import Path

import pytest
from validate_s2_postproc_measurements import validate_run


def _metric(count: int) -> dict:
    return {"status": "available", "value": {"count": count, "mean": 1.0}}


def _write_fixture(root: Path, run_id: str) -> None:
    for degree in (2, 4):
        lane = root / run_id / "s0-measurement" / f"sp{degree}"
        lane.mkdir(parents=True)
        off = {
            "workload": {"measured_chunks": 200},
            "metrics": {
                "profiler_off": {
                    "dit_wall_ms": _metric(200),
                    "vae_wall_ms": _metric(200),
                }
            },
        }
        for repeat in (1, 2):
            (lane / f"profiler-off-repeat{repeat}.json").write_text(
                json.dumps(off), encoding="utf-8"
            )
        profile_dir = lane / "profiler-on"
        profile_dir.mkdir()
        profile = {
            "workload": {"measured_chunks": 10},
            "metrics": {
                "profiler_on": {
                    "dit_cuda_ms": _metric(10),
                    "vae_cuda_ms": _metric(10),
                }
            },
        }
        (profile_dir / "measurement.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )


def test_s2_exact_count_guard_accepts_complete_sp2_sp4_results(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "complete")
    summary = validate_run(tmp_path, "complete")
    assert summary["lanes"]["sp2"] == {
        "profiler_off_repeats": 2,
        "wall_count": 200,
        "cuda_count": 10,
    }
    assert summary["lanes"]["sp4"] == summary["lanes"]["sp2"]


@pytest.mark.parametrize(
    ("relative_path", "container", "metric", "bad_count", "expected"),
    [
        (
            "sp2/profiler-off-repeat1.json",
            "profiler_off",
            "dit_wall_ms",
            199,
            "dit_wall_ms.value.count=199, expected 200",
        ),
        (
            "sp4/profiler-on/measurement.json",
            "profiler_on",
            "vae_cuda_ms",
            9,
            "vae_cuda_ms.value.count=9, expected 10",
        ),
    ],
)
def test_s2_exact_count_guard_rejects_partial_results(
    tmp_path: Path,
    relative_path: str,
    container: str,
    metric: str,
    bad_count: int,
    expected: str,
) -> None:
    run_id = "partial"
    _write_fixture(tmp_path, run_id)
    result_path = tmp_path / run_id / "s0-measurement" / relative_path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"][container][metric]["value"]["count"] = bad_count
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        validate_run(tmp_path, run_id)
