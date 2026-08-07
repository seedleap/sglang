#!/usr/bin/env python3
"""Apply S2's exact DiT/VAE count guard on top of the S0 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_latency_count(
    result: dict,
    result_path: Path,
    *,
    container: str,
    metric_names: tuple[str, ...],
    expected_count: int,
) -> None:
    measured_chunks = result["workload"]["measured_chunks"]
    if measured_chunks != expected_count:
        raise ValueError(
            f"{result_path}: measured_chunks={measured_chunks}, expected {expected_count}"
        )
    metrics = result["metrics"][container]
    for name in metric_names:
        metric = metrics[name]
        if metric["status"] != "available":
            raise ValueError(f"{result_path}: {name} is not available: {metric}")
        count = metric["value"].get("count")
        if count != expected_count:
            raise ValueError(
                f"{result_path}: {name}.value.count={count}, expected {expected_count}"
            )


def validate_run(results_root: Path, run_id: str) -> dict:
    root = results_root / run_id / "s0-measurement"
    summary = {}
    for degree in (2, 4):
        lane_root = root / f"sp{degree}"
        repeats = sorted(lane_root.glob("profiler-off-repeat*.json"))
        if len(repeats) < 2:
            raise ValueError(f"{lane_root}: expected at least two profiler-off repeats")
        for result_path in repeats:
            _require_latency_count(
                _read(result_path),
                result_path,
                container="profiler_off",
                metric_names=("dit_wall_ms", "vae_wall_ms"),
                expected_count=200,
            )
        profile_path = lane_root / "profiler-on/measurement.json"
        _require_latency_count(
            _read(profile_path),
            profile_path,
            container="profiler_on",
            metric_names=("dit_cuda_ms", "vae_cuda_ms"),
            expected_count=10,
        )
        summary[f"sp{degree}"] = {
            "profiler_off_repeats": len(repeats),
            "wall_count": 200,
            "cuda_count": 10,
        }
    return {"run_id": run_id, "lanes": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(validate_run(args.results_root, args.run_id), sort_keys=True))


if __name__ == "__main__":
    main()
