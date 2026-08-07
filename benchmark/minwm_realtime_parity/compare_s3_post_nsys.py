#!/usr/bin/env python3
"""Compare exact-window Nsight records for the MinWM post-A2A fast lane."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from measurement import API_BOUNDARY_ATTRIBUTION_POLICY, validate_measurement
from measurement_tool import require_complete_stable_nsys


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
FUSED_POST_RE = re.compile(r"fused_rope_cache_update", re.IGNORECASE)
ROPE_CACHE_RE = re.compile(
    r"(?:fused_rope_cache_update|rotary|rope|cache)", re.IGNORECASE
)
NCCL_A2A_RE = re.compile(
    r"(?:nccl.*(?:sendrecv|alltoall)|(?:sendrecv|alltoall).*nccl)", re.IGNORECASE
)


def _load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    validate_measurement(record)
    require_complete_stable_nsys(record)
    return record


def _load_off_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary.get("schema_version") != "minwm-realtime-repeat-summary/v1":
        raise ValueError(f"unexpected profiler-off summary schema: {path}")
    if summary.get("acceptance", {}).get("passes_cv_target") is not True:
        raise ValueError(f"profiler-off source did not pass CV: {path}")
    return summary


def _available_value(metric: dict[str, Any], name: str) -> Any:
    if metric.get("status") != "available":
        raise ValueError(f"{name} is unavailable: {metric}")
    return metric["value"]


def _assert_contract(record: dict[str, Any], degree: int) -> None:
    if record["mode"] != "profiler_on":
        raise ValueError(f"expected profiler_on, got {record['mode']}")
    workload = record["workload"]
    expected = {
        "sp_degree": degree,
        "warmup_chunks": 1,
        "precondition_warmup_chunks": 20,
        "measured_chunks": 10,
        "precision": "bf16",
        "dmd_forwards_per_chunk": 4,
        "clean_cache_forwards_per_chunk": 1,
    }
    for key, value in expected.items():
        if workload[key] != value:
            raise ValueError(f"workload.{key}={workload[key]!r}, expected {value!r}")
    if record["comparison_contract"]["kv_cache_num_frames"] != 45:
        raise ValueError("comparison_contract.kv_cache_num_frames must be 45")
    gpu = record["provenance"]["gpu"]
    if gpu["count"] != degree or gpu["allocated_count"] != 8:
        raise ValueError(f"expected active/allocated GPUs {degree}/8, got {gpu}")

    on = record["metrics"]["profiler_on"]
    for name in (
        "dit_cuda_ms",
        "vae_cuda_ms",
        "kernel_count",
        "cuda_api_count",
        "kernel_launch_api_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
        "stable_window_coverage",
        "capture_coverage",
    ):
        _available_value(on[name], name)
    observed = on["observed_wall_with_profiler_overhead"]
    for name in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"):
        value = _available_value(observed[name], f"observed_wall.{name}")
        if value["count"] != 10:
            raise ValueError(f"observed_wall.{name}.count must be 10")
    for name in ("dit_cuda_ms", "vae_cuda_ms"):
        if _available_value(on[name], name)["count"] != 10:
            raise ValueError(f"{name}.count must be 10")
    for name in ("cuda_api_count", "kernel_launch_api_count"):
        value = _available_value(on[name], name)
        if value["boundary_attribution_policy"] != API_BOUNDARY_ATTRIBUTION_POLICY:
            raise ValueError(f"{name}: unexpected API boundary attribution policy")

    expected_devices = list(range(degree))
    for name in ("sm_active", "tensor_active"):
        value = _available_value(on["gpu_metrics"][name], f"gpu_metrics.{name}")
        if value["collected_target_count"] != 8:
            raise ValueError(f"{name}: expected all 8 collected targets")
        if value["allocated_target_count"] != 8:
            raise ValueError(f"{name}: allocated target count is not 8")
        if value["active_target_count"] != degree:
            raise ValueError(f"{name}: active target count is not SP{degree}")
        if value["active_cuda_device_ids"] != expected_devices:
            raise ValueError(f"{name}: active CUDA device mapping is wrong")
        if len(value["active_pw_gpu_ids"]) != degree:
            raise ValueError(f"{name}: active PerfWorks target count is wrong")
        if len(value["target_mapping"]) != 8:
            raise ValueError(f"{name}: target mapping does not cover all 8 GPUs")
        if set(value["per_device_per_chunk_sample_count"]) != {
            str(device) for device in expected_devices
        }:
            raise ValueError(f"{name}: per-device coverage is not SP{degree}")
        if any(
            len(per_chunk) != 10
            for per_chunk in value["per_device_per_chunk_sample_count"].values()
        ):
            raise ValueError(f"{name}: every active device must cover 10 chunks")

    window = _available_value(on["stable_window_coverage"], "stable_window")
    expected_indices = list(range(1, 11))
    if window["expected_stable_chunk_indices"] != expected_indices:
        raise ValueError("expected stable chunk indices must be 1..10")
    if window["observed_stable_chunk_indices"] != expected_indices:
        raise ValueError("observed stable chunk indices must be 1..10")
    if window["normalization_denominator"] != 10 or len(window["intervals"]) != 10:
        raise ValueError("stable window must contain exactly 10 ranges")


def _kernel_name_column(connection: sqlite3.Connection) -> str:
    columns = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info('{KERNEL_TABLE}')")
    }
    for name in ("demangledName", "shortName", "mangledName", "name"):
        if name in columns:
            return name
    raise ValueError(f"{KERNEL_TABLE} has no supported name column: {sorted(columns)}")


def _resolve_names(
    connection: sqlite3.Connection, raw_names: list[Any]
) -> dict[Any, str]:
    integer_ids = {int(value) for value in raw_names if isinstance(value, int)}
    resolved: dict[Any, str] = {
        value: str(value) for value in raw_names if not isinstance(value, int)
    }
    if not integer_ids:
        return resolved
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "StringIds" not in tables:
        raise ValueError("kernel name column contains IDs but StringIds is absent")
    placeholders = ",".join("?" for _ in integer_ids)
    for identifier, value in connection.execute(
        f"SELECT id, value FROM StringIds WHERE id IN ({placeholders})",
        tuple(sorted(integer_ids)),
    ):
        resolved[int(identifier)] = str(value)
    missing = integer_ids - set(resolved)
    if missing:
        raise ValueError(f"unresolved kernel name IDs: {sorted(missing)}")
    return resolved


def _kernel_events(
    sqlite_path: Path,
    intervals: list[dict[str, Any]],
    active_devices: set[int],
) -> list[dict[str, Any]]:
    with sqlite3.connect(sqlite_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if KERNEL_TABLE not in tables:
            raise ValueError(f"{KERNEL_TABLE} is absent from {sqlite_path}")
        name_column = _kernel_name_column(connection)
        rows = list(
            connection.execute(
                f"SELECT deviceId, start, end, {name_column} FROM {KERNEL_TABLE}"
            )
        )
        names = _resolve_names(connection, [row[3] for row in rows])

    events = []
    boundary_overlap = 0
    for raw_device, raw_start, raw_end, raw_name in rows:
        device = int(raw_device)
        if device not in active_devices:
            continue
        start, end = int(raw_start), int(raw_end)
        containing = [
            interval
            for interval in intervals
            if start >= int(interval["start_ns"]) and end <= int(interval["end_ns"])
        ]
        overlaps = any(
            start < int(interval["end_ns"]) and end > int(interval["start_ns"])
            for interval in intervals
        )
        if len(containing) == 1:
            events.append(
                {
                    "device": device,
                    "chunk_index": int(containing[0]["chunk_index"]),
                    "chunk_start_ns": int(containing[0]["start_ns"]),
                    "chunk_end_ns": int(containing[0]["end_ns"]),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "name": names[raw_name],
                }
            )
        elif overlaps:
            boundary_overlap += 1
    if boundary_overlap:
        raise ValueError(
            f"{sqlite_path}: {boundary_overlap} kernel events cross stable boundaries"
        )
    return events


def _event_group_summary(
    events: list[dict[str, Any]],
    predicate: Callable[[str], bool],
    active_device_count: int,
) -> dict[str, Any]:
    selected = [event for event in events if predicate(event["name"])]
    names: dict[str, dict[str, float | int]] = {}
    for name in sorted({event["name"] for event in selected}):
        matching = [event for event in selected if event["name"] == name]
        names[name] = {
            "raw_total": len(matching),
            "duration_ms_raw_total": sum(event["duration_ns"] for event in matching)
            / 1e6,
        }
    duration_ms = sum(event["duration_ns"] for event in selected) / 1e6
    return {
        "raw_total": len(selected),
        "per_stable_chunk": len(selected) / 10,
        "per_active_device_per_stable_chunk": len(selected)
        / (10 * active_device_count),
        "duration_ms_raw_total": duration_ms,
        "duration_ms_per_stable_chunk": duration_ms / 10,
        "duration_ms_per_active_device_per_stable_chunk": duration_ms
        / (10 * active_device_count),
        "kernel_names": names,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999) - 1))
    return ordered[index]


def _nccl_a2a_summary(
    events: list[dict[str, Any]], active_device_count: int
) -> dict[str, Any]:
    summary = _event_group_summary(
        events, lambda name: bool(NCCL_A2A_RE.search(name)), active_device_count
    )
    nccl_events = [event for event in events if NCCL_A2A_RE.search(event["name"])]
    by_device_chunk: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_device_chunk[(event["device"], event["chunk_index"])].append(event)

    pre_gaps_us = []
    post_gaps_us = []
    for nccl in nccl_events:
        peers = by_device_chunk[(nccl["device"], nccl["chunk_index"])]
        overlaps_start = any(
            event is not nccl and event["start_ns"] < nccl["start_ns"] < event["end_ns"]
            for event in peers
        )
        overlaps_end = any(
            event is not nccl and event["start_ns"] < nccl["end_ns"] < event["end_ns"]
            for event in peers
        )
        previous_ends = [
            event["end_ns"] for event in peers if event["end_ns"] <= nccl["start_ns"]
        ]
        next_starts = [
            event["start_ns"] for event in peers if event["start_ns"] >= nccl["end_ns"]
        ]
        previous_end = max(previous_ends, default=nccl["chunk_start_ns"])
        next_start = min(next_starts, default=nccl["chunk_end_ns"])
        pre_gaps_us.append(
            0.0 if overlaps_start else (nccl["start_ns"] - previous_end) / 1e3
        )
        post_gaps_us.append(
            0.0 if overlaps_end else (next_start - nccl["end_ns"]) / 1e3
        )

    def gap_summary(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean_us": statistics.fmean(values) if values else None,
            "p95_us": _percentile(values, 0.95),
            "max_us": max(values) if values else None,
            "raw_total_ms": sum(values) / 1e3,
            "per_stable_chunk_ms": sum(values) / 1e3 / 10,
        }

    summary["device_visible_predecessor_gap"] = gap_summary(pre_gaps_us)
    summary["device_visible_successor_gap"] = gap_summary(post_gaps_us)
    summary["gap_method"] = (
        "For every active-device NCCL SendRecv/AllToAll kernel fully contained in "
        "an exact stable chunk, measure to the nearest completed predecessor and "
        "nearest following kernel on that device. Separate predecessor/successor "
        "totals may describe the same inter-kernel gap and are not added together."
    )
    return summary


def _metric_summary(record: dict[str, Any]) -> dict[str, Any]:
    on = record["metrics"]["profiler_on"]
    observed = on["observed_wall_with_profiler_overhead"]
    scheduler = _available_value(
        observed["scheduler_chunk_wall_ms"], "scheduler_chunk_wall_ms"
    )["mean"]
    dit_wall = _available_value(observed["dit_wall_ms"], "dit_wall_ms")["mean"]
    vae_wall = _available_value(observed["vae_wall_ms"], "vae_wall_ms")["mean"]
    cuda_api = _available_value(on["cuda_api_count"], "cuda_api_count")
    launch_api = _available_value(
        on["kernel_launch_api_count"], "kernel_launch_api_count"
    )
    return {
        "scheduler_observed_wall_ms": scheduler,
        "dit_observed_wall_ms": dit_wall,
        "vae_observed_wall_ms": vae_wall,
        "scheduler_unclassified_observed_ms": scheduler - dit_wall - vae_wall,
        "dit_cuda_ms": _available_value(on["dit_cuda_ms"], "dit_cuda_ms")["mean"],
        "vae_cuda_ms": _available_value(on["vae_cuda_ms"], "vae_cuda_ms")["mean"],
        "kernel_per_chunk": _available_value(on["kernel_count"], "kernel_count")[
            "per_stable_chunk"
        ],
        "cuda_api_per_chunk": cuda_api["total_per_chunk"],
        "launch_api_per_chunk": launch_api["total_per_chunk"],
        "cuda_api_boundary_evidence": cuda_api,
        "launch_api_boundary_evidence": launch_api,
        "short_kernel_buckets_per_chunk": _available_value(
            on["short_kernel_buckets"], "short_kernel_buckets"
        )["per_stable_chunk"],
        "gpu_kernel_busy": _available_value(on["gpu_kernel_busy"], "gpu_kernel_busy"),
        "sm_active": _available_value(
            on["gpu_metrics"]["sm_active"], "gpu_metrics.sm_active"
        ),
        "tensor_active": _available_value(
            on["gpu_metrics"]["tensor_active"], "gpu_metrics.tensor_active"
        ),
        "dram": on["gpu_metrics"]["dram"],
    }


def _off_metric_summary(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["metrics"]
    values = {
        name: float(metrics[name]["mean"])
        for name in (
            "client_fps",
            "scheduler_fps",
            "scheduler_chunk_wall_ms",
            "dit_wall_ms",
            "vae_wall_ms",
        )
    }
    values["scheduler_unclassified_ms"] = (
        values["scheduler_chunk_wall_ms"]
        - values["dit_wall_ms"]
        - values["vae_wall_ms"]
    )
    return values


def _scalar_comparison(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    comparison = {}
    for name in baseline:
        before = float(baseline[name])
        after = float(candidate[name])
        comparison[name] = {
            "baseline": before,
            "candidate": after,
            "candidate_minus_baseline": after - before,
            "candidate_delta_pct": (after / before - 1.0) * 100.0 if before else None,
        }
    return comparison


def _name_deltas(
    baseline_events: list[dict[str, Any]], candidate_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def summarize(events: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, int]]:
        counts = Counter(event["name"] for event in events)
        durations: dict[str, int] = defaultdict(int)
        for event in events:
            durations[event["name"]] += event["duration_ns"]
        return counts, durations

    baseline_counts, baseline_durations = summarize(baseline_events)
    candidate_counts, candidate_durations = summarize(candidate_events)
    deltas = []
    for name in sorted(set(baseline_counts) | set(candidate_counts)):
        count_delta = candidate_counts[name] - baseline_counts[name]
        duration_delta_ms = (candidate_durations[name] - baseline_durations[name]) / 1e6
        if count_delta:
            deltas.append(
                {
                    "name": name,
                    "baseline_raw_total": baseline_counts[name],
                    "candidate_raw_total": candidate_counts[name],
                    "candidate_minus_baseline": count_delta,
                    "baseline_per_chunk": baseline_counts[name] / 10,
                    "candidate_per_chunk": candidate_counts[name] / 10,
                    "baseline_duration_ms_raw_total": baseline_durations[name] / 1e6,
                    "candidate_duration_ms_raw_total": candidate_durations[name] / 1e6,
                    "candidate_minus_baseline_duration_ms": duration_delta_ms,
                }
            )
    deltas.sort(
        key=lambda item: (
            -abs(item["candidate_minus_baseline"]),
            -abs(item["candidate_minus_baseline_duration_ms"]),
            item["name"],
        )
    )
    return deltas


def compare(
    degree: int,
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any],
    baseline_sqlite: Path,
    candidate_sqlite: Path,
    baseline_off_summary: dict[str, Any],
    candidate_off_summary: dict[str, Any],
) -> dict[str, Any]:
    for record in (baseline_record, candidate_record):
        _assert_contract(record, degree)
    for key in ("sglang_commit", "minwm_commit", "container_image"):
        if baseline_record["provenance"][key] != candidate_record["provenance"][key]:
            raise ValueError(f"provenance mismatch: {key}")
    if baseline_off_summary["contract"]["sp_degree"] != degree:
        raise ValueError("baseline profiler-off SP degree mismatch")
    if candidate_off_summary["contract"]["sp_degree"] != degree:
        raise ValueError("candidate profiler-off SP degree mismatch")

    def window(record: dict[str, Any]) -> dict[str, Any]:
        return record["metrics"]["profiler_on"]["stable_window_coverage"]["value"]

    def active_devices(record: dict[str, Any]) -> set[int]:
        value = record["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"]["value"]
        return set(value["active_cuda_device_ids"])

    baseline_events = _kernel_events(
        baseline_sqlite,
        window(baseline_record)["intervals"],
        active_devices(baseline_record),
    )
    candidate_events = _kernel_events(
        candidate_sqlite,
        window(candidate_record)["intervals"],
        active_devices(candidate_record),
    )
    baseline_fused = _event_group_summary(
        baseline_events, lambda name: bool(FUSED_POST_RE.search(name)), degree
    )
    candidate_fused = _event_group_summary(
        candidate_events, lambda name: bool(FUSED_POST_RE.search(name)), degree
    )
    if baseline_fused["raw_total"] != 0:
        raise ValueError("baseline unexpectedly launched the fused post-A2A kernel")
    if candidate_fused["raw_total"] == 0:
        raise ValueError("candidate did not launch the fused post-A2A kernel")
    baseline_nccl = _nccl_a2a_summary(baseline_events, degree)
    candidate_nccl = _nccl_a2a_summary(candidate_events, degree)
    if baseline_nccl["raw_total"] == 0 or candidate_nccl["raw_total"] == 0:
        raise ValueError("NCCL SendRecv/AllToAll kernels were not captured")

    baseline_metrics = _metric_summary(baseline_record)
    candidate_metrics = _metric_summary(candidate_record)
    scalar_names = (
        "scheduler_observed_wall_ms",
        "dit_observed_wall_ms",
        "vae_observed_wall_ms",
        "scheduler_unclassified_observed_ms",
        "dit_cuda_ms",
        "vae_cuda_ms",
        "kernel_per_chunk",
        "cuda_api_per_chunk",
        "launch_api_per_chunk",
    )
    scalar_comparison = _scalar_comparison(
        {name: baseline_metrics[name] for name in scalar_names},
        {name: candidate_metrics[name] for name in scalar_names},
    )
    name_deltas = _name_deltas(baseline_events, candidate_events)

    return {
        "schema_version": "minwm-s3-post-nsys-comparison/v1",
        "comparison_contract": {
            "sp_degree": degree,
            "allocated_gpu_count": 8,
            "kv_cache_num_frames": 45,
            "precondition_warmup_chunks": 20,
            "discard_chunks": 1,
            "exact_stable_chunks": 10,
            "precision": "bf16",
            "lane_flags": {
                "baseline": {"MINWM_FUSED_POST_A2A_ROPE_CACHE": "0"},
                "candidate": {"MINWM_FUSED_POST_A2A_ROPE_CACHE": "1"},
            },
        },
        "run_ids": {
            "baseline": baseline_record["run_id"],
            "candidate": candidate_record["run_id"],
        },
        "profiler_off_headline": _scalar_comparison(
            _off_metric_summary(baseline_off_summary),
            _off_metric_summary(candidate_off_summary),
        ),
        "profiler_on_scalar_metrics": scalar_comparison,
        "short_kernel_buckets_per_chunk": {
            "baseline": baseline_metrics["short_kernel_buckets_per_chunk"],
            "candidate": candidate_metrics["short_kernel_buckets_per_chunk"],
        },
        "api_boundary_evidence": {
            "attribution_policy": API_BOUNDARY_ATTRIBUTION_POLICY,
            "baseline": {
                "cuda_api_count": baseline_metrics["cuda_api_boundary_evidence"],
                "kernel_launch_api_count": baseline_metrics[
                    "launch_api_boundary_evidence"
                ],
            },
            "candidate": {
                "cuda_api_count": candidate_metrics["cuda_api_boundary_evidence"],
                "kernel_launch_api_count": candidate_metrics[
                    "launch_api_boundary_evidence"
                ],
            },
        },
        "gpu_metrics": {
            "baseline": {
                key: baseline_metrics[key]
                for key in ("gpu_kernel_busy", "sm_active", "tensor_active", "dram")
            },
            "candidate": {
                key: candidate_metrics[key]
                for key in ("gpu_kernel_busy", "sm_active", "tensor_active", "dram")
            },
        },
        "post_a2a_rope_cache_kernels": {
            "attribution_note": (
                "The Triton fused kernel has a semantic name. Baseline eager RoPE/"
                "cache pointwise and copy launches may have generic generated names, "
                "so candidate-removed names are retained as differential evidence."
            ),
            "fused_exact": {
                "baseline": baseline_fused,
                "candidate": candidate_fused,
            },
            "semantic_name_matches": {
                "baseline": _event_group_summary(
                    baseline_events,
                    lambda name: bool(ROPE_CACHE_RE.search(name)),
                    degree,
                ),
                "candidate": _event_group_summary(
                    candidate_events,
                    lambda name: bool(ROPE_CACHE_RE.search(name)),
                    degree,
                ),
            },
            "candidate_removed_launches_by_name": [
                item for item in name_deltas if item["candidate_minus_baseline"] < 0
            ],
        },
        "nccl_a2a": {"baseline": baseline_nccl, "candidate": candidate_nccl},
        "kernel_name_deltas": name_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, choices=(2, 4), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--candidate-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-off-summary", type=Path, required=True)
    parser.add_argument("--candidate-off-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = compare(
        args.degree,
        _load_record(args.baseline),
        _load_record(args.candidate),
        args.baseline_sqlite,
        args.candidate_sqlite,
        _load_off_summary(args.baseline_off_summary),
        _load_off_summary(args.candidate_off_summary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
