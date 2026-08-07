#!/usr/bin/env python3
"""Compare exact-window Nsight records for the MinWM timestep hoist A/B."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from measurement import API_BOUNDARY_ATTRIBUTION_POLICY, validate_measurement
from measurement_tool import require_complete_stable_nsys


INDEX_KERNEL_RE = re.compile(
    r"(?:index_elementwise|gpu_index|index_kernel|gather|take_along)",
    re.IGNORECASE,
)
KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"


def _load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    validate_measurement(record)
    require_complete_stable_nsys(record)
    return record


def _available_value(metric: dict[str, Any], name: str) -> Any:
    if metric.get("status") != "available":
        raise ValueError(f"{name} is unavailable: {metric}")
    return metric["value"]


def _assert_off_resume_contract(
    record: dict[str, Any], expected_sglang_ref: str, variant: str
) -> None:
    if record["mode"] != "profiler_off":
        raise ValueError(f"expected profiler_off, got {record['mode']}")
    if record["provenance"]["sglang_commit"] != expected_sglang_ref:
        raise ValueError("profiler-off resume source has an unexpected sglang commit")
    workload = record["workload"]
    gpu = record["provenance"]["gpu"]
    expected = {
        "sp_degree": 2,
        "warmup_chunks": 20,
        "measured_chunks": 200,
        "precision": "bf16",
        "fast_lane": True,
        "dmd_forwards_per_chunk": 4,
        "clean_cache_forwards_per_chunk": 1,
    }
    for key, value in expected.items():
        if workload[key] != value:
            raise ValueError(f"workload.{key}={workload[key]!r}, expected {value!r}")
    if gpu["count"] != 2 or gpu["allocated_count"] != 8:
        raise ValueError(f"expected active/allocated GPUs 2/8, got {gpu}")
    if record["comparison_contract"]["kv_cache_num_frames"] != 45:
        raise ValueError("comparison_contract.kv_cache_num_frames must be 45")
    expected_run_label = f"-temb-hoist-{variant}-sp2-off-"
    if expected_run_label not in record["run_id"]:
        raise ValueError(f"profiler-off resume source is not labeled {variant}")
    if record["profile_name"] != "bf16-fast-sp2":
        raise ValueError("profiler-off resume source has an unexpected profile name")
    off = record["metrics"]["profiler_off"]
    for name in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"):
        value = _available_value(off[name], name)
        if value["count"] != 200:
            raise ValueError(f"{name}: expected 200 complete profiler-off samples")


def _assert_historical_runner_flag_guard(contents: str, variant: str) -> None:
    expected_by_variant = {
        "legacy": (
            'if [[ "${lane}" == "legacy" ]]; then',
            "export MINWM_HOIST_TIMESTEP_MODULATION=0",
            'export MINWM_S0_RUN_LABEL="temb-hoist-${lane}"',
            'bash "${SCRIPT_DIR}/run_s0_measurement.sh"',
        ),
        "candidate": (
            "export MINWM_HOIST_TIMESTEP_MODULATION=1",
            "export MINWM_S0_RUN_LABEL=temb-hoist-candidate",
            'bash "${SCRIPT_DIR}/run_s0_measurement.sh"',
        ),
    }
    try:
        expected = expected_by_variant[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported historical variant: {variant}") from exc
    search_start = 0
    for item in expected:
        try:
            position = contents.index(item, search_start)
        except ValueError as exc:
            raise ValueError(
                f"historical {variant} runner lost its flag guard or reordered it"
            ) from exc
        search_start = position + len(item)


def _assert_contract(record: dict[str, Any]) -> None:
    if record["mode"] != "profiler_on":
        raise ValueError(f"expected profiler_on, got {record['mode']}")
    workload = record["workload"]
    contract = record["comparison_contract"]
    gpu = record["provenance"]["gpu"]
    expected = {
        "sp_degree": 2,
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
    if contract["kv_cache_num_frames"] != 45:
        raise ValueError("comparison_contract.kv_cache_num_frames must be 45")
    if gpu["count"] != 2 or gpu["allocated_count"] != 8:
        raise ValueError(f"expected active/allocated GPUs 2/8, got {gpu}")

    on = record["metrics"]["profiler_on"]
    for name in ("dit_wall_ms", "vae_wall_ms", "dit_cuda_ms", "vae_cuda_ms"):
        value = _available_value(on[name], name)
        if value["count"] != 10:
            raise ValueError(f"{name}: expected 10 complete stage samples")
    for name in (
        "kernel_count",
        "cuda_api_count",
        "kernel_launch_api_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
        "stable_window_coverage",
        "capture_coverage",
    ):
        _available_value(on[name], name)
    for name in ("cuda_api_count", "kernel_launch_api_count"):
        value = _available_value(on[name], name)
        if value["boundary_attribution_policy"] != API_BOUNDARY_ATTRIBUTION_POLICY:
            raise ValueError(f"{name}: unexpected API boundary attribution policy")
    for name in ("sm_active", "tensor_active"):
        value = _available_value(on["gpu_metrics"][name], f"gpu_metrics.{name}")
        if value["collected_target_count"] != 8:
            raise ValueError(f"{name}: expected all 8 collected targets")
        if value["allocated_target_count"] != 8:
            raise ValueError(f"{name}: allocated target count is not 8")
        if value["active_target_count"] != 2:
            raise ValueError(f"{name}: active target count is not 2")
        if value["active_cuda_device_ids"] != [0, 1]:
            raise ValueError(f"{name}: active CUDA devices are not [0, 1]")
        if len(value["active_pw_gpu_ids"]) != 2:
            raise ValueError(f"{name}: expected two active PerfWorks targets")
        if len(value["target_mapping"]) != 8:
            raise ValueError(f"{name}: target mapping does not cover all 8 GPUs")
        if set(value["per_device_per_chunk_sample_count"]) != {"0", "1"}:
            raise ValueError(f"{name}: per-device coverage is not SP2")
        if any(
            len(per_chunk) != 10
            for per_chunk in value["per_device_per_chunk_sample_count"].values()
        ):
            raise ValueError(f"{name}: every active device must cover 10 chunks")

    window = _available_value(on["stable_window_coverage"], "stable_window")
    if window["expected_stable_chunk_indices"] != list(range(1, 11)):
        raise ValueError("expected stable chunk indices must be 1..10")
    if window["observed_stable_chunk_indices"] != list(range(1, 11)):
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


def _kernel_name_counts(
    sqlite_path: Path,
    intervals: list[dict[str, Any]],
    active_devices: set[int],
) -> Counter[str]:
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

    counts: Counter[str] = Counter()
    boundary_overlap = 0
    for raw_device, raw_start, raw_end, raw_name in rows:
        device = int(raw_device)
        if device not in active_devices:
            continue
        start, end = int(raw_start), int(raw_end)
        contained = [
            interval
            for interval in intervals
            if start >= int(interval["start_ns"]) and end <= int(interval["end_ns"])
        ]
        overlaps = any(
            start < int(interval["end_ns"]) and end > int(interval["start_ns"])
            for interval in intervals
        )
        if len(contained) == 1:
            counts[names[raw_name]] += 1
        elif overlaps:
            boundary_overlap += 1
    if boundary_overlap:
        raise ValueError(
            f"{sqlite_path}: {boundary_overlap} kernel events cross stable boundaries"
        )
    return counts


def _metric_summary(record: dict[str, Any]) -> dict[str, Any]:
    on = record["metrics"]["profiler_on"]
    cuda_api = _available_value(on["cuda_api_count"], "cuda_api_count")
    launch_api = _available_value(
        on["kernel_launch_api_count"], "kernel_launch_api_count"
    )
    return {
        "dit_wall_ms": _available_value(on["dit_wall_ms"], "dit_wall_ms")["mean"],
        "vae_wall_ms": _available_value(on["vae_wall_ms"], "vae_wall_ms")["mean"],
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


def compare(
    legacy_record: dict[str, Any],
    candidate_record: dict[str, Any],
    legacy_sqlite: Path,
    candidate_sqlite: Path,
) -> dict[str, Any]:
    for record in (legacy_record, candidate_record):
        _assert_contract(record)
    required_equal = (
        "sglang_commit",
        "minwm_commit",
        "container_image",
    )
    for key in required_equal:
        if legacy_record["provenance"][key] != candidate_record["provenance"][key]:
            raise ValueError(f"provenance mismatch: {key}")

    def window(record: dict[str, Any]) -> dict[str, Any]:
        return record["metrics"]["profiler_on"]["stable_window_coverage"]["value"]

    def active_devices(record: dict[str, Any]) -> set[int]:
        value = record["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"]["value"]
        return set(value["active_cuda_device_ids"])

    legacy_names = _kernel_name_counts(
        legacy_sqlite, window(legacy_record)["intervals"], active_devices(legacy_record)
    )
    candidate_names = _kernel_name_counts(
        candidate_sqlite,
        window(candidate_record)["intervals"],
        active_devices(candidate_record),
    )
    names = sorted(set(legacy_names) | set(candidate_names))
    deltas = [
        {
            "name": name,
            "legacy_raw_total": legacy_names[name],
            "candidate_raw_total": candidate_names[name],
            "candidate_minus_legacy": candidate_names[name] - legacy_names[name],
            "legacy_per_chunk": legacy_names[name] / 10,
            "candidate_per_chunk": candidate_names[name] / 10,
        }
        for name in names
        if legacy_names[name] != candidate_names[name]
    ]
    deltas.sort(key=lambda item: (-abs(item["candidate_minus_legacy"]), item["name"]))
    suspected = [item for item in deltas if INDEX_KERNEL_RE.search(item["name"])]

    legacy_metrics = _metric_summary(legacy_record)
    candidate_metrics = _metric_summary(candidate_record)
    scalar_names = (
        "dit_wall_ms",
        "vae_wall_ms",
        "dit_cuda_ms",
        "vae_cuda_ms",
        "kernel_per_chunk",
        "cuda_api_per_chunk",
        "launch_api_per_chunk",
    )
    scalar_comparison = {}
    for name in scalar_names:
        before = float(legacy_metrics[name])
        after = float(candidate_metrics[name])
        scalar_comparison[name] = {
            "legacy": before,
            "candidate": after,
            "candidate_minus_legacy": after - before,
            "candidate_delta_pct": (after / before - 1.0) * 100.0 if before else None,
        }

    return {
        "schema_version": "minwm-temb-hoist-nsys-comparison/v1",
        "comparison_contract": {
            "sp_degree": 2,
            "allocated_gpu_count": 8,
            "kv_cache_num_frames": 45,
            "precondition_warmup_chunks": 20,
            "discard_chunks": 1,
            "exact_stable_chunks": 10,
            "precision": "bf16",
            "dmd_forwards_per_chunk": 4,
            "clean_cache_forwards_per_chunk": 1,
            "static_upper_bound_removed_block_materializations_per_chunk": 145,
        },
        "run_ids": {
            "legacy": legacy_record["run_id"],
            "candidate": candidate_record["run_id"],
        },
        "scalar_metrics": scalar_comparison,
        "short_kernel_buckets_per_chunk": {
            "legacy": legacy_metrics["short_kernel_buckets_per_chunk"],
            "candidate": candidate_metrics["short_kernel_buckets_per_chunk"],
        },
        "api_boundary_evidence": {
            "attribution_policy": API_BOUNDARY_ATTRIBUTION_POLICY,
            "legacy": {
                "cuda_api_count": legacy_metrics["cuda_api_boundary_evidence"],
                "kernel_launch_api_count": legacy_metrics[
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
            "legacy": {
                key: legacy_metrics[key]
                for key in ("gpu_kernel_busy", "sm_active", "tensor_active", "dram")
            },
            "candidate": {
                key: candidate_metrics[key]
                for key in ("gpu_kernel_busy", "sm_active", "tensor_active", "dram")
            },
        },
        "kernel_name_deltas": deltas,
        "suspected_indexing_materialization_deltas": suspected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--legacy-sqlite", type=Path, required=True)
    parser.add_argument("--candidate-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = compare(
        _load_record(args.legacy),
        _load_record(args.candidate),
        args.legacy_sqlite,
        args.candidate_sqlite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
