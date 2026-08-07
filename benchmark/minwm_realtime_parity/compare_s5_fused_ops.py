#!/usr/bin/env python3
"""Validate and compare the S5 stacked-binary exact-window Nsight matrix."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from measurement import API_BOUNDARY_ATTRIBUTION_POLICY, validate_measurement
from measurement_tool import require_complete_stable_nsys


PRIMARY_CONFIGS = ("000", "100", "010", "001", "111")
PAIR_CONFIGS = ("110", "101", "011")
SCALAR_METRICS = (
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
    "dit_cuda_ms",
    "vae_cuda_ms",
    "kernel_per_chunk",
    "cuda_api_per_chunk",
    "launch_api_per_chunk",
    "gpu_kernel_busy_mean_pct",
)
WALL_METRICS = ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms")
ADDITIVE_COUNT_METRICS = ("kernel_per_chunk", "launch_api_per_chunk")


def _available(metric: Any, name: str) -> Any:
    if not isinstance(metric, dict) or metric.get("status") != "available":
        raise ValueError(f"{name} is unavailable: {metric}")
    return metric["value"]


def _assert_record(record: dict[str, Any], degree: int) -> None:
    validate_measurement(record)
    require_complete_stable_nsys(record)
    if record["mode"] != "profiler_on":
        raise ValueError(f"expected profiler_on, got {record['mode']!r}")
    workload = record["workload"]
    expected = {
        "sp_degree": degree,
        "precondition_warmup_chunks": 20,
        "warmup_chunks": 1,
        "measured_chunks": 10,
        "precision": "bf16",
        "dmd_forwards_per_chunk": 4,
        "clean_cache_forwards_per_chunk": 1,
    }
    for name, value in expected.items():
        if workload.get(name) != value:
            raise ValueError(
                f"workload.{name}={workload.get(name)!r}, expected {value!r}"
            )
    if record["comparison_contract"]["kv_cache_num_frames"] != 45:
        raise ValueError("comparison_contract.kv_cache_num_frames must be 45")
    gpu = record["provenance"]["gpu"]
    if gpu["count"] != degree or gpu["allocated_count"] != 8:
        raise ValueError(f"expected active/allocated GPUs {degree}/8, got {gpu}")

    on = record["metrics"]["profiler_on"]
    observed = on["observed_wall_with_profiler_overhead"]
    for name in WALL_METRICS:
        if _available(observed[name], name)["count"] != 10:
            raise ValueError(f"{name}.count must be 10")
    for name in ("dit_cuda_ms", "vae_cuda_ms"):
        if _available(on[name], name)["count"] != 10:
            raise ValueError(f"{name}.count must be 10")
    for name in (
        "kernel_count",
        "cuda_api_count",
        "kernel_launch_api_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
        "stable_window_coverage",
        "capture_coverage",
    ):
        _available(on[name], name)
    for name in ("cuda_api_count", "kernel_launch_api_count"):
        value = _available(on[name], name)
        if value["boundary_attribution_policy"] != API_BOUNDARY_ATTRIBUTION_POLICY:
            raise ValueError(f"{name} uses a non-canonical boundary policy")

    expected_devices = list(range(degree))
    for name in ("sm_active", "tensor_active", "dram"):
        value = _available(on["gpu_metrics"][name], f"gpu_metrics.{name}")
        expected_coverage = {
            "collected_target_count": 8,
            "allocated_target_count": 8,
            "active_target_count": degree,
        }
        for field, expected_value in expected_coverage.items():
            if value[field] != expected_value:
                raise ValueError(
                    f"gpu_metrics.{name}.{field}={value[field]!r}, "
                    f"expected {expected_value}"
                )
        if value["active_cuda_device_ids"] != expected_devices:
            raise ValueError(f"gpu_metrics.{name} active CUDA mapping is incomplete")
        if len(value["active_pw_gpu_ids"]) != degree:
            raise ValueError(
                f"gpu_metrics.{name} active PerfWorks coverage is incomplete"
            )
        if len(value["target_mapping"]) != 8:
            raise ValueError(f"gpu_metrics.{name} target mapping must cover all 8 GPUs")
        per_device = value["per_device_per_chunk_sample_count"]
        if set(per_device) != {str(index) for index in expected_devices}:
            raise ValueError(f"gpu_metrics.{name} per-device coverage is incomplete")
        if any(len(per_chunk) != 10 for per_chunk in per_device.values()):
            raise ValueError(f"gpu_metrics.{name} must cover exact 10 chunks per GPU")

    window = _available(on["stable_window_coverage"], "stable_window_coverage")
    expected_indices = list(range(1, 11))
    if window["expected_stable_chunk_indices"] != expected_indices:
        raise ValueError("expected stable chunk indices must be 1..10")
    if window["observed_stable_chunk_indices"] != expected_indices:
        raise ValueError("observed stable chunk indices must be 1..10")
    if window["normalization_denominator"] != 10 or len(window["intervals"]) != 10:
        raise ValueError("stable window must contain exactly 10 ranges")


def _metric_summary(record: dict[str, Any]) -> dict[str, Any]:
    on = record["metrics"]["profiler_on"]
    observed = on["observed_wall_with_profiler_overhead"]
    kernel = _available(on["kernel_count"], "kernel_count")
    cuda_api = _available(on["cuda_api_count"], "cuda_api_count")
    launch_api = _available(on["kernel_launch_api_count"], "kernel_launch_api_count")
    busy = _available(on["gpu_kernel_busy"], "gpu_kernel_busy")
    return {
        "scheduler_chunk_wall_ms": _available(
            observed["scheduler_chunk_wall_ms"], "scheduler_chunk_wall_ms"
        )["mean"],
        "dit_wall_ms": _available(observed["dit_wall_ms"], "dit_wall_ms")["mean"],
        "vae_wall_ms": _available(observed["vae_wall_ms"], "vae_wall_ms")["mean"],
        "dit_cuda_ms": _available(on["dit_cuda_ms"], "dit_cuda_ms")["mean"],
        "vae_cuda_ms": _available(on["vae_cuda_ms"], "vae_cuda_ms")["mean"],
        "kernel_per_chunk": kernel["per_stable_chunk"],
        "cuda_api_per_chunk": cuda_api["total_per_chunk"],
        "launch_api_per_chunk": launch_api["total_per_chunk"],
        "gpu_kernel_busy_mean_pct": busy["mean_pct"],
        "kernel_count": kernel,
        "cuda_api_count": cuda_api,
        "kernel_launch_api_count": launch_api,
        "short_kernel_buckets": _available(
            on["short_kernel_buckets"], "short_kernel_buckets"
        ),
        "gpu_kernel_busy": busy,
        "gpu_metrics": {
            name: _available(on["gpu_metrics"][name], f"gpu_metrics.{name}")
            for name in ("sm_active", "tensor_active", "dram")
        },
        "capture_coverage": _available(on["capture_coverage"], "capture_coverage"),
        "stable_window_coverage": _available(
            on["stable_window_coverage"], "stable_window_coverage"
        ),
    }


def _fused_post_kernel_count(
    sqlite_path: Path, record: dict[str, Any]
) -> dict[str, Any]:
    intervals = _available(
        record["metrics"]["profiler_on"]["stable_window_coverage"],
        "stable_window_coverage",
    )["intervals"]
    active_devices = set(
        _available(
            record["metrics"]["profiler_on"]["gpu_metrics"]["sm_active"],
            "gpu_metrics.sm_active",
        )["active_cuda_device_ids"]
    )
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    with sqlite3.connect(sqlite_path) as connection:
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info('{table}')")
        }
        for candidate in ("demangledName", "shortName", "mangledName", "name"):
            if candidate in columns:
                name_column = candidate
                break
        else:
            raise ValueError(f"{table} has no supported name column")
        rows = list(
            connection.execute(
                f"SELECT deviceId, start, end, {name_column} FROM {table}"
            )
        )
        integer_ids = {int(row[3]) for row in rows if isinstance(row[3], int)}
        resolved: dict[Any, str] = {
            row[3]: str(row[3]) for row in rows if not isinstance(row[3], int)
        }
        if integer_ids:
            placeholders = ",".join("?" for _ in integer_ids)
            for identifier, value in connection.execute(
                f"SELECT id, value FROM StringIds WHERE id IN ({placeholders})",
                tuple(sorted(integer_ids)),
            ):
                resolved[int(identifier)] = str(value)
    count = 0
    names: dict[str, int] = {}
    for device, start, end, raw_name in rows:
        if int(device) not in active_devices:
            continue
        if not any(
            int(start) >= int(interval["start_ns"])
            and int(end) <= int(interval["end_ns"])
            for interval in intervals
        ):
            continue
        name = resolved[raw_name]
        if "fused_rope_cache_update" in name.lower():
            count += 1
            names[name] = names.get(name, 0) + 1
    return {
        "raw_total": count,
        "per_stable_chunk": count / 10,
        "kernel_names": names,
    }


def _delta(after: float, before: float) -> dict[str, float | None]:
    return {
        "absolute": after - before,
        "pct": (after / before - 1.0) * 100.0 if before else None,
    }


def _interaction_residual(
    summaries: dict[str, dict[str, Any]], config: str, components: tuple[str, ...]
) -> dict[str, Any]:
    base = summaries["000"]
    residuals = {}
    for metric in SCALAR_METRICS:
        base_value = float(base[metric])
        combined_delta = float(summaries[config][metric]) - base_value
        component_delta = sum(
            float(summaries[item][metric]) - base_value for item in components
        )
        absolute = combined_delta - component_delta
        residuals[metric] = {
            "combined_delta": combined_delta,
            "component_delta_sum": component_delta,
            "absolute": absolute,
            "percentage_points_of_000": (
                absolute / base_value * 100.0 if base_value else None
            ),
        }
    return residuals


def compare(root: Path, degree: int) -> dict[str, Any]:
    configs = list(PRIMARY_CONFIGS)
    configs.extend(
        config
        for config in PAIR_CONFIGS
        if (root / f"nsys-sp{degree}-{config}" / f"sp{degree}").is_dir()
    )
    records = {}
    paths = {}
    for config in configs:
        lane = root / f"nsys-sp{degree}-{config}" / f"sp{degree}" / "profiler-on"
        path = lane / "measurement.json"
        record = json.loads(path.read_text())
        _assert_record(record, degree)
        records[config] = record
        paths[config] = {
            "measurement": str(path),
            "sqlite": str(lane / f"sp{degree}.sqlite"),
            "nsys_rep": str(lane / f"sp{degree}.nsys-rep"),
            "server_log": str(lane / "server.log"),
        }

    provenance_keys = ("sglang_commit", "minwm_commit", "container_image")
    base_provenance = records["000"]["provenance"]
    for config, record in records.items():
        for key in provenance_keys:
            if record["provenance"][key] != base_provenance[key]:
                raise ValueError(f"{config}: provenance mismatch for {key}")

    summaries = {config: _metric_summary(record) for config, record in records.items()}
    for config, record in records.items():
        fused_post = _fused_post_kernel_count(Path(paths[config]["sqlite"]), record)
        summaries[config]["fused_post_kernel"] = fused_post
        if config[1] == "1" and fused_post["raw_total"] == 0:
            raise ValueError(
                f"{config}: post-A2A fusion was requested but its kernel did not launch"
            )
        if config[1] == "0" and fused_post["raw_total"] != 0:
            raise ValueError(
                f"{config}: post-A2A fusion kernel launched while the flag was disabled"
            )
    deltas = {
        config: {
            metric: _delta(
                float(summaries[config][metric]), float(summaries["000"][metric])
            )
            for metric in SCALAR_METRICS
        }
        for config in configs
        if config != "000"
    }
    triple = _interaction_residual(summaries, "111", ("100", "010", "001"))
    wall_trigger_metrics = [
        name
        for name in WALL_METRICS
        if abs(float(triple[name]["percentage_points_of_000"])) > 0.5
    ]
    count_trigger_metrics = [
        name
        for name in ADDITIVE_COUNT_METRICS
        if abs(float(triple[name]["absolute"])) > 1e-9
    ]
    pair_residuals = {}
    for config, components in {
        "110": ("100", "010"),
        "101": ("100", "001"),
        "011": ("010", "001"),
    }.items():
        if config in summaries:
            pair_residuals[config] = _interaction_residual(
                summaries, config, components
            )

    return {
        "schema_version": "minwm-s5-fused-ops-nsys/v1",
        "comparison_contract": {
            "sp_degree": degree,
            "allocated_gpu_count": 8,
            "kv_cache_num_frames": 45,
            "precondition_warmup_chunks": 20,
            "discard_chunks": 1,
            "exact_stable_chunks": 10,
            "precision": "bf16",
            "dmd_forwards_per_chunk": 4,
            "clean_cache_forwards_per_chunk": 1,
            "wall_residual_trigger_abs_percentage_points": 0.5,
            "count_residual_trigger": "any non-zero per-chunk residual",
        },
        "paths": paths,
        "metrics": summaries,
        "deltas_from_000": deltas,
        "triple_interaction_residual": triple,
        "pair_interaction_residuals": pair_residuals,
        "pairwise_required": bool(wall_trigger_metrics or count_trigger_metrics),
        "pairwise_trigger": {
            "wall_metrics": wall_trigger_metrics,
            "count_metrics": count_trigger_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--degree", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.root, args.degree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["pairwise_trigger"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
