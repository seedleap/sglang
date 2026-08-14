#!/usr/bin/env python3
"""Summarize the MinWM local-shard QKV Nsight Compute sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LANE_PATTERN = re.compile(r"m(?P<m>\d+)-gpu(?P<gpu>\d+)-(?P<mode>baseline|fused)")
GEMM_PREFIXES = ("nvjet", "cutlass")
PERCENT_METRICS = {
    "tensor_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "active_warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
SCALAR_METRICS = {
    "registers_per_thread": "launch__registers_per_thread",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "local_load_instructions": "sass__inst_executed_local_loads",
    "local_store_instructions": "sass__inst_executed_local_stores",
}
BYTE_METRICS = {
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
}
BYTE_SCALES = {
    "byte": 1.0,
    "Kbyte": 1_000.0,
    "Mbyte": 1_000_000.0,
    "Gbyte": 1_000_000_000.0,
}


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _mean(records: list[dict], key: str) -> float:
    return statistics.fmean(record[key] for record in records)


def _parse_ncu(path: Path) -> dict:
    match = LANE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized lane name: {path.name}")
    rows = list(csv.reader(path.open(errors="replace")))
    if len(rows) < 3:
        raise ValueError(f"Incomplete NCU CSV: {path}")
    header, units = rows[0], rows[1]
    index = {name: position for position, name in enumerate(header)}
    kernels = []
    for row in rows[2:]:
        if len(row) != len(header) or not row[index["ID"]]:
            continue
        kernel = {
            "name": row[index["Kernel Name"]],
            "duration_us": _number(row[index["gpu__time_duration.sum"]]),
        }
        for output_name, metric_name in PERCENT_METRICS.items():
            kernel[output_name] = _number(row[index[metric_name]])
        for output_name, metric_name in SCALAR_METRICS.items():
            kernel[output_name] = _number(row[index[metric_name]])
        for output_name, metric_name in BYTE_METRICS.items():
            unit = units[index[metric_name]]
            kernel[output_name] = _number(row[index[metric_name]]) * BYTE_SCALES[unit]
        kernels.append(kernel)

    mode = match.group("mode")
    gemms = [kernel for kernel in kernels if kernel["name"].startswith(GEMM_PREFIXES)]
    if len(gemms) != (3 if mode == "baseline" else 1):
        raise ValueError(f"Unexpected GEMM count in {path}: {len(gemms)}")
    value_copy_us = kernels[-1]["duration_us"] if mode == "fused" else 0.0
    total_us = sum(kernel["duration_us"] for kernel in kernels)
    gemm_us = sum(kernel["duration_us"] for kernel in gemms)
    result = {
        "m": int(match.group("m")),
        "gpu": int(match.group("gpu")),
        "mode": mode,
        "kernel_count": len(kernels),
        "total_duration_us": total_us,
        "gemm_duration_us": gemm_us,
        "qk_norm_duration_us": total_us - gemm_us - value_copy_us,
        "value_copy_duration_us": value_copy_us,
        "gemm": {},
    }
    for output_name in (*PERCENT_METRICS, *SCALAR_METRICS):
        result["gemm"][output_name] = _mean(gemms, output_name)
    for output_name in BYTE_METRICS:
        result["gemm"][output_name] = sum(kernel[output_name] for kernel in gemms)
    return result


def _parse_timing(path: Path) -> dict:
    payload = json.loads(path.read_text())
    baseline_ms = payload["timing"]["baseline"]["mean_ms"]
    fused_ms = payload["timing"]["fused"]["mean_ms"]
    return {
        "path": path.name,
        "m": payload["shape"]["M"],
        "profiled_mode": payload["mode"],
        "order": payload["order"],
        "baseline_mean_ms": baseline_ms,
        "fused_mean_ms": fused_ms,
        "improvement_pct": (baseline_ms - fused_ms) / baseline_ms * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lanes = [_parse_ncu(path) for path in sorted(args.root.glob("*-raw.csv"))]
    timings = [
        _parse_timing(path) for path in sorted(args.root.glob("*-timing.json"))
    ]
    result = {
        "schema_version": "minwm-qkv-local-boundary-ncu-summary/v1",
        "lanes": lanes,
        "timings": timings,
        "by_m": {},
    }
    for m in sorted({lane["m"] for lane in lanes}):
        baseline = [lane for lane in lanes if lane["m"] == m and lane["mode"] == "baseline"]
        fused = [lane for lane in lanes if lane["m"] == m and lane["mode"] == "fused"]
        if len(baseline) != 2 or len(fused) != 2:
            raise ValueError(f"M={m} requires two baseline and two fused NCU lanes")
        aggregate = {}
        for name, records in (("baseline", baseline), ("fused", fused)):
            aggregate[name] = {
                key: _mean(records, key)
                for key in (
                    "kernel_count",
                    "total_duration_us",
                    "gemm_duration_us",
                    "qk_norm_duration_us",
                    "value_copy_duration_us",
                )
            }
            aggregate[name]["gemm"] = {
                key: statistics.fmean(record["gemm"][key] for record in records)
                for key in records[0]["gemm"]
            }
        baseline_total = aggregate["baseline"]["total_duration_us"]
        fused_total = aggregate["fused"]["total_duration_us"]
        baseline_gemm = aggregate["baseline"]["gemm_duration_us"]
        fused_gemm = aggregate["fused"]["gemm_duration_us"]
        timing_samples = [item for item in timings if item["m"] == m]
        result["by_m"][str(m)] = {
            **aggregate,
            "kernel_boundary_improvement_pct": (baseline_total - fused_total)
            / baseline_total
            * 100.0,
            "gemm_improvement_pct": (baseline_gemm - fused_gemm)
            / baseline_gemm
            * 100.0,
            "decomposition_us": {
                "gemm_saving": baseline_gemm - fused_gemm,
                "qk_norm_penalty": aggregate["fused"]["qk_norm_duration_us"]
                - aggregate["baseline"]["qk_norm_duration_us"],
                "value_copy_cost": aggregate["fused"]["value_copy_duration_us"],
                "net_saving": baseline_total - fused_total,
            },
            "paired_event_improvement_pct": {
                "count": len(timing_samples),
                "mean": _mean(timing_samples, "improvement_pct"),
                "median": statistics.median(
                    item["improvement_pct"] for item in timing_samples
                ),
                "samples": [item["improvement_pct"] for item in timing_samples],
            },
        }

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
