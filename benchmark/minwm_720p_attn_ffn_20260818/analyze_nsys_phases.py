#!/usr/bin/env python3
"""Attribute CUDA kernels to complete MinWM DiT and block-phase NVTX ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

PHASE_PREFIX = "minwm.block."
DIT_BLOCKS_LABEL = "minwm.dit.blocks"
DIT_FORWARD_LABEL = "minwm.dit.forward"
REQUIRED_PHASES = {"self_attention", "cross_attention", "ffn"}


def kernel_family(name: str) -> str:
    lower = name.lower()
    if "flash" in lower or "fmha" in lower or "attention" in lower:
        return "attention"
    if "gemm" in lower or "cutlass" in lower or "nvjet" in lower:
        return "gemm"
    if "norm" in lower or "reduce" in lower:
        return "normalization_reduction"
    if "elementwise" in lower or "pointwise" in lower or "triton_poi" in lower:
        return "pointwise"
    if "copy" in lower or "memcpy" in lower:
        return "copy"
    return "other"


def load_nvtx_records(
    cursor: sqlite3.Cursor, label: str, *, prefix: bool = False
) -> dict[int, dict]:
    operator = "LIKE" if prefix else "="
    value = f"{label}%" if prefix else label
    rows = cursor.execute(
        f"""
        SELECT n.eventId, n.start, n.end, n.globalTid,
               COALESCE(n.text, s.value) AS label
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL
          AND COALESCE(n.text, s.value) {operator} ?
        ORDER BY n.start
        """,
        (value,),
    ).fetchall()
    records = {
        row["eventId"]: {
            "label": row["label"],
            "start": row["start"],
            "end": row["end"],
            "global_tid": row["globalTid"],
            "cpu_range_ns": row["end"] - row["start"],
            "kernels": {},
        }
        for row in rows
    }
    if not records:
        return records
    mapped = cursor.execute(
        f"""
        SELECT DISTINCT n.eventId, k.rowid AS kernel_rowid, k.start, k.end,
               k.deviceId,
               COALESCE(d.value, short.value, 'unknown') AS kernel_name
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds nvtx_string ON nvtx_string.id = n.textId
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime
          ON runtime.globalTid = n.globalTid
         AND runtime.start >= n.start
         AND runtime.start <= n.end
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = runtime.correlationId
        LEFT JOIN StringIds d ON d.id = k.demangledName
        LEFT JOIN StringIds short ON short.id = k.shortName
        WHERE n.end IS NOT NULL
          AND COALESCE(n.text, nvtx_string.value) {operator} ?
        ORDER BY n.eventId, k.start
        """,
        (value,),
    ).fetchall()
    for row in mapped:
        if row["eventId"] not in records:
            continue
        records[row["eventId"]]["kernels"][row["kernel_rowid"]] = {
            "start": row["start"],
            "end": row["end"],
            "device": row["deviceId"],
            "name": row["kernel_name"],
        }
    return records


def is_contained(inner: dict, outer: dict) -> bool:
    return (
        inner["global_tid"] == outer["global_tid"]
        and outer["start"] <= inner["start"]
        and outer["end"] >= inner["end"]
    )


def aggregate_ranges(records: dict[int, dict]) -> dict:
    kernel_ns = span_ns = kernel_calls = ranges_without_kernels = 0
    for record in records.values():
        kernels = list(record["kernels"].values())
        if not kernels:
            ranges_without_kernels += 1
            continue
        kernel_ns += sum(kernel["end"] - kernel["start"] for kernel in kernels)
        span_ns += max(kernel["end"] for kernel in kernels) - min(
            kernel["start"] for kernel in kernels
        )
        kernel_calls += len(kernels)
    return {
        "range_count": len(records),
        "ranges_without_kernels": ranges_without_kernels,
        "cpu_range_ns": sum(record["cpu_range_ns"] for record in records.values()),
        "gpu_kernel_ns": kernel_ns,
        "gpu_span_ns": span_ns,
        "kernel_calls": kernel_calls,
    }


def render_range_aggregate(item: dict) -> dict:
    return {
        "range_count": item["range_count"],
        "ranges_without_kernels": item["ranges_without_kernels"],
        "cpu_launch_range_ms_sum": item["cpu_range_ns"] / 1e6,
        "gpu_kernel_ms_sum": item["gpu_kernel_ns"] / 1e6,
        "gpu_span_ms_sum": item["gpu_span_ns"] / 1e6,
        "kernel_calls": item["kernel_calls"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-dit-block-ranges", type=int, default=1)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    all_forward_ranges = load_nvtx_records(cursor, DIT_FORWARD_LABEL)
    all_block_ranges = load_nvtx_records(cursor, DIT_BLOCKS_LABEL)
    all_phase_ranges = load_nvtx_records(cursor, PHASE_PREFIX, prefix=True)

    # A delayed capture can begin inside a range. Keep only fully nested
    # forward -> block-loop -> phase trees so a partial first/last forward
    # cannot inflate a numerator or denominator.
    block_ranges = {
        event_id: record
        for event_id, record in all_block_ranges.items()
        if any(is_contained(record, outer) for outer in all_forward_ranges.values())
    }
    forward_ranges = {
        event_id: record
        for event_id, record in all_forward_ranges.items()
        if any(is_contained(inner, record) for inner in block_ranges.values())
    }
    phase_ranges = {
        event_id: record
        for event_id, record in all_phase_ranges.items()
        if any(is_contained(record, outer) for outer in block_ranges.values())
    }
    if len(block_ranges) < args.min_dit_block_ranges:
        raise SystemExit(
            "too few complete MinWM DiT block-loop ranges for a stable profile: "
            f"found {len(block_ranges)}, require {args.min_dit_block_ranges}"
        )
    if len(forward_ranges) != len(block_ranges):
        raise SystemExit(
            "expected one complete DiT forward per block loop, found "
            f"{len(forward_ranges)} forwards and {len(block_ranges)} block loops"
        )

    block_totals = aggregate_ranges(block_ranges)
    forward_totals = aggregate_ranges(forward_ranges)
    if block_totals["ranges_without_kernels"] or not block_totals["gpu_kernel_ns"]:
        raise SystemExit(
            "one or more DiT block-loop ranges have no mapped CUDA kernels"
        )
    if forward_totals["ranges_without_kernels"] or not forward_totals["gpu_kernel_ns"]:
        raise SystemExit("one or more DiT forward ranges have no mapped CUDA kernels")

    aggregate: dict[str, dict] = defaultdict(
        lambda: {
            "range_count": 0,
            "ranges_without_kernels": 0,
            "cpu_range_ns": 0,
            "gpu_kernel_ns": 0,
            "gpu_span_ns": 0,
            "kernel_calls": 0,
            "families_ns": Counter(),
            "families_calls": Counter(),
        }
    )
    for record in phase_ranges.values():
        phase = record["label"].removeprefix(PHASE_PREFIX)
        item = aggregate[phase]
        item["range_count"] += 1
        item["cpu_range_ns"] += record["cpu_range_ns"]
        kernels = list(record["kernels"].values())
        if not kernels:
            item["ranges_without_kernels"] += 1
            continue
        item["gpu_kernel_ns"] += sum(
            kernel["end"] - kernel["start"] for kernel in kernels
        )
        item["gpu_span_ns"] += max(kernel["end"] for kernel in kernels) - min(
            kernel["start"] for kernel in kernels
        )
        item["kernel_calls"] += len(kernels)
        for kernel in kernels:
            family = kernel_family(kernel["name"])
            item["families_ns"][family] += kernel["end"] - kernel["start"]
            item["families_calls"][family] += 1

    if set(aggregate) != REQUIRED_PHASES:
        raise SystemExit(
            f"expected phases {sorted(REQUIRED_PHASES)}, found {sorted(aggregate)}"
        )
    expected_phase_ranges = len(block_ranges) * 30
    for phase, item in aggregate.items():
        if item["range_count"] != expected_phase_ranges:
            raise SystemExit(
                f"{phase}: expected {expected_phase_ranges} complete ranges, "
                f"found {item['range_count']}"
            )
        if item["ranges_without_kernels"] or not item["gpu_kernel_ns"]:
            raise SystemExit(f"{phase}: one or more ranges have no mapped CUDA kernels")

    total_kernel_ns = sum(item["gpu_kernel_ns"] for item in aggregate.values())
    total_span_ns = sum(item["gpu_span_ns"] for item in aggregate.values())
    rendered = {}
    for phase, item in sorted(aggregate.items()):
        count = item["range_count"]
        rendered[phase] = {
            "range_count": count,
            "expected_range_count": expected_phase_ranges,
            "range_coverage_fraction": count / expected_phase_ranges,
            "ranges_without_kernels": item["ranges_without_kernels"],
            "cpu_launch_range_ms_sum": item["cpu_range_ns"] / 1e6,
            "cpu_launch_range_ms_mean": item["cpu_range_ns"] / count / 1e6,
            "gpu_kernel_ms_sum": item["gpu_kernel_ns"] / 1e6,
            "gpu_kernel_ms_mean_per_range": item["gpu_kernel_ns"] / count / 1e6,
            "gpu_kernel_share_of_three_pct": (
                100 * item["gpu_kernel_ns"] / total_kernel_ns
            ),
            "gpu_kernel_share_of_dit_blocks_pct": (
                100 * item["gpu_kernel_ns"] / block_totals["gpu_kernel_ns"]
            ),
            "gpu_kernel_share_of_dit_forward_pct": (
                100 * item["gpu_kernel_ns"] / forward_totals["gpu_kernel_ns"]
            ),
            "gpu_span_ms_sum": item["gpu_span_ns"] / 1e6,
            "gpu_span_ms_mean_per_range": item["gpu_span_ns"] / count / 1e6,
            "gpu_span_share_of_three_pct": 100 * item["gpu_span_ns"] / total_span_ns,
            "gpu_span_share_of_dit_blocks_pct": (
                100 * item["gpu_span_ns"] / block_totals["gpu_span_ns"]
            ),
            "gpu_span_share_of_dit_forward_pct": (
                100 * item["gpu_span_ns"] / forward_totals["gpu_span_ns"]
            ),
            "kernel_calls": item["kernel_calls"],
            "kernel_calls_mean_per_range": item["kernel_calls"] / count,
            "kernel_families": {
                family: {
                    "calls": item["families_calls"][family],
                    "gpu_kernel_ms": duration / 1e6,
                }
                for family, duration in sorted(item["families_ns"].items())
            },
        }

    result = {
        "schema_version": "minwm-nsys-block-phases/v3",
        "sqlite": str(args.sqlite),
        "method": (
            "CUDA kernels are joined to runtime launches by correlationId; only "
            "complete same-thread forward -> block-loop -> phase NVTX trees are used."
        ),
        "phases": rendered,
        "dit_blocks": render_range_aggregate(block_totals),
        "dit_forward": render_range_aggregate(forward_totals),
        "totals": {
            "phase_range_count": len(phase_ranges),
            "mapped_phase_kernel_calls": sum(
                item["kernel_calls"] for item in aggregate.values()
            ),
            "phase_gpu_kernel_ms": total_kernel_ns / 1e6,
            "phase_gpu_span_ms_sum": total_span_ns / 1e6,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    connection.close()


if __name__ == "__main__":
    main()
