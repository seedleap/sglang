"""Extract version-tolerant CUDA and GPU metrics from an Nsight SQLite export."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from measurement import available, unavailable, validate_measurement


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _permission_reason(evidence: str) -> str:
    lowered = evidence.lower()
    if "permission" in lowered or "perf_event" in lowered or "access denied" in lowered:
        return "permission_denied"
    return "gpu_metrics_not_collected"


def _metric_evidence(names: list[str], external_evidence: str) -> str:
    exposed = ", ".join(names) if names else "none"
    return f"Nsight exposed GPU metric names: {exposed}. {external_evidence}".strip()


def _resolve_string_ids(
    connection: sqlite3.Connection, ids: set[int], tables: set[str]
) -> dict[int, str]:
    if not ids or "StringIds" not in tables:
        return {}
    placeholders = ",".join("?" for _ in ids)
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f"SELECT id, value FROM StringIds WHERE id IN ({placeholders})",
            tuple(sorted(ids)),
        )
    }


def _merged_busy(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    if not intervals:
        return 0, 0
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    span = intervals[-1][1] - intervals[0][0]
    return busy, span


def _kernel_metrics(
    connection: sqlite3.Connection, tables: set[str]
) -> dict[str, dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in tables:
        missing = unavailable(
            "cuda_kernel_table_missing", f"{table} is absent from the Nsight export"
        )
        return {
            "kernel_count": missing,
            "short_kernel_buckets": missing,
            "gpu_kernel_busy": missing,
        }

    rows = [
        (int(device), int(start), int(end))
        for device, start, end in connection.execute(
            f"SELECT deviceId, start, end FROM {table} ORDER BY deviceId, start"
        )
    ]
    durations_us = [(end - start) / 1000.0 for _device, start, end in rows]
    buckets = {
        "lt_10_us": sum(value < 10 for value in durations_us),
        "10_to_lt_50_us": sum(10 <= value < 50 for value in durations_us),
        "50_to_lt_100_us": sum(50 <= value < 100 for value in durations_us),
        "gte_100_us": sum(value >= 100 for value in durations_us),
    }
    by_device: dict[int, list[tuple[int, int]]] = {}
    for device, start, end in rows:
        by_device.setdefault(device, []).append((start, end))
    busy_pct: dict[str, float] = {}
    for device, intervals in sorted(by_device.items()):
        busy, span = _merged_busy(intervals)
        if span:
            busy_pct[str(device)] = 100.0 * busy / span
    busy_value = {
        "per_device_pct": busy_pct,
        "mean_pct": sum(busy_pct.values()) / len(busy_pct) if busy_pct else None,
        "window": "first kernel start to last kernel end per device",
    }
    return {
        "kernel_count": available(len(rows), "count", table),
        "short_kernel_buckets": available(
            buckets, "count", f"{table}.end-start; fixed microsecond buckets"
        ),
        "gpu_kernel_busy": available(
            busy_value, "percent", f"merged {table} intervals"
        ),
    }


def _api_metrics(
    connection: sqlite3.Connection, tables: set[str]
) -> dict[str, dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in tables:
        missing = unavailable(
            "cuda_api_table_missing", f"{table} is absent from the Nsight export"
        )
        return {"cuda_api_count": missing, "kernel_launch_api_count": missing}
    name_ids = [
        int(row[0]) for row in connection.execute(f"SELECT nameId FROM {table}")
    ]
    names = _resolve_string_ids(connection, set(name_ids), tables)
    launch_pattern = re.compile(
        r"(?:cu|cuda).*(?:launch|graphlaunch).*(?:kernel|graph)?", re.I
    )
    launch_count = sum(
        bool(launch_pattern.search(names.get(name_id, ""))) for name_id in name_ids
    )
    return {
        "cuda_api_count": available(len(name_ids), "count", table),
        "kernel_launch_api_count": available(
            launch_count,
            "count",
            f"{table} names matching CUDA kernel/graph launch APIs",
        ),
    }


def _gpu_metrics(
    connection: sqlite3.Connection,
    tables: set[str],
    external_evidence: str,
) -> dict[str, dict[str, Any]]:
    required_tables = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}
    if not required_tables.issubset(tables):
        missing = sorted(required_tables - tables)
        evidence = (
            f"Missing Nsight tables: {', '.join(missing)}. {external_evidence}".strip()
        )
        status = unavailable(_permission_reason(external_evidence), evidence)
        return {"sm_active": status, "tensor_active": status, "dram": status}

    id_to_name = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        )
    }
    samples: dict[int, list[tuple[int, float]]] = {}
    for type_id, metric_id, value in connection.execute(
        "SELECT typeId, metricId, value FROM GPU_METRICS"
    ):
        samples.setdefault(int(metric_id), []).append((int(type_id), float(value)))

    def select(label: str, predicate) -> dict[str, Any]:
        candidates = [
            metric_id
            for metric_id, name in id_to_name.items()
            if predicate(name.lower()) and samples.get(metric_id)
        ]
        if not candidates:
            return unavailable(
                "metric_not_exposed",
                _metric_evidence(sorted(id_to_name.values()), external_evidence),
            )
        metric_id = candidates[0]
        per_type: dict[str, list[float]] = {}
        for type_id, value in samples[metric_id]:
            per_type.setdefault(str(type_id), []).append(value)
        per_type_mean = {
            type_id: sum(values) / len(values)
            for type_id, values in sorted(per_type.items())
        }
        all_values = [value for _type_id, value in samples[metric_id]]
        return available(
            {
                "metric_name": id_to_name[metric_id],
                "mean": sum(all_values) / len(all_values),
                "samples": len(all_values),
                "per_type_mean": per_type_mean,
            },
            "nsys_native",
            f"GPU_METRICS metricId={metric_id} selected for {label}",
        )

    return {
        "sm_active": select("sm_active", lambda name: "sm active" in name),
        "tensor_active": select("tensor_active", lambda name: "tensor active" in name),
        "dram": select(
            "dram",
            lambda name: "dram" in name
            and any(marker in name for marker in ("throughput", "bandwidth", "active")),
        ),
    }


def merge_nsys_metrics(
    result: dict[str, Any], sqlite_path: Path, evidence: str = ""
) -> dict[str, Any]:
    validate_measurement(result)
    if result["mode"] != "profiler_on":
        raise ValueError("Nsight metrics can only be merged into profiler_on records")
    connection = sqlite3.connect(str(sqlite_path))
    try:
        tables = _tables(connection)
        on = result["metrics"]["profiler_on"]
        on.update(_kernel_metrics(connection, tables))
        on.update(_api_metrics(connection, tables))
        on["gpu_metrics"] = _gpu_metrics(connection, tables, evidence)
    finally:
        connection.close()
    result["artifacts"]["nsys_sqlite"] = str(sqlite_path)
    validate_measurement(result)
    return result
