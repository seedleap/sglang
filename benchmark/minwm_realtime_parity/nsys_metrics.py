"""Extract strictly windowed CUDA and GPU metrics from an Nsight SQLite export."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from measurement import available, unavailable, validate_measurement

_MARKER_PREFIX = "sglang.realtime.chunk"
_MARKER_RE = re.compile(
    rf"^{re.escape(_MARKER_PREFIX)}\|trace_id=(?P<trace_id>[^|]+)"
    r"\|request_id=(?P<request_id>[^|]+)"
    r"\|chunk_index=(?P<chunk_index>-?\d+)"
    r"\|role=(?P<role>discard|measured|outside)$"
)
_CAPTURE_SCOPE = "union of exact measured outer chunk NVTX ranges"


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info('{table}')")}


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


def _nvtx_text_rows(
    connection: sqlite3.Connection, tables: set[str]
) -> tuple[list[tuple[int, int, str]], str]:
    table = "NVTX_EVENTS"
    if table not in tables:
        raise ValueError(f"{table} is absent")
    columns = _columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        raise ValueError(f"{table} lacks start/end columns: {sorted(columns)}")
    rows: list[tuple[int, int, str]] = []
    sources = []
    if "text" in columns:
        rows.extend(
            [
                (int(start), int(end), str(text))
                for start, end, text in connection.execute(
                    f"SELECT start, end, text FROM {table} "
                    "WHERE text IS NOT NULL AND end IS NOT NULL"
                )
            ]
        )
        sources.append(f"{table}.text")
    if "textId" in columns:
        raw = [
            (int(start), int(end), int(text_id))
            for start, end, text_id in connection.execute(
                f"SELECT start, end, textId FROM {table} "
                "WHERE textId IS NOT NULL AND end IS NOT NULL"
            )
        ]
        names = _resolve_string_ids(
            connection, {text_id for _start, _end, text_id in raw}, tables
        )
        rows.extend(
            (start, end, names[text_id])
            for start, end, text_id in raw
            if text_id in names
        )
        sources.append(f"{table}.textId -> StringIds.value")
    if not sources:
        raise ValueError(f"{table} lacks text/textId columns: {sorted(columns)}")
    return rows, " + ".join(sources)


def _stable_window(
    connection: sqlite3.Connection,
    tables: set[str],
    result: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workload = result["workload"]
    warmup = int(workload["warmup_chunks"])
    measured = int(workload["measured_chunks"])
    trace_id = str(result["run_id"])
    expected_measured = list(range(warmup, warmup + measured))
    expected_discard = list(range(warmup))
    try:
        rows, source = _nvtx_text_rows(connection, tables)
    except ValueError as exc:
        return unavailable("stable_window_marker_missing", str(exc)), []

    markers: list[dict[str, Any]] = []
    for start, end, text in rows:
        match = _MARKER_RE.fullmatch(text)
        if not match or match.group("trace_id") != trace_id:
            continue
        markers.append(
            {
                "start_ns": start,
                "end_ns": end,
                "trace_id": trace_id,
                "request_id": match.group("request_id"),
                "chunk_index": int(match.group("chunk_index")),
                "role": match.group("role"),
            }
        )

    by_role_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for marker in markers:
        by_role_index.setdefault(
            (str(marker["role"]), int(marker["chunk_index"])), []
        ).append(marker)
    observed_measured = sorted(
        index for role, index in by_role_index if role == "measured"
    )
    observed_discard = sorted(
        index for role, index in by_role_index if role == "discard"
    )
    duplicate = sorted(
        {
            index
            for (role, index), values in by_role_index.items()
            if role == "measured" and len(values) != 1
        }
    )
    missing = sorted(set(expected_measured) - set(observed_measured))
    unexpected = sorted(set(observed_measured) - set(expected_measured))
    discard_duplicate = sorted(
        {
            index
            for (role, index), values in by_role_index.items()
            if role == "discard" and len(values) != 1
        }
    )
    discard_missing = sorted(set(expected_discard) - set(observed_discard))
    discard_unexpected = sorted(set(observed_discard) - set(expected_discard))
    selected = [
        by_role_index[("measured", index)][0]
        for index in expected_measured
        if len(by_role_index.get(("measured", index), [])) == 1
    ]
    invalid_ranges = [
        int(marker["chunk_index"])
        for marker in selected
        if int(marker["end_ns"]) <= int(marker["start_ns"])
    ]
    ordered = sorted(selected, key=lambda item: int(item["start_ns"]))
    overlaps = [
        [int(left["chunk_index"]), int(right["chunk_index"])]
        for left, right in zip(ordered, ordered[1:])
        if int(right["start_ns"]) < int(left["end_ns"])
    ]
    chronological_indices = [int(marker["chunk_index"]) for marker in ordered]
    request_ids = [str(marker["request_id"]) for marker in selected]
    duplicate_request_ids = sorted(
        {request_id for request_id in request_ids if request_ids.count(request_id) > 1}
    )
    diagnostics = {
        "missing_measured_indices": missing,
        "unexpected_measured_indices": unexpected,
        "duplicate_measured_indices": duplicate,
        "missing_discard_indices": discard_missing,
        "unexpected_discard_indices": discard_unexpected,
        "duplicate_discard_indices": discard_duplicate,
        "invalid_range_indices": invalid_ranges,
        "overlapping_measured_ranges": overlaps,
        "chronological_measured_indices": (
            [] if chronological_indices == expected_measured else chronological_indices
        ),
        "duplicate_measured_request_ids": duplicate_request_ids,
    }
    if any(diagnostics.values()) or len(selected) != measured:
        return (
            unavailable(
                "stable_window_marker_incomplete",
                f"trace_id={trace_id}; source={source}; diagnostics={diagnostics}",
            ),
            [],
        )

    value = {
        "window_source": source,
        "marker_prefix": _MARKER_PREFIX,
        "trace_id": trace_id,
        "clock_domain": (
            "Nsight SQLite nanoseconds in one launch/start/stop session; outer "
            "NVTX ranges are emitted by the API process and CUDA rows by traced workers"
        ),
        "expected_stable_chunk_indices": expected_measured,
        "observed_stable_chunk_indices": observed_measured,
        "expected_stable_chunk_count": measured,
        "observed_stable_chunk_count": len(observed_measured),
        "excluded_precondition_chunks": int(workload["precondition_warmup_chunks"]),
        "excluded_warmup_chunk_indices": expected_discard,
        "observed_discard_chunk_indices": observed_discard,
        "normalization_denominator": measured,
        "attribution_policy": (
            "Events must be fully contained in exactly one measured range; GPU "
            "samples use timestamps in [start,end); gaps and non-target traces are excluded"
        ),
        "intervals": [
            {
                "chunk_index": int(marker["chunk_index"]),
                "request_id": str(marker["request_id"]),
                "role": "measured",
                "start_ns": int(marker["start_ns"]),
                "end_ns": int(marker["end_ns"]),
            }
            for marker in sorted(selected, key=lambda item: int(item["chunk_index"]))
        ],
    }
    return available(value, "capture_window", source), value["intervals"]


def _event_membership(
    start: int, end: int, intervals: list[dict[str, Any]]
) -> tuple[int | None, bool]:
    contained = [
        index
        for index, interval in enumerate(intervals)
        if start >= interval["start_ns"] and end <= interval["end_ns"]
    ]
    overlaps = any(
        start < interval["end_ns"] and end > interval["start_ns"]
        for interval in intervals
    )
    if len(contained) == 1:
        return contained[0], False
    return None, overlaps


def _point_membership(timestamp: int, intervals: list[dict[str, Any]]) -> int | None:
    for index, interval in enumerate(intervals):
        if interval["start_ns"] <= timestamp < interval["end_ns"]:
            return index
    return None


def _window_failure_metrics(window: dict[str, Any]) -> dict[str, Any]:
    evidence = window.get("evidence", "stable window is unavailable")
    missing = unavailable("stable_window_unproven", evidence)
    return {
        "kernel_count": missing,
        "short_kernel_buckets": missing,
        "gpu_kernel_busy": missing,
        "cuda_api_count": missing,
        "kernel_launch_api_count": missing,
        "capture_coverage": missing,
        "gpu_metrics": {
            "sm_active": missing,
            "tensor_active": missing,
            "dram": missing,
        },
    }


def _bucket_counts(selected: list[tuple[int, int, int, int]]) -> dict[str, int]:
    durations_us = [(end - start) / 1000.0 for _device, start, end, _chunk in selected]
    return {
        "lt_10_us": sum(value < 10 for value in durations_us),
        "10_to_lt_50_us": sum(10 <= value < 50 for value in durations_us),
        "50_to_lt_100_us": sum(50 <= value < 100 for value in durations_us),
        "gte_100_us": sum(value >= 100 for value in durations_us),
    }


def _merged_duration(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _kernel_metrics(
    connection: sqlite3.Connection,
    tables: set[str],
    intervals: list[dict[str, Any]],
    expected_device_count: int,
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in tables:
        missing = unavailable("cuda_kernel_table_missing", f"{table} is absent")
        return {
            "kernel_count": missing,
            "short_kernel_buckets": missing,
            "gpu_kernel_busy": missing,
        }, []
    rows = [
        (int(device), int(start), int(end))
        for device, start, end in connection.execute(
            f"SELECT deviceId, start, end FROM {table} ORDER BY deviceId, start"
        )
    ]
    selected: list[tuple[int, int, int, int]] = []
    boundary = 0
    for device, start, end in rows:
        chunk, crosses = _event_membership(start, end, intervals)
        if chunk is not None:
            selected.append((device, start, end, chunk))
        elif crosses:
            boundary += 1
    if boundary:
        evidence = (
            f"{table}: captured_raw_total={len(rows)}; selected_raw_total={len(selected)}; "
            f"boundary_overlap_count={boundary}; strict containment required"
        )
        missing = unavailable("event_crosses_stable_window_boundary", evidence)
        return {
            "kernel_count": missing,
            "short_kernel_buckets": missing,
            "gpu_kernel_busy": missing,
        }, sorted({device for device, *_rest in selected})

    captured_device_ids = sorted({device for device, *_rest in selected})
    if len(captured_device_ids) != expected_device_count:
        evidence = (
            f"{table}: captured_raw_total={len(rows)}; selected_raw_total={len(selected)}; "
            f"stable-window device_ids={captured_device_ids}; "
            f"expected_device_count={expected_device_count}"
        )
        missing = unavailable("device_capture_coverage_incomplete", evidence)
        return {
            "kernel_count": missing,
            "short_kernel_buckets": missing,
            "gpu_kernel_busy": missing,
        }, captured_device_ids

    denominator = len(intervals)
    by_device: dict[int, list[tuple[int, int, int, int]]] = {}
    for row in selected:
        by_device.setdefault(row[0], []).append(row)
    count_value = {
        "raw_total": len(selected),
        "captured_raw_total": len(rows),
        "excluded_raw_total": len(rows) - len(selected),
        "boundary_overlap_count": 0,
        "per_stable_chunk": len(selected) / denominator,
        "per_device": {
            str(device): {
                "raw_total": len(device_rows),
                "per_stable_chunk": len(device_rows) / denominator,
            }
            for device, device_rows in sorted(by_device.items())
        },
        "stable_chunk_denominator": denominator,
        "capture_scope": _CAPTURE_SCOPE,
    }
    buckets = _bucket_counts(selected)
    bucket_value = {
        "raw_total": buckets,
        "captured_raw_total": len(rows),
        "excluded_raw_total": len(rows) - len(selected),
        "boundary_overlap_count": 0,
        "per_stable_chunk": {
            name: count / denominator for name, count in buckets.items()
        },
        "per_device": {
            str(device): {
                "raw_total": _bucket_counts(device_rows),
                "per_stable_chunk": {
                    name: count / denominator
                    for name, count in _bucket_counts(device_rows).items()
                },
            }
            for device, device_rows in sorted(by_device.items())
        },
        "stable_chunk_denominator": denominator,
        "capture_scope": _CAPTURE_SCOPE,
    }
    window_ns = sum(interval["end_ns"] - interval["start_ns"] for interval in intervals)
    per_device_pct: dict[str, float] = {}
    for device, device_rows in sorted(by_device.items()):
        busy_ns = sum(
            _merged_duration(
                [
                    (start, end)
                    for _device, start, end, row_chunk in device_rows
                    if row_chunk == chunk
                ]
            )
            for chunk in range(denominator)
        )
        per_device_pct[str(device)] = 100.0 * busy_ns / window_ns
    busy_value = {
        "per_device_pct": per_device_pct,
        "mean_pct": (
            sum(per_device_pct.values()) / len(per_device_pct)
            if per_device_pct
            else None
        ),
        "window_duration_ns": window_ns,
        "window": _CAPTURE_SCOPE,
        "boundary_overlap_count": 0,
    }
    return {
        "kernel_count": available(count_value, "count", table),
        "short_kernel_buckets": available(
            bucket_value, "count", f"{table}.end-start; fixed microsecond buckets"
        ),
        "gpu_kernel_busy": available(
            busy_value,
            "percent",
            f"merged {table} intervals divided by measured NVTX range duration sum",
        ),
    }, sorted(by_device)


def _capture_coverage(
    selected_process_ids: list[int],
    process_id_source: str,
    expected_ranks: int,
    captured_device_ids: list[int],
    kernel_process_ids: list[int],
    kernel_processes_by_device: dict[str, list[int]],
) -> dict[str, Any]:
    evidence = (
        f"stable-window runtime_process_ids={selected_process_ids}; "
        f"runtime_process_id_source={process_id_source}; "
        f"kernel_process_ids={kernel_process_ids}; "
        f"kernel_processes_by_device={kernel_processes_by_device}; "
        f"stable-window device_ids={captured_device_ids}; expected_ranks={expected_ranks}"
    )
    if (
        len(selected_process_ids) != expected_ranks
        or len(captured_device_ids) != expected_ranks
        or len(kernel_process_ids) != expected_ranks
        or set(selected_process_ids) != set(kernel_process_ids)
        or any(len(processes) != 1 for processes in kernel_processes_by_device.values())
    ):
        return unavailable("rank_capture_coverage_unconfirmed", evidence)
    return available(
        {
            "expected_ranks": expected_ranks,
            "observed_process_count": len(selected_process_ids),
            "process_id_source": process_id_source,
            "observed_process_ids": selected_process_ids,
            "kernel_process_ids": kernel_process_ids,
            "kernel_processes_by_device": kernel_processes_by_device,
            "captured_device_ids": captured_device_ids,
            "capture_scope": _CAPTURE_SCOPE,
        },
        "capture_coverage",
        evidence,
    )


def _api_metrics(
    connection: sqlite3.Connection,
    tables: set[str],
    intervals: list[dict[str, Any]],
    active_gpu_count: int,
    captured_device_ids: list[int],
    kernel_process_ids: list[int],
    kernel_processes_by_device: dict[str, list[int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in tables:
        missing = unavailable("cuda_api_table_missing", f"{table} is absent")
        coverage = unavailable(
            "rank_capture_coverage_unconfirmed", f"{table} is absent"
        )
        return {"cuda_api_count": missing, "kernel_launch_api_count": missing}, coverage
    columns = _columns(connection, table)
    direct_process_column = next(
        (name for name in ("globalPid", "processId") if name in columns), None
    )
    has_global_tid_mapping = "globalTid" in columns and "PROCESSES" in tables
    query_columns = ["nameId", "start", "end"]
    if direct_process_column:
        query_columns.append(direct_process_column)
        process_id_source = f"{table}.{direct_process_column}"
    elif has_global_tid_mapping:
        query_columns.append("globalTid")
        process_id_source = f"{table}.globalTid masked to PROCESSES.globalPid"
    else:
        process_id_source = "unavailable"
    raw = list(connection.execute(f"SELECT {', '.join(query_columns)} FROM {table}"))
    known_global_pids = (
        {int(row[0]) for row in connection.execute("SELECT globalPid FROM PROCESSES")}
        if has_global_tid_mapping
        else set()
    )
    selected: list[tuple[int, int | None]] = []
    boundary = 0
    for row in raw:
        name_id, start, end = int(row[0]), int(row[1]), int(row[2])
        _chunk, crosses = _event_membership(start, end, intervals)
        if _chunk is not None:
            process_id = int(row[3]) if len(row) > 3 else None
            if (
                process_id is not None
                and has_global_tid_mapping
                and not direct_process_column
            ):
                process_id &= ~0xFFFFFF
                if process_id not in known_global_pids:
                    process_id = None
            selected.append((name_id, process_id))
        elif crosses:
            boundary += 1
    selected_process_ids = sorted(
        {process_id for _name_id, process_id in selected if process_id is not None}
    )
    coverage = _capture_coverage(
        selected_process_ids,
        process_id_source,
        active_gpu_count,
        captured_device_ids,
        kernel_process_ids,
        kernel_processes_by_device,
    )
    if direct_process_column is None and not has_global_tid_mapping:
        coverage = unavailable(
            "rank_capture_coverage_unconfirmed",
            f"{table} columns={sorted(columns)}; stable-window device_ids={captured_device_ids}",
        )
    if boundary:
        evidence = (
            f"{table}: captured_raw_total={len(raw)}; selected_raw_total={len(selected)}; "
            f"boundary_overlap_count={boundary}; strict containment required"
        )
        missing = unavailable("event_crosses_stable_window_boundary", evidence)
        return {"cuda_api_count": missing, "kernel_launch_api_count": missing}, coverage

    name_ids = [name_id for name_id, _process_id in selected]
    names = _resolve_string_ids(connection, set(name_ids), tables)
    launch_pattern = re.compile(
        r"(?:cu|cuda).*(?:launch|graphlaunch).*(?:kernel|graph)?", re.I
    )
    launch_count = sum(
        bool(launch_pattern.search(names.get(name_id, ""))) for name_id in name_ids
    )
    denominator = len(intervals)

    def normalized_count(raw_total: int, source: str) -> dict[str, Any]:
        per_rank = (
            available(
                raw_total / denominator / active_gpu_count,
                "count_per_rank_per_chunk",
                f"{source}; {coverage['source']}",
            )
            if coverage["status"] == "available"
            else unavailable("rank_capture_coverage_unconfirmed", coverage["evidence"])
        )
        return available(
            {
                "raw_total": raw_total,
                "captured_raw_total": len(raw),
                "excluded_raw_total": len(raw) - len(selected),
                "boundary_overlap_count": 0,
                "total_per_chunk": raw_total / denominator,
                "per_rank_per_chunk": per_rank,
                "stable_chunk_denominator": denominator,
                "capture_scope": _CAPTURE_SCOPE,
            },
            "count",
            source,
        )

    return {
        "cuda_api_count": normalized_count(len(name_ids), table),
        "kernel_launch_api_count": normalized_count(
            launch_count, f"{table} names matching CUDA kernel/graph launch APIs"
        ),
    }, coverage


def _kernel_process_coverage(
    connection: sqlite3.Connection,
    tables: set[str],
    intervals: list[dict[str, Any]],
) -> tuple[list[int], dict[str, list[int]]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in tables:
        return [], {}
    columns = _columns(connection, table)
    process_column = next(
        (name for name in ("globalPid", "processId") if name in columns), None
    )
    if process_column is None:
        return [], {}
    processes_by_device: dict[int, set[int]] = {}
    for device, process_id, start, end in connection.execute(
        f"SELECT deviceId, {process_column}, start, end FROM {table}"
    ):
        chunk, _crosses = _event_membership(int(start), int(end), intervals)
        if chunk is not None:
            processes_by_device.setdefault(int(device), set()).add(int(process_id))
    normalized = {
        str(device): sorted(processes)
        for device, processes in sorted(processes_by_device.items())
    }
    return (
        sorted({item for values in normalized.values() for item in values}),
        normalized,
    )


def _gpu_metrics(
    connection: sqlite3.Connection,
    tables: set[str],
    intervals: list[dict[str, Any]],
    active_gpu_count: int,
    external_evidence: str,
) -> dict[str, dict[str, Any]]:
    required_tables = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}
    if not required_tables.issubset(tables):
        missing_tables = sorted(required_tables - tables)
        evidence = f"Missing Nsight tables: {', '.join(missing_tables)}. {external_evidence}".strip()
        status = unavailable(_permission_reason(external_evidence), evidence)
        return {"sm_active": status, "tensor_active": status, "dram": status}
    gpu_columns = _columns(connection, "GPU_METRICS")
    timestamp_column = next(
        (name for name in ("timestamp", "start") if name in gpu_columns), None
    )
    if timestamp_column is None:
        status = unavailable(
            "gpu_metric_timestamp_missing",
            f"GPU_METRICS columns lack timestamp/start: {sorted(gpu_columns)}",
        )
        return {"sm_active": status, "tensor_active": status, "dram": status}

    id_to_name = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        )
    }
    captured: dict[int, list[tuple[int, float, int]]] = {}
    selected: dict[int, list[tuple[int, float, int, int]]] = {}
    for type_id, metric_id, value, timestamp in connection.execute(
        f"SELECT typeId, metricId, value, {timestamp_column} FROM GPU_METRICS"
    ):
        metric_id = int(metric_id)
        sample = (int(type_id), float(value), int(timestamp))
        captured.setdefault(metric_id, []).append(sample)
        chunk = _point_membership(sample[2], intervals)
        if chunk is not None:
            selected.setdefault(metric_id, []).append((*sample, chunk))

    def select(label: str, predicate: Callable[[str], bool]) -> dict[str, Any]:
        candidates = [
            metric_id
            for metric_id, name in id_to_name.items()
            if predicate(name.lower()) and selected.get(metric_id)
        ]
        if not candidates:
            return unavailable(
                "metric_not_exposed",
                _metric_evidence(sorted(id_to_name.values()), external_evidence),
            )
        metric_id = candidates[0]
        samples = selected[metric_id]
        per_chunk_sample_count = {
            str(interval["chunk_index"]): sum(sample[3] == chunk for sample in samples)
            for chunk, interval in enumerate(intervals)
        }
        observed_type_ids = sorted({sample[0] for sample in samples})
        per_type_per_chunk_sample_count = {
            str(type_id): {
                str(interval["chunk_index"]): sum(
                    sample[0] == type_id and sample[3] == chunk for sample in samples
                )
                for chunk, interval in enumerate(intervals)
            }
            for type_id in observed_type_ids
        }
        missing_type_chunks = {
            type_id: [
                chunk_index for chunk_index, count in chunk_counts.items() if count == 0
            ]
            for type_id, chunk_counts in per_type_per_chunk_sample_count.items()
            if any(count == 0 for count in chunk_counts.values())
        }
        if len(observed_type_ids) != active_gpu_count or missing_type_chunks:
            return unavailable(
                "gpu_metric_window_coverage_incomplete",
                f"metric={id_to_name[metric_id]}; expected_type_count={active_gpu_count}; "
                f"observed_type_ids={observed_type_ids}; "
                f"missing_type_chunks={missing_type_chunks}; "
                f"per_chunk_sample_count={per_chunk_sample_count}",
            )
        per_type: dict[str, list[float]] = {}
        for type_id, value, _timestamp, _chunk in samples:
            per_type.setdefault(str(type_id), []).append(value)
        per_type_mean = {
            type_id: sum(values) / len(values)
            for type_id, values in sorted(per_type.items())
        }
        all_values = [value for _type_id, value, _timestamp, _chunk in samples]
        captured_count = len(captured.get(metric_id, []))
        return available(
            {
                "metric_name": id_to_name[metric_id],
                "raw_metric_name": id_to_name[metric_id],
                "mean": sum(all_values) / len(all_values),
                "samples": len(all_values),
                "sample_count": len(all_values),
                "captured_sample_count": captured_count,
                "excluded_sample_count": captured_count - len(all_values),
                "per_chunk_sample_count": per_chunk_sample_count,
                "per_type_per_chunk_sample_count": per_type_per_chunk_sample_count,
                "observed_type_ids": observed_type_ids,
                "type_id_coverage_semantics": (
                    "Nsight GPU_METRICS target identity; formal acceptance requires "
                    "one distinct typeId per active GPU and samples for every typeId/chunk"
                ),
                "per_type_mean": per_type_mean,
                "per_type_sample_count": {
                    type_id: len(values) for type_id, values in sorted(per_type.items())
                },
                "exposed_metric_names": sorted(id_to_name.values()),
                "capture_scope": _CAPTURE_SCOPE,
                "timestamp_column": timestamp_column,
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
        stable_window, intervals = _stable_window(connection, tables, result)
        on["stable_window_coverage"] = stable_window
        if stable_window["status"] != "available":
            on.update(_window_failure_metrics(stable_window))
        else:
            kernel_metrics, captured_device_ids = _kernel_metrics(
                connection,
                tables,
                intervals,
                int(result["provenance"]["gpu"]["count"]),
            )
            kernel_process_ids, kernel_processes_by_device = _kernel_process_coverage(
                connection, tables, intervals
            )
            api_metrics, coverage = _api_metrics(
                connection,
                tables,
                intervals,
                int(result["provenance"]["gpu"]["count"]),
                captured_device_ids,
                kernel_process_ids,
                kernel_processes_by_device,
            )
            on.update(kernel_metrics)
            on.update(api_metrics)
            on["capture_coverage"] = coverage
            on["gpu_metrics"] = _gpu_metrics(
                connection,
                tables,
                intervals,
                int(result["provenance"]["gpu"]["count"]),
                evidence,
            )
    finally:
        connection.close()
    result["artifacts"]["nsys_sqlite"] = str(sqlite_path)
    validate_measurement(result)
    return result
