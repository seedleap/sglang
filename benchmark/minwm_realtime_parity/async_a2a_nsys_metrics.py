#!/usr/bin/env python3
"""Measure MinWM Ulysses A2A exposure from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nsys_metrics import (
    _api_metrics,
    _gpu_metrics,
    _kernel_metrics,
    _tables,
)

_CHUNK_RE = re.compile(
    r"^sglang\.realtime\.chunk\|trace_id=(?P<trace_id>[^|]+)"
    r"\|request_id=(?P<request_id>[^|]+)"
    r"\|chunk_index=(?P<chunk_index>\d+)"
    r"\|role=(?P<role>discard|measured|outside)$"
)
_TARGET_RANGES = {
    "qkv_pack",
    "qkv_pack_qk",
    "qkv_pack_v",
    "input_a2a_launch_qk",
    "input_a2a_launch_v",
    "input_a2a_wait_qk",
    "input_a2a_wait_v",
    "input_a2a_overlap_v_projection",
    "post_input_a2a_cache_rope",
    "attention",
    "output_a2a_launch_wait_sync",
    "output_a2a_launch",
    "output_a2a_wait",
    "output_projection",
    "ffn",
}
_NCCL_RE = re.compile(r"nccl|alltoall|sendrecv", re.I)
_SYNC_RE = re.compile(
    r"(?:device|context|ctx|stream|event).*synchroniz|synchroniz.*(?:device|context|ctx|stream|event)",
    re.I,
)
_GLOBAL_TID_MASK = -16777216  # Nsight encodes the OS TID in the low 24 bits.


@dataclass(frozen=True)
class NvtxRange:
    identifier: int
    name: str
    start: int
    end: int
    global_tid: int
    chunk: int


@dataclass(frozen=True)
class RuntimeCall:
    start: int
    end: int
    correlation: int
    name: str


@dataclass(frozen=True)
class Kernel:
    start: int
    end: int
    device: int
    stream: int
    correlation: int
    global_pid: int
    name: str


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    return {
        int(identifier): str(value)
        for identifier, value in connection.execute("SELECT id, value FROM StringIds")
    }


def _nvtx_rows(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> Iterable[tuple[int, int, str, int]]:
    for start, end, text, text_id, global_tid in connection.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND globalTid IS NOT NULL"
    ):
        name = str(text) if text is not None else strings.get(int(text_id), "")
        yield int(start), int(end), name, int(global_tid)


def _chunk_intervals(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    trace_id: str,
    warmup_chunks: int,
    measured_chunks: int,
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for start, end, name, _global_tid in _nvtx_rows(connection, strings):
        match = _CHUNK_RE.fullmatch(name)
        if not match or match.group("trace_id") != trace_id:
            continue
        if match.group("role") != "measured":
            continue
        index = int(match.group("chunk_index"))
        if index in selected:
            raise ValueError(f"duplicate measured chunk NVTX range {index}")
        selected[index] = {
            "chunk_index": index,
            "request_id": match.group("request_id"),
            "start_ns": start,
            "end_ns": end,
        }
    expected = list(range(warmup_chunks, warmup_chunks + measured_chunks))
    if sorted(selected) != expected:
        raise ValueError(
            f"measured chunk NVTX coverage mismatch: expected={expected}, "
            f"observed={sorted(selected)}"
        )
    intervals = [selected[index] for index in expected]
    for left, right in zip(intervals, intervals[1:]):
        if left["end_ns"] > right["start_ns"]:
            raise ValueError("measured chunk NVTX ranges overlap")
    return intervals


def _chunk_membership(
    start: int, end: int, intervals: list[dict[str, Any]]
) -> int | None:
    for ordinal, interval in enumerate(intervals):
        if start >= interval["start_ns"] and end <= interval["end_ns"]:
            return ordinal
    return None


def _target_ranges(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    intervals: list[dict[str, Any]],
) -> list[NvtxRange]:
    result = []
    for start, end, name, global_tid in _nvtx_rows(connection, strings):
        if name not in _TARGET_RANGES:
            continue
        chunk = _chunk_membership(start, end, intervals)
        if chunk is None:
            continue
        result.append(NvtxRange(len(result), name, start, end, global_tid, chunk))
    return result


def _runtime_calls(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    intervals: list[dict[str, Any]],
) -> tuple[dict[int, list[RuntimeCall]], dict[str, dict[str, float | int]]]:
    launch_ids = {
        identifier for identifier, name in strings.items() if "launch" in name.lower()
    }
    sync_ids = {
        identifier for identifier, name in strings.items() if _SYNC_RE.search(name)
    }
    selected_ids = sorted(launch_ids | sync_ids)
    if not selected_ids:
        return {}, {}
    placeholders = ",".join("?" for _ in selected_ids)
    lower = min(interval["start_ns"] for interval in intervals)
    upper = max(interval["end_ns"] for interval in intervals)
    by_tid: dict[int, list[RuntimeCall]] = defaultdict(list)
    sync: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration_ns": 0}
    )
    for start, end, global_tid, correlation, name_id in connection.execute(
        "SELECT start, end, globalTid, correlationId, nameId "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        f"WHERE nameId IN ({placeholders}) AND start >= ? AND end <= ?",
        (*selected_ids, lower, upper),
    ):
        call = RuntimeCall(
            int(start), int(end), int(correlation), strings.get(int(name_id), "")
        )
        by_tid[int(global_tid)].append(call)
        if (
            int(name_id) in sync_ids
            and _chunk_membership(call.start, call.end, intervals) is not None
        ):
            item = sync[call.name]
            item["count"] = int(item["count"]) + 1
            item["duration_ns"] = int(item["duration_ns"]) + call.end - call.start
    for calls in by_tid.values():
        calls.sort(key=lambda item: item.start)
    return by_tid, dict(sync)


def _range_correlations(
    ranges: list[NvtxRange], calls_by_tid: dict[int, list[RuntimeCall]]
) -> dict[tuple[int, int], set[int]]:
    correlations: dict[tuple[int, int], set[int]] = defaultdict(set)
    for span in ranges:
        calls = calls_by_tid.get(span.global_tid, [])
        starts = [call.start for call in calls]
        cursor = bisect.bisect_left(starts, span.start)
        while cursor < len(calls) and calls[cursor].start <= span.end:
            call = calls[cursor]
            if call.end <= span.end:
                global_pid = span.global_tid & _GLOBAL_TID_MASK
                correlations[(global_pid, call.correlation)].add(span.identifier)
            cursor += 1
    return correlations


def _range_kernels(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    intervals: list[dict[str, Any]],
    correlations: dict[tuple[int, int], set[int]],
) -> tuple[dict[int, list[Kernel]], dict[str, Any]]:
    lower = min(interval["start_ns"] for interval in intervals)
    upper = max(interval["end_ns"] for interval in intervals)
    by_range: dict[int, list[Kernel]] = defaultdict(list)
    captured = 0
    nccl = 0
    for row in connection.execute(
        "SELECT start, end, deviceId, streamId, correlationId, globalPid, shortName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ?",
        (lower, upper),
    ):
        kernel = Kernel(
            start=int(row[0]),
            end=int(row[1]),
            device=int(row[2]),
            stream=int(row[3]),
            correlation=int(row[4]),
            global_pid=int(row[5]),
            name=strings.get(int(row[6]), ""),
        )
        if _chunk_membership(kernel.start, kernel.end, intervals) is None:
            continue
        captured += 1
        if _NCCL_RE.search(kernel.name):
            nccl += 1
        for range_id in correlations.get((kernel.global_pid, kernel.correlation), ()):
            by_range[range_id].append(kernel)
    for kernels in by_range.values():
        kernels.sort(key=lambda item: item.start)
    return dict(by_range), {
        "cuda_kernel_count": captured,
        "nccl_kernel_count": nccl,
    }


def _merged_duration(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _intersection_duration(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    first = sorted(left)
    second = sorted(right)
    i = j = total = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            total += end - start
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    selected = list(values)
    if not selected:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": len(selected),
        "mean": statistics.fmean(selected),
        "median": statistics.median(selected),
        "p10": _percentile(selected, 0.10),
        "p90": _percentile(selected, 0.90),
    }


def _intervals(kernels: Iterable[Kernel]) -> list[tuple[int, int]]:
    return [(kernel.start, kernel.end) for kernel in kernels]


def _nccl(kernels: Iterable[Kernel]) -> list[Kernel]:
    return [kernel for kernel in kernels if _NCCL_RE.search(kernel.name)]


def _compute(kernels: Iterable[Kernel]) -> list[Kernel]:
    return [kernel for kernel in kernels if not _NCCL_RE.search(kernel.name)]


def _first_kernel_after(
    ranges: list[NvtxRange],
    kernels: dict[int, list[Kernel]],
    timestamp: int,
    limit: int,
) -> Kernel | None:
    candidates = [
        kernel
        for span in ranges
        if span.start >= timestamp and span.start < limit
        for kernel in kernels.get(span.identifier, ())
        if kernel.start >= timestamp
    ]
    return min(candidates, key=lambda item: item.start) if candidates else None


def _rank_accumulator() -> dict[str, Any]:
    return {
        "input_instances": 0,
        "input_a2a_total_ns": 0,
        "input_qk_a2a_ns": 0,
        "input_v_a2a_ns": 0,
        "input_compute_overlap_ns": 0,
        "input_exposed_kernel_ns": 0,
        "input_exposed_stage_ns": 0,
        "input_pre_idle_ns": 0,
        "input_post_idle_ns": 0,
        "output_instances": 0,
        "output_a2a_total_ns": 0,
        "output_exposed_stage_ns": 0,
        "output_pre_idle_ns": 0,
        "output_post_idle_ns": 0,
        "launch_start_to_wait_start_ns": [],
        "launch_end_to_wait_start_ns": [],
        "wait_cpu_range_ns": [],
    }


def _add_input_candidate(
    item: dict[str, Any],
    qk: NvtxRange,
    v_launch: NvtxRange,
    wait_qk: NvtxRange,
    wait_v: NvtxRange,
    v_projection: NvtxRange,
    qk_pack: NvtxRange | None,
    post_ranges: list[NvtxRange],
    next_qk_start: int,
    kernels: dict[int, list[Kernel]],
) -> None:
    qk_comm = _nccl(kernels.get(qk.identifier, ()))
    v_comm = _nccl(kernels.get(v_launch.identifier, ()))
    v_compute = _compute(kernels.get(v_projection.identifier, ()))
    if not qk_comm or not v_comm or not v_compute:
        return
    all_comm = _intervals((*qk_comm, *v_comm))
    overlap = _intersection_duration(_intervals(qk_comm), _intervals(v_compute))
    total = _merged_duration(all_comm)
    independent_end = max(kernel.end for kernel in v_compute)
    dependent = _first_kernel_after(post_ranges, kernels, wait_v.end, next_qk_start)
    if dependent is None:
        return
    first_comm = min(kernel.start for kernel in (*qk_comm, *v_comm))
    last_comm = max(kernel.end for kernel in (*qk_comm, *v_comm))
    pack_compute = _compute(kernels.get(qk_pack.identifier, ())) if qk_pack else []
    pack_end = max((kernel.end for kernel in pack_compute), default=first_comm)
    item["input_instances"] += 1
    item["input_a2a_total_ns"] += total
    item["input_qk_a2a_ns"] += _merged_duration(_intervals(qk_comm))
    item["input_v_a2a_ns"] += _merged_duration(_intervals(v_comm))
    item["input_compute_overlap_ns"] += overlap
    item["input_exposed_kernel_ns"] += total - overlap
    item["input_exposed_stage_ns"] += max(0, dependent.start - independent_end)
    item["input_pre_idle_ns"] += max(0, first_comm - pack_end)
    item["input_post_idle_ns"] += max(0, dependent.start - last_comm)
    item["launch_start_to_wait_start_ns"].append(max(0, wait_qk.start - qk.start))
    item["launch_end_to_wait_start_ns"].append(max(0, wait_qk.start - qk.end))
    item["wait_cpu_range_ns"].extend(
        (wait_qk.end - wait_qk.start, wait_v.end - wait_v.start)
    )


def _add_input_baseline(
    item: dict[str, Any],
    input_span: NvtxRange,
    post_ranges: list[NvtxRange],
    next_input_start: int,
    kernels: dict[int, list[Kernel]],
) -> None:
    span_kernels = kernels.get(input_span.identifier, ())
    comm = _nccl(span_kernels)
    if not comm:
        return
    dependent = _first_kernel_after(
        post_ranges, kernels, input_span.end, next_input_start
    )
    if dependent is None:
        return
    first_comm = min(kernel.start for kernel in comm)
    last_comm = max(kernel.end for kernel in comm)
    pack = [kernel for kernel in _compute(span_kernels) if kernel.end <= first_comm]
    pack_end = max((kernel.end for kernel in pack), default=first_comm)
    duration = _merged_duration(_intervals(comm))
    item["input_instances"] += 1
    item["input_a2a_total_ns"] += duration
    item["input_qk_a2a_ns"] += duration
    item["input_exposed_kernel_ns"] += duration
    item["input_exposed_stage_ns"] += max(0, dependent.start - first_comm)
    item["input_pre_idle_ns"] += max(0, first_comm - pack_end)
    item["input_post_idle_ns"] += max(0, dependent.start - last_comm)


def _add_output(
    item: dict[str, Any],
    output_span: NvtxRange,
    projection_ranges: list[NvtxRange],
    next_output_start: int,
    kernels: dict[int, list[Kernel]],
) -> None:
    span_kernels = kernels.get(output_span.identifier, ())
    comm = _nccl(span_kernels)
    if not comm:
        return
    dependent = _first_kernel_after(
        projection_ranges, kernels, output_span.end, next_output_start
    )
    if dependent is None:
        return
    first_comm = min(kernel.start for kernel in comm)
    last_comm = max(kernel.end for kernel in comm)
    pack = [kernel for kernel in _compute(span_kernels) if kernel.end <= first_comm]
    pack_end = max((kernel.end for kernel in pack), default=first_comm)
    item["output_instances"] += 1
    item["output_a2a_total_ns"] += _merged_duration(_intervals(comm))
    item["output_exposed_stage_ns"] += max(0, dependent.start - first_comm)
    item["output_pre_idle_ns"] += max(0, first_comm - pack_end)
    item["output_post_idle_ns"] += max(0, dependent.start - last_comm)


def _operation_metrics(
    ranges: list[NvtxRange],
    kernels: dict[int, list[Kernel]],
    intervals: list[dict[str, Any]],
    lane: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[int, int], list[NvtxRange]] = defaultdict(list)
    for span in ranges:
        by_key[(span.chunk, span.global_tid)].append(span)
    accumulators: dict[tuple[int, int], dict[str, Any]] = {}
    missing: dict[str, int] = defaultdict(int)
    for (chunk, global_tid), spans in by_key.items():
        spans.sort(key=lambda item: item.start)
        by_name: dict[str, list[NvtxRange]] = defaultdict(list)
        for span in spans:
            by_name[span.name].append(span)
        device_candidates = [
            kernel.device
            for span in spans
            for kernel in kernels.get(span.identifier, ())
        ]
        if not device_candidates:
            continue
        device = statistics.mode(device_candidates)
        item = accumulators.setdefault((chunk, device), _rank_accumulator())
        post = by_name["post_input_a2a_cache_rope"]
        if lane == "candidate":
            qks = by_name["input_a2a_launch_qk"]
            vs = by_name["input_a2a_launch_v"]
            wait_qks = by_name["input_a2a_wait_qk"]
            wait_vs = by_name["input_a2a_wait_v"]
            projections = by_name["input_a2a_overlap_v_projection"]
            packs = by_name["qkv_pack_qk"]
            count = min(
                len(qks), len(vs), len(wait_qks), len(wait_vs), len(projections)
            )
            missing["candidate_unpaired_input_ranges"] += (
                max(len(qks), len(vs), len(wait_qks), len(wait_vs), len(projections))
                - count
            )
            for index in range(count):
                next_start = (
                    qks[index + 1].start if index + 1 < count else spans[-1].end + 1
                )
                _add_input_candidate(
                    item,
                    qks[index],
                    vs[index],
                    wait_qks[index],
                    wait_vs[index],
                    projections[index],
                    packs[index] if index < len(packs) else None,
                    post,
                    next_start,
                    kernels,
                )
        else:
            inputs = by_name["qkv_pack"]
            for index, input_span in enumerate(inputs):
                next_start = (
                    inputs[index + 1].start
                    if index + 1 < len(inputs)
                    else spans[-1].end + 1
                )
                _add_input_baseline(item, input_span, post, next_start, kernels)
        outputs = by_name["output_a2a_launch_wait_sync"] or by_name["output_a2a_launch"]
        projections = by_name["output_projection"]
        for index, output_span in enumerate(outputs):
            next_start = (
                outputs[index + 1].start
                if index + 1 < len(outputs)
                else spans[-1].end + 1
            )
            _add_output(item, output_span, projections, next_start, kernels)

    chunks = []
    metric_names = [
        "input_a2a_total_ns",
        "input_qk_a2a_ns",
        "input_v_a2a_ns",
        "input_compute_overlap_ns",
        "input_exposed_kernel_ns",
        "input_exposed_stage_ns",
        "input_pre_idle_ns",
        "input_post_idle_ns",
        "output_a2a_total_ns",
        "output_exposed_stage_ns",
        "output_pre_idle_ns",
        "output_post_idle_ns",
    ]
    for chunk, interval in enumerate(intervals):
        ranks = {
            device: value
            for (item_chunk, device), value in accumulators.items()
            if item_chunk == chunk
        }
        rank_json = {}
        for device, value in sorted(ranks.items()):
            converted = {
                name.removesuffix("_ns") + "_ms": value[name] / 1e6
                for name in metric_names
            }
            qk_ms = converted["input_qk_a2a_ms"]
            converted["input_qk_overlap_ratio"] = (
                converted["input_compute_overlap_ms"] / qk_ms if qk_ms else 0.0
            )
            converted.update(
                {
                    "input_instances": value["input_instances"],
                    "output_instances": value["output_instances"],
                    "launch_start_to_wait_start_ms": _summary(
                        number / 1e6
                        for number in value["launch_start_to_wait_start_ns"]
                    ),
                    "launch_end_to_wait_start_ms": _summary(
                        number / 1e6 for number in value["launch_end_to_wait_start_ns"]
                    ),
                    "wait_cpu_range_ms": _summary(
                        number / 1e6 for number in value["wait_cpu_range_ns"]
                    ),
                }
            )
            rank_json[str(device)] = converted
        rank_metric_names = [name.removesuffix("_ns") + "_ms" for name in metric_names]
        rank_metric_names.append("input_qk_overlap_ratio")
        chunks.append(
            {
                "chunk_index": interval["chunk_index"],
                "request_id": interval["request_id"],
                "duration_ms": (interval["end_ns"] - interval["start_ns"]) / 1e6,
                "ranks": rank_json,
                "rank_mean": {
                    name: statistics.fmean(rank[name] for rank in rank_json.values())
                    if rank_json
                    else None
                    for name in rank_metric_names
                },
                "critical_rank_max": {
                    name: max(rank[name] for rank in rank_json.values())
                    if rank_json
                    else None
                    for name in rank_metric_names
                },
            }
        )
    aggregate = {"rank_mean": {}, "critical_rank_max": {}}
    for group in aggregate:
        for metric in chunks[0][group] if chunks else ():
            aggregate[group][metric] = _summary(
                chunk[group][metric]
                for chunk in chunks
                if chunk[group][metric] is not None
            )
    return chunks, {"metrics": aggregate, "diagnostics": dict(missing)}


def analyze(
    sqlite_path: Path,
    client_path: Path,
    lane: str,
    evidence: str = "",
) -> dict[str, Any]:
    client = _read(client_path)
    if lane not in {"baseline", "candidate"}:
        raise ValueError("lane must be baseline or candidate")
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _tables(connection)
        required = {
            "StringIds",
            "NVTX_EVENTS",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_KERNEL",
        }
        if missing := required - tables:
            raise ValueError(f"Nsight SQLite is missing tables: {sorted(missing)}")
        strings = _string_ids(connection)
        workload = client["workload"]
        intervals = _chunk_intervals(
            connection,
            strings,
            str(client["run_id"]),
            int(workload["warmup_chunks"]),
            int(workload["measured_chunks"]),
        )
        ranges = _target_ranges(connection, strings, intervals)
        calls, sync = _runtime_calls(connection, strings, intervals)
        correlations = _range_correlations(ranges, calls)
        kernels, raw_counts = _range_kernels(
            connection, strings, intervals, correlations
        )
        chunks, operation = _operation_metrics(ranges, kernels, intervals, lane)
        active_gpu_count = int(client["provenance"]["gpu"]["count"])
        general_kernel, captured_devices = _kernel_metrics(
            connection, tables, intervals, active_gpu_count
        )
        general_api, coverage = _api_metrics(
            connection, tables, intervals, active_gpu_count, captured_devices
        )
        gpu_metrics = _gpu_metrics(
            connection, tables, intervals, active_gpu_count, evidence
        )
    finally:
        connection.close()
    return {
        "schema_version": "minwm-async-a2a-nsys/v1",
        "lane": lane,
        "trace_id": client["run_id"],
        "source": {
            "sqlite": str(sqlite_path),
            "client": str(client_path),
            "profiler_wall_headline_eligible": False,
        },
        "workload": client["workload"],
        "provenance": client["provenance"],
        "stable_chunks": intervals,
        "chunks": chunks,
        "aggregate": operation,
        "raw_counts": raw_counts,
        "synchronization_apis": {
            name: {
                "count": int(value["count"]),
                "duration_ms": int(value["duration_ns"]) / 1e6,
            }
            for name, value in sorted(sync.items())
        },
        "general_nsys": {
            **general_kernel,
            **general_api,
            "capture_coverage": coverage,
            "gpu_metrics": gpu_metrics,
        },
    }


def _median(result: dict[str, Any], metric: str) -> float:
    value = result["aggregate"]["metrics"]["critical_rank_max"][metric]["median"]
    if value is None:
        raise ValueError(f"metric {metric} has no values")
    return float(value)


def _reduction(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline if baseline else 0.0


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    exposed_baseline = _median(baseline, "input_exposed_kernel_ms")
    exposed_candidate = _median(candidate, "input_exposed_kernel_ms")
    stage_baseline = _median(baseline, "input_exposed_stage_ms")
    stage_candidate = _median(candidate, "input_exposed_stage_ms")
    overlap_chunks = sum(
        float(chunk["critical_rank_max"]["input_compute_overlap_ms"] or 0) > 0
        for chunk in candidate["chunks"]
    )
    candidate_chunk_count = len(candidate["chunks"])
    repeatable_overlap = overlap_chunks >= max(
        1, math.ceil(0.8 * candidate_chunk_count)
    )
    exposed_reduction = _reduction(exposed_baseline, exposed_candidate)
    stage_reduction = _reduction(stage_baseline, stage_candidate)
    baseline_sync = sum(
        value["count"] for value in baseline["synchronization_apis"].values()
    )
    candidate_sync = sum(
        value["count"] for value in candidate["synchronization_apis"].values()
    )
    no_new_global_sync = candidate_sync <= baseline_sync
    mechanism_pass = (
        repeatable_overlap
        and no_new_global_sync
        and (exposed_reduction >= 20.0 or stage_reduction >= 10.0)
    )
    return {
        "schema_version": "minwm-async-a2a-nsys-comparison/v1",
        "baseline_trace_id": baseline["trace_id"],
        "candidate_trace_id": candidate["trace_id"],
        "input_exposed_kernel_ms": {
            "baseline_median": exposed_baseline,
            "candidate_median": exposed_candidate,
            "reduction_percent": exposed_reduction,
            "required_reduction_percent": 20.0,
        },
        "input_a2a_plus_adjacent_idle_critical_ms": {
            "baseline_median": stage_baseline,
            "candidate_median": stage_candidate,
            "reduction_percent": stage_reduction,
            "required_reduction_percent": 10.0,
        },
        "repeatable_compute_communication_overlap": {
            "nonzero_chunks": overlap_chunks,
            "total_chunks": candidate_chunk_count,
            "required_fraction": 0.8,
            "passes": repeatable_overlap,
        },
        "global_synchronization_api_count": {
            "baseline": baseline_sync,
            "candidate": candidate_sync,
            "passes_no_increase": no_new_global_sync,
        },
        "mechanism_acceptance": {
            "passes": mechanism_pass,
            "rule": (
                "repeatable nonzero QK-A2A/V-compute overlap, no increase in CUDA "
                "synchronization APIs, and either exposed A2A -20% or exposed "
                "A2A+adjacent-idle critical path -10%"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--sqlite", required=True, type=Path)
    analyze_parser.add_argument("--client", required=True, type=Path)
    analyze_parser.add_argument(
        "--lane", required=True, choices=("baseline", "candidate")
    )
    analyze_parser.add_argument("--status-log", type=Path)
    analyze_parser.add_argument("--output", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        evidence = ""
        if args.status_log and args.status_log.exists():
            evidence = args.status_log.read_text(encoding="utf-8", errors="replace")[
                -4000:
            ]
        result = analyze(args.sqlite, args.client, args.lane, evidence)
    else:
        result = compare(_read(args.baseline), _read(args.candidate))
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
