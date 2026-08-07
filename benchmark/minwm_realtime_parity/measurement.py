"""Shared MinWM realtime measurement schema and metric helpers."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "minwm-realtime-measurement/v1"
SHORT_KERNEL_BUCKETS_US = ((0, 10), (10, 50), (50, 100), (100, None))

TIMING_DOMAINS = {
    "client": (
        "Client monotonic wall time between complete steady-chunk payloads. "
        "It includes server work, transport, and client receive overhead."
    ),
    "scheduler": (
        "Server monotonic wall time around scheduler request execution. It excludes "
        "client transport and output serialization."
    ),
    "stage_wall": (
        "Server monotonic wall time around the named pipeline stage. The realtime "
        "stage hook synchronizes its ending CUDA event, so queued GPU work is included."
    ),
    "cuda": (
        "CUDA-event or Nsight device time. It is not interchangeable with wall time "
        "and does not include CPU gaps or transport."
    ),
}


class MeasurementValidationError(ValueError):
    pass


def available(value: Any, unit: str, source: str) -> dict[str, Any]:
    return {
        "status": "available",
        "value": value,
        "unit": unit,
        "source": source,
    }


def unavailable(reason: str, evidence: str) -> dict[str, str]:
    return {
        "status": "unavailable",
        "reason": reason,
        "evidence": evidence,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def stage_trace_values(
    trace_events: list[dict[str, Any]],
    *,
    event: str,
    field: str,
    measured_indices: set[int],
    source: str | None = None,
    component: str | None = None,
) -> list[float]:
    """Return one selected trace value per measured chunk in chunk order."""
    by_chunk: dict[int, float] = {}
    for item in trace_events:
        if item.get("event") != event or field not in item:
            continue
        if source is not None and item.get("source") != source:
            continue
        if component is not None and item.get("component") != component:
            continue
        try:
            chunk_index = int(item["chunk_index"])
            value = float(item[field])
        except (KeyError, TypeError, ValueError):
            continue
        if chunk_index not in measured_indices:
            continue
        if chunk_index in by_chunk:
            raise MeasurementValidationError(
                f"duplicate {event}/{field} trace for chunk {chunk_index}"
            )
        by_chunk[chunk_index] = value
    return [by_chunk[index] for index in sorted(by_chunk)]


def optional_latency_metric(
    values: list[float], *, unit: str, source: str, missing_evidence: str
) -> dict[str, Any]:
    if values:
        return available(latency_summary(values), unit, source)
    return unavailable("trace_metric_missing", missing_evidence)


def build_measurement(
    *,
    mode: str,
    run_id: str,
    profile_name: str,
    timestamp_utc: str | None,
    sglang_commit: str,
    minwm_commit: dict[str, Any],
    container_image: dict[str, Any],
    gpu_model: dict[str, Any],
    gpu_count: int,
    allocated_gpu_count: int,
    sp_degree: int,
    checkpoint_id: str,
    checkpoint_step: int,
    width: int,
    height: int,
    warmup_chunks: int,
    measured_chunks: int,
    precondition_warmup_chunks: int,
    precision: str,
    fast_lane: bool,
    comparison_contract: dict[str, Any],
    profiler_off_metrics: dict[str, Any],
    profiler_on_cuda_metrics: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    if mode not in {"profiler_off", "profiler_on"}:
        raise ValueError(f"unsupported measurement mode: {mode}")
    if gpu_count < 1 or allocated_gpu_count < 1 or sp_degree < 1:
        raise ValueError("GPU counts and sp_degree must be positive")
    if allocated_gpu_count < gpu_count:
        raise ValueError("allocated_gpu_count cannot be smaller than active gpu_count")

    pending_nsys = unavailable(
        "nsys_result_not_merged",
        "Run measurement_tool.py merge-nsys with the .sqlite export and status log.",
    )
    metrics: dict[str, Any]
    if mode == "profiler_off":
        metrics = {"profiler_off": profiler_off_metrics, "profiler_on": None}
        headline_eligible = True
    else:
        metrics = {
            "profiler_off": None,
            "profiler_on": {
                "observed_wall_with_profiler_overhead": profiler_off_metrics,
                **profiler_on_cuda_metrics,
                "kernel_count": pending_nsys,
                "cuda_api_count": pending_nsys,
                "kernel_launch_api_count": pending_nsys,
                "short_kernel_buckets": pending_nsys,
                "gpu_kernel_busy": pending_nsys,
                "capture_coverage": pending_nsys,
                "gpu_metrics": {
                    "sm_active": pending_nsys,
                    "tensor_active": pending_nsys,
                    "dram": pending_nsys,
                },
            },
        }
        headline_eligible = False

    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_id,
        "profile_name": profile_name,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "sglang_commit": sglang_commit,
            "minwm_commit": minwm_commit,
            "container_image": container_image,
            "gpu": {
                "model": gpu_model,
                "count": gpu_count,
                "allocated_count": allocated_gpu_count,
            },
        },
        "workload": {
            "model": "MinWM 5B",
            "checkpoint": {"id": checkpoint_id, "step": checkpoint_step},
            "resolution": {"width": width, "height": height},
            "precision": precision,
            "fast_lane": fast_lane,
            "sp_degree": sp_degree,
            "dmd_forwards_per_chunk": 4,
            "clean_cache_forwards_per_chunk": 1,
            "latent_frames_per_chunk": 4,
            "pixel_frames_per_chunk": 16,
            "warmup_chunks": warmup_chunks,
            "precondition_warmup_chunks": precondition_warmup_chunks,
            "measured_chunks": measured_chunks,
        },
        "measurement_contract": {
            "headline_eligible": headline_eligible,
            "headline_rule": (
                "Only profiler_off Client/Scheduler FPS may be used as headline. "
                "Profiler-on wall/FPS values include Nsight overhead."
            ),
            "timing_domains": TIMING_DOMAINS,
            "gpu_count_semantics": {
                "provenance.gpu.count": "active GPUs used by the workload",
                "provenance.gpu.allocated_count": (
                    "GPUs reserved by the job, including idle isolation capacity"
                ),
            },
            "profiler_on": {
                "tool": "nsys launch/start/stop",
                "trace": ["cuda", "nvtx"],
                "trace_fork_before_exec": True,
                "cuda_graph_trace": "node",
                "torch_profiler_concurrent": False,
                "minimum_stable_chunks": 10,
            },
        },
        "comparison_contract": comparison_contract,
        "metrics": metrics,
        "artifacts": artifacts,
        "degradations": [],
    }
    validate_measurement(result)
    return result


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("CV requires at least two values")
    mean = statistics.fmean(values)
    if mean == 0:
        raise ValueError("CV is undefined for zero mean")
    return statistics.stdev(values) / mean


def _require_availability(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an availability object")
        return
    status = value.get("status")
    if status == "available":
        if (
            value.get("value") is None
            or not value.get("unit")
            or not value.get("source")
        ):
            errors.append(f"{path} available metric requires value, unit, and source")
    elif status == "unavailable":
        if not value.get("reason") or not value.get("evidence"):
            errors.append(f"{path} unavailable metric requires reason and evidence")
    else:
        errors.append(f"{path}.status must be available or unavailable")


def _require_normalized_count(
    metric: Any, path: str, fields: tuple[str, ...], errors: list[str]
) -> None:
    _require_availability(metric, path, errors)
    if not isinstance(metric, dict) or metric.get("status") != "available":
        return
    value = metric.get("value")
    if not isinstance(value, dict):
        errors.append(f"{path}.value must contain raw and normalized counts")
        return
    missing = [field for field in fields if field not in value]
    if missing:
        errors.append(f"{path}.value missing normalized fields: {missing}")


def validate_measurement(result: dict[str, Any]) -> None:
    errors: list[str] = []
    if result.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if result.get("mode") not in {"profiler_off", "profiler_on"}:
        errors.append("mode must be profiler_off or profiler_on")
    for path, value in (
        ("run_id", result.get("run_id")),
        ("timestamp_utc", result.get("timestamp_utc")),
        ("provenance.sglang_commit", result.get("provenance", {}).get("sglang_commit")),
    ):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path} must be a non-empty string")

    provenance = result.get("provenance", {})
    _require_availability(
        provenance.get("container_image"), "provenance.container_image", errors
    )
    _require_availability(
        provenance.get("minwm_commit"), "provenance.minwm_commit", errors
    )
    gpu = provenance.get("gpu", {})
    _require_availability(gpu.get("model"), "provenance.gpu.model", errors)
    if not isinstance(gpu.get("count"), int) or gpu.get("count", 0) < 1:
        errors.append("provenance.gpu.count must be a positive integer")
    if (
        not isinstance(gpu.get("allocated_count"), int)
        or gpu.get("allocated_count", 0) < 1
    ):
        errors.append("provenance.gpu.allocated_count must be a positive integer")
    elif isinstance(gpu.get("count"), int) and gpu["allocated_count"] < gpu["count"]:
        errors.append("provenance.gpu.allocated_count must be >= active count")

    workload = result.get("workload", {})
    required_workload = {
        "precision": str,
        "fast_lane": bool,
        "sp_degree": int,
        "warmup_chunks": int,
        "measured_chunks": int,
    }
    for key, expected_type in required_workload.items():
        if not isinstance(workload.get(key), expected_type):
            errors.append(f"workload.{key} must be {expected_type.__name__}")
    if isinstance(gpu.get("count"), int) and isinstance(workload.get("sp_degree"), int):
        if gpu["count"] != workload["sp_degree"]:
            errors.append(
                "provenance.gpu.count must equal workload.sp_degree for this S0 lane"
            )
    if workload.get("dmd_forwards_per_chunk") != 4:
        errors.append("workload.dmd_forwards_per_chunk must be 4")
    if workload.get("clean_cache_forwards_per_chunk") != 1:
        errors.append("workload.clean_cache_forwards_per_chunk must be 1")
    if result.get("mode") == "profiler_on" and workload.get("measured_chunks", 0) < 10:
        errors.append("profiler_on requires at least 10 measured stable chunks")

    contract = result.get("measurement_contract", {})
    expected_headline = result.get("mode") == "profiler_off"
    if contract.get("headline_eligible") is not expected_headline:
        errors.append("headline_eligible must be true only for profiler_off")

    metrics = result.get("metrics", {})
    if result.get("mode") == "profiler_off":
        if metrics.get("profiler_on") is not None:
            errors.append("profiler_off record must not contain profiler_on metrics")
        off = metrics.get("profiler_off")
        if not isinstance(off, dict):
            errors.append("profiler_off record requires profiler_off metrics")
        else:
            for name in (
                "client_fps",
                "scheduler_fps",
                "scheduler_chunk_wall_ms",
                "dit_wall_ms",
                "vae_wall_ms",
            ):
                _require_availability(
                    off.get(name), f"metrics.profiler_off.{name}", errors
                )
    elif result.get("mode") == "profiler_on":
        if metrics.get("profiler_off") is not None:
            errors.append(
                "profiler_on record must not contain headline profiler_off metrics"
            )
        on = metrics.get("profiler_on")
        if not isinstance(on, dict):
            errors.append("profiler_on record requires profiler_on metrics")
        else:
            for name in (
                "dit_cuda_ms",
                "vae_cuda_ms",
                "gpu_kernel_busy",
                "capture_coverage",
            ):
                _require_availability(
                    on.get(name), f"metrics.profiler_on.{name}", errors
                )
            _require_normalized_count(
                on.get("kernel_count"),
                "metrics.profiler_on.kernel_count",
                (
                    "raw_total",
                    "per_stable_chunk",
                    "per_device",
                    "stable_chunk_denominator",
                    "capture_scope",
                ),
                errors,
            )
            for name in ("cuda_api_count", "kernel_launch_api_count"):
                _require_normalized_count(
                    on.get(name),
                    f"metrics.profiler_on.{name}",
                    (
                        "raw_total",
                        "total_per_stable_chunk",
                        "per_rank_per_stable_chunk",
                        "stable_chunk_denominator",
                        "capture_scope",
                    ),
                    errors,
                )
                metric = on.get(name)
                if isinstance(metric, dict) and metric.get("status") == "available":
                    nested = metric.get("value", {}).get("per_rank_per_stable_chunk")
                    _require_availability(
                        nested,
                        f"metrics.profiler_on.{name}.value.per_rank_per_stable_chunk",
                        errors,
                    )
            _require_normalized_count(
                on.get("short_kernel_buckets"),
                "metrics.profiler_on.short_kernel_buckets",
                (
                    "raw_total",
                    "per_stable_chunk",
                    "per_device",
                    "stable_chunk_denominator",
                    "capture_scope",
                ),
                errors,
            )
            gpu_metrics = on.get("gpu_metrics", {})
            for name in ("sm_active", "tensor_active", "dram"):
                _require_availability(
                    gpu_metrics.get(name),
                    f"metrics.profiler_on.gpu_metrics.{name}",
                    errors,
                )
                metric = gpu_metrics.get(name)
                if isinstance(metric, dict) and metric.get("status") == "available":
                    metric_value = metric.get("value")
                    required = {
                        "raw_metric_name",
                        "sample_count",
                        "exposed_metric_names",
                    }
                    if not isinstance(metric_value, dict) or not required.issubset(
                        metric_value
                    ):
                        errors.append(
                            f"metrics.profiler_on.gpu_metrics.{name}.value must "
                            "retain raw_metric_name, sample_count, and "
                            "exposed_metric_names"
                        )

    if errors:
        raise MeasurementValidationError("; ".join(errors))
