#!/usr/bin/env python3
"""Validate, merge, and aggregate MinWM realtime measurement records."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from measurement import (
    MeasurementValidationError,
    coefficient_of_variation,
    validate_measurement,
)
from nsys_metrics import merge_nsys_metrics


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_invalid_marker(
    root: Path,
    *,
    reason: str,
    marker_path: Path,
    preserved_root: Path | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("invalid-attempt reason must be non-empty")
    root = root.resolve()
    marker_path = marker_path.resolve()
    if not root.is_dir():
        raise ValueError(f"invalid-attempt root is not a directory: {root}")
    preserved_root = preserved_root.resolve() if preserved_root else root
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == marker_path:
            continue
        relative = path.relative_to(root)
        files.append(
            {
                "original_path": str(path),
                "preserved_path": str(preserved_root / relative),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "recoverable": True,
            }
        )
    return {
        "schema_version": "minwm-realtime-invalid-attempt/v1",
        "reason": reason,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "original_root": str(root),
        "preserved_root": str(preserved_root),
        "recoverability": (
            "moved_to_attempt_invalid"
            if preserved_root != root
            else "preserved_in_place"
        ),
        "files": files,
    }


def _is_invalid_result(path: Path) -> bool:
    if "invalid" in path.parts:
        return True
    measurement_root = next(
        (parent for parent in path.parents if parent.name == "s0-measurement"), None
    )
    if measurement_root is None:
        scopes = [path.parent]
    else:
        scopes = []
        scope = path.parent
        while True:
            scopes.append(scope)
            if scope == measurement_root:
                break
            scope = scope.parent
    return any(
        next(scope.glob("invalid-marker*.json"), None) is not None for scope in scopes
    )


def load_aggregate_records(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[Path]]:
    excluded = [path for path in paths if _is_invalid_result(path)]
    accepted = [path for path in paths if path not in excluded]
    return [_read(path) for path in accepted], excluded


def require_complete_stable_nsys(result: dict[str, Any]) -> None:
    validate_measurement(result)
    if result.get("mode") != "profiler_on":
        raise MeasurementValidationError(
            "complete stable Nsight acceptance requires a profiler_on record"
        )
    profiler = result["metrics"]["profiler_on"]
    required = (
        "stable_window_coverage",
        "kernel_count",
        "cuda_api_count",
        "kernel_launch_api_count",
        "short_kernel_buckets",
        "gpu_kernel_busy",
        "capture_coverage",
    )
    unavailable_metrics = {
        name: {
            "reason": profiler[name].get("reason"),
            "evidence": profiler[name].get("evidence"),
        }
        for name in required
        if profiler[name].get("status") != "available"
    }
    for name in ("sm_active", "tensor_active"):
        metric = profiler["gpu_metrics"][name]
        if metric.get("status") != "available":
            unavailable_metrics[f"gpu_metrics.{name}"] = {
                "reason": metric.get("reason"),
                "evidence": metric.get("evidence"),
            }
    dram = profiler["gpu_metrics"]["dram"]
    if dram.get("status") == "unavailable" and (
        dram.get("reason") != "metric_not_exposed"
        or "Nsight exposed GPU metric names:" not in dram.get("evidence", "")
    ):
        unavailable_metrics["gpu_metrics.dram"] = {
            "reason": dram.get("reason"),
            "evidence": dram.get("evidence"),
        }
    if unavailable_metrics:
        raise MeasurementValidationError(
            "formal Nsight result lacks complete stable-window metrics: "
            f"{json.dumps(unavailable_metrics, sort_keys=True)}"
        )


def _off_scalar(record: dict[str, Any], name: str) -> float:
    metric = record["metrics"]["profiler_off"][name]
    if metric["status"] != "available":
        raise ValueError(f"{record['run_id']} metric {name} is unavailable")
    value = metric["value"]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def aggregate(records: list[dict[str, Any]], explanation: str | None) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("repeat aggregation requires at least two records")
    for record in records:
        validate_measurement(record)
        if record["mode"] != "profiler_off":
            raise ValueError("repeat aggregation accepts profiler_off records only")
    contract_keys = ("sp_degree", "precision", "fast_lane", "measured_chunks")
    reference = records[0]["workload"]
    for record in records[1:]:
        mismatched = [
            key for key in contract_keys if record["workload"][key] != reference[key]
        ]
        if mismatched:
            raise ValueError(f"repeat contract mismatch: {mismatched}")

    metric_names = (
        "client_fps",
        "scheduler_fps",
        "scheduler_chunk_wall_ms",
        "dit_wall_ms",
        "vae_wall_ms",
    )
    metrics = {}
    for name in metric_names:
        values = [_off_scalar(record, name) for record in records]
        cv = coefficient_of_variation(values)
        metrics[name] = {
            "values": values,
            "mean": sum(values) / len(values),
            "cv": cv,
            "target_cv_lte": 0.03,
            "passes": cv <= 0.03,
        }
    acceptance_metrics = (
        "client_fps",
        "scheduler_fps",
        "dit_wall_ms",
        "vae_wall_ms",
    )
    passes = all(metrics[name]["passes"] for name in acceptance_metrics)
    return {
        "schema_version": "minwm-realtime-repeat-summary/v1",
        "run_ids": [record["run_id"] for record in records],
        "contract": {key: reference[key] for key in contract_keys},
        "metrics": metrics,
        "acceptance": {
            "passes_cv_target": passes,
            "required_metrics": list(acceptance_metrics),
            "explanation_required": not passes,
            "environment_noise_explanation": explanation,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("result", type=Path)
    validate_parser.add_argument("--require-complete-stable-nsys", action="store_true")

    merge_parser = subparsers.add_parser("merge-nsys")
    merge_parser.add_argument("--result", required=True, type=Path)
    merge_parser.add_argument("--sqlite", required=True, type=Path)
    merge_parser.add_argument("--status-log", type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("results", nargs="+", type=Path)
    aggregate_parser.add_argument("--noise-explanation")
    aggregate_parser.add_argument("--output", required=True, type=Path)

    invalid_parser = subparsers.add_parser("mark-invalid")
    invalid_parser.add_argument("--root", required=True, type=Path)
    invalid_parser.add_argument("--reason", required=True)
    invalid_parser.add_argument("--marker", required=True, type=Path)
    invalid_parser.add_argument("--preserved-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "mark-invalid":
        if args.marker.exists():
            raise ValueError(f"refusing to overwrite invalid marker: {args.marker}")
        marker = build_invalid_marker(
            args.root,
            reason=args.reason,
            marker_path=args.marker,
            preserved_root=args.preserved_root,
        )
        _write(args.marker, marker)
        print(f"invalid attempt marked: {args.marker}")
        return
    if args.command == "validate":
        result = _read(args.result)
        validate_measurement(result)
        if args.require_complete_stable_nsys:
            require_complete_stable_nsys(result)
        print(f"valid: {args.result}")
        return
    if args.command == "merge-nsys":
        evidence = ""
        if args.status_log and args.status_log.exists():
            evidence = args.status_log.read_text(encoding="utf-8", errors="replace")[
                -4000:
            ]
        result = merge_nsys_metrics(_read(args.result), args.sqlite, evidence)
        if args.status_log:
            result["artifacts"]["nsys_status_log"] = str(args.status_log)
        _write(args.output, result)
        return
    records, excluded = load_aggregate_records(args.results)
    for path in excluded:
        print(f"excluded invalid result: {path}")
    summary = aggregate(records, args.noise_explanation)
    _write(args.output, summary)


if __name__ == "__main__":
    main()
