#!/usr/bin/env python3
"""Validate, merge, and aggregate MinWM realtime measurement records."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from measurement import coefficient_of_variation, validate_measurement
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


def load_aggregate_records(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[Path]]:
    excluded = [path for path in paths if "invalid" in path.parts]
    accepted = [path for path in paths if path not in excluded]
    return [_read(path) for path in accepted], excluded


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
        validate_measurement(_read(args.result))
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
