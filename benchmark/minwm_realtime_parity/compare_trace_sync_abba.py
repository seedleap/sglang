#!/usr/bin/env python3
"""Validate and summarize paired realtime CUDA trace-sync A/B runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from measurement import coefficient_of_variation, validate_measurement

ARMS = ("control", "candidate")
REPEATS = (1, 2)
METRICS = (
    "client_fps",
    "scheduler_fps",
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
)


def _metric_value(record: dict[str, Any], name: str) -> float:
    metric = record["metrics"]["profiler_off"][name]
    if metric.get("status") != "available":
        raise ValueError(f"{name} is unavailable: {metric}")
    value = metric["value"]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def _payload_hashes(record: dict[str, Any]) -> dict[str, str]:
    hashes = record.get("client", {}).get("payload_sha256_by_chunk")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("record has no client.payload_sha256_by_chunk")
    return {str(key): str(value) for key, value in hashes.items()}


def _fixed_contract(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sglang_commit": record["provenance"]["sglang_commit"],
        "minwm_commit": record["provenance"]["minwm_commit"],
        "container_image": record["provenance"]["container_image"],
        "gpu": record["provenance"]["gpu"],
        "workload": record["workload"],
        "comparison_contract": record.get("comparison_contract"),
    }


def _arm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in METRICS:
        values = [_metric_value(record, name) for record in records]
        summary[name] = {
            "values": values,
            "mean": statistics.fmean(values),
            "cv": coefficient_of_variation(values),
        }
    return summary


def _paired_delta_percent(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]], name: str
) -> list[float]:
    return [
        (
            _metric_value(candidate_record, name) / _metric_value(control_record, name)
            - 1
        )
        * 100
        for control_record, candidate_record in zip(control, candidate)
    ]


def build_trace_sync_summary(
    records_by_sp: dict[int, dict[str, list[dict[str, Any]]]],
    telemetry_by_sp: dict[int, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    overall_go = True
    for degree, arms in sorted(records_by_sp.items()):
        if set(arms) != set(ARMS):
            raise ValueError(f"SP{degree} must contain control and candidate records")
        for arm in ARMS:
            if len(arms[arm]) != len(REPEATS):
                raise ValueError(f"SP{degree} {arm} must contain two repeats")
            for record in arms[arm]:
                validate_measurement(record)
                workload = record["workload"]
                if record["mode"] != "profiler_off":
                    raise ValueError(
                        "trace-sync A/B headline records must be profiler_off"
                    )
                if workload["sp_degree"] != degree:
                    raise ValueError(
                        f"record SP mismatch: {workload['sp_degree']} != {degree}"
                    )
                if workload["warmup_chunks"] != 20 or workload["measured_chunks"] < 200:
                    raise ValueError(
                        "trace-sync A/B requires 20 warmup and >=200 measured"
                    )
                if workload["precision"].lower() != "bf16":
                    raise ValueError("trace-sync A/B requires BF16")
                if record["measurement_contract"]["headline_eligible"] is not True:
                    raise ValueError("profiler-off A/B record is not headline eligible")

        control = arms["control"]
        candidate = arms["candidate"]
        fixed_contract = _fixed_contract(control[0])
        if any(
            _fixed_contract(record) != fixed_contract
            for arm in ARMS
            for record in arms[arm]
        ):
            raise ValueError(f"SP{degree} changed a required fixed A/B field")
        if any(
            record.get("artifacts", {}).get("server_trace_sync_cuda")
            != (1 if arm == "control" else 0)
            for arm in ARMS
            for record in arms[arm]
        ):
            raise ValueError(f"SP{degree} trace-sync arm metadata is missing or wrong")
        workload = control[0]["workload"]
        if workload["resolution"] != {"width": 1248, "height": 704}:
            raise ValueError("trace-sync headline requires 1248x704")
        if workload["checkpoint"]["step"] != 3200:
            raise ValueError("trace-sync headline requires MinWM step-3200")
        if control[0].get("comparison_contract", {}).get("kv_cache_num_frames") != 45:
            raise ValueError("trace-sync headline requires KV45")
        reference_hashes = _payload_hashes(control[0])
        expected_hash_count = (
            control[0]["workload"]["warmup_chunks"]
            + control[0]["workload"]["measured_chunks"]
        )
        bitwise = len(reference_hashes) == expected_hash_count and all(
            _payload_hashes(record) == reference_hashes
            for arm in ARMS
            for record in arms[arm]
        )
        arm_summaries = {arm: _arm_summary(arms[arm]) for arm in ARMS}
        cv_pass = all(
            arm_summaries[arm][name]["cv"] <= 0.03 for arm in ARMS for name in METRICS
        )
        client_no_regression = (
            arm_summaries["candidate"]["client_fps"]["mean"]
            >= arm_summaries["control"]["client_fps"]["mean"]
        )
        scheduler_no_regression = (
            arm_summaries["candidate"]["scheduler_fps"]["mean"]
            >= arm_summaries["control"]["scheduler_fps"]["mean"]
        )
        control_dit = arm_summaries["control"]["dit_wall_ms"]["mean"]
        candidate_dit = arm_summaries["candidate"]["dit_wall_ms"]["mean"]
        dit_improvement_percent = (control_dit / candidate_dit - 1) * 100
        lane_go = (
            bitwise and cv_pass and client_no_regression and scheduler_no_regression
        )
        overall_go = overall_go and lane_go
        lane = {
            "workload": {
                "sp_degree": degree,
                "warmup_chunks": control[0]["workload"]["warmup_chunks"],
                "measured_chunks": control[0]["workload"]["measured_chunks"],
                "precision": control[0]["workload"]["precision"],
                "kv_cache_num_frames": control[0]
                .get("comparison_contract", {})
                .get("kv_cache_num_frames"),
            },
            "arms": arm_summaries,
            "paired_delta_percent": {
                name: _paired_delta_percent(control, candidate, name)
                for name in METRICS
            },
            "dit_improvement_percent": dit_improvement_percent,
            "payload_bitwise": {
                "passes": bitwise,
                "chunk_count": len(reference_hashes),
                "algorithm": "sha256(batch_index || sha256(raw_rgb_batch))",
            },
            "acceptance": {
                "cv_le_3_percent": cv_pass,
                "client_fps_no_regression": client_no_regression,
                "scheduler_fps_no_regression": scheduler_no_regression,
                "dit_improvement_ge_2_percent": dit_improvement_percent >= 2.0,
                "go": lane_go,
            },
        }
        if telemetry_by_sp is not None and degree in telemetry_by_sp:
            lane["telemetry"] = telemetry_by_sp[degree]
        lanes[f"sp{degree}"] = lane

    return {
        "schema_version": "minwm-realtime-trace-sync-abba/v1",
        "comparison": {
            "control": "SGLANG_REALTIME_TRACE_SYNC_CUDA=1",
            "candidate": "SGLANG_REALTIME_TRACE_SYNC_CUDA=0",
            "order": "control1,candidate1,candidate2,control2",
        },
        "lanes": lanes,
        "acceptance": {"go": overall_go},
    }


def _load_inputs(root: Path) -> tuple[
    dict[int, dict[str, list[dict[str, Any]]]],
    dict[int, dict[str, list[dict[str, Any]]]],
]:
    records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    telemetry: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for degree in (2, 4):
        lane = root / f"sp{degree}"
        records[degree] = {}
        telemetry[degree] = {}
        for arm in ARMS:
            records[degree][arm] = [
                json.loads((lane / f"{arm}-repeat{repeat}.json").read_text())
                for repeat in REPEATS
            ]
            telemetry[degree][arm] = [
                json.loads(
                    (lane / f"{arm}-repeat{repeat}-telemetry-summary.json").read_text()
                )
                for repeat in REPEATS
            ]
    return records, telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, telemetry = _load_inputs(args.root)
    summary = build_trace_sync_summary(records, telemetry)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["acceptance"], sort_keys=True))
    raise SystemExit(not summary["acceptance"]["go"])


if __name__ == "__main__":
    main()
