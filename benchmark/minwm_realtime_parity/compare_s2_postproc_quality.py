#!/usr/bin/env python3
"""Compare S2 self-attention fast-lane frames and diagnostic latent dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from compare_results import evaluate, metric_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--baseline-prefix", default="self_post_baseline")
    parser.add_argument("--candidate-prefix", default="self_post_fast")
    parser.add_argument("--baseline-dumps", required=True, type=Path)
    parser.add_argument("--candidate-dumps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def latent_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"latent shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.dtype != candidate.dtype:
        raise ValueError(
            f"latent dtype mismatch: {reference.dtype} != {candidate.dtype}"
        )
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    absolute = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference_float.double()).item()
    difference_norm = torch.linalg.vector_norm(difference.double()).item()
    denominator = (
        reference_norm * torch.linalg.vector_norm(candidate_float.double()).item()
    )
    cosine = (
        1.0
        if denominator == 0
        else torch.dot(
            reference_float.double().flatten(), candidate_float.double().flatten()
        ).item()
        / denominator
    )
    changed_value_count = int(torch.count_nonzero(difference).item())
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "numel": reference.numel(),
        "all_finite": bool(
            torch.isfinite(reference_float).all()
            and torch.isfinite(candidate_float).all()
        ),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "changed_value_count": changed_value_count,
        "changed_value_fraction": float(changed_value_count / difference.numel()),
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_l2": float(
            0.0 if reference_norm == 0 else difference_norm / reference_norm
        ),
        "cosine_similarity": float(cosine),
    }


def collect_latents(
    baseline_root: Path, candidate_root: Path
) -> tuple[list[dict], list[str]]:
    pattern = "sglang/sp_*_rank_*/chunk_*_latents.pt"
    baseline = {
        path.relative_to(baseline_root): path
        for path in sorted(baseline_root.glob(pattern))
    }
    candidate = {
        path.relative_to(candidate_root): path
        for path in sorted(candidate_root.glob(pattern))
    }
    missing = [str(path) for path in sorted(baseline.keys() - candidate.keys())]
    unexpected = [str(path) for path in sorted(candidate.keys() - baseline.keys())]
    if not baseline:
        raise ValueError("no baseline chunk latent dumps found")
    if missing or unexpected:
        raise ValueError(
            f"latent dump set mismatch: missing={missing}, unexpected={unexpected}"
        )
    records = []
    failures = []
    for relative in sorted(baseline):
        reference = torch.load(
            baseline[relative], map_location="cpu", weights_only=True
        )
        value = torch.load(candidate[relative], map_location="cpu", weights_only=True)
        metrics = latent_metrics(reference, value)
        if not metrics["all_finite"]:
            failures.append(f"{relative}: non-finite latent")
        records.append({"path": str(relative), "metrics": metrics})
    return records, failures


def summarize_latents(records: list[dict]) -> dict:
    metrics = [record["metrics"] for record in records]
    total_values = sum(item["numel"] for item in metrics)
    changed_values = sum(item["changed_value_count"] for item in metrics)
    squared_error_sum = sum(item["rmse"] ** 2 * item["numel"] for item in metrics)
    return {
        "file_count": len(records),
        "all_finite": all(item["all_finite"] for item in metrics),
        "all_bitwise_equal": all(item["bitwise_equal"] for item in metrics),
        "changed_value_fraction": changed_values / total_values,
        "max_abs": max(item["max_abs"] for item in metrics),
        "rmse": math.sqrt(squared_error_sum / total_values),
        "max_relative_l2": max(item["relative_l2"] for item in metrics),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics),
    }


def main() -> None:
    args = parse_args()
    case_root = args.results / "cases" / args.case
    baseline_frames = np.load(
        case_root / f"{args.baseline_prefix}.npy", allow_pickle=False
    )
    candidate_frames = np.load(
        case_root / f"{args.candidate_prefix}.npy", allow_pickle=False
    )
    frame_metrics = metric_block(baseline_frames[1:], candidate_frames[1:])
    profile_manifest = json.loads(args.thresholds.read_text(encoding="utf-8"))
    profile = profile_manifest["profiles"]["bf16_backend_candidate"]
    frames_passed, frame_failures = evaluate(
        {"generated_frames": frame_metrics}, profile
    )
    latent_records, latent_failures = collect_latents(
        args.baseline_dumps, args.candidate_dumps
    )
    latent_summary = summarize_latents(latent_records)
    failures = [*frame_failures, *latent_failures]
    report = {
        "schema_version": "minwm-s2-postproc-quality/v1",
        "acceptance": {
            "passed": frames_passed and not latent_failures,
            "failures": failures,
            "normative_contract": "bf16_backend_candidate generated-frame thresholds",
            "latent_contract": (
                "diagnostic: require matching files/shapes/dtypes and finite values; "
                "report numeric drift without inventing an unreviewed threshold"
            ),
        },
        "frame_contract": profile,
        "generated_frames": frame_metrics,
        "latents": {
            "summary": latent_summary,
            "records": latent_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "acceptance": report["acceptance"],
                "generated_frames": frame_metrics,
                "latents": latent_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
