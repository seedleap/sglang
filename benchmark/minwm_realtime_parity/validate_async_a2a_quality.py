#!/usr/bin/env python3
"""Validate saved MinWM async-A2A video reports and per-rank tensor probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REQUIRED_PROBES = {
    "self_q_norm_000.pt",
    "self_k_norm_000.pt",
    "self_q_roped_000.pt",
    "self_k_roped_000.pt",
    "self_attention_output_000.pt",
    "block0_output_000.pt",
    "output_proj_output_000.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sp-degrees", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    summary: dict[str, dict] = {"video_bitwise": {}, "tensor_parity": {}}
    for degree in args.sp_degrees:
        sp_name = f"sp{degree}"
        report = json.loads((root / sp_name / "report.json").read_text())
        case = next(item for item in report["cases"] if item["id"] == args.case_id)
        if not case["metrics"]["generated_frames"]["bitwise_equal"]:
            raise AssertionError(
                f"{sp_name}: baseline/candidate video is not bitwise exact"
            )
        summary["video_bitwise"][sp_name] = True

        lane_roots = {}
        for lane in ("baseline", "candidate"):
            matches = sorted(
                (root / "layer-probes" / lane / sp_name).glob(
                    f"sglang/sp_{degree:02d}_rank_*"
                )
            )
            if len(matches) != degree:
                raise AssertionError(
                    f"{sp_name}/{lane}: expected {degree} rank dumps, "
                    f"got {len(matches)}"
                )
            lane_roots[lane] = matches

        rank_summaries = {}
        for rank, (baseline_dir, candidate_dir) in enumerate(
            zip(lane_roots["baseline"], lane_roots["candidate"])
        ):
            baseline_names = {path.name for path in baseline_dir.glob("*.pt")}
            candidate_names = {path.name for path in candidate_dir.glob("*.pt")}
            missing_candidate = baseline_names - candidate_names
            if missing_candidate:
                raise AssertionError(
                    f"{sp_name}/rank{rank}: candidate is missing baseline probes "
                    f"{sorted(missing_candidate)}"
                )
            missing_required = REQUIRED_PROBES - baseline_names
            if missing_required:
                raise AssertionError(
                    f"{sp_name}/rank{rank}: missing required baseline probes "
                    f"{sorted(missing_required)}"
                )
            metrics = {}
            for name in sorted(baseline_names):
                baseline = torch.load(
                    baseline_dir / name, map_location="cpu", weights_only=True
                )
                candidate = torch.load(
                    candidate_dir / name, map_location="cpu", weights_only=True
                )
                if not isinstance(baseline, torch.Tensor) or not isinstance(
                    candidate, torch.Tensor
                ):
                    continue
                if (
                    baseline.shape != candidate.shape
                    or baseline.dtype != candidate.dtype
                ):
                    raise AssertionError(
                        f"{sp_name}/rank{rank}/{name}: tensor contract differs"
                    )
                equal = torch.equal(baseline, candidate)
                difference = baseline.float() - candidate.float()
                metrics[name] = {
                    "shape": list(baseline.shape),
                    "dtype": str(baseline.dtype),
                    "bitwise_equal": bool(equal),
                    "max_abs": float(difference.abs().max().item()),
                    "rmse": float(difference.square().mean().sqrt().item()),
                }
                if not equal:
                    raise AssertionError(
                        f"{sp_name}/rank{rank}/{name}: not bitwise exact"
                    )
            rank_summaries[f"rank{rank}"] = {
                "candidate_extra_probes": sorted(
                    candidate_names - baseline_names
                ),
                "metrics": metrics,
            }
        summary["tensor_parity"][sp_name] = rank_summaries
    return summary


def main() -> None:
    args = parse_args()
    summary = validate(args)
    output = args.output or args.root / "async-a2a-quality-summary.json"
    with output.open("x") as output_file:
        output_file.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "video_bitwise": summary["video_bitwise"],
                "tensor_probe_counts": {
                    degree: {
                        rank: len(record["metrics"])
                        for rank, record in ranks.items()
                    }
                    for degree, ranks in summary["tensor_parity"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
