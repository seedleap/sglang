#!/usr/bin/env python3
"""Analyze a same-GPU FA2-reference versus mandatory-FA3 MinWM quality A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PREFIXES = ("fa2_a", "fa2_b", "fa3_a", "fa3_b")
SHORT_MAX_ABS = 8
SHORT_MAX_RMSE = 1.0
SHORT_MIN_SSIM = 0.995
SHORT_MAX_LPIPS = 0.05
ACTION_MIN_DELTA_COSINE = 0.95
ACTION_MIN_NORM_RATIO = 0.8
ACTION_MAX_NORM_RATIO = 1.25
TEMPORAL_MIN_ACTIVITY_RATIO = 0.5
TEMPORAL_MAX_ACTIVITY_RATIO = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-results", type=Path, required=True)
    parser.add_argument("--long-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-lpips", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_indices(frame_count: int, count: int = 32) -> list[int]:
    if frame_count <= count:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, count, dtype=int).tolist()))


def pixel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    count = 0
    changed = 0
    maximum = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    dot = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    for reference_frame, candidate_frame in zip(reference, candidate):
        difference = reference_frame.astype(np.int16) - candidate_frame.astype(np.int16)
        absolute = np.abs(difference)
        difference_float = difference.astype(np.float64)
        reference_float = reference_frame.astype(np.float64)
        candidate_float = candidate_frame.astype(np.float64)
        count += difference.size
        changed += int(np.count_nonzero(difference))
        maximum = max(maximum, int(absolute.max(initial=0)))
        absolute_sum += float(absolute.sum(dtype=np.float64))
        squared_sum += float(np.square(difference_float).sum(dtype=np.float64))
        dot += float(
            np.multiply(reference_float, candidate_float).sum(dtype=np.float64)
        )
        reference_squared += float(np.square(reference_float).sum(dtype=np.float64))
        candidate_squared += float(np.square(candidate_float).sum(dtype=np.float64))
    rmse = math.sqrt(squared_sum / count) if count else 0.0
    denominator = math.sqrt(reference_squared * candidate_squared)
    indices = sample_indices(len(reference), 32)
    if maximum == 0:
        return {
            "bitwise_equal": True,
            "changed_value_fraction": 0.0,
            "cosine_similarity": 1.0,
            "max_abs": 0,
            "mean_abs": 0.0,
            "min_sampled_ssim": 1.0,
            "psnr_db": 999.0,
            "rmse": 0.0,
            "sample_indices": indices,
            "sampled_ssim": 1.0,
        }
    from skimage.metrics import structural_similarity

    ssim = [
        float(
            structural_similarity(
                reference[index], candidate[index], channel_axis=2, data_range=255
            )
        )
        for index in indices
    ]
    return {
        "bitwise_equal": maximum == 0,
        "changed_value_fraction": changed / count if count else 0.0,
        "cosine_similarity": 1.0 if denominator == 0 else dot / denominator,
        "max_abs": maximum,
        "mean_abs": absolute_sum / count if count else 0.0,
        "min_sampled_ssim": min(ssim) if ssim else 1.0,
        "psnr_db": 999.0 if rmse == 0 else 20 * math.log10(255.0 / rmse),
        "rmse": rmse,
        "sample_indices": indices,
        "sampled_ssim": sum(ssim) / len(ssim) if ssim else 1.0,
    }


def lpips_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    import lpips
    import torch

    model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    indices = sample_indices(len(reference), 24)
    values = []
    with torch.inference_mode():
        for index in indices:
            ref = torch.from_numpy(np.asarray(reference[index])).cuda().float()
            cand = torch.from_numpy(np.asarray(candidate[index])).cuda().float()
            ref = ref.permute(2, 0, 1).unsqueeze(0).mul_(2 / 255).sub_(1)
            cand = cand.permute(2, 0, 1).unsqueeze(0).mul_(2 / 255).sub_(1)
            values.append(float(model(ref, cand).item()))
    return {
        "max": max(values) if values else 0.0,
        "mean": sum(values) / len(values) if values else 0.0,
        "sample_indices": indices,
        "values": values,
    }


def temporal_activity(frames: np.ndarray) -> dict:
    values = []
    for previous, current in zip(frames[:-1], frames[1:]):
        difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        values.append(float(difference.mean()))
    array = np.asarray(values, dtype=np.float64)
    return {
        "frozen_transition_fraction": float(np.mean(array < 0.01)),
        "max_mean_abs_delta": float(array.max(initial=0.0)),
        "mean_abs_delta": float(array.mean()) if len(array) else 0.0,
        "min_mean_abs_delta": float(array.min(initial=0.0)),
        "p05_mean_abs_delta": float(np.percentile(array, 5)) if len(array) else 0.0,
        "p95_mean_abs_delta": float(np.percentile(array, 95)) if len(array) else 0.0,
        "transition_count": len(values),
    }


def delta_metrics(
    fa2_action: np.ndarray,
    fa2_idle: np.ndarray,
    fa3_action: np.ndarray,
    fa3_idle: np.ndarray,
) -> dict:
    dot = 0.0
    fa2_squared = 0.0
    fa3_squared = 0.0
    fa2_onset = None
    fa3_onset = None
    for frame_index, (action2, idle2, action3, idle3) in enumerate(
        zip(fa2_action, fa2_idle, fa3_action, fa3_idle)
    ):
        delta2 = action2.astype(np.float32) - idle2.astype(np.float32)
        delta3 = action3.astype(np.float32) - idle3.astype(np.float32)
        if fa2_onset is None and float(np.abs(delta2).mean()) > 0.01:
            fa2_onset = frame_index
        if fa3_onset is None and float(np.abs(delta3).mean()) > 0.01:
            fa3_onset = frame_index
        dot += float(np.multiply(delta2, delta3).sum(dtype=np.float64))
        fa2_squared += float(np.square(delta2).sum(dtype=np.float64))
        fa3_squared += float(np.square(delta3).sum(dtype=np.float64))
    denominator = math.sqrt(fa2_squared * fa3_squared)
    norm_ratio = math.sqrt(fa3_squared / fa2_squared) if fa2_squared else math.inf
    return {
        "delta_cosine_similarity": 1.0 if denominator == 0 else dot / denominator,
        "fa2_effect_l2": math.sqrt(fa2_squared),
        "fa2_first_effect_frame": fa2_onset,
        "fa3_effect_l2": math.sqrt(fa3_squared),
        "fa3_first_effect_frame": fa3_onset,
        "fa3_to_fa2_effect_norm_ratio": norm_ratio,
    }


def load_case(results: Path, case_id: str, prefix: str) -> tuple[np.ndarray, Path]:
    path = results / "cases" / case_id / f"{prefix}.npy"
    return np.load(path, mmap_mode="r", allow_pickle=False), path


def case_ids(results: Path) -> list[str]:
    return sorted(path.name for path in (results / "cases").iterdir() if path.is_dir())


def analyze_actions(results: Path, require_lpips: bool) -> dict:
    ids = case_ids(results)
    if not ids or ids[0] != "00_idle_street":
        raise ValueError(f"unexpected action cases: {ids}")
    records = []
    all_passed = True
    idle = {prefix: load_case(results, ids[0], prefix)[0][1:] for prefix in PREFIXES}
    for case_id in ids:
        frames = {prefix: load_case(results, case_id, prefix)[0] for prefix in PREFIXES}
        paths = {prefix: load_case(results, case_id, prefix)[1] for prefix in PREFIXES}
        replay = {
            "fa2": bool(np.array_equal(frames["fa2_a"], frames["fa2_b"])),
            "fa3": bool(np.array_equal(frames["fa3_a"], frames["fa3_b"])),
        }
        cross = pixel_metrics(frames["fa2_a"][1:], frames["fa3_a"][1:])
        lpips = lpips_metrics(frames["fa2_a"][1:], frames["fa3_a"][1:])
        cross_passed = (
            cross["max_abs"] <= SHORT_MAX_ABS
            and cross["rmse"] <= SHORT_MAX_RMSE
            and cross["sampled_ssim"] >= SHORT_MIN_SSIM
            and lpips["max"] <= SHORT_MAX_LPIPS
        )
        action = None
        action_passed = True
        if case_id != ids[0]:
            action = delta_metrics(
                frames["fa2_a"][1:],
                idle["fa2_a"],
                frames["fa3_a"][1:],
                idle["fa3_a"],
            )
            action_passed = (
                action["fa2_first_effect_frame"] == action["fa3_first_effect_frame"]
                and action["delta_cosine_similarity"] >= ACTION_MIN_DELTA_COSINE
                and ACTION_MIN_NORM_RATIO
                <= action["fa3_to_fa2_effect_norm_ratio"]
                <= ACTION_MAX_NORM_RATIO
            )
        passed = all(replay.values()) and cross_passed and action_passed
        all_passed = all_passed and passed
        records.append(
            {
                "action_effect": action,
                "case_id": case_id,
                "cross_backend": cross,
                "file_sha256": {
                    prefix: sha256_file(path) for prefix, path in paths.items()
                },
                "lpips": lpips,
                "passed": passed,
                "replay_bitwise_equal": replay,
            }
        )
    if require_lpips and any(record["lpips"] is None for record in records):
        raise RuntimeError("LPIPS evidence is required")
    return {"all_passed": all_passed, "cases": records}


def analyze_long(results: Path) -> dict:
    ids = case_ids(results)
    if ids != ["00_street_action_schedule_60s"]:
        raise ValueError(f"unexpected long cases: {ids}")
    case_id = ids[0]
    frames = {prefix: load_case(results, case_id, prefix)[0] for prefix in PREFIXES}
    paths = {prefix: load_case(results, case_id, prefix)[1] for prefix in PREFIXES}
    replay = {
        "fa2": bool(np.array_equal(frames["fa2_a"], frames["fa2_b"])),
        "fa3": bool(np.array_equal(frames["fa3_a"], frames["fa3_b"])),
    }
    cross_windows = []
    generated2 = frames["fa2_a"][1:]
    generated3 = frames["fa3_a"][1:]
    for start in range(0, len(generated2), 360):
        end = min(start + 360, len(generated2))
        cross_windows.append(
            {
                "end_frame": end,
                "metrics": pixel_metrics(generated2[start:end], generated3[start:end]),
                "start_frame": start,
            }
        )
    activity2 = temporal_activity(generated2)
    activity3 = temporal_activity(generated3)
    activity_ratio = (
        activity3["mean_abs_delta"] / activity2["mean_abs_delta"]
        if activity2["mean_abs_delta"]
        else math.inf
    )
    stable = (
        all(replay.values())
        and activity2["frozen_transition_fraction"] == 0.0
        and activity3["frozen_transition_fraction"] == 0.0
        and TEMPORAL_MIN_ACTIVITY_RATIO <= activity_ratio <= TEMPORAL_MAX_ACTIVITY_RATIO
    )
    return {
        "cross_backend_full": pixel_metrics(generated2, generated3),
        "cross_backend_lpips": lpips_metrics(generated2, generated3),
        "cross_backend_windows": cross_windows,
        "fa2_temporal_activity": activity2,
        "fa3_temporal_activity": activity3,
        "fa3_to_fa2_activity_ratio": activity_ratio,
        "file_sha256": {prefix: sha256_file(path) for prefix, path in paths.items()},
        "replay_bitwise_equal": replay,
        "stable": stable,
    }


def main() -> None:
    args = parse_args()
    actions = analyze_actions(args.actions_results, args.require_lpips)
    long = analyze_long(args.long_results)
    report = {
        "acceptance_thresholds": {
            "action_delta_cosine_gte": ACTION_MIN_DELTA_COSINE,
            "action_effect_norm_ratio": [
                ACTION_MIN_NORM_RATIO,
                ACTION_MAX_NORM_RATIO,
            ],
            "short_lpips_lte": SHORT_MAX_LPIPS,
            "short_max_abs_lte": SHORT_MAX_ABS,
            "short_rmse_lte": SHORT_MAX_RMSE,
            "short_ssim_gte": SHORT_MIN_SSIM,
            "temporal_activity_ratio": [
                TEMPORAL_MIN_ACTIVITY_RATIO,
                TEMPORAL_MAX_ACTIVITY_RATIO,
            ],
        },
        "actions": actions,
        "all_passed": actions["all_passed"] and long["stable"],
        "long_rollout": long,
        "schema_version": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "action_cases_passed": sum(case["passed"] for case in actions["cases"]),
                "action_cases_total": len(actions["cases"]),
                "all_passed": report["all_passed"],
                "long_rollout_stable": long["stable"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
