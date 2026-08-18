from __future__ import annotations

from pathlib import Path

import analyze_fa3_quality as quality
import numpy as np


def _write_case(
    results: Path,
    case_id: str,
    frames_by_prefix: dict[str, np.ndarray],
) -> None:
    case_dir = results / "cases" / case_id
    case_dir.mkdir(parents=True)
    for prefix, frames in frames_by_prefix.items():
        np.save(case_dir / f"{prefix}.npy", frames, allow_pickle=False)


def _zero_lpips(reference: np.ndarray, candidate: np.ndarray) -> dict:
    assert reference.shape == candidate.shape
    return {"max": 0.0, "mean": 0.0, "sample_indices": [0], "values": [0.0]}


def test_analyzer_accepts_replay_exact_and_preserved_action_effect(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(quality, "lpips_metrics", _zero_lpips)
    actions = tmp_path / "actions"
    long = tmp_path / "long"
    idle = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    action = idle.copy()
    action[1:] = 4
    _write_case(
        actions, "00_idle_street", {prefix: idle for prefix in quality.PREFIXES}
    )
    _write_case(
        actions,
        "01_forward_w_street",
        {prefix: action for prefix in quality.PREFIXES},
    )
    trajectory = np.stack(
        [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(9)]
    )
    _write_case(
        long,
        "00_street_action_schedule_60s",
        {prefix: trajectory for prefix in quality.PREFIXES},
    )

    action_report = quality.analyze_actions(actions, require_lpips=True)
    long_report = quality.analyze_long(long)

    assert action_report["all_passed"] is True
    assert all(case["passed"] for case in action_report["cases"])
    effect = action_report["cases"][1]["action_effect"]
    assert effect["delta_cosine_similarity"] == 1.0
    assert effect["fa3_to_fa2_effect_norm_ratio"] == 1.0
    assert long_report["stable"] is True
    assert long_report["replay_bitwise_equal"] == {"fa2": True, "fa3": True}


def test_analyzer_rejects_nonrepeatable_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(quality, "lpips_metrics", _zero_lpips)
    actions = tmp_path / "actions"
    idle = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    fa3_replay = idle.copy()
    fa3_replay[-1, 0, 0, 0] = 1
    _write_case(
        actions,
        "00_idle_street",
        {
            "fa2_a": idle,
            "fa2_b": idle,
            "fa3_a": idle,
            "fa3_b": fa3_replay,
        },
    )

    report = quality.analyze_actions(actions, require_lpips=True)

    assert report["all_passed"] is False
    assert report["cases"][0]["replay_bitwise_equal"]["fa3"] is False
