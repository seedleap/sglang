import json

import pytest

from run_paired_crossover import (
    assignment,
    concurrency_is_safe,
    parse_cpu_set,
    plan,
    read_config,
)


def config(tmp_path):
    payload = {
        "sglang_git_ref": "deadbeef",
        "checkpoint_sha256": "a" * 64,
        "nvme_root": str(tmp_path / "nvme"),
        "artifact_root": str(tmp_path / "artifacts"),
        "base_port": 31000,
        "paired_reps": 3,
        "gpu_slots": [
            {"gpu": 0, "cpu_set": "0-7", "numa_node": 0},
            {"gpu": 1, "cpu_set": "8-15", "numa_node": 1},
        ],
        "cases": [
            {
                "name": "480-eager",
                "size": "832x480",
                "mode": "eager",
                "control": {"command": ["serve", "--port", "{port}"]},
                "candidate": {"command": ["serve", "--port", "{port}"]},
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_assignment_crosses_gpus():
    assert assignment(0) == {"control": 0, "candidate": 1}
    assert assignment(1) == {"control": 1, "candidate": 0}
    assert assignment(2) == {"control": 0, "candidate": 1}


def test_plan_contains_three_paired_reps(tmp_path):
    output = plan(read_config(config(tmp_path)))
    assert len(output["cases"][0]["assignments"]) == 3
    assert output["topology"] == "two_gpu_paired"


def test_plan_supports_single_gpu_sequential_abba(tmp_path):
    path = config(tmp_path)
    payload = json.loads(path.read_text())
    payload["paired_reps"] = 4
    payload["gpu_slots"] = payload["gpu_slots"][:1]
    path.write_text(json.dumps(payload))

    output = plan(read_config(path))

    assert output["topology"] == "single_gpu_sequential"
    assert output["cases"][0]["assignments"] == [
        {"control": 0, "candidate": 0},
    ] * 4


def test_rejects_too_few_reps(tmp_path):
    path = config(tmp_path)
    payload = json.loads(path.read_text())
    payload["paired_reps"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="at least 3"):
        read_config(path)


def test_concurrency_threshold_checks_both_rates():
    assert concurrency_is_safe(
        {
            "control": {"client": 0.02, "scheduler": 0.01},
            "candidate": {"client": 0.0, "scheduler": 0.02},
        },
        0.02,
    )
    assert not concurrency_is_safe(
        {
            "control": {"client": 0.01, "scheduler": 0.021},
            "candidate": {"client": 0.0, "scheduler": 0.0},
        },
        0.02,
    )


def test_cpu_sets_are_parsed_and_must_not_overlap(tmp_path):
    assert parse_cpu_set("0-2,5") == {0, 1, 2, 5}
    path = config(tmp_path)
    payload = json.loads(path.read_text())
    payload["gpu_slots"][1]["cpu_set"] = "7-15"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must not overlap"):
        read_config(path)
