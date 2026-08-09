from thirdperson_actions import (
    ACTION_SEED,
    ACTION_SOURCE,
    build_action_trajectory,
    sampled_trajectory_index,
    validate_action_trajectories,
)


def _traj(traj_id: str, first_key: str, last_key: str | None = None) -> dict:
    last_key = last_key or first_key
    return {
        "traj_id": traj_id,
        "fps": 24,
        "num_frames": 129,
        "traj_type": "test_traj",
        "condition_inputs": {
            "camera_actions": (
                [[first_key] for _ in range(64)]
                + [[last_key] for _ in range(65)]
            )
        },
        "segments": [
            {"key": first_key, "start_frame": 0, "end_frame": 63, "num_frames": 64},
            {"key": last_key, "start_frame": 64, "end_frame": 128, "num_frames": 65},
        ],
    }


def _trajs() -> tuple[dict, ...]:
    return (
        _traj("traj-0", "w", "a"),
        _traj("traj-1", "a", "s"),
        _traj("traj-2", "s", "d"),
        _traj("traj-3", "d", "w"),
    )


def test_sampling_from_trajs_jsonl_is_seeded_random_without_replacement() -> None:
    ids = [
        build_action_trajectory(i, seed=7, trajectories=_trajs())["traj_id"]
        for i in range(4)
    ]

    assert ids == ["traj-3", "traj-1", "traj-0", "traj-2"]
    assert ids == [
        build_action_trajectory(i, seed=7, trajectories=_trajs())["traj_id"]
        for i in range(4)
    ]
    assert len(set(ids)) == 4
    assert ids != ["traj-0", "traj-1", "traj-2", "traj-3"]


def test_sampling_wraps_with_a_new_seeded_shuffle_when_pool_is_exhausted() -> None:
    first_cycle = [
        sampled_trajectory_index(i, pool_size=4, seed=ACTION_SEED) for i in range(4)
    ]
    second_cycle = [
        sampled_trajectory_index(i, pool_size=4, seed=ACTION_SEED) for i in range(4, 8)
    ]

    assert sorted(first_cycle) == [0, 1, 2, 3]
    assert sorted(second_cycle) == [0, 1, 2, 3]
    assert second_cycle != first_cycle


def test_action_metadata_comes_from_selected_traj() -> None:
    action = build_action_trajectory(0, seed=7, trajectories=_trajs())
    frames = action["condition_inputs"]["camera_actions"]

    assert action["traj_id"] == "traj-3"
    assert action["action_id"] == "traj-3"
    assert action["action_source"] == ACTION_SOURCE
    assert action["action_index"] == 3
    assert action["movement_key"] == "d"
    assert action["ending_movement_key"] == "w"
    assert action["movement_pair"] == "d+w"
    assert action["camera_key"] == ""
    assert action["action_seed"] == 7
    assert action["action_pattern"] == "trajs.jsonl:test_traj"
    assert len(frames) == 129
    assert frames == _trajs()[3]["condition_inputs"]["camera_actions"]


def test_validate_action_trajectories_rejects_bad_shape() -> None:
    bad = dict(_traj("bad", "w"))
    bad["num_frames"] = 128

    try:
        validate_action_trajectories([bad])
    except ValueError as error:
        assert "expected 24 FPS and 129 frames" in str(error)
    else:
        raise AssertionError("expected invalid trajectory to be rejected")
