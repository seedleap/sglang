from thirdperson_actions import (
    ACTION_PATTERN,
    ACTION_SEED,
    build_action_trajectory,
    combo_for_case,
    combo_schedule,
    validate_assignment,
)


def test_schedule_is_seeded_and_contains_all_pairs_once() -> None:
    schedule = combo_schedule(ACTION_SEED)
    assert len(schedule) == 12
    assert len(set(schedule)) == 12
    assert all(start in "wasd" and end in "wasd" for start, end in schedule)
    assert all(start != end for start, end in schedule)
    assert schedule == combo_schedule(ACTION_SEED)


def test_each_image_gets_five_unique_pairs() -> None:
    for image_index in range(3699):
        pairs = {
            combo_for_case(image_index * 5 + case_slot, ACTION_SEED)
            for case_slot in range(5)
        }
        assert len(pairs) == 5


def test_remaining_batch_is_globally_balanced() -> None:
    result = validate_assignment(3699, 5, ACTION_SEED)
    assert sum(result["pair_case_counts"].values()) == 18495
    assert result["pair_count_min"] == 1541
    assert result["pair_count_max"] == 1542
    assert sum(result["movement_key_counts"].values()) == 36990
    assert result["movement_key_count_min"] == 9247
    assert result["movement_key_count_max"] == 9249


def test_action_pattern_and_metadata_are_exact() -> None:
    action = build_action_trajectory(1234, ACTION_SEED)
    frames = action["condition_inputs"]["camera_actions"]
    assert action["movement_key"] != action["ending_movement_key"]
    assert action["camera_key"] == ""
    assert len(frames) == 129
    assert frames[:57] == [[action["movement_key"]] for _ in range(57)]
    assert frames[57:72] == [[] for _ in range(15)]
    assert frames[72:] == [[action["ending_movement_key"]] for _ in range(57)]
    assert action["action_seed"] == ACTION_SEED
    assert action["action_pattern"] == ACTION_PATTERN
    assert action == build_action_trajectory(1234, ACTION_SEED)
