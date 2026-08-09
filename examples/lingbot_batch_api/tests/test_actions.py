from collections import Counter

import pytest

from lingbot_batch_api.actions import (
    ALL_ACTION_PAIRS,
    ActionPair,
    realtime_latent_actions,
    select_action_pairs,
    video_frame_actions,
)


def test_five_pairs_are_unique_and_reproducible_per_image():
    first = select_action_pairs(image_index=123, variants=5, action_seed=7)
    second = select_action_pairs(image_index=123, variants=5, action_seed=7)
    assert first == second
    assert len(set(first)) == 5


def test_3699x5_is_globally_balanced_within_one():
    counts = Counter(
        pair
        for image_index in range(3699)
        for pair in select_action_pairs(image_index=image_index, variants=5)
    )
    assert set((pair.movement_key, pair.camera_key) for pair in counts) == set(
        ALL_ACTION_PAIRS
    )
    assert sum(counts.values()) == 18495
    assert max(counts.values()) - min(counts.values()) == 1


def test_action_schedule_matches_production_contract():
    pair = ActionPair("w", "j")
    video = video_frame_actions(pair)
    latent = realtime_latent_actions(pair)
    assert len(video) == 129
    assert video[:57] == [["w"]] * 57
    assert video[57:72] == [[]] * 15
    assert video[72:] == [["j"]] * 57
    assert len(latent) == 33
    assert latent[0] == []
    assert latent[1:15] == [["w"]] * 14
    assert latent[15:19] == [[]] * 4
    assert latent[19:] == [["j"]] * 14


@pytest.mark.parametrize("variants", [0, 17])
def test_invalid_variant_count(variants):
    with pytest.raises(ValueError):
        select_action_pairs(image_index=0, variants=variants)
