import pytest

from tianpeng_runtime_alignment_gate import validate_alignment


def canonical():
    return {
        "local_attn_size": 32,
        "sink_size": 8,
        "rope_position_mode": "block_relative",
        "rope_max_frame_gap": 12,
        "prompt_first_frame_pin_enabled": True,
    }


def model_config():
    return {
        **canonical(),
        "sliding_window_num_frames": 32,
        "scene_cut_rope_offset": 0,
        "scene_cut_sink_enabled": False,
    }


def test_validate_alignment_accepts_tianpeng_runtime_contract():
    rows = validate_alignment(canonical(), model_config())
    assert rows
    assert all(row["pass"] for row in rows)


def test_validate_alignment_rejects_absolute_rope():
    config = model_config()
    config["rope_position_mode"] = "absolute"
    with pytest.raises(ValueError, match="rope_position_mode"):
        validate_alignment(canonical(), config)
