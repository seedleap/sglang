import ast
from pathlib import Path

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


def test_runtime_alignment_log_is_in_effective_minwm_cache_hook():
    source_path = (
        Path(__file__).parents[2]
        / "python/sglang/multimodal_gen/runtime/pipelines_core/stages/"
        "model_specific_stages/minwm/minwm_causal_denoising.py"
    )
    tree = ast.parse(source_path.read_text())
    stage = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MinWMCausalDMDDenoisingStage"
    )
    hooks = [
        node
        for node in stage.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_realtime_causal_caches"
    ]
    assert len(hooks) == 1
    hook_source = ast.get_source_segment(source_path.read_text(), hooks[0])
    assert hook_source is not None
    assert "MINWM_RUNTIME_ALIGNMENT" in hook_source
