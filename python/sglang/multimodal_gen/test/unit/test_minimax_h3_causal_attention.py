# SPDX-License-Identifier: Apache-2.0
"""Contracts for the experimental MiniMax H3 block-causal attention path."""

import importlib.util
import math

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalAttentionSpec,
    minimax_h3_build_causal_attention_plan,
    minimax_h3_dense_causal_mask,
    minimax_h3_flex_causal_attention,
    minimax_h3_reference_causal_attention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)


def _packed(*, latent_t: int = 30, audio_t: int = 192, first_frame: bool = False):
    return minimax_h3_packed_sequence(
        text_len=5,
        latent_t=latent_t,
        latent_h=4,
        latent_w=4,
        audio_t=audio_t,
        include_keyframe_cond=first_frame,
        keyframe_frame_indices=[0] if first_frame else None,
        frame_count=90 if first_frame else None,
    )


def test_causal_spec_rounds_frame_limits_to_whole_blocks():
    spec = MiniMaxH3CausalAttentionSpec(
        mode="flex",
        block_frames=3,
        sink_frames=4,
        window_frames=20,
    )
    assert spec.sink_blocks == 2
    assert spec.window_blocks == 7
    assert spec.effective_sink_frames == 6
    assert spec.effective_window_frames == 21


def test_causal_spec_parses_mask_cache_boolean():
    spec = MiniMaxH3CausalAttentionSpec.from_attention_backend_config(
        {"minimax_h3_causal_cache_block_mask": "false"}
    )
    assert not spec.cache_block_mask


def test_causal_plan_keeps_prefix_global_and_couples_audio_video_blocks():
    packed = _packed()
    spec = MiniMaxH3CausalAttentionSpec(mode="reference")
    plan = minimax_h3_build_causal_attention_plan(packed, spec)
    assert plan is not None
    assert plan.num_target_frames == 30
    assert plan.num_target_blocks == 10

    mask = minimax_h3_dense_causal_mask(plan)
    text_query = int(packed["text_pos"][0])
    text_key = int(packed["text_pos"][-1])
    assert mask[text_query, text_key]

    used_length = int(packed["cu_seqlens"][1])
    padding_row = used_length
    assert mask[padding_row, padding_row]
    assert mask[padding_row].sum() == 1
    assert not mask[text_query, padding_row]

    target_img_pos = packed["img_pos"][packed["update_mask"]]
    video_blocks = plan.block_ids[target_img_pos]
    block9_video = int(target_img_pos[(video_blocks == 9).nonzero()[0]])
    block2_video = int(target_img_pos[(video_blocks == 2).nonzero()[0]])
    block3_video = int(target_img_pos[(video_blocks == 3).nonzero()[0]])
    block0_video = int(target_img_pos[(video_blocks == 0).nonzero()[0]])
    assert mask[block9_video, text_key]
    assert mask[block9_video, block0_video]
    assert not mask[block9_video, block2_video]
    assert mask[block9_video, block3_video]

    audio_blocks = plan.block_ids[packed["audio_pos"]]
    assert set(range(plan.num_target_blocks)).issubset(set(audio_blocks.tolist()))
    shared_block = 5
    audio_row = int(packed["audio_pos"][(audio_blocks == shared_block).nonzero()[0]])
    video_row = int(target_img_pos[(video_blocks == shared_block).nonzero()[0]])
    assert mask[audio_row, video_row]
    assert mask[video_row, audio_row]

    block4_video = int(target_img_pos[(video_blocks == 4).nonzero()[0]])
    assert not mask[block4_video, audio_row]


def test_fl2va_first_frame_condition_is_a_global_prefix():
    packed = _packed(latent_t=9, first_frame=True)
    plan = minimax_h3_build_causal_attention_plan(
        packed,
        MiniMaxH3CausalAttentionSpec(mode="reference"),
    )
    assert plan is not None
    condition_rows = packed["img_pos"][~packed["update_mask"]]
    assert torch.equal(
        plan.block_ids[condition_rows],
        torch.full_like(condition_rows, -1),
    )

    target_row = int(packed["img_pos"][packed["update_mask"]][0])
    mask = minimax_h3_dense_causal_mask(plan)
    assert mask[target_row, int(condition_rows[0])]
    assert not mask[int(condition_rows[0]), target_row]


def test_dense_reference_matches_explicit_masked_softmax():
    packed = _packed(latent_t=6)
    plan = minimax_h3_build_causal_attention_plan(
        packed,
        MiniMaxH3CausalAttentionSpec(mode="reference"),
    )
    assert plan is not None
    torch.manual_seed(7)
    seq_len = int(packed["seq_len"])
    query = torch.randn(seq_len, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = 1.0 / math.sqrt(query.shape[-1])

    actual = minimax_h3_reference_causal_attention(
        query,
        key,
        value,
        plan=plan,
        softmax_scale=scale,
    )
    allowed = minimax_h3_dense_causal_mask(plan)
    scores = torch.einsum("shd,thd->hst", query, key) * scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    expected = torch.einsum("hst,thd->shd", torch.softmax(scores, dim=-1), value)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("torch.nn.attention.flex_attention") is None,
    reason="requires CUDA FlexAttention",
)
def test_bf16_flex_matches_dense_reference():
    packed = _packed(latent_t=6)
    plan = minimax_h3_build_causal_attention_plan(
        packed,
        MiniMaxH3CausalAttentionSpec(mode="flex"),
    )
    assert plan is not None
    torch.manual_seed(11)
    seq_len = int(packed["seq_len"])
    query = torch.randn(seq_len, 2, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = 1.0 / math.sqrt(query.shape[-1])

    expected = minimax_h3_reference_causal_attention(
        query,
        key,
        value,
        plan=plan,
        softmax_scale=scale,
    )
    actual = minimax_h3_flex_causal_attention(
        query,
        key,
        value,
        plan=plan,
        softmax_scale=scale,
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
