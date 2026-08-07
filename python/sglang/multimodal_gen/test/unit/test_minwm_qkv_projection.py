from __future__ import annotations

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.usp import _usp_pack_peer_first_qkv
from sglang.multimodal_gen.runtime.models.dits import (
    causal_wanvideo as causal_wanvideo_module,
)
from sglang.multimodal_gen.runtime.models.dits import minwm as minwm_module
from sglang.multimodal_gen.runtime.models.dits.minwm import (
    MinWMCausalTransformer3DModel,
    MinWMCausalTransformerBlock,
    _minwm_fused_qkv_param_names_mapping,
    _minwm_qkv_load_state_dict_pre_hook,
    _minwm_should_use_fused_qkv_projection,
)


class _TupleLinear(nn.Linear):
    def forward(self, hidden_states: torch.Tensor):
        return super().forward(hidden_states), None


class _NoopAttention(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()


def _make_projection_block(dim: int, *, fused: bool):
    block = MinWMCausalTransformerBlock.__new__(MinWMCausalTransformerBlock)
    nn.Module.__init__(block)
    block.use_fused_qkv_projection = fused
    if fused:
        block.to_qkv = _TupleLinear(dim, 3 * dim, bias=True)
    else:
        block.to_q = _TupleLinear(dim, dim, bias=True)
        block.to_k = _TupleLinear(dim, dim, bias=True)
        block.to_v = _TupleLinear(dim, dim, bias=True)
    block.register_load_state_dict_pre_hook(_minwm_qkv_load_state_dict_pre_hook)
    return block


def _copy_split_weights_into_fused(split_block, fused_block) -> None:
    with torch.no_grad():
        fused_block.to_qkv.weight.copy_(
            torch.cat(
                [
                    split_block.to_q.weight,
                    split_block.to_k.weight,
                    split_block.to_v.weight,
                ],
                dim=0,
            )
        )
        fused_block.to_qkv.bias.copy_(
            torch.cat(
                [
                    split_block.to_q.bias,
                    split_block.to_k.bias,
                    split_block.to_v.bias,
                ],
                dim=0,
            )
        )


def test_minwm_fast_lane_constructs_one_physical_qkv_parameter(monkeypatch):
    monkeypatch.setattr(minwm_module, "_MINWM_FUSED_QKV_PROJECTION", True)
    monkeypatch.setattr(
        MinWMCausalTransformerBlock, "self_attention_cls", _NoopAttention
    )
    monkeypatch.setattr(
        MinWMCausalTransformerBlock, "cross_attention_cls", _NoopAttention
    )
    monkeypatch.setattr(causal_wanvideo_module, "MLP", _NoopAttention)
    block = MinWMCausalTransformerBlock(
        24,
        48,
        6,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        supported_attention_backends=set(),
        prefix="Wan.blocks.0",
    )

    assert block.use_fused_qkv_projection
    assert isinstance(block.to_qkv, ReplicatedLinear)
    assert not hasattr(block, "to_q")
    assert not hasattr(block, "to_k")
    assert not hasattr(block, "to_v")
    self_attention_keys = {
        key for key in block.state_dict() if key.startswith(("to_q", "to_k.", "to_v."))
    }
    assert self_attention_keys == {"to_qkv.weight", "to_qkv.bias"}


def test_minwm_fused_qkv_projection_is_one_linear_without_forward_cat(monkeypatch):
    torch.manual_seed(17)
    split_block = _make_projection_block(16, fused=False)
    fused_block = _make_projection_block(16, fused=True)
    _copy_split_weights_into_fused(split_block, fused_block)
    hidden_states = torch.randn(2, 5, 16)
    expected = split_block._project_qkv(hidden_states)

    original_cat = torch.cat

    def fail_forward_cat(*_args, **_kwargs):
        raise AssertionError("the fused projection must not concatenate in forward")

    monkeypatch.setattr(torch, "cat", fail_forward_cat)
    actual = fused_block._project_qkv(hidden_states)
    monkeypatch.setattr(torch, "cat", original_cat)

    for actual_shard, expected_shard in zip(actual, expected):
        torch.testing.assert_close(actual_shard, expected_shard, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 17, 8), (2, 5, 8), (1, 1, 7, 8)])
def test_minwm_fused_qkv_projection_handles_sequence_shapes(shape):
    torch.manual_seed(23)
    block = _make_projection_block(8, fused=True)
    actual = block._project_qkv(torch.randn(shape))
    assert [tensor.shape for tensor in actual] == [shape, shape, shape]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.compile coverage runs in the H200 image",
)
def test_minwm_fused_qkv_projection_compiles_on_cuda():
    torch.manual_seed(29)
    block = _make_projection_block(64, fused=True).to(
        device="cuda", dtype=torch.bfloat16
    )
    hidden_states = torch.randn((1, 17, 64), device="cuda", dtype=torch.bfloat16)
    expected = block._project_qkv(hidden_states)
    compiled = torch.compile(block._project_qkv, fullgraph=True)
    actual = compiled(hidden_states)

    for actual_shard, expected_shard in zip(actual, expected):
        torch.testing.assert_close(actual_shard, expected_shard, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("sequence_length", "num_heads", "head_dim", "sp_degree"),
    [(17, 24, 2, 1), (11, 12, 2, 2), (5, 8, 4, 4)],
)
def test_minwm_fused_qkv_views_feed_peer_first_layout(
    sequence_length, num_heads, head_dim, sp_degree
):
    torch.manual_seed(30 + sp_degree)
    dim = num_heads * head_dim
    block = _make_projection_block(dim, fused=True)
    with torch.inference_mode():
        projected = block._project_qkv(torch.randn(1, sequence_length, dim))
        query, key, value = (
            tensor.unflatten(-1, (num_heads, head_dim)).contiguous()
            for tensor in projected
        )
        output_buffer = torch.empty(3 * query.numel())
        packed = _usp_pack_peer_first_qkv(query, key, value, sp_degree, output_buffer)
    local_heads = num_heads // sp_degree
    expected = torch.cat(
        tuple(
            tensor.unflatten(2, (sp_degree, local_heads)).permute(2, 0, 1, 3, 4)
            for tensor in (query, key, value)
        ),
        dim=-1,
    )

    assert packed.shape == (
        sp_degree,
        1,
        sequence_length,
        local_heads,
        3 * head_dim,
    )
    assert packed.data_ptr() == output_buffer.data_ptr()
    assert torch.equal(packed, expected)


def test_minwm_fused_qkv_state_dict_round_trips_across_toggle_and_dtype_move():
    torch.manual_seed(31)
    split_block = _make_projection_block(8, fused=False)
    fused_block = _make_projection_block(8, fused=True)

    split_state = split_block.state_dict()
    assert set(split_state) == {
        "to_q.weight",
        "to_q.bias",
        "to_k.weight",
        "to_k.bias",
        "to_v.weight",
        "to_v.bias",
    }
    fused_block.load_state_dict(split_state, strict=True)
    fused_state = fused_block.state_dict()
    assert set(fused_state) == {"to_qkv.weight", "to_qkv.bias"}

    restored_split = _make_projection_block(8, fused=False)
    restored_split.load_state_dict(fused_state, strict=True)
    for key, expected in split_state.items():
        torch.testing.assert_close(
            restored_split.state_dict()[key], expected, rtol=0, atol=0
        )

    fused_block.to(dtype=torch.float64)
    assert fused_block.to_qkv.weight.dtype == torch.float64
    assert fused_block.to_qkv.bias.dtype == torch.float64


def test_minwm_native_and_internal_checkpoint_keys_merge_at_load_time():
    mapping = get_param_names_mapping(_minwm_fused_qkv_param_names_mapping())
    for shard_index, shard_name in enumerate(("q", "k", "v")):
        assert mapping(f"blocks.7.self_attn.{shard_name}.weight") == (
            "blocks.7.to_qkv.weight",
            shard_index,
            3,
        )
        assert mapping(f"blocks.7.to_{shard_name}.bias") == (
            "blocks.7.to_qkv.bias",
            shard_index,
            3,
        )

    source = {}
    for shard_index, shard_name in enumerate(("q", "k", "v"), start=1):
        source[f"blocks.0.self_attn.{shard_name}.weight"] = torch.full(
            (2, 4), shard_index
        )
        source[f"blocks.0.self_attn.{shard_name}.bias"] = torch.full((2,), shard_index)
    merged, _ = hf_to_custom_state_dict(source, mapping)
    assert torch.equal(
        merged["blocks.0.to_qkv.weight"],
        torch.tensor([[1] * 4] * 2 + [[2] * 4] * 2 + [[3] * 4] * 2),
    )
    assert torch.equal(merged["blocks.0.to_qkv.bias"], torch.tensor([1, 1, 2, 2, 3, 3]))


def test_minwm_saved_fused_checkpoint_splits_for_fallback_loader():
    model = MinWMCausalTransformer3DModel.__new__(MinWMCausalTransformer3DModel)
    nn.Module.__init__(model)
    model.use_fused_qkv_projection = False
    weight = torch.arange(24).reshape(6, 4)
    bias = torch.arange(6)

    split = dict(
        model.preprocess_loaded_state_dict(
            {
                "blocks.0.to_qkv.weight": weight,
                "blocks.0.to_qkv.bias": bias,
            }
        )
    )

    for shard_index, shard_name in enumerate(("q", "k", "v")):
        assert torch.equal(
            split[f"blocks.0.to_{shard_name}.weight"],
            weight.chunk(3, dim=0)[shard_index],
        )
        assert torch.equal(
            split[f"blocks.0.to_{shard_name}.bias"],
            bias.chunk(3, dim=0)[shard_index],
        )


def test_minwm_quantized_qkv_uses_safe_fallback(monkeypatch):
    monkeypatch.setattr(minwm_module, "_MINWM_FUSED_QKV_PROJECTION", True)
    assert _minwm_should_use_fused_qkv_projection(None)
    assert not _minwm_should_use_fused_qkv_projection(object())
