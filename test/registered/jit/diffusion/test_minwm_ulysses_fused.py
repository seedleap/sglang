"""Bit-exact tests for MinWM kernels around the Ulysses collective."""

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.minwm_ulysses import (
    fused_qk_rmsnorm_pack_peer_first,
    fused_rope_cache_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference_rms_norm(x, weight, eps):
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.type_as(x) * weight


def _reference_rope(x, cos, sin):
    real = x[..., 0::2].float()
    imaginary = x[..., 1::2].float()
    return (
        torch.stack(
            (
                real * cos[None, :, None] - imaginary * sin[None, :, None],
                real * sin[None, :, None] + imaginary * cos[None, :, None],
            ),
            dim=-1,
        )
        .flatten(-2)
        .type_as(x)
    )


@pytest.mark.parametrize(
    ("shape", "world_size"),
    [
        ((1, 17, 24, 128), 2),
        ((1, 5, 24, 128), 4),
    ],
)
def test_fused_qk_rmsnorm_pack_peer_first_is_bit_exact(shape, world_size):
    torch.manual_seed(71)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    hidden_size = shape[2] * shape[3]
    query_weight = torch.randn(hidden_size, dtype=query.dtype, device=query.device)
    key_weight = torch.randn_like(query_weight)

    output = fused_qk_rmsnorm_pack_peer_first(
        query,
        key,
        value,
        query_weight,
        key_weight,
        1e-5,
        world_size,
    )

    batch, sequence, global_heads, head_dim = shape
    local_heads = global_heads // world_size
    normalized_query = _reference_rms_norm(query.flatten(2), query_weight, 1e-5).view(
        shape
    )
    normalized_key = _reference_rms_norm(key.flatten(2), key_weight, 1e-5).view(shape)
    expected = torch.cat(
        tuple(
            tensor.unflatten(2, (world_size, local_heads)).permute(2, 0, 1, 3, 4)
            for tensor in (normalized_query, normalized_key, value)
        ),
        dim=-1,
    )
    assert output.shape == (
        world_size,
        batch,
        sequence,
        local_heads,
        3 * head_dim,
    )
    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    ("world_size", "current_tokens", "local_heads"),
    [(2, 17, 12), (4, 5, 6)],
)
def test_fused_rope_cache_update_first_block_is_bit_exact(
    world_size, current_tokens, local_heads
):
    del world_size
    torch.manual_seed(73)
    head_dim = 128
    qkv = torch.randn(
        1,
        current_tokens,
        local_heads,
        3 * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = qkv.chunk(3, dim=-1)
    cache_k = torch.zeros_like(key, memory_format=torch.contiguous_format)
    cache_v = torch.zeros_like(value, memory_format=torch.contiguous_format)
    rotated_k = torch.empty_like(cache_k)
    angles = torch.randn(
        current_tokens, head_dim // 2, dtype=torch.float32, device="cuda"
    )
    cos, sin = angles.cos(), angles.sin()

    expected_query = _reference_rope(query, cos, sin)
    expected_key = _reference_rope(key, cos, sin)
    output = fused_rope_cache_update(
        query,
        key,
        value,
        cache_k,
        cache_v,
        rotated_k,
        cos,
        sin,
        cos,
        sin,
        0,
        rotate_all_keys=True,
    )

    assert torch.equal(output, expected_query)
    assert torch.equal(cache_k, key)
    assert torch.equal(cache_v, value)
    assert torch.equal(rotated_k, expected_key)


def test_fused_rope_cache_update_recompute_is_deterministic():
    torch.manual_seed(77)
    current_tokens, visible_tokens, write_start = 3, 7, 4
    heads, head_dim = 6, 128
    qkv = torch.randn(
        1,
        current_tokens,
        heads,
        3 * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = qkv.chunk(3, dim=-1)
    cache_k = torch.randn(
        1, visible_tokens, heads, head_dim, dtype=query.dtype, device=query.device
    )
    cache_v = torch.randn_like(cache_k)
    rotated_k = torch.randn_like(cache_k)
    angles = torch.randn(
        visible_tokens, head_dim // 2, dtype=torch.float32, device=query.device
    )
    key_cos, key_sin = angles.cos(), angles.sin()
    query_cos = key_cos[write_start:].contiguous()
    query_sin = key_sin[write_start:].contiguous()

    first_cache_k, second_cache_k = cache_k.clone(), cache_k.clone()
    first_cache_v, second_cache_v = cache_v.clone(), cache_v.clone()
    first_rotated_k, second_rotated_k = rotated_k.clone(), rotated_k.clone()
    first = fused_rope_cache_update(
        query,
        key,
        value,
        first_cache_k,
        first_cache_v,
        first_rotated_k,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        write_start,
        rotate_all_keys=False,
    )
    second = fused_rope_cache_update(
        query,
        key,
        value,
        second_cache_k,
        second_cache_v,
        second_rotated_k,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        write_start,
        rotate_all_keys=False,
    )

    _assert_bitwise_exact(first, second, "deterministic_query")
    _assert_bitwise_exact(first_cache_k, second_cache_k, "deterministic_raw_key")
    _assert_bitwise_exact(first_cache_v, second_cache_v, "deterministic_raw_value")
    _assert_bitwise_exact(
        first_rotated_k, second_rotated_k, "deterministic_rotated_key"
    )


def test_fused_rope_cache_update_append_and_recompute_are_bit_exact():
    torch.manual_seed(79)
    old_tokens, current_tokens = 7, 5
    visible_tokens = old_tokens + current_tokens
    heads, head_dim = 6, 128
    cache_k = torch.randn(
        1, visible_tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    cache_v = torch.randn_like(cache_k)
    angles = torch.randn(
        visible_tokens, head_dim // 2, dtype=torch.float32, device="cuda"
    )
    key_cos, key_sin = angles.cos(), angles.sin()
    rotated_k = torch.empty_like(cache_k)
    rotated_k[:, :old_tokens] = _reference_rope(
        cache_k[:, :old_tokens], key_cos[:old_tokens], key_sin[:old_tokens]
    )

    qkv = torch.randn(
        1,
        current_tokens,
        heads,
        3 * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = qkv.chunk(3, dim=-1)
    query_cos = key_cos[old_tokens:].contiguous()
    query_sin = key_sin[old_tokens:].contiguous()
    expected_cache_k = cache_k.clone()
    expected_cache_v = cache_v.clone()
    expected_cache_k[:, old_tokens:] = key
    expected_cache_v[:, old_tokens:] = value
    expected_rotated = _reference_rope(expected_cache_k, key_cos, key_sin)
    expected_query = _reference_rope(query, query_cos, query_sin)

    output = fused_rope_cache_update(
        query,
        key,
        value,
        cache_k,
        cache_v,
        rotated_k,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        old_tokens,
        rotate_all_keys=True,
    )
    assert torch.equal(output, expected_query)
    assert torch.equal(cache_k, expected_cache_k)
    assert torch.equal(cache_v, expected_cache_v)
    assert torch.equal(rotated_k, expected_rotated)

    replacement_qkv = torch.randn_like(qkv)
    replacement_query, replacement_key, replacement_value = replacement_qkv.chunk(
        3, dim=-1
    )
    expected_cache_k[:, old_tokens:] = replacement_key
    expected_cache_v[:, old_tokens:] = replacement_value
    expected_rotated[:, old_tokens:] = _reference_rope(
        replacement_key, query_cos, query_sin
    )
    recompute_output = fused_rope_cache_update(
        replacement_query,
        replacement_key,
        replacement_value,
        cache_k,
        cache_v,
        rotated_k,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        old_tokens,
        rotate_all_keys=False,
    )
    assert torch.equal(
        recompute_output,
        _reference_rope(replacement_query, query_cos, query_sin),
    )
    assert torch.equal(cache_k, expected_cache_k)
    assert torch.equal(cache_v, expected_cache_v)
    assert torch.equal(rotated_k, expected_rotated)
