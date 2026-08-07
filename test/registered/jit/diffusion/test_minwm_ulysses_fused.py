"""Bit-exact tests for MinWM kernels around the Ulysses collective."""

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.minwm_ulysses import (
    can_fuse_rope_cache_update,
    fused_rope_cache_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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


def test_fused_rope_cache_update_nonzero_positions_are_bit_exact():
    torch.manual_seed(83)
    current_tokens, visible_tokens, write_start = 3, 8, 5
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
    rotated_k = torch.empty_like(cache_k)
    angles = torch.randn(
        visible_tokens, head_dim // 2, dtype=torch.float32, device=query.device
    )
    key_cos, key_sin = angles.cos(), angles.sin()
    query_cos = key_cos[write_start:].contiguous()
    query_sin = key_sin[write_start:].contiguous()
    expected_k = cache_k.clone()
    expected_v = cache_v.clone()
    expected_k[:, write_start:] = key
    expected_v[:, write_start:] = value

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
        write_start,
        rotate_all_keys=True,
    )

    _assert_bitwise_exact(
        output,
        _reference_rope(query, query_cos, query_sin),
        "offset_query",
    )
    _assert_bitwise_exact(cache_k, expected_k, "offset_raw_key")
    _assert_bitwise_exact(cache_v, expected_v, "offset_raw_value")
    _assert_bitwise_exact(
        rotated_k,
        _reference_rope(expected_k, key_cos, key_sin),
        "offset_rotated_key",
    )


def test_fused_rope_cache_update_rejects_unsupported_strides():
    torch.manual_seed(89)
    current_tokens, visible_tokens = 3, 5
    heads, head_dim = 6, 128
    qkv_storage = torch.randn(
        1,
        current_tokens,
        heads,
        6 * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = qkv_storage[..., ::2].chunk(3, dim=-1)
    cache_k = torch.randn(
        1, visible_tokens, heads, head_dim, dtype=query.dtype, device=query.device
    )
    cache_v = torch.randn_like(cache_k)
    rotated_k = torch.empty_like(cache_k)
    angles = torch.randn(
        visible_tokens, head_dim // 2, dtype=torch.float32, device=query.device
    )
    key_cos, key_sin = angles.cos(), angles.sin()
    query_cos = key_cos[:current_tokens].contiguous()
    query_sin = key_sin[:current_tokens].contiguous()

    assert query.stride(-1) == 2
    assert not can_fuse_rope_cache_update(
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
        0,
    )


def test_fused_rope_cache_update_rejects_malformed_cache_rank():
    current_tokens, heads, head_dim = 3, 6, 128
    qkv = torch.randn(
        1,
        current_tokens,
        heads,
        3 * head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = qkv.chunk(3, dim=-1)
    malformed_cache = torch.empty((), dtype=query.dtype, device=query.device)
    angles = torch.randn(
        current_tokens, head_dim // 2, dtype=torch.float32, device=query.device
    )
    cos, sin = angles.cos(), angles.sin()

    assert not can_fuse_rope_cache_update(
        query,
        key,
        value,
        malformed_cache,
        malformed_cache,
        malformed_cache,
        cos,
        sin,
        cos,
        sin,
        0,
    )
