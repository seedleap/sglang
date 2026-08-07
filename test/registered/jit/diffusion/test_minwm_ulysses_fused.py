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


def _assert_bitwise_exact(actual, expected, name):
    assert actual.shape == expected.shape, name
    assert actual.dtype == expected.dtype, name
    assert actual.dtype in {torch.bfloat16, torch.float16}, name
    actual_bits = actual.view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.view(torch.int16).to(torch.int32) & 0xFFFF
    mismatch = actual_bits != expected_bits
    mismatch_count = int(mismatch.sum().item())
    if mismatch_count == 0:
        return

    def ordered(bits):
        magnitude = bits & 0x7FFF
        return torch.where(bits & 0x8000 != 0, 0x8000 - magnitude, 0x8000 + magnitude)

    ulp = (ordered(actual_bits) - ordered(expected_bits)).abs()
    absolute = (actual.float() - expected.float()).abs()
    first_flat = int(mismatch.flatten().nonzero()[0].item())
    first_index = []
    remaining = first_flat
    for size in reversed(actual.shape):
        first_index.append(remaining % size)
        remaining //= size
    first_index.reverse()
    raise AssertionError(
        f"{name}: mismatch_count={mismatch_count}, "
        f"mismatch_fraction={mismatch_count / actual.numel():.9f}, "
        f"max_abs={absolute[mismatch].max().item():.9g}, "
        f"max_ulp={ulp[mismatch].max().item()}, "
        f"first_index={tuple(first_index)}, "
        f"first_actual={actual.flatten()[first_flat].item():.9g}, "
        f"first_expected={expected.flatten()[first_flat].item():.9g}, "
        f"first_ulp={ulp.flatten()[first_flat].item()}"
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
    _assert_bitwise_exact(output, expected, "pre_qkv")


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

    _assert_bitwise_exact(output, expected_query, "first_query")
    _assert_bitwise_exact(cache_k, key, "first_raw_key")
    _assert_bitwise_exact(cache_v, value, "first_raw_value")
    _assert_bitwise_exact(rotated_k, expected_key, "first_rotated_key")


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
    _assert_bitwise_exact(output, expected_query, "append_query")
    _assert_bitwise_exact(cache_k, expected_cache_k, "append_raw_key")
    _assert_bitwise_exact(cache_v, expected_cache_v, "append_raw_value")
    _assert_bitwise_exact(rotated_k, expected_rotated, "append_rotated_key")

    replacement_qkv = torch.randn_like(qkv)
    replacement_query, replacement_key, replacement_value = replacement_qkv.chunk(
        3, dim=-1
    )
    expected_cache_k[:, old_tokens:] = replacement_key
    expected_cache_v[:, old_tokens:] = replacement_value
    expected_rotated[:, old_tokens:] = _reference_rope(
        replacement_key, query_cos, query_sin
    )
    cache_k_before_recompute = cache_k.clone()
    cache_v_before_recompute = cache_v.clone()
    rotated_k_before_recompute = rotated_k.clone()
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
    _assert_bitwise_exact(
        recompute_output,
        _reference_rope(replacement_query, query_cos, query_sin),
        "recompute_query",
    )
    _assert_bitwise_exact(cache_k, expected_cache_k, "recompute_raw_key")
    _assert_bitwise_exact(cache_v, expected_cache_v, "recompute_raw_value")
    _assert_bitwise_exact(rotated_k, expected_rotated, "recompute_rotated_key")

    repeat_output = fused_rope_cache_update(
        replacement_query,
        replacement_key,
        replacement_value,
        cache_k_before_recompute,
        cache_v_before_recompute,
        rotated_k_before_recompute,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        old_tokens,
        rotate_all_keys=False,
    )
    _assert_bitwise_exact(repeat_output, recompute_output, "recompute_query_repeat")
    _assert_bitwise_exact(cache_k_before_recompute, cache_k, "recompute_key_repeat")
    _assert_bitwise_exact(cache_v_before_recompute, cache_v, "recompute_value_repeat")
    _assert_bitwise_exact(
        rotated_k_before_recompute, rotated_k, "recompute_rotated_repeat"
    )
