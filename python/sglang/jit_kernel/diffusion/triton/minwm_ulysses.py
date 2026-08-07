"""MinWM kernels on either side of the Ulysses all-to-all boundary."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _fp32_mul_rn(left, right):
    """Match one eager CUDA FP32 multiply without FMA contraction."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_add_rn(left, right):
    """Match one eager CUDA FP32 add without FMA contraction."""
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_sub_rn(left, right):
    """Match one eager CUDA FP32 subtract without FMA contraction."""
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _minwm_rope_pair(real, imaginary, cos, sin):
    """Preserve the four eager pointwise arithmetic rounding boundaries."""
    real_cos = _fp32_mul_rn(real, cos)
    imaginary_sin = _fp32_mul_rn(imaginary, sin)
    real_sin = _fp32_mul_rn(real, sin)
    imaginary_cos = _fp32_mul_rn(imaginary, cos)
    return (
        _fp32_sub_rn(real_cos, imaginary_sin),
        _fp32_add_rn(real_sin, imaginary_cos),
    )


@triton.jit
def _write_fresh_key_value(
    K_ptr,
    V_ptr,
    CacheK_ptr,
    CacheV_ptr,
    fresh_key_base,
    fresh_value_base,
    cache_base,
    even_features,
    odd_features,
    half_mask,
):
    key_real = tl.load(K_ptr + fresh_key_base + even_features, mask=half_mask).to(
        tl.float32
    )
    key_imaginary = tl.load(K_ptr + fresh_key_base + odd_features, mask=half_mask).to(
        tl.float32
    )
    tl.store(CacheK_ptr + cache_base + even_features, key_real, mask=half_mask)
    tl.store(CacheK_ptr + cache_base + odd_features, key_imaginary, mask=half_mask)
    value_even = tl.load(V_ptr + fresh_value_base + even_features, mask=half_mask)
    value_odd = tl.load(V_ptr + fresh_value_base + odd_features, mask=half_mask)
    tl.store(CacheV_ptr + cache_base + even_features, value_even, mask=half_mask)
    tl.store(CacheV_ptr + cache_base + odd_features, value_odd, mask=half_mask)
    return key_real, key_imaginary


@triton.jit
def _fused_rope_cache_update_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    CacheK_ptr,
    CacheV_ptr,
    RotatedK_ptr,
    QueryCos_ptr,
    QuerySin_ptr,
    KeyCos_ptr,
    KeySin_ptr,
    QueryOut_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    cache_stride_b,
    cache_stride_s,
    cache_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    S_CURRENT: tl.constexpr,
    S_VISIBLE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    WRITE_START: tl.constexpr,
    ROTATE_ALL_KEYS: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    head = batch_head % H
    half_features = tl.arange(0, HALF_BLOCK)
    half_mask = half_features < D // 2
    even_features = 2 * half_features
    odd_features = even_features + 1

    if sequence < S_CURRENT:
        query_base = batch * q_stride_b + sequence * q_stride_s + head * q_stride_h
        real = tl.load(Q_ptr + query_base + even_features, mask=half_mask).to(
            tl.float32
        )
        imaginary = tl.load(Q_ptr + query_base + odd_features, mask=half_mask).to(
            tl.float32
        )
        cos = tl.load(
            QueryCos_ptr + sequence * (D // 2) + half_features, mask=half_mask
        ).to(tl.float32)
        sin = tl.load(
            QuerySin_ptr + sequence * (D // 2) + half_features, mask=half_mask
        ).to(tl.float32)
        output_base = (
            batch * out_stride_b + sequence * out_stride_s + head * out_stride_h
        )
        output_real, output_imaginary = _minwm_rope_pair(real, imaginary, cos, sin)
        tl.store(
            QueryOut_ptr + output_base + even_features,
            output_real,
            mask=half_mask,
        )
        tl.store(
            QueryOut_ptr + output_base + odd_features,
            output_imaginary,
            mask=half_mask,
        )

    if ROTATE_ALL_KEYS:
        key_sequence = sequence
        is_fresh = (sequence >= WRITE_START) & (sequence < WRITE_START + S_CURRENT)
    else:
        key_sequence = WRITE_START + sequence

    cache_base = (
        batch * cache_stride_b + key_sequence * cache_stride_s + head * cache_stride_h
    )

    if ROTATE_ALL_KEYS:
        if is_fresh:
            fresh_sequence = sequence - WRITE_START
            fresh_key_base = (
                batch * k_stride_b + fresh_sequence * k_stride_s + head * k_stride_h
            )
            fresh_value_base = (
                batch * v_stride_b + fresh_sequence * v_stride_s + head * v_stride_h
            )
            key_real, key_imaginary = _write_fresh_key_value(
                K_ptr,
                V_ptr,
                CacheK_ptr,
                CacheV_ptr,
                fresh_key_base,
                fresh_value_base,
                cache_base,
                even_features,
                odd_features,
                half_mask,
            )
        else:
            key_real = tl.load(
                CacheK_ptr + cache_base + even_features, mask=half_mask
            ).to(tl.float32)
            key_imaginary = tl.load(
                CacheK_ptr + cache_base + odd_features, mask=half_mask
            ).to(tl.float32)
    else:
        fresh_key_base = batch * k_stride_b + sequence * k_stride_s + head * k_stride_h
        fresh_value_base = (
            batch * v_stride_b + sequence * v_stride_s + head * v_stride_h
        )
        key_real, key_imaginary = _write_fresh_key_value(
            K_ptr,
            V_ptr,
            CacheK_ptr,
            CacheV_ptr,
            fresh_key_base,
            fresh_value_base,
            cache_base,
            even_features,
            odd_features,
            half_mask,
        )

    key_cos = tl.load(
        KeyCos_ptr + key_sequence * (D // 2) + half_features,
        mask=half_mask,
    ).to(tl.float32)
    key_sin = tl.load(
        KeySin_ptr + key_sequence * (D // 2) + half_features,
        mask=half_mask,
    ).to(tl.float32)
    rotated_real, rotated_imaginary = _minwm_rope_pair(
        key_real, key_imaginary, key_cos, key_sin
    )
    tl.store(
        RotatedK_ptr + cache_base + even_features,
        rotated_real,
        mask=half_mask,
    )
    tl.store(
        RotatedK_ptr + cache_base + odd_features,
        rotated_imaginary,
        mask=half_mask,
    )


def can_fuse_rope_cache_update(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    rotated_k: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    write_start: int,
) -> bool:
    """Return whether the post-A2A RoPE/cache layout is supported."""
    if (
        query.ndim != 4
        or query.shape != key.shape
        or query.shape != value.shape
        or cache_k.ndim != 4
        or cache_v.ndim != 4
        or rotated_k.ndim != 4
    ):
        return False
    batch, current_tokens, heads, head_dim = query.shape
    visible_tokens = cache_k.shape[1]
    return bool(
        query.is_cuda
        and torch.version.hip is None
        and query.dtype in {torch.bfloat16, torch.float16}
        and query.dtype == key.dtype == value.dtype == cache_k.dtype == cache_v.dtype
        and rotated_k.dtype == query.dtype
        and query.device
        == key.device
        == value.device
        == cache_k.device
        == cache_v.device
        == rotated_k.device
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
        and cache_k.shape == cache_v.shape == rotated_k.shape
        and cache_k.shape[0] == batch
        and cache_k.shape[2:] == (heads, head_dim)
        and cache_k.is_contiguous()
        and cache_v.is_contiguous()
        and rotated_k.is_contiguous()
        and head_dim % 2 == 0
        and 0 <= write_start <= visible_tokens
        and write_start + current_tokens <= visible_tokens
        and query_cos.shape == query_sin.shape == (current_tokens, head_dim // 2)
        and key_cos.shape == key_sin.shape == (visible_tokens, head_dim // 2)
        and query_cos.dtype
        == query_sin.dtype
        == key_cos.dtype
        == key_sin.dtype
        == torch.float32
        and query_cos.device
        == query_sin.device
        == key_cos.device
        == key_sin.device
        == query.device
        and query_cos.is_contiguous()
        and query_sin.is_contiguous()
        and key_cos.is_contiguous()
        and key_sin.is_contiguous()
    )


def fused_rope_cache_update(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    rotated_k: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    write_start: int,
    *,
    rotate_all_keys: bool,
) -> torch.Tensor:
    """Apply post-A2A RoPE while writing raw and rotated causal cache state."""
    if not can_fuse_rope_cache_update(
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
    ):
        raise ValueError("unsupported MinWM post-A2A RoPE/cache layout")

    batch, current_tokens, heads, head_dim = query.shape
    visible_tokens = cache_k.shape[1]
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    rows = max(current_tokens, visible_tokens if rotate_all_keys else current_tokens)
    half_block = triton.next_power_of_2(head_dim // 2)
    with torch.cuda.device(query.device):
        _fused_rope_cache_update_kernel[(rows, batch * heads)](
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
            output,
            *query.stride()[:3],
            *key.stride()[:3],
            *value.stride()[:3],
            *cache_k.stride()[:3],
            *output.stride()[:3],
            S_CURRENT=current_tokens,
            S_VISIBLE=visible_tokens,
            H=heads,
            D=head_dim,
            WRITE_START=write_start,
            ROTATE_ALL_KEYS=rotate_all_keys,
            HALF_BLOCK=half_block,
            num_warps=4,
        )
    return output
