"""Fused CUDA pack kernels for MinWM's peer-first Ulysses layouts."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _fused_pack_peer_first_qkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    NUMEL: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    H_GLOBAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    feature = offsets % D
    rows = offsets // D
    local_head = rows % H_LOCAL
    rows = rows // H_LOCAL
    sequence = rows % S
    rows = rows // S
    batch = rows % B
    peer = rows // B
    global_head = peer * H_LOCAL + local_head
    source_offsets = ((batch * S + sequence) * H_GLOBAL + global_head) * D + feature

    output_offsets = (offsets // D) * (3 * D) + feature
    tl.store(
        Out_ptr + output_offsets, tl.load(Q_ptr + source_offsets, mask=mask), mask=mask
    )
    tl.store(
        Out_ptr + output_offsets + D,
        tl.load(K_ptr + source_offsets, mask=mask),
        mask=mask,
    )
    tl.store(
        Out_ptr + output_offsets + 2 * D,
        tl.load(V_ptr + source_offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _fused_pack_peer_first_qk_kernel(
    Q_ptr,
    K_ptr,
    Out_ptr,
    NUMEL: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    H_GLOBAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    feature = offsets % D
    rows = offsets // D
    local_head = rows % H_LOCAL
    rows = rows // H_LOCAL
    sequence = rows % S
    rows = rows // S
    batch = rows % B
    peer = rows // B
    global_head = peer * H_LOCAL + local_head
    source_offsets = ((batch * S + sequence) * H_GLOBAL + global_head) * D + feature

    output_offsets = (offsets // D) * (2 * D) + feature
    tl.store(
        Out_ptr + output_offsets, tl.load(Q_ptr + source_offsets, mask=mask), mask=mask
    )
    tl.store(
        Out_ptr + output_offsets + D,
        tl.load(K_ptr + source_offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _fused_pack_peer_first_tensor_kernel(
    Input_ptr,
    Out_ptr,
    NUMEL: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    H_GLOBAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    feature = offsets % D
    rows = offsets // D
    local_head = rows % H_LOCAL
    rows = rows // H_LOCAL
    sequence = rows % S
    rows = rows // S
    batch = rows % B
    peer = rows // B
    global_head = peer * H_LOCAL + local_head
    source_offsets = ((batch * S + sequence) * H_GLOBAL + global_head) * D + feature
    tl.store(
        Out_ptr + offsets, tl.load(Input_ptr + source_offsets, mask=mask), mask=mask
    )


def fused_pack_peer_first_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack contiguous ``[B, S, H, D]`` Q/K/V into ``[P, B, S, H/P, 3D]``."""
    if not query.is_cuda or torch.version.hip is not None:
        raise ValueError("peer-first QKV Triton packing requires CUDA")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Q/K/V must have identical [B, S, H, D] shapes")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V must have identical dtypes")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q/K/V must be on the same device")
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
    ):
        raise ValueError("peer-first QKV Triton packing requires contiguous inputs")

    batch, sequence, global_heads, head_dim = query.shape
    if global_heads % world_size != 0:
        raise ValueError("global head count must be divisible by world_size")
    local_heads = global_heads // world_size
    packed_shape = (
        world_size,
        batch,
        sequence,
        local_heads,
        3 * head_dim,
    )
    if output is None:
        output = query.new_empty(3 * query.numel())
    elif (
        output.numel() != 3 * query.numel()
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be a matching contiguous QKV buffer")

    block_size = 256
    with torch.cuda.device(query.device):
        _fused_pack_peer_first_qkv_kernel[(triton.cdiv(query.numel(), block_size),)](
            query,
            key,
            value,
            output,
            NUMEL=query.numel(),
            B=batch,
            S=sequence,
            H_GLOBAL=global_heads,
            H_LOCAL=local_heads,
            D=head_dim,
            BLOCK_SIZE=block_size,
        )
    return output.view(packed_shape)


def _validate_inputs(
    tensors: tuple[torch.Tensor, ...], world_size: int
) -> tuple[int, int, int, int]:
    reference = tensors[0]
    if not reference.is_cuda or torch.version.hip is not None:
        raise ValueError("peer-first Triton packing requires CUDA")
    if reference.ndim != 4 or any(t.shape != reference.shape for t in tensors[1:]):
        raise ValueError("inputs must have identical [B, S, H, D] shapes")
    if any(t.dtype != reference.dtype for t in tensors[1:]):
        raise ValueError("inputs must have identical dtypes")
    if any(t.device != reference.device for t in tensors[1:]):
        raise ValueError("inputs must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("peer-first Triton packing requires contiguous inputs")
    batch, sequence, global_heads, head_dim = reference.shape
    if global_heads % world_size != 0:
        raise ValueError("global head count must be divisible by world_size")
    return batch, sequence, global_heads, head_dim


def fused_pack_peer_first_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    world_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack contiguous ``[B, S, H, D]`` Q/K into ``[P, B, S, H/P, 2D]``."""
    batch, sequence, global_heads, head_dim = _validate_inputs((query, key), world_size)
    local_heads = global_heads // world_size
    packed_shape = world_size, batch, sequence, local_heads, 2 * head_dim
    if output is None:
        output = query.new_empty(2 * query.numel())
    elif (
        output.numel() != 2 * query.numel()
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be a matching contiguous QK buffer")

    block_size = 256
    with torch.cuda.device(query.device):
        _fused_pack_peer_first_qk_kernel[(triton.cdiv(query.numel(), block_size),)](
            query,
            key,
            output,
            NUMEL=query.numel(),
            B=batch,
            S=sequence,
            H_GLOBAL=global_heads,
            H_LOCAL=local_heads,
            D=head_dim,
            BLOCK_SIZE=block_size,
        )
    return output.view(packed_shape)


def fused_pack_peer_first_tensor(
    tensor: torch.Tensor,
    world_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack contiguous ``[B, S, H, D]`` into ``[P, B, S, H/P, D]``."""
    batch, sequence, global_heads, head_dim = _validate_inputs((tensor,), world_size)
    local_heads = global_heads // world_size
    packed_shape = world_size, batch, sequence, local_heads, head_dim
    if output is None:
        output = tensor.new_empty(tensor.numel())
    elif (
        output.numel() != tensor.numel()
        or output.dtype != tensor.dtype
        or output.device != tensor.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be a matching contiguous tensor buffer")

    block_size = 256
    with torch.cuda.device(tensor.device):
        _fused_pack_peer_first_tensor_kernel[
            (triton.cdiv(tensor.numel(), block_size),)
        ](
            tensor,
            output,
            NUMEL=tensor.numel(),
            B=batch,
            S=sequence,
            H_GLOBAL=global_heads,
            H_LOCAL=local_heads,
            D=head_dim,
            BLOCK_SIZE=block_size,
        )
    return output.view(packed_shape)
