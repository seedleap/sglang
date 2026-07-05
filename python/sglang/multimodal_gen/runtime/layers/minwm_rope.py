# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM wan/modules/causal_model.py (causal_rope_apply, rope_params)

"""Temporal RoPE helpers for the MinWM causal DiT.

Standalone (torch-only) so the block-relative window semantics can be
unit-tested against the minWM reference without the full runtime import chain.
"""

from __future__ import annotations

import torch


def minwm_rope_params(
    max_seq_len: int, dim: int, theta: float = 10000
) -> torch.Tensor:
    """Verbatim port of minWM ``rope_params`` (float64 complex frequencies)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def minwm_rope_apply(
    x: torch.Tensor,  # [B, L, n, d], frame-major token order
    frame_indices: torch.Tensor,  # [F] temporal rope position per frame
    height: int,
    width: int,
    freqs: torch.Tensor,  # [max_seq_len, d/2] complex128
) -> torch.Tensor:
    """Verbatim port of minWM ``causal_rope_apply`` with explicit frame indices.

    Complex multiplication in float64, [t, h, w] frequency split
    ``[c - 2*(c//3), c//3, c//3]`` over ``c = d // 2``.
    """
    b, seq_len, n, d = x.shape
    c = d // 2
    f = frame_indices.shape[0]
    assert seq_len == f * height * width, (
        f"rope input length {seq_len} != {f} frames * {height}x{width}"
    )

    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_temporal = (
        freqs_split[0][frame_indices.long()]
        .view(f, 1, 1, -1)
        .expand(f, height, width, -1)
    )
    freqs_i = torch.cat(
        [
            freqs_temporal,
            freqs_split[1][:height].view(1, height, 1, -1).expand(f, height, width, -1),
            freqs_split[2][:width].view(1, 1, width, -1).expand(f, height, width, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    output = []
    for i in range(b):
        x_i = torch.view_as_complex(
            x[i].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(x_i)
    return torch.stack(output).type_as(x)


def minwm_cache_rope_frame_indices(
    *,
    num_cache_frames: int,
    global_end_frame: int,
    sink_size: int,
    local_attn_size: int,
    rope_position_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Temporal rope positions for the assembled cache window.

    Port of minWM ``cache_frame_indices``. With the stage4 config the cache
    buffer equals the attention window (16 frames), so the block_relative
    branch reduces to ``arange(num_cache_frames)``; the remap branches are kept
    for cache-size overrides larger than the window.
    """
    if rope_position_mode == "block_relative":
        max_attention_frames = (
            local_attn_size if local_attn_size != -1 else num_cache_frames
        )
        if num_cache_frames <= max_attention_frames:
            return torch.arange(0, num_cache_frames, device=device)

        indices = torch.zeros(num_cache_frames, dtype=torch.long, device=device)
        if sink_size > 0:
            sink_frames = min(sink_size, max_attention_frames, num_cache_frames)
            tail_frames = max_attention_frames - sink_frames
            if tail_frames > 0:
                tail_start = num_cache_frames - tail_frames
                indices[tail_start:num_cache_frames] = torch.arange(
                    sink_frames, max_attention_frames, device=device
                )
            indices[:sink_frames] = torch.arange(0, sink_frames, device=device)
        else:
            tail_start = num_cache_frames - max_attention_frames
            indices[tail_start:num_cache_frames] = torch.arange(
                0, max_attention_frames, device=device
            )
        return indices

    end_frame = global_end_frame
    if sink_size > 0 and end_frame > num_cache_frames:
        sink_frames = min(sink_size, num_cache_frames)
        tail_frames = num_cache_frames - sink_frames
        sink_indices = torch.arange(0, sink_frames, device=device)
        if tail_frames == 0:
            return sink_indices
        tail_start = end_frame - tail_frames
        tail_indices = torch.arange(tail_start, end_frame, device=device)
        return torch.cat([sink_indices, tail_indices], dim=0)

    start_frame = end_frame - num_cache_frames
    return torch.arange(start_frame, end_frame, device=device)


def minwm_query_rope_frame_indices(
    *,
    local_start_frame: int,
    local_end_frame: int,
    num_new_frames: int,
    current_start_frame: int,
    local_attn_size: int,
    rope_position_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Temporal rope positions for the current chunk's queries.

    Port of minWM ``query_frame_indices``.
    """
    if rope_position_mode == "block_relative":
        if local_attn_size != -1 and local_end_frame > local_attn_size:
            start_frame = max(0, local_attn_size - num_new_frames)
        else:
            start_frame = local_start_frame
    else:
        start_frame = current_start_frame
    return torch.arange(start_frame, start_frame + num_new_frames, device=device)
