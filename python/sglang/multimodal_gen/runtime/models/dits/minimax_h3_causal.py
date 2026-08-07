# SPDX-License-Identifier: Apache-2.0
"""Experimental block-causal attention helpers for MiniMax H3.

This module intentionally changes only the attention visibility of the
released checkpoint.  It does not claim that the checkpoint was trained for
causal generation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import torch

_PREFIX_BLOCK_ID = -1
_PADDING_BLOCK_ID = -2
_FLEX_BLOCK_SIZE = 128
_FLEX_BLOCK_MASK_CACHE_SIZE = 8
_FLEX_BLOCK_MASK_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def _config_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"MiniMax H3 causal {name} must be a boolean")


@dataclass(frozen=True)
class MiniMaxH3CausalAttentionSpec:
    mode: str = "off"
    block_frames: int = 3
    sink_frames: int = 4
    window_frames: int = 20
    cache_block_mask: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"off", "flex", "reference"}:
            raise ValueError(
                "MiniMax H3 causal attention mode must be one of off/flex/reference"
            )
        if self.block_frames <= 0:
            raise ValueError("MiniMax H3 causal block_frames must be positive")
        if self.sink_frames < 0:
            raise ValueError("MiniMax H3 causal sink_frames must be non-negative")
        if self.window_frames <= 0:
            raise ValueError("MiniMax H3 causal window_frames must be positive")
        if not isinstance(self.cache_block_mask, bool):
            raise ValueError("MiniMax H3 causal cache_block_mask must be a boolean")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def sink_blocks(self) -> int:
        return math.ceil(self.sink_frames / self.block_frames)

    @property
    def window_blocks(self) -> int:
        return math.ceil(self.window_frames / self.block_frames)

    @property
    def effective_sink_frames(self) -> int:
        return self.sink_blocks * self.block_frames

    @property
    def effective_window_frames(self) -> int:
        return self.window_blocks * self.block_frames

    @classmethod
    def from_attention_backend_config(
        cls, config: Mapping[str, Any] | None
    ) -> "MiniMaxH3CausalAttentionSpec":
        config = config or {}
        return cls(
            mode=str(config.get("minimax_h3_causal_mode", "off")).strip().lower(),
            block_frames=int(config.get("minimax_h3_causal_block_frames", 3)),
            sink_frames=int(config.get("minimax_h3_causal_sink_frames", 4)),
            window_frames=int(config.get("minimax_h3_causal_window_frames", 20)),
            cache_block_mask=_config_bool(
                config.get("minimax_h3_causal_cache_block_mask", True),
                name="cache_block_mask",
            ),
        )


@dataclass
class MiniMaxH3CausalAttentionPlan:
    spec: MiniMaxH3CausalAttentionSpec
    block_ids: torch.Tensor
    used_length: int
    num_target_frames: int
    num_target_blocks: int
    _block_masks: dict[tuple[str, int | None, int], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _block_signature: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_ids.ndim != 1:
            raise ValueError("MiniMax H3 causal block_ids must be rank 1")
        if self.block_ids.dtype != torch.long:
            self.block_ids = self.block_ids.to(torch.long)
        if self.used_length <= 0 or self.used_length > self.block_ids.numel():
            raise ValueError("MiniMax H3 causal used_length is out of range")

    def padded_length(self) -> int:
        seq_len = int(self.block_ids.numel())
        return math.ceil(seq_len / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE

    def block_ids_for_length(
        self, length: int, *, device: torch.device
    ) -> torch.Tensor:
        if length < self.block_ids.numel():
            raise ValueError(
                f"causal attention length {length} is shorter than plan length "
                f"{self.block_ids.numel()}"
            )
        block_ids = self.block_ids.to(device=device)
        if length == block_ids.numel():
            return block_ids
        return torch.cat(
            [
                block_ids,
                torch.full(
                    (length - block_ids.numel(),),
                    _PADDING_BLOCK_ID,
                    dtype=torch.long,
                    device=device,
                ),
            ]
        )

    def get_flex_block_mask(self, *, device: torch.device, length: int):
        local_key = (device.type, device.index, length)
        cached = self._block_masks.get(local_key)
        if cached is not None:
            return cached

        global_key = None
        if self.spec.cache_block_mask:
            if self._block_signature is None:
                block_bytes = (
                    self.block_ids.detach()
                    .to(device="cpu")
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
                self._block_signature = sha256(block_bytes).digest()
            global_key = (
                device.type,
                device.index,
                length,
                self.spec.sink_blocks,
                self.spec.window_blocks,
                self._block_signature,
            )
            cached = _FLEX_BLOCK_MASK_CACHE.get(global_key)
            if cached is not None:
                _FLEX_BLOCK_MASK_CACHE.move_to_end(global_key)
                self._block_masks[local_key] = cached
                return cached

        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax H3 causal flex mode requires a PyTorch build with "
                "torch.nn.attention.flex_attention"
            ) from exc

        block_ids = self.block_ids_for_length(length, device=device)
        sink_blocks = self.spec.sink_blocks
        window_blocks = self.spec.window_blocks

        def mask_mod(_batch, _head, query_index, key_index):
            query_block = block_ids[query_index]
            key_block = block_ids[key_index]
            prefix_visible = (query_block == _PREFIX_BLOCK_ID) & (
                key_block == _PREFIX_BLOCK_ID
            )
            target_key_visible = (
                (key_block >= 0)
                & (key_block <= query_block)
                & (
                    (key_block < sink_blocks)
                    | (key_block >= query_block - window_blocks + 1)
                )
            )
            target_visible = (query_block >= 0) & (
                (key_block == _PREFIX_BLOCK_ID) | target_key_visible
            )
            padding_visible = (query_block == _PADDING_BLOCK_ID) & (
                query_index == key_index
            )
            return prefix_visible | target_visible | padding_visible

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=length,
            KV_LEN=length,
            device=device,
            _compile=device.type == "cuda",
        )
        self._block_masks[local_key] = block_mask
        if global_key is not None:
            _FLEX_BLOCK_MASK_CACHE[global_key] = block_mask
            _FLEX_BLOCK_MASK_CACHE.move_to_end(global_key)
            while len(_FLEX_BLOCK_MASK_CACHE) > _FLEX_BLOCK_MASK_CACHE_SIZE:
                _FLEX_BLOCK_MASK_CACHE.popitem(last=False)
        return block_mask


def minimax_h3_build_causal_attention_plan(
    packed: Mapping[str, Any],
    spec: MiniMaxH3CausalAttentionSpec,
) -> MiniMaxH3CausalAttentionPlan | None:
    if not spec.enabled:
        return None

    seq_len = int(packed["seq_len"])
    cu_seqlens = packed["cu_seqlens"].view(-1)
    if cu_seqlens.numel() < 2:
        raise ValueError("MiniMax H3 causal plan requires packed cu_seqlens")
    used_length = int(cu_seqlens[1])
    block_ids = torch.full((seq_len,), _PREFIX_BLOCK_ID, dtype=torch.long)
    block_ids[used_length:] = _PADDING_BLOCK_ID

    img_pos = packed["img_pos"].view(-1).to(torch.long)
    update_mask = packed["update_mask"].view(-1).to(torch.bool)
    target_img_pos = img_pos[update_mask]
    if target_img_pos.numel() == 0:
        raise ValueError("MiniMax H3 causal plan requires target video rows")

    position_ids = packed["img_position_ids"]
    target_video_times = position_ids[target_img_pos, 0]
    unique_video_times, video_frame_ids = torch.unique_consecutive(
        target_video_times,
        return_inverse=True,
    )
    num_target_frames = int(unique_video_times.numel())
    num_target_blocks = math.ceil(num_target_frames / spec.block_frames)
    block_ids[target_img_pos] = video_frame_ids.to(torch.long) // spec.block_frames

    audio_pos = packed["audio_pos"].view(-1).to(torch.long)
    raw_audio_update_mask = packed.get("audio_update_mask")
    audio_update_mask = (
        torch.ones(audio_pos.numel(), dtype=torch.bool)
        if raw_audio_update_mask is None
        else raw_audio_update_mask.view(-1).to(torch.bool)
    )
    target_audio_pos = audio_pos[audio_update_mask]
    if target_audio_pos.numel() > 0:
        target_audio_times = position_ids[target_audio_pos, 0]
        boundary_times = unique_video_times[spec.block_frames :: spec.block_frames]
        audio_block_ids = torch.bucketize(
            target_audio_times.contiguous(),
            boundary_times.contiguous(),
            right=True,
        ).clamp(max=num_target_blocks - 1)
        block_ids[target_audio_pos] = audio_block_ids.to(torch.long)

    return MiniMaxH3CausalAttentionPlan(
        spec=spec,
        block_ids=block_ids,
        used_length=used_length,
        num_target_frames=num_target_frames,
        num_target_blocks=num_target_blocks,
    )


def minimax_h3_dense_causal_mask(
    plan: MiniMaxH3CausalAttentionPlan,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    block_ids = plan.block_ids
    if device is not None:
        block_ids = block_ids.to(device=device)
    query_block = block_ids[:, None]
    key_block = block_ids[None, :]
    prefix_visible = (query_block == _PREFIX_BLOCK_ID) & (key_block == _PREFIX_BLOCK_ID)
    target_key_visible = (
        (key_block >= 0)
        & (key_block <= query_block)
        & (
            (key_block < plan.spec.sink_blocks)
            | (key_block >= query_block - plan.spec.window_blocks + 1)
        )
    )
    target_visible = (query_block >= 0) & (
        (key_block == _PREFIX_BLOCK_ID) | target_key_visible
    )
    padding_visible = (query_block == _PADDING_BLOCK_ID) & torch.eye(
        block_ids.numel(), dtype=torch.bool, device=block_ids.device
    )
    return prefix_visible | target_visible | padding_visible


def minimax_h3_reference_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    plan: MiniMaxH3CausalAttentionPlan,
    softmax_scale: float,
) -> torch.Tensor:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("MiniMax H3 causal reference requires matching Q/K/V shapes")
    if query.ndim != 3:
        raise ValueError("MiniMax H3 causal reference expects [S, H, D] Q/K/V")
    if query.shape[0] != plan.block_ids.numel():
        raise ValueError("MiniMax H3 causal reference Q/K/V length mismatch")
    if query.shape[0] > 4096:
        raise ValueError(
            "MiniMax H3 dense causal reference is limited to 4096 rows; use it "
            "for parity probes, not full-resolution generation"
        )

    mask = minimax_h3_dense_causal_mask(plan, device=query.device)
    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    )
    return output[0].transpose(0, 1)


_COMPILED_FLEX_ATTENTION = None


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax H3 causal flex mode requires a PyTorch build with "
                "torch.nn.attention.flex_attention"
            ) from exc
        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
    return _COMPILED_FLEX_ATTENTION


def minimax_h3_flex_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    plan: MiniMaxH3CausalAttentionPlan,
    softmax_scale: float,
) -> torch.Tensor:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("MiniMax H3 causal flex mode requires matching Q/K/V shapes")
    if query.ndim != 3:
        raise ValueError("MiniMax H3 causal flex mode expects [S, H, D] Q/K/V")
    if query.device.type != "cuda":
        raise RuntimeError("MiniMax H3 causal flex mode requires CUDA")
    if query.shape[0] != plan.block_ids.numel():
        raise ValueError("MiniMax H3 causal flex Q/K/V length mismatch")

    seq_len = int(query.shape[0])
    padded_len = plan.padded_length()
    if padded_len != seq_len:
        pad_shape = (padded_len - seq_len, query.shape[1], query.shape[2])
        query = torch.cat([query, query.new_zeros(pad_shape)])
        key = torch.cat([key, key.new_zeros(pad_shape)])
        value = torch.cat([value, value.new_zeros(pad_shape)])

    block_mask = plan.get_flex_block_mask(device=query.device, length=padded_len)
    output = _compiled_flex_attention()(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        scale=softmax_scale,
    )
    return output[0, :, :seq_len].transpose(0, 1)


__all__ = [
    "MiniMaxH3CausalAttentionPlan",
    "MiniMaxH3CausalAttentionSpec",
    "minimax_h3_build_causal_attention_plan",
    "minimax_h3_dense_causal_mask",
    "minimax_h3_flex_causal_attention",
    "minimax_h3_reference_causal_attention",
]
