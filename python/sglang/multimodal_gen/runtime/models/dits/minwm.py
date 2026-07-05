# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM (Wan2.1-1.3B causal DMD world model):
# wan/modules/causal_model.py + wan/modules/prope.py

"""MinWM causal Wan2.1-1.3B DiT with PRoPE camera conditioning.

Differences from ``CausalWanTransformer3DModel`` that make this a separate
implementation rather than a subclass override of the blocks:

1. The self-attention KV cache stores *un-roped* keys. Temporal RoPE is applied
   to the assembled cache window on every forward, with frame positions
   remapped per ``rope_position_mode`` ("block_relative" in the stage4 eval
   config: sink frames keep positions ``[0, sink)`` and the rolled tail maps to
   ``[sink, local_attn_size)`` once the window saturates). The stock cache
   assumes rope-then-cache, which cannot express this.
2. A second, PRoPE-transformed attention path with its own KV cache
   (``prope_kv_cache``). Its output is fused through a learned zero-init
   projection: ``out = to_out(x) + prope_o(apply_fn_o(x_prope))``.
3. Temporal RoPE is computed in float64 complex, matching minWM's
   ``causal_rope_apply`` bit-for-bit (cross-stack parity requirement).
"""

import math
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.minwm import MinWMVideoConfig
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.minwm_rope import (
    minwm_cache_rope_frame_indices,
    minwm_query_rope_frame_indices,
    minwm_rope_apply,
    minwm_rope_params,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.prope import (
    PropeApplyFn,
    expand_camera_params_to_tokens,
    prope_prepare_apply_fns,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MinWMTransformerBlock(nn.Module):
    """One causal Wan block with the PRoPE second attention path.

    Attention algebra mirrors minWM ``CausalWanAttentionBlock`` /
    ``CausalWanSelfAttention`` in inference (kv-cache) mode.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str,
        cross_attn_norm: bool,
        eps: float,
        supported_attention_backends: set[AttentionBackendEnum] | None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        assert qk_norm == "rms_norm_across_heads", (
            "MinWM uses WanRMSNorm over the full hidden dim"
        )
        assert cross_attn_norm is True

        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.dim_head = dim // num_heads

        self.to_q = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config)
        self.to_k = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config)
        self.to_v = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config)
        self.to_out = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config)
        # PRoPE fusion projection; zero-init in minWM but always present in
        # stage4 checkpoints, so no init special-casing is needed here.
        self.prope_o = nn.Linear(dim, dim, bias=True)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.dim_head,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FA,
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

        cross_attn_backends = {
            b for b in (supported_attention_backends or set()) if not b.is_sparse
        }
        self.attn2 = WanT2VCrossAttention(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            supported_attention_backends=cross_attn_backends,
            quant_config=quant_config,
        )

        self.ffn = MLP(
            dim, ffn_dim, act_type="gelu_pytorch_tanh", quant_config=quant_config
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _cross_attn_with_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        crossattn_cache: CrossAttentionKVCache | None,
    ) -> torch.Tensor:
        attn2 = self.attn2
        q, _ = attn2.to_q(hidden_states)
        q = attn2.norm_q(q)
        q = q.unflatten(2, (attn2.local_num_heads, attn2.head_dim))

        if crossattn_cache is not None and crossattn_cache.is_init:
            k = crossattn_cache.k
            v = crossattn_cache.v
        else:
            k, _ = attn2.to_k(encoder_hidden_states)
            k = attn2.norm_k(k)
            k = k.unflatten(2, (attn2.local_num_heads, attn2.head_dim))
            v, _ = attn2.to_v(encoder_hidden_states)
            v = v.unflatten(2, (attn2.local_num_heads, attn2.head_dim))
            if crossattn_cache is not None:
                crossattn_cache.store(k, v)

        out = attn2.attn(q, k, v)
        out = out.flatten(2)
        out, _ = attn2.to_out(out)
        return out

    def _windowed_self_attention(
        self,
        *,
        query: torch.Tensor,  # [B, L, H, D] normed, un-roped
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: CausalSelfAttentionKVCache,
        prope_kv_cache: CausalSelfAttentionKVCache | None,
        current_start_tokens: int,
        current_start_frame: int,
        num_new_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        freqs: torch.Tensor,
        rope_position_mode: str,
        local_attn_size: int,
        sink_size: int,
        prope_apply_fns: tuple[PropeApplyFn, PropeApplyFn, PropeApplyFn] | None,
        sink_protected_rewrite: bool,
        update_cache_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        frame_seqlen = post_patch_height * post_patch_width

        prope_view: CausalAttentionKVView | None = None
        if prope_apply_fns is not None:
            apply_fn_q, apply_fn_kv, apply_fn_o = prope_apply_fns
            # prope transforms expect (B, H, L, D)
            q_p = apply_fn_q(query.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            k_p = apply_fn_kv(key.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            v_p = apply_fn_kv(value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            assert prope_kv_cache is not None
            prope_view = prope_kv_cache.update_and_get_attention_kv(
                key=k_p,
                value=v_p,
                current_chunk_start=current_start_tokens,
                sink_protected_rewrite=sink_protected_rewrite,
                debug_name="MinWM PRoPE KV cache",
            )

        if (
            local_attn_size != -1
            and kv_cache.cache_size > local_attn_size * frame_seqlen
        ):
            raise NotImplementedError(
                "MinWM attention windows the cache to local_attn_size with a "
                "sink+tail split; a KV cache larger than "
                "local_attn_size * frame_seqlen is not supported yet "
                f"(cache_size={kv_cache.cache_size}, local_attn_size={local_attn_size})."
            )

        # Main path: cache raw keys, rope the assembled window each call.
        view = kv_cache.update_and_get_attention_kv(
            key=key,
            value=value,
            current_chunk_start=current_start_tokens,
            sink_protected_rewrite=sink_protected_rewrite,
            debug_name="MinWM KV cache",
        )
        if update_cache_only:
            return None

        device = query.device
        num_window_frames = view.k.shape[1] // frame_seqlen
        cache_indices = minwm_cache_rope_frame_indices(
            num_cache_frames=num_window_frames,
            global_end_frame=view.visible_global_end // frame_seqlen,
            sink_size=sink_size,
            local_attn_size=local_attn_size,
            rope_position_mode=rope_position_mode,
            device=device,
        )
        query_indices = minwm_query_rope_frame_indices(
            local_start_frame=view.local_start_index // frame_seqlen,
            local_end_frame=view.visible_local_end // frame_seqlen,
            num_new_frames=num_new_frames,
            current_start_frame=current_start_frame,
            local_attn_size=local_attn_size,
            rope_position_mode=rope_position_mode,
            device=device,
        )
        roped_query = minwm_rope_apply(
            query, query_indices, post_patch_height, post_patch_width, freqs
        ).type_as(value)
        roped_window_k = minwm_rope_apply(
            view.k, cache_indices, post_patch_height, post_patch_width, freqs
        ).type_as(value)

        x = self.attn(roped_query, roped_window_k, view.v)

        if prope_view is not None:
            x_prope = self.attn(q_p, prope_view.k, prope_view.v)
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            return x, x_prope
        return x, None

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, C]
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,  # [B, F_t, 6, C]
        *,
        kv_cache: CausalSelfAttentionKVCache,
        crossattn_cache: CrossAttentionKVCache | None,
        current_start_tokens: int,
        current_start_frame: int,
        num_new_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        freqs: torch.Tensor,
        rope_position_mode: str,
        local_attn_size: int,
        sink_size: int,
        prope_kv_cache: CausalSelfAttentionKVCache | None = None,
        prope_apply_fns: tuple[PropeApplyFn, PropeApplyFn, PropeApplyFn] | None = None,
        sink_protected_rewrite: bool = False,
        update_cache_only: bool = False,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_mod_frames = temb.shape[1]
        frame_seqlen_mod = hidden_states.shape[1] // num_mod_frames
        orig_dtype = hidden_states.dtype

        e = (self.scale_shift_table + temb.float()).chunk(6, dim=2)
        # e[i]: [B, F_t, 1, C]

        norm_hidden_states = (
            (
                self.norm1(hidden_states.float()).unflatten(
                    dim=1, sizes=(num_mod_frames, frame_seqlen_mod)
                )
                * (1 + e[1])
                + e[0]
            )
            .flatten(1, 2)
            .to(orig_dtype)
        )

        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query.unflatten(2, (self.num_attention_heads, self.dim_head))
        key = key.unflatten(2, (self.num_attention_heads, self.dim_head))
        value = value.unflatten(2, (self.num_attention_heads, self.dim_head))

        attn_result = self._windowed_self_attention(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            prope_kv_cache=prope_kv_cache,
            current_start_tokens=current_start_tokens,
            current_start_frame=current_start_frame,
            num_new_frames=num_new_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
            freqs=freqs,
            rope_position_mode=rope_position_mode,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            prope_apply_fns=prope_apply_fns,
            sink_protected_rewrite=sink_protected_rewrite,
            update_cache_only=update_cache_only,
        )
        if attn_result is None:
            # KV caches updated; block output is discarded by the caller.
            return hidden_states
        x, x_prope = attn_result

        attn_output, _ = self.to_out(x.flatten(2))
        if x_prope is not None:
            attn_output = attn_output + self.prope_o(x_prope.flatten(2))

        # x = x + y * e2   (residual with per-frame gate, minWM order)
        hidden_states = hidden_states + (
            attn_output.unflatten(dim=1, sizes=(num_mod_frames, frame_seqlen_mod))
            * e[2]
        ).flatten(1, 2).to(orig_dtype)

        # Cross-attention (pre-norm norm3, plain residual).
        cross_out = self._cross_attn_with_cache(
            self.norm3(hidden_states.float()).to(orig_dtype),
            encoder_hidden_states,
            crossattn_cache,
        )
        hidden_states = hidden_states + cross_out

        # FFN with modulation.
        ffn_input = (
            (
                self.norm2(hidden_states.float()).unflatten(
                    dim=1, sizes=(num_mod_frames, frame_seqlen_mod)
                )
                * (1 + e[4])
                + e[3]
            )
            .flatten(1, 2)
            .to(orig_dtype)
        )
        ffn_output = self.ffn(ffn_input)
        hidden_states = hidden_states + (
            ffn_output.unflatten(dim=1, sizes=(num_mod_frames, frame_seqlen_mod))
            * e[5]
        ).flatten(1, 2).to(orig_dtype)

        return hidden_states.to(orig_dtype)


class MinWMCausalTransformer3DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = MinWMVideoConfig()._fsdp_shard_conditions
    _compile_conditions = MinWMVideoConfig()._compile_conditions
    _supported_attention_backends = MinWMVideoConfig()._supported_attention_backends
    param_names_mapping = MinWMVideoConfig().param_names_mapping
    reverse_param_names_mapping = MinWMVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = MinWMVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: MinWMVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.num_channels_latents
        self.patch_size = arch.patch_size
        self.text_len = arch.text_len
        self.local_attn_size = arch.local_attn_size
        self.sink_size = arch.sink_size
        self.rope_position_mode = arch.rope_position_mode
        self.num_frame_per_block = arch.num_frames_per_block
        self.independent_first_frame = False

        self.patch_embedding = PatchEmbed(
            in_chans=arch.in_channels,
            embed_dim=inner_dim,
            patch_size=arch.patch_size,
            flatten=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=arch.freq_dim,
            text_embed_dim=arch.text_dim,
            image_embed_dim=arch.image_dim,
        )
        self.blocks = nn.ModuleList(
            [
                MinWMTransformerBlock(
                    inner_dim,
                    arch.ffn_dim,
                    arch.num_attention_heads,
                    arch.qk_norm,
                    arch.cross_attn_norm,
                    arch.eps,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(arch.num_layers)
            ]
        )
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=arch.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, arch.out_channels * math.prod(arch.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # minWM rope frequency table: [t, h, w] split of the head dim.
        d = self.hidden_size // self.num_attention_heads
        self.freqs = torch.cat(
            [
                minwm_rope_params(arch.rope_max_seq_len, d - 4 * (d // 6)),
                minwm_rope_params(arch.rope_max_seq_len, 2 * (d // 6)),
                minwm_rope_params(arch.rope_max_seq_len, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.__post_init__()
        self.layer_names = ["blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, C, F, H, W]
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[CausalSelfAttentionKVCache] | None = None,
        crossattn_cache: list[CrossAttentionKVCache] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        viewmats: torch.Tensor | None = None,  # [B, F, 4, 4] w2c
        Ks: torch.Tensor | None = None,  # [B, F, 3, 3]
        prope_kv_cache: list[CausalSelfAttentionKVCache] | None = None,
        sink_protected_rewrite: bool = False,
        skip_final_projection: bool = False,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise NotImplementedError(
                "MinWMCausalTransformer3DModel only supports kv-cache (streaming) "
                "inference; the offline flex-attention path is not ported."
            )
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_seqlen = post_patch_height * post_patch_width

        if self.freqs.device != hidden_states.device:
            self.freqs = self.freqs.to(hidden_states.device)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(
            dim=0, sizes=timestep.shape
        )
        assert encoder_hidden_states.dtype == orig_dtype

        prope_apply_fns = None
        if viewmats is not None:
            assert Ks is not None, "PRoPE requires both viewmats and Ks"
            assert prope_kv_cache is not None, (
                "PRoPE streaming inference requires prope_kv_cache"
            )
            viewmats_tok, ks_tok = expand_camera_params_to_tokens(
                viewmats.to(device=hidden_states.device),
                Ks.to(device=hidden_states.device),
                frame_seqlen=frame_seqlen,
            )
            # The P matrices depend only on the camera params, so build the
            # transforms once per forward and share them across all 30 layers
            # (minWM recomputes per layer; the values are identical).
            prope_apply_fns = prope_prepare_apply_fns(
                head_dim=self.attention_head_dim,
                viewmats=viewmats_tok,
                Ks=ks_tok,
            )

        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                kv_cache=kv_cache[block_index],
                crossattn_cache=(
                    crossattn_cache[block_index]
                    if crossattn_cache is not None
                    else None
                ),
                current_start_tokens=current_start,
                current_start_frame=start_frame,
                num_new_frames=post_patch_num_frames,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
                freqs=self.freqs,
                rope_position_mode=self.rope_position_mode,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                prope_kv_cache=(
                    prope_kv_cache[block_index] if prope_kv_cache is not None else None
                ),
                prope_apply_fns=prope_apply_fns,
                sink_protected_rewrite=sink_protected_rewrite,
                update_cache_only=skip_final_projection
                and block_index == len(self.blocks) - 1,
            )

        if skip_final_projection:
            return hidden_states

        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = MinWMCausalTransformer3DModel
