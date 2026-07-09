# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM (Wan21/pipeline/causal_inference.py, realtime_session.py)

"""MinWM causal DMD denoising stage (realtime, pure T2V + PRoPE camera).

Per-chunk semantics mirror minWM ``CausalInferencePipeline``:

1. optional prompt-switch recache: replay the last ``local_attn_size`` clean
   latent frames through the model at ``context_noise`` with the *new* prompt
   (LongLive recache; sink K/V preserved under ``global_sink``),
2. 4-step DMD denoising of the chunk (re-noising between steps),
3. one clean-context forward at ``context_noise`` to refill the KV caches.

The PRoPE attention path keeps a second KV cache list, allocated and reset in
lockstep with the main cache.
"""

from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_PROMPT_UPDATED_CONDITION,
    MinWMCameraState,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
    CausalDMDForwardContext,
    CausalDMDRealtimeCacheContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MINWM_PROPE_KV_CACHE = "minwm_prope_kv_cache"
MINWM_LATENT_HISTORY = "minwm_latent_history"


class MinWMChunkNoisePreparationStage(PipelineStage):
    """Sample one chunk of starting noise for the causal T2V model.

    Draws in minWM's ``[B, F, C, H', W']`` layout before permuting to the
    SGLang ``[B, C, F, H', W']`` latent convention, so a generator seeded the
    same way as minWM's RNG produces element-identical noise (RNG fills are
    row-major over the drawn shape). This is what makes cross-stack seed
    parity possible without a noise-injection side channel.
    """

    def __init__(self, transformer, vae_config) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae_config = vae_config

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is not None:
            return batch

        arch = self.transformer.config.arch_config
        chunk_size = int(batch.realtime_chunk_size or arch.num_frames_per_block)
        spatial_ratio = int(self.vae_config.arch_config.spatial_compression_ratio)
        if batch.height is None or batch.width is None:
            raise ValueError("MinWM chunk noise preparation requires height/width")
        latent_h = int(batch.height) // spatial_ratio
        latent_w = int(batch.width) // spatial_ratio

        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0]
        device = get_local_torch_device()
        noise = torch.randn(
            (batch.batch_size, chunk_size, arch.out_channels, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        batch.latents = noise.permute(0, 2, 1, 3, 4)
        batch.raw_latent_shape = batch.latents.shape
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result


class MinWMCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising for MinWM (Wan2.1-1.3B, PRoPE camera, pure T2V)."""

    # ------------------------------------------------------------------
    # Prepared-input overrides (DMDTimestepPreparationStage supplies these)
    # ------------------------------------------------------------------

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        latents = batch.latents
        assert latents is not None, (
            "MinWM causal DMD requires prepared chunk latents. "
            "Ensure MinWMChunkNoisePreparationStage runs before this stage."
        )
        return latents

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        scheduler = batch.scheduler
        assert scheduler is not None, (
            "MinWM causal DMD requires a prepared scheduler. "
            "Ensure DMDTimestepPreparationStage runs before this stage."
        )
        return scheduler

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = batch.timesteps
        assert timesteps is not None
        return timesteps.to(device)

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        return server_args.pipeline_config.get_pos_prompt_embeds(batch)

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict[str, Any]:
        # minWM default is allow_sink_write_on_recache=false, which maps to
        # sink-protected rewrites on every forward (the protection only fires
        # when re-writing an already-cached range that overlaps the sink).
        kwargs: dict[str, Any] = {"sink_protected_rewrite": True}
        camera = server_args.pipeline_config.prepare_minwm_camera_chunk(
            batch, self.device, target_dtype
        )
        if camera is not None:
            viewmats, ks = camera
            kwargs["viewmats"] = viewmats
            kwargs["Ks"] = ks
        return kwargs

    # ------------------------------------------------------------------
    # Cache geometry
    # ------------------------------------------------------------------

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        # Restore checkpoint defaults before applying request overrides so a
        # per-request override does not leak into later sessions.
        arch_config = self.transformer.config.arch_config
        self.sink_size = int(arch_config.sink_size)
        self.sliding_window_num_frames = int(arch_config.sliding_window_num_frames)
        super()._apply_causal_cache_overrides(batch, server_args)

    # ------------------------------------------------------------------
    # PRoPE second KV cache, kept in lockstep with the main cache
    # ------------------------------------------------------------------

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        cache_ctx = super()._prepare_realtime_causal_caches(batch, server_args, ctx)
        runtime_cache = cache_ctx.cache_state.runtime_cache

        main_cache = cache_ctx.kv_cache
        prope_cache = runtime_cache.get(MINWM_PROPE_KV_CACHE)
        needs_rebuild = (
            prope_cache is None
            or len(prope_cache) != len(main_cache)
            or prope_cache[0].k.shape != main_cache[0].k.shape
        )
        if needs_rebuild:
            prope_cache = self._allocate_causal_kv_cache(
                batch_size=main_cache[0].k.shape[0],
                kv_cache_size=main_cache[0].cache_size,
                num_attention_heads=main_cache[0].k.shape[2],
                attention_head_dim=main_cache[0].k.shape[3],
                dtype=ctx.target_dtype,
                device=ctx.device,
                sink_tokens=main_cache[0].sink_tokens,
                attention_window_size=main_cache[0].attention_window_size,
            )
            runtime_cache[MINWM_PROPE_KV_CACHE] = prope_cache
        elif cache_ctx.chunk_idx == 0:
            # Main cache was reset (fresh session on same geometry); reset the
            # PRoPE lane too.
            self._reset_kv_cache(prope_cache)

        self._maybe_recache_after_prompt_switch(batch, server_args, ctx, cache_ctx)
        return cache_ctx

    # ------------------------------------------------------------------
    # Prompt-switch recache (LongLive semantics)
    # ------------------------------------------------------------------

    def _maybe_recache_after_prompt_switch(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
        cache_ctx: CausalDMDRealtimeCacheContext,
    ) -> None:
        condition_inputs = batch.condition_inputs or {}
        if not condition_inputs.get(MINWM_PROMPT_UPDATED_CONDITION):
            return
        pipeline_config = server_args.pipeline_config

        # New prompt embeds were already produced by RealtimeTextEncodingStage;
        # drop the cached cross-attn K/V so they are re-encoded (minWM resets
        # the cross-attn cache both before and after the recache forward).
        self._reset_crossattn_cache(cache_ctx.crossattn_cache)

        if not pipeline_config.recache_on_prompt_switch:
            return

        current_start_frame = cache_ctx.current_start_frame
        if current_start_frame <= 0:
            return

        runtime_cache = cache_ctx.cache_state.runtime_cache
        latent_history = runtime_cache.get(MINWM_LATENT_HISTORY)
        if latent_history is None or latent_history.shape[2] == 0:
            logger.warning(
                "MinWM prompt switch at block %s has no latent history; "
                "skipping recache",
                batch.block_idx,
            )
            return

        prope_cache = runtime_cache.get(MINWM_PROPE_KV_CACHE)
        if not pipeline_config.global_sink:
            for cache in cache_ctx.kv_cache:
                cache.k.zero_()
                cache.v.zero_()
            if prope_cache is not None:
                for cache in prope_cache:
                    cache.k.zero_()
                    cache.v.zero_()

        local_attn_size = int(
            self.transformer.config.arch_config.local_attn_size
        )
        max_recache = (
            current_start_frame
            if local_attn_size == -1
            else min(local_attn_size, current_start_frame)
        )
        num_recache_frames = min(max_recache, latent_history.shape[2])
        if num_recache_frames <= 0:
            return
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = latent_history[:, :, -num_recache_frames:]

        recache_kwargs: dict[str, Any] = dict(ctx.pos_cond_kwargs)
        recache_kwargs["sink_protected_rewrite"] = (
            not pipeline_config.allow_sink_write_on_recache
        )
        camera_state = (
            batch.session.get_or_create_state(MinWMCameraState)
            if batch.session is not None
            else None
        )
        if camera_state is not None and camera_state.viewmats_history is not None:
            if camera_state.viewmats_history.shape[1] < num_recache_frames:
                raise ValueError(
                    "MinWM recache camera history shorter than latent history: "
                    f"{camera_state.viewmats_history.shape[1]} < {num_recache_frames}"
                )
            recache_kwargs["viewmats"] = camera_state.viewmats_history[
                :, -num_recache_frames:
            ]
            recache_kwargs["Ks"] = camera_state.ks_history[:, -num_recache_frames:]
            recache_kwargs["prope_kv_cache"] = prope_cache
        else:
            recache_kwargs.pop("viewmats", None)
            recache_kwargs.pop("Ks", None)
            recache_kwargs["prope_kv_cache"] = None

        self._minwm_context_forward(
            batch,
            server_args,
            context_input=frames_to_recache.to(ctx.target_dtype),
            prompt_embeds=ctx.prompt_embeds,
            kv_cache=cache_ctx.kv_cache,
            crossattn_cache=cache_ctx.crossattn_cache,
            current_start_frame=recache_start_frame,
            pos_cond_kwargs=recache_kwargs,
            target_dtype=ctx.target_dtype,
            autocast_enabled=ctx.autocast_enabled,
        )
        self._reset_crossattn_cache(cache_ctx.crossattn_cache)
        logger.info(
            "MinWM prompt-switch recache: session_id=%s block_idx=%s "
            "recache_frames=%s recache_start_frame=%s global_sink=%s",
            batch.realtime_session_id,
            batch.block_idx,
            num_recache_frames,
            recache_start_frame,
            pipeline_config.global_sink,
        )

    # ------------------------------------------------------------------
    # Context (clean KV refill) forward with skip_final_projection
    # ------------------------------------------------------------------

    def _minwm_context_forward(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        pos_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        context_noise = server_args.pipeline_config.context_noise
        timestep = torch.full(
            (context_input.shape[0], 1),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=-1,
                attn_metadata=None,
                forward_batch=batch,
            ),
        ):
            self.transformer(
                context_input.to(target_dtype),
                prompt_embeds,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.num_token_per_frame,
                start_frame=current_start_frame,
                skip_final_projection=True,
                **pos_cond_kwargs,
            )

    def _update_causal_context_cache(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        del image_kwargs, attn_metadata
        self._minwm_context_forward(
            batch,
            server_args,
            context_input=context_input,
            prompt_embeds=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=start_frame,
            pos_cond_kwargs=pos_cond_kwargs,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

    # ------------------------------------------------------------------
    # Main chunk forward
    # ------------------------------------------------------------------

    def _append_chunk_histories(
        self,
        batch: Req,
        cache_ctx: CausalDMDRealtimeCacheContext,
        ctx: CausalDMDForwardContext,
        clean_latents: torch.Tensor,
    ) -> None:
        """Track clean latents (+ camera already tracked) for future recache."""
        runtime_cache = cache_ctx.cache_state.runtime_cache
        history = runtime_cache.get(MINWM_LATENT_HISTORY)
        if history is None:
            history = clean_latents.detach()
        else:
            history = torch.cat([history, clean_latents.detach()], dim=2)
        max_frames = int(self.transformer.config.arch_config.local_attn_size)
        if max_frames != -1 and history.shape[2] > max_frames:
            history = history[:, :, -max_frames:]
        runtime_cache[MINWM_LATENT_HISTORY] = history

        if batch.session is not None:
            camera_state = batch.session.get_or_create_state(MinWMCameraState)
            viewmats = ctx.pos_cond_kwargs.get("viewmats")
            ks = ctx.pos_cond_kwargs.get("Ks")
            if viewmats is not None and ks is not None:
                camera_state.append_history(viewmats, ks)
                if max_frames != -1:
                    camera_state.trim_history(max_frames)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        cache_ctx = self._prepare_realtime_causal_caches(batch, server_args, ctx)

        prope_cache = cache_ctx.cache_state.runtime_cache[MINWM_PROPE_KV_CACHE]
        ctx.pos_cond_kwargs["prope_kv_cache"] = (
            prope_cache if "viewmats" in ctx.pos_cond_kwargs else None
        )

        def prepare_model_input(current_latents: torch.Tensor) -> torch.Tensor:
            return current_latents

        current_latents = self._denoise_realtime_causal_chunk(
            batch,
            server_args,
            ctx=ctx,
            cache_ctx=cache_ctx,
            chunk_latents=ctx.latents,
            prepare_model_input=prepare_model_input,
            prepare_context_input=prepare_model_input,
        )

        self._append_chunk_histories(batch, cache_ctx, ctx, current_latents)
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)

        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result
