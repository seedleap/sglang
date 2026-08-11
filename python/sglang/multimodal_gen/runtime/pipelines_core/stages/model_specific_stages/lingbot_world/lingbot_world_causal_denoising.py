# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""LingBot-World causal DMD denoising stage."""

from collections.abc import Callable
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import get_sp_parallel_rank
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
    CausalDMDDenoisingStage,
    CausalDMDForwardContext,
    CausalDMDRealtimeCacheContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.constants import (
    LINGBOT_C2WS_PLUCKER_EMB_CACHE,
    LINGBOT_CAM_CONDITIONER_CACHE,
    LINGBOT_CAMERA_ACTIONS_CONDITION,
    LINGBOT_INTERACTIVE_KV_WINDOW_CACHE,
    LINGBOT_PROMPT_UPDATED_CONDITION,
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


def _cuda_graph_tensor_signature(value: torch.Tensor) -> tuple:
    return (value.shape, value.stride(), value.dtype, value.device)


def _static_cuda_graph_tensor(value: torch.Tensor) -> torch.Tensor:
    static = torch.empty_strided(
        value.shape,
        value.stride(),
        dtype=value.dtype,
        device=value.device,
    )
    static.copy_(value)
    return static


def _static_cuda_graph_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    c2ws = inputs["c2ws_plucker_emb"]
    cam_shifts = inputs["cam_conditioner_scale_shifts"]
    return {
        "freqs_cis": tuple(
            _static_cuda_graph_tensor(value) for value in inputs["freqs_cis"]
        ),
        "time_embeddings": tuple(
            _static_cuda_graph_tensor(value) for value in inputs["time_embeddings"]
        ),
        "c2ws_plucker_emb": (None if c2ws is None else _static_cuda_graph_tensor(c2ws)),
        "cam_conditioner_scale_shifts": (
            None
            if cam_shifts is None
            else [
                (
                    _static_cuda_graph_tensor(scale),
                    _static_cuda_graph_tensor(shift),
                )
                for scale, shift in cam_shifts
            ]
        ),
    }


def _cuda_graph_input_source_key(inputs: dict[str, Any], *names: str) -> tuple:
    def tensor_key(value: torch.Tensor) -> tuple:
        # Tensors allocated under torch.inference_mode() deliberately do not
        # expose a version counter.  Identity and storage address are enough
        # for the request-scoped prepared-input caches used here; retain the
        # version when PyTorch makes it available so ordinary in-place updates
        # still invalidate the copy cache.
        try:
            version = value._version
        except RuntimeError:
            version = None
        return (id(value), value.data_ptr(), version)

    keys = []
    for name in names:
        value = inputs[name]
        if value is None:
            keys.append((name, None))
        elif isinstance(value, torch.Tensor):
            keys.append((name, tensor_key(value)))
        else:
            keys.append(
                (
                    name,
                    tuple(
                        tensor_key(item)
                        for group in value
                        for item in (group if isinstance(group, tuple) else (group,))
                    ),
                )
            )
    return tuple(keys)


def _copy_cuda_graph_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    if _cuda_graph_tensor_signature(target) != _cuda_graph_tensor_signature(source):
        raise RuntimeError("LingBot CUDA graph input shape changed")
    target.copy_(source)


class _LingBotCudaGraphRunner:
    """One full-DiT graph bound to a saturated LingBot session cache."""

    def __init__(self, key: tuple) -> None:
        self.key = key
        self.graph = None
        self.output = None
        self.static_latent = None
        self.static_prompt = None
        self.static_timestep = None
        self.static_inputs = None
        self.capture_stream = None
        self.pool = None
        self.replay_count = 0
        self._last_rope_source = None
        self._last_time_source = None
        self._last_camera_source = None
        self._source_inputs = None

    def _copy_prepared_inputs(self, inputs: dict[str, Any]) -> None:
        static_inputs = self.static_inputs
        if static_inputs is None:
            raise RuntimeError("LingBot CUDA graph has no static inputs")

        rope_source = _cuda_graph_input_source_key(inputs, "freqs_cis")
        if rope_source != self._last_rope_source:
            for target, source in zip(
                static_inputs["freqs_cis"], inputs["freqs_cis"], strict=True
            ):
                _copy_cuda_graph_tensor(target, source)
            self._last_rope_source = rope_source

        time_source = _cuda_graph_input_source_key(inputs, "time_embeddings")
        if time_source != self._last_time_source:
            for target, source in zip(
                static_inputs["time_embeddings"],
                inputs["time_embeddings"],
                strict=True,
            ):
                _copy_cuda_graph_tensor(target, source)
            self._last_time_source = time_source

        camera_source = _cuda_graph_input_source_key(
            inputs,
            "c2ws_plucker_emb",
            "cam_conditioner_scale_shifts",
        )
        if camera_source != self._last_camera_source:
            static_c2ws = static_inputs["c2ws_plucker_emb"]
            current_c2ws = inputs["c2ws_plucker_emb"]
            if static_c2ws is None or current_c2ws is None:
                if static_c2ws is not current_c2ws:
                    raise RuntimeError("LingBot CUDA graph camera input changed")
            else:
                _copy_cuda_graph_tensor(static_c2ws, current_c2ws)

            static_shifts = static_inputs["cam_conditioner_scale_shifts"]
            current_shifts = inputs["cam_conditioner_scale_shifts"]
            if static_shifts is None or current_shifts is None:
                if static_shifts is not current_shifts:
                    raise RuntimeError("LingBot CUDA graph camera shifts changed")
            else:
                if len(static_shifts) != len(current_shifts):
                    raise RuntimeError("LingBot CUDA graph camera block count changed")
                for static_pair, current_pair in zip(
                    static_shifts, current_shifts, strict=True
                ):
                    for target, source in zip(static_pair, current_pair, strict=True):
                        _copy_cuda_graph_tensor(target, source)
            self._last_camera_source = camera_source

        # Retain the source tensors until the next update so allocator pointer
        # reuse cannot make a new chunk look identical to the previous source.
        self._source_inputs = inputs

    def run(
        self,
        *,
        latent: torch.Tensor,
        prompt: torch.Tensor,
        timestep: torch.Tensor,
        prepared_inputs: dict[str, Any],
        capture_forward: (
            Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
                torch.Tensor,
            ]
            | None
        ),
    ) -> torch.Tensor:
        if self.graph is not None:
            self.static_latent.copy_(latent)
            self.static_prompt.copy_(prompt)
            self.static_timestep.copy_(timestep)
            self._copy_prepared_inputs(prepared_inputs)
            self.graph.replay()
            self.replay_count += 1
            if self.replay_count == 1 or self.replay_count % 100 == 0:
                logger.info(
                    "LingBot CUDA graph replay rank=%d count=%d",
                    get_sp_parallel_rank(),
                    self.replay_count,
                )
            return self.output

        self.static_latent = _static_cuda_graph_tensor(latent)
        self.static_prompt = _static_cuda_graph_tensor(prompt)
        self.static_timestep = _static_cuda_graph_tensor(timestep)
        self.static_inputs = _static_cuda_graph_inputs(prepared_inputs)
        self._last_rope_source = _cuda_graph_input_source_key(
            prepared_inputs, "freqs_cis"
        )
        self._last_time_source = _cuda_graph_input_source_key(
            prepared_inputs, "time_embeddings"
        )
        self._last_camera_source = _cuda_graph_input_source_key(
            prepared_inputs,
            "c2ws_plucker_emb",
            "cam_conditioner_scale_shifts",
        )
        self._source_inputs = prepared_inputs
        if capture_forward is None:
            raise RuntimeError("LingBot CUDA graph capture callable is missing")

        self.capture_stream = torch.cuda.Stream(device=latent.device)
        current_stream = torch.cuda.current_stream(latent.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):
                capture_forward(
                    self.static_latent,
                    self.static_prompt,
                    self.static_timestep,
                    self.static_inputs,
                )
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(latent.device)

        self.graph = torch.cuda.CUDAGraph()
        self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(self.graph, pool=self.pool, stream=self.capture_stream):
            self.output = capture_forward(
                self.static_latent,
                self.static_prompt,
                self.static_timestep,
                self.static_inputs,
            )
        current_stream.wait_stream(self.capture_stream)
        self.graph.replay()
        logger.info(
            "Captured LingBot saturated recompute CUDA graph rank=%d",
            get_sp_parallel_rank(),
        )
        return self.output


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects
    ``[noise(16ch), condition(20ch)]`` concatenated along channel dim.
    Each call processes one chunk (num_frames_per_block frames).
    """

    def _get_causal_kv_cache_size(
        self,
        *,
        sequence_shard_enabled: bool = False,
    ) -> int:
        if self.local_attn_size != -1:
            return self.local_attn_size * self.num_token_per_frame

        return self.sliding_window_num_frames * self.num_token_per_frame

    def _causal_sequence_shard_enabled(self, batch: Req) -> bool:
        return bool(
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )

    def _num_causal_cache_attention_heads(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> int:
        num_attention_heads = self.transformer.num_attention_heads
        if not sequence_shard_enabled:
            return num_attention_heads

        ulysses_world_size = get_ulysses_parallel_world_size()
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
            )
        if ulysses_world_size <= 1:
            raise ValueError(
                "LingBot causal sequence sharding requires ulysses_degree > 1."
            )
        if num_attention_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
            )
        return num_attention_heads // ulysses_world_size

    def _causal_kv_cache_kwargs(
        self,
        policy: CausalDMDCachePolicy,
    ) -> dict[str, Any]:
        return {
            "sequence_shard_enabled": policy.sequence_shard_enabled,
            "kv_cache_size": policy.expected_cache_tokens,
        }

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return True

    @staticmethod
    def _chunk_has_camera_motion(actions) -> bool:
        if not actions:
            return False
        for frame_actions in actions:
            if frame_actions:
                return True
        return False

    def _uses_interactive_kv_window(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        if not self._interactive_kv_window_enabled(server_args):
            return False
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        return LINGBOT_CAMERA_ACTIONS_CONDITION in condition_inputs

    @staticmethod
    def _interactive_kv_window_enabled(server_args: ServerArgs) -> bool:
        config_enabled = bool(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_window_enable",
                False,
            )
        )
        return config_enabled or envs.SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        self._reset_causal_cache_config_defaults()
        super()._apply_causal_cache_overrides(batch, server_args)
        self._sync_interactive_kv_cache_window(server_args)

    def _reset_causal_cache_config_defaults(self) -> None:
        arch_config = getattr(
            getattr(getattr(self, "transformer", None), "config", None),
            "arch_config",
            None,
        )
        if arch_config is None:
            return
        if hasattr(arch_config, "sink_size"):
            self.sink_size = int(arch_config.sink_size)
        if hasattr(arch_config, "sliding_window_num_frames"):
            self.sliding_window_num_frames = int(arch_config.sliding_window_num_frames)

    def _sync_interactive_kv_cache_window(self, server_args: ServerArgs) -> None:
        if not self._interactive_kv_window_enabled(server_args):
            return
        if self.local_attn_size != -1:
            return
        self.sliding_window_num_frames = (
            self._effective_interactive_kv_cache_num_frames(server_args)
        )

    def _effective_interactive_kv_cache_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int:
        cache_window = int(self.sliding_window_num_frames)
        if self.local_attn_size != -1:
            return cache_window

        moving_window = self._moving_kv_sample_num_frames(server_args) or 0
        still_window = self._still_kv_sample_num_frames(server_args) or 0
        return max(
            cache_window,
            int(self.sink_size)
            + max(moving_window, still_window)
            + int(self.num_frames_per_block),
        )

    def _build_realtime_causal_cache_policy(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalDMDCachePolicy:
        policy = super()._build_realtime_causal_cache_policy(batch, server_args)
        if self._interactive_kv_window_enabled(server_args):
            policy.expected_cache_tokens = (
                self._effective_interactive_kv_cache_num_frames(server_args)
                * self.num_token_per_frame
            )
        return policy

    @staticmethod
    def _should_reset_lingbot_crossattn_cache(batch: Req) -> bool:
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        return bool(condition_inputs.get(LINGBOT_PROMPT_UPDATED_CONDITION))

    def _sync_lingbot_crossattn_cache(
        self,
        batch: Req,
        cache_ctx: CausalDMDRealtimeCacheContext,
    ) -> None:
        if self._should_reset_lingbot_crossattn_cache(batch):
            self._reset_crossattn_cache(cache_ctx.crossattn_cache)

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        cache_ctx = super()._prepare_realtime_causal_caches(batch, server_args, ctx)
        self._sync_lingbot_crossattn_cache(batch, cache_ctx)
        return cache_ctx

    def _base_kv_sample_num_frames(self) -> int | None:
        sample_frames = (
            int(self.sliding_window_num_frames)
            - int(self.sink_size)
            - int(self.num_frames_per_block)
        )
        return sample_frames if sample_frames > 0 else None

    @staticmethod
    def _optional_non_negative_int(value: Any) -> int | None:
        if value is None:
            return None
        return max(0, int(value))

    def _moving_kv_sample_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int | None:
        return self._optional_non_negative_int(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_moving_window",
                None,
            )
        )

    def _still_kv_sample_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int | None:
        return self._optional_non_negative_int(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_still_window",
                3,
            )
        )

    def _get_interactive_kv_sample_num_frames(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        pipeline_config = server_args.pipeline_config
        if not self._interactive_kv_window_enabled(server_args):
            return None
        if not self._uses_interactive_kv_window(batch, server_args):
            return self._base_kv_sample_num_frames()

        dynamic_state = cache_state.runtime_cache.setdefault(
            LINGBOT_INTERACTIVE_KV_WINDOW_CACHE,
            {
                "consecutive_still_chunks": 0,
                "sample_num_frames": None,
            },
        )
        if cache_state.chunk_idx == 0:
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = None

        moving_window = self._moving_kv_sample_num_frames(server_args)
        if moving_window is None:
            return None
        still_window = self._still_kv_sample_num_frames(server_args)
        still_chunks_threshold = max(
            1, int(getattr(pipeline_config, "interactive_kv_still_chunks", 2))
        )
        if dynamic_state["sample_num_frames"] is None:
            dynamic_state["sample_num_frames"] = moving_window

        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        if self._chunk_has_camera_motion(
            condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
        ):
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = moving_window
        else:
            dynamic_state["consecutive_still_chunks"] += 1
            if (
                still_window is not None
                and dynamic_state["consecutive_still_chunks"] >= still_chunks_threshold
            ):
                dynamic_state["sample_num_frames"] = still_window

        return int(dynamic_state["sample_num_frames"])

    def _log_lingbot_kv_window(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
        *,
        sample_frames: int | None,
    ) -> None:
        if not self._interactive_kv_window_enabled(server_args):
            return

        mode = "base"
        still_chunks = None
        if self._uses_interactive_kv_window(batch, server_args):
            dynamic_state = cache_state.runtime_cache.get(
                LINGBOT_INTERACTIVE_KV_WINDOW_CACHE, {}
            )
            still_chunks = dynamic_state.get("consecutive_still_chunks")
            condition_inputs = getattr(batch, "condition_inputs", None) or {}
            if self._chunk_has_camera_motion(
                condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
            ):
                mode = "moving"
            else:
                still_window = self._still_kv_sample_num_frames(server_args)
                still_chunks_threshold = max(
                    1,
                    int(
                        getattr(
                            server_args.pipeline_config,
                            "interactive_kv_still_chunks",
                            2,
                        )
                    ),
                )
                if (
                    still_window is not None
                    and sample_frames == still_window
                    and still_chunks is not None
                    and still_chunks >= still_chunks_threshold
                ):
                    mode = "still"
                else:
                    mode = "moving"

        window_frames = (
            int(self.sliding_window_num_frames)
            if sample_frames is None
            else int(self.sink_size)
            + int(sample_frames)
            + int(self.num_frames_per_block)
        )
        sample_tokens = (
            None
            if sample_frames is None
            else int(sample_frames) * int(self.num_token_per_frame)
        )
        logger.info(
            "LingBot interactive KV window: session_id=%s request_id=%s "
            "chunk_idx=%s mode=%s window_frames=%s sample_frames=%s "
            "cache_frames=%s sink_frames=%s current_frames=%s sample_tokens=%s "
            "cache_tokens=%s still_chunks=%s",
            getattr(batch, "realtime_session_id", None),
            getattr(batch, "request_id", None),
            getattr(batch, "block_idx", None),
            mode,
            window_frames,
            sample_frames,
            int(self.sliding_window_num_frames),
            int(self.sink_size),
            int(self.num_frames_per_block),
            sample_tokens,
            int(self.sliding_window_num_frames) * int(self.num_token_per_frame),
            still_chunks,
        )

    def _set_lingbot_kv_sample_tokens(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        self._sync_interactive_kv_cache_window(server_args)
        sample_frames = self._get_interactive_kv_sample_num_frames(
            cache_state,
            batch,
            server_args,
        )
        sample_tokens = (
            None
            if sample_frames is None
            else int(sample_frames) * self.num_token_per_frame
        )
        self._log_lingbot_kv_window(
            cache_state,
            batch,
            server_args,
            sample_frames=sample_frames,
        )
        previous = getattr(batch, "realtime_causal_kv_sample_tokens", None)
        batch.realtime_causal_kv_sample_tokens = sample_tokens
        return previous

    @staticmethod
    def _clear_lingbot_dynamic_condition_cache(cache_state) -> None:
        runtime_cache = getattr(cache_state, "runtime_cache", None)
        if runtime_cache is None:
            return
        runtime_cache.pop(LINGBOT_C2WS_PLUCKER_EMB_CACHE, None)
        runtime_cache.pop(LINGBOT_CAM_CONDITIONER_CACHE, None)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result

    def _denoise_realtime_causal_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx,
        cache_ctx,
        chunk_latents: torch.Tensor,
        prepare_model_input,
        prepare_context_input,
    ) -> torch.Tensor:
        previous_sample_tokens = self._set_lingbot_kv_sample_tokens(
            cache_ctx.cache_state,
            batch,
            server_args,
        )
        try:
            return super()._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=chunk_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_context_input,
            )
        finally:
            batch.realtime_causal_kv_sample_tokens = previous_sample_tokens

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        latents = batch.latents
        assert latents is not None, (
            "LingBot-World causal DMD requires prepared chunk latents. "
            "Ensure RealtimeChunkLatentPreparationStage runs before this stage."
        )
        return latents

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        scheduler = batch.scheduler
        assert scheduler is not None, (
            "LingBot-World causal DMD requires prepared DMD timesteps. "
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

    def _prepare_causal_dmd_image_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        image_embeds = getattr(batch, "image_embeds", [])
        if len(image_embeds) > 0:
            image_embeds = [ie.to(target_dtype) for ie in image_embeds]
        return {
            "encoder_hidden_states_image": image_embeds,
        }

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        # lingbot transformer forward uses varargs, so inspect filtering drops valid kwargs
        return server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            getattr(self.transformer, "rotary_emb", None),
            dtype=target_dtype,
        )

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        return server_args.pipeline_config.get_pos_prompt_embeds(batch)

    @staticmethod
    def _single_prompt_tensor(prompt_embeds) -> torch.Tensor | None:
        if isinstance(prompt_embeds, torch.Tensor):
            return prompt_embeds
        if (
            isinstance(prompt_embeds, (list, tuple))
            and len(prompt_embeds) == 1
            and isinstance(prompt_embeds[0], torch.Tensor)
        ):
            return prompt_embeds[0]
        return None

    def _lingbot_cuda_graph_key(
        self,
        batch: Req,
        *,
        latent_model_input: torch.Tensor,
        prompt_embeds,
        timestep: torch.Tensor,
        kv_cache,
        crossattn_cache,
        pos_cond_kwargs: dict[str, Any],
        current_timestep: int,
        attn_metadata,
    ) -> tuple | None:
        if (
            not getattr(self, "_lingbot_cuda_graph_enabled", False)
            or current_timestep == 0
            or attn_metadata is not None
            or not latent_model_input.is_cuda
            or torch.version.hip is not None
            or set(pos_cond_kwargs) - {"c2ws_plucker_emb"}
        ):
            return None
        prompt = self._single_prompt_tensor(prompt_embeds)
        if prompt is None or not kv_cache or not crossattn_cache:
            return None
        if not all(
            isinstance(cache, CausalSelfAttentionKVCache)
            and not cache.allow_growth
            and cache.local_end_index_int == cache.cache_size
            for cache in kv_cache
        ):
            return None
        if not all(
            isinstance(cache, CrossAttentionKVCache) and cache.is_init
            for cache in crossattn_cache
        ):
            return None

        c2ws = pos_cond_kwargs.get("c2ws_plucker_emb")
        if c2ws is not None and not isinstance(c2ws, torch.Tensor):
            return None
        cache_pointers = tuple(
            (
                cache.k.data_ptr(),
                cache.v.data_ptr(),
                cache.cache_size,
                cache.sink_tokens,
                cache.attention_window_size,
            )
            for cache in kv_cache
        )
        crossattn_pointers = tuple(
            (cache.k.data_ptr(), cache.v.data_ptr()) for cache in crossattn_cache
        )
        return (
            cache_pointers,
            crossattn_pointers,
            bool(getattr(batch, "enable_sequence_shard", False)),
            tuple(getattr(batch, "sequence_shard_splits", ()) or ()),
            getattr(batch, "realtime_causal_kv_sample_tokens", None),
            _cuda_graph_tensor_signature(latent_model_input),
            _cuda_graph_tensor_signature(prompt),
            _cuda_graph_tensor_signature(timestep),
            None if c2ws is None else _cuda_graph_tensor_signature(c2ws),
        )

    def _forward_causal_transformer(
        self,
        batch: Req,
        *,
        latent_model_input: torch.Tensor,
        prompt_embeds,
        timestep: torch.Tensor,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        current_timestep: int,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> torch.Tensor:
        graph_key = self._lingbot_cuda_graph_key(
            batch,
            latent_model_input=latent_model_input,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
        )
        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=current_timestep,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ),
        ):
            if graph_key is None:
                return self.transformer(
                    latent_model_input,
                    prompt_embeds,
                    timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_tokens,
                    start_frame=start_frame,
                    **image_kwargs,
                    **pos_cond_kwargs,
                )

            prompt = self._single_prompt_tensor(prompt_embeds)
            assert prompt is not None
            prepared_inputs = self.transformer.prepare_lingbot_cuda_graph_inputs(
                hidden_states=latent_model_input,
                timestep=timestep,
                c2ws_plucker_emb=pos_cond_kwargs.get("c2ws_plucker_emb"),
                start_frame=start_frame,
                forward_batch=batch,
            )
            runner = getattr(self, "_lingbot_cuda_graph_runner", None)
            if runner is None or runner.key != graph_key:
                runner = _LingBotCudaGraphRunner(graph_key)
                self._lingbot_cuda_graph_runner = runner

            if runner.graph is None:

                def capture_forward(
                    static_latent: torch.Tensor,
                    static_prompt: torch.Tensor,
                    static_timestep: torch.Tensor,
                    static_inputs: dict[str, Any],
                ) -> torch.Tensor:
                    return self.transformer(
                        static_latent,
                        static_prompt,
                        static_timestep,
                        encoder_hidden_states_image=[],
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_tokens,
                        start_frame=start_frame,
                        c2ws_plucker_emb=None,
                        precomputed_lingbot_inputs=static_inputs,
                    )

            else:
                capture_forward = None

            graph_output = runner.run(
                latent=latent_model_input,
                prompt=prompt,
                timestep=timestep,
                prepared_inputs=prepared_inputs,
                capture_forward=capture_forward,
            )
            # The graph reuses its output allocation on every replay.  The
            # scheduler still consumes the previous result when the next step
            # starts, so return an independent snapshot.
            return graph_output.clone()

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
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
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
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ),
        ):
            self.transformer(
                context_input.to(target_dtype),
                prompt_embeds,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_tokens,
                start_frame=start_frame,
                skip_final_projection=True,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    @staticmethod
    def _select_i2v_condition_chunk(
        condition_full: torch.Tensor,
        chunk_idx: int,
        chunk_size: int,
    ) -> torch.Tensor:
        condition_chunks = condition_full.split(chunk_size, dim=2)
        condition = condition_chunks[min(chunk_idx, len(condition_chunks) - 1)]

        if condition.shape[2] == chunk_size:
            return condition
        pad_frames = chunk_size - condition.shape[2]
        return torch.cat(
            [
                condition,
                condition.new_zeros(
                    condition.shape[0],
                    condition.shape[1],
                    pad_frames,
                    condition.shape[3],
                    condition.shape[4],
                ),
            ],
            dim=2,
        )

    @staticmethod
    def _build_i2v_model_input_writer(
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device,
    ):
        b, latent_channels, t, h, w = latents.shape
        condition = condition.to(device=device, dtype=target_dtype)
        model_input = torch.empty(
            (
                b,
                latent_channels + condition.shape[1],
                t,
                h,
                w,
            ),
            dtype=target_dtype,
            device=device,
        )
        model_input[:, latent_channels:].copy_(condition)

        def write(current_latents: torch.Tensor) -> torch.Tensor:
            model_input[:, :latent_channels].copy_(current_latents)
            return model_input

        return write

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._lingbot_cuda_graph_enabled = bool(
            getattr(server_args, "enable_cuda_graph", False)
        )
        if self._lingbot_cuda_graph_enabled and bool(
            getattr(server_args, "enable_torch_compile", False)
        ):
            raise ValueError(
                "LingBot CUDA graph cannot be combined with whole-DiT torch.compile."
            )
        # --- Condition: take current chunk's slice ---
        condition_full = batch.image_latent
        assert condition_full is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        latents = ctx.latents
        cache_ctx = self._prepare_realtime_causal_caches(batch, server_args, ctx)

        # Keep cross-attention K/V cache across realtime chunks; LingBot text/image
        # conditions are session-static and are invalidated by cache reset.

        # Slice condition to current chunk
        condition = self._select_i2v_condition_chunk(
            condition_full,
            cache_ctx.chunk_idx,
            ctx.num_frames,
        )

        # --- Denoising loop (single chunk) ---
        current_latents = latents
        prepare_model_input = self._build_i2v_model_input_writer(
            latents=current_latents,
            condition=condition,
            target_dtype=ctx.target_dtype,
            device=ctx.device,
        )

        try:
            current_latents = self._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=current_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_model_input,
            )
        finally:
            self._clear_lingbot_dynamic_condition_cache(cache_ctx.cache_state)

        # Advance cumulative frame position
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch
