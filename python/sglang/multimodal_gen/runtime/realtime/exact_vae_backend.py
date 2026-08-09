# SPDX-License-Identifier: Apache-2.0

"""Native causal VAE engine for the unified realtime worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class _ExactDecoder:
    decode_state: RealtimeVAEDecodeState
    reset_causal_state: Callable[[], Any] | None

    def reset(self) -> None:
        if callable(self.reset_causal_state):
            self.reset_causal_state()


class ExactCausalVAEEngine:
    """One exact native VAE with a model-global causal cache."""

    backend = "exact"
    max_sessions = 1
    rgb_quantization = "truncate"
    _DEFAULT_WARMUP_SHAPE = (1, 48, 1, 30, 52)

    def __init__(self, server_args: ServerArgs, vae_path: str) -> None:
        self.server_args = server_args
        vae_config = getattr(server_args.pipeline_config, "vae_config", None)
        if CausalVaeDecodingStage._taehv_checkpoint_path(vae_config) is not None:
            raise ValueError(
                "exact realtime VAE backend cannot use taehv_checkpoint_path"
            )

        loader = VAELoader()
        self.vae, _ = loader.load(
            vae_path,
            server_args,
            component_name="vae",
            transformers_or_diffusers=loader.expected_library,
        )
        stage_cls = CausalVaeDecodingStage
        if server_args.pipeline_class_name in {
            "MinWMCausalDMDPipeline",
            "MinWMCausalUniPCPipeline",
        }:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm import (
                MinWMCausalVaeDecodingStage,
            )

            stage_cls = MinWMCausalVaeDecodingStage
        self.stage = stage_cls(vae=self.vae)
        self._reset_causal_state = self.stage._get_causal_decode_reset_fn()
        logger.info("exact realtime VAE backend loaded %s", vae_path)

    def create_decoder(self, identity: tuple[str, str]) -> _ExactDecoder:
        del identity
        state = RealtimeVAEDecodeState()
        state.reset_causal_decode_state = self._reset_causal_state
        return _ExactDecoder(state, self._reset_causal_state)

    @torch.no_grad()
    def warmup(
        self,
        latent_shape: tuple[int, int, int, int, int] = _DEFAULT_WARMUP_SHAPE,
    ) -> None:
        decoder = self.create_decoder(("warmup", "warmup"))
        try:
            self.decode(
                decoder,
                torch.zeros(latent_shape, dtype=torch.bfloat16),
                first_chunk=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            decoder.reset()

    @torch.no_grad()
    def decode(
        self,
        decoder: _ExactDecoder,
        latents: torch.Tensor,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        if first_chunk:
            decoder.reset()
        frames = self.stage.decode_causal(
            latents,
            self.server_args,
            first_chunk=first_chunk,
            decode_state=decoder.decode_state,
        )
        return self.server_args.pipeline_config.post_decoding(
            frames,
            self.server_args,
        )
