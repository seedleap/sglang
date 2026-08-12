# SPDX-License-Identifier: Apache-2.0

"""Native causal VAE engine for the unified realtime worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ExactVAEParallelController:
    """Small rank-0 control plane for one spatially sharded exact decoder."""

    RESET = 1
    DECODE = 2
    STOP = 3
    _META_SIZE = 8
    _DTYPE_TO_CODE = {
        torch.bfloat16: 1,
        torch.float16: 2,
        torch.float32: 3,
    }
    _CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}

    def __init__(self, *, rank: int, world_size: int, device: torch.device) -> None:
        if world_size < 2:
            raise ValueError("exact VAE parallel controller requires at least 2 ranks")
        if rank < 0 or rank >= world_size:
            raise ValueError("exact VAE parallel rank is out of range")
        self.rank = rank
        self.world_size = world_size
        self.device = device

    @property
    def is_driver(self) -> bool:
        return self.rank == 0

    def _send_meta(self, values: list[int]) -> None:
        if not self.is_driver:
            raise RuntimeError("only exact VAE rank 0 may send commands")
        if len(values) != self._META_SIZE:
            raise ValueError("invalid exact VAE command metadata")
        metadata = torch.tensor(values, dtype=torch.int64, device=self.device)
        dist.broadcast(metadata, src=0)

    def send_reset(self) -> None:
        self._send_meta([self.RESET, 0, 0, 0, 0, 0, 0, 0])

    def send_decode(self, latents: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError("exact VAE parallel decode expects BCTHW latents")
        dtype_code = self._DTYPE_TO_CODE.get(latents.dtype)
        if dtype_code is None:
            raise ValueError(f"unsupported exact VAE latent dtype: {latents.dtype}")
        source = latents.detach().contiguous().to(self.device, non_blocking=True)
        self._send_meta(
            [
                self.DECODE,
                int(first_chunk),
                dtype_code,
                *(int(value) for value in source.shape),
            ]
        )
        dist.broadcast(source, src=0)
        return source

    def send_stop(self) -> None:
        self._send_meta([self.STOP, 0, 0, 0, 0, 0, 0, 0])

    def receive(self) -> tuple[int, bool, torch.Tensor | None]:
        if self.is_driver:
            raise RuntimeError("exact VAE rank 0 cannot receive follower commands")
        metadata = torch.empty(self._META_SIZE, dtype=torch.int64, device=self.device)
        dist.broadcast(metadata, src=0)
        values = [int(value) for value in metadata.tolist()]
        command = values[0]
        if command != self.DECODE:
            if command not in {self.RESET, self.STOP}:
                raise RuntimeError(f"unknown exact VAE parallel command: {command}")
            return command, False, None

        dtype = self._CODE_TO_DTYPE.get(values[2])
        if dtype is None:
            raise RuntimeError(f"unknown exact VAE latent dtype code: {values[2]}")
        shape = tuple(values[3:])
        if any(value <= 0 for value in shape):
            raise RuntimeError(f"invalid exact VAE latent shape: {shape}")
        latents = torch.empty(shape, dtype=dtype, device=self.device)
        dist.broadcast(latents, src=0)
        return command, bool(values[1]), latents


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
    gpu_rgb8_d2h = True
    _DEFAULT_WARMUP_SHAPE = (1, 48, 1, 30, 52)
    # Five 1248x704-tier latents cross Wan's auto spatial-decode threshold for
    # SP2. Multi-rank readiness must exercise the collective path, not only load
    # two identical model replicas and report healthy.
    _PARALLEL_WARMUP_SHAPE = (1, 48, 5, 44, 78)

    def __init__(
        self,
        server_args: ServerArgs,
        vae_path: str,
        *,
        use_dedicated_cuda_stream: bool = False,
    ) -> None:
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
        self._parallel_driver: ExactVAEParallelController | None = None
        self.decode_parallel_size = 1
        self.use_dedicated_cuda_stream = bool(use_dedicated_cuda_stream)
        self._decode_stream = None
        if self.use_dedicated_cuda_stream:
            if not torch.cuda.is_available():
                raise ValueError("dedicated exact VAE CUDA stream requires CUDA")
            self._decode_stream = torch.cuda.Stream()
        logger.info("exact realtime VAE backend loaded %s", vae_path)

    def attach_parallel_driver(self, controller: ExactVAEParallelController) -> None:
        if not controller.is_driver:
            raise ValueError("only exact VAE rank 0 can attach the parallel driver")
        self._parallel_driver = controller
        self.decode_parallel_size = controller.world_size

    def set_decode_parallel_size(self, world_size: int) -> None:
        self.decode_parallel_size = world_size

    def _reset_decoder_state(self) -> None:
        if self._parallel_driver is not None:
            self._parallel_driver.send_reset()
        if callable(self._reset_causal_state):
            self._reset_causal_state()

    def create_decoder(self, identity: tuple[str, str]) -> _ExactDecoder:
        del identity
        state = RealtimeVAEDecodeState()
        reset = (
            self._reset_decoder_state
            if self._parallel_driver is not None
            else self._reset_causal_state
        )
        state.reset_causal_decode_state = reset
        return _ExactDecoder(state, reset)

    @torch.no_grad()
    def warmup(
        self,
        latent_shape: tuple[int, int, int, int, int] = _DEFAULT_WARMUP_SHAPE,
    ) -> None:
        if self.decode_parallel_size > 1 and latent_shape == self._DEFAULT_WARMUP_SHAPE:
            latent_shape = self._PARALLEL_WARMUP_SHAPE
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
        if self._parallel_driver is not None:
            latents = self._parallel_driver.send_decode(
                latents, first_chunk=first_chunk
            )
        return self._decode_local(
            decoder,
            latents,
            first_chunk=first_chunk,
        )

    def _decode_local(
        self,
        decoder: _ExactDecoder,
        latents: torch.Tensor,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        if self._decode_stream is None:
            return self._decode_and_postprocess(
                decoder,
                latents,
                first_chunk=first_chunk,
            )

        if latents.device.type == "cuda":
            producer_stream = torch.cuda.current_stream(latents.device)
            self._decode_stream.wait_stream(producer_stream)
        with torch.cuda.stream(self._decode_stream):
            frames = self._decode_and_postprocess(
                decoder,
                latents,
                first_chunk=first_chunk,
            )
            complete = torch.cuda.Event()
            complete.record(self._decode_stream)
        # The worker may hand the tensor to a CPU encoder thread immediately.
        # Keep the synchronization boundary explicit instead of relying on an
        # implicit default-stream or D2H synchronization.
        complete.synchronize()
        return frames

    def _decode_and_postprocess(
        self,
        decoder: _ExactDecoder,
        latents: torch.Tensor,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        frames = self.stage.decode_causal(
            latents,
            self.server_args,
            first_chunk=first_chunk,
            decode_state=decoder.decode_state,
        )
        return self.server_args.pipeline_config.post_decoding(frames, self.server_args)


@torch.no_grad()
def run_exact_vae_follower(
    engine: ExactCausalVAEEngine,
    controller: ExactVAEParallelController,
) -> None:
    """Execute rank-0 commands in lockstep so causal cache stays rank-local."""

    if controller.is_driver:
        raise ValueError("exact VAE rank 0 cannot run the follower loop")
    engine.set_decode_parallel_size(controller.world_size)
    decoder = engine.create_decoder(("parallel", f"rank-{controller.rank}"))
    try:
        while True:
            command, first_chunk, latents = controller.receive()
            if command == controller.STOP:
                return
            if command == controller.RESET:
                decoder.reset()
                continue
            assert latents is not None
            engine._decode_local(
                decoder,
                latents,
                first_chunk=first_chunk,
            )
    finally:
        decoder.reset()
