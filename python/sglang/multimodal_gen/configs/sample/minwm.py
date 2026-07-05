# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM (Wan2.1-1.3B causal DMD world model)
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams


@dataclass
class MinWMSamplingParams(WanT2V_1_3B_SamplingParams):
    """Sampling defaults for the MinWM realtime causal DMD pipeline.

    minWM is a 4-step DMD generator: no CFG, no negative prompt. 16 latent
    frames per request default keeps offline `sglang generate` usable; the
    realtime path generates one 4-frame chunk per scheduler round-trip.
    """

    negative_prompt: str | None = None
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    height: int = 480
    width: int = 832
    num_frames: int = 61  # (16 latent frames - 1) * 4 + 1 pixel frames
    fps: int = 16

    # Realtime controls: per-latent-frame held-key lists (minWM key vocabulary
    # w/s/a/d/u/dn + i/k/j/l) and normalized camera intrinsics (fx, fy, cx, cy).
    actions: list[list[str]] | None = None
    camera_intrinsics: tuple[float, float, float, float] | None = None
    chunk_size: int | None = None

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),
        ]
    )

    def _adjust(self, server_args):
        super()._adjust(server_args)
        if self.chunk_size is None:
            self.chunk_size = max(
                1,
                int(
                    server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
                ),
            )
        if self.actions is not None:
            self.condition_inputs["minwm_camera_actions"] = self.actions
        if self.camera_intrinsics is not None:
            self.condition_inputs["minwm_camera_intrinsics"] = list(
                self.camera_intrinsics
            )
        self.realtime_chunk_size = self.chunk_size
