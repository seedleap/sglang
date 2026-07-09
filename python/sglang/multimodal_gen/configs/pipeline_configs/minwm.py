# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM (Wan2.1-1.3B causal DMD world model, stage4 eval config)

from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import MinWMVideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    SelfForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.minwm_camera import (
    MINWM_DEFAULT_INTRINSICS,
    advance_camera_chunk,
)

logger = init_logger(__name__)

# condition_inputs keys used by the MinWM realtime adapter / stages
MINWM_CAMERA_ACTIONS_CONDITION = "minwm_camera_actions"
MINWM_CAMERA_INTRINSICS_CONDITION = "minwm_camera_intrinsics"
MINWM_PROMPT_UPDATED_CONDITION = "minwm_prompt_updated"


class MinWMCameraState(BaseRealtimeState):
    """Cumulative camera pose plus per-frame history for prompt-switch recache."""

    def __init__(self):
        super().__init__()
        self.current_c2w: np.ndarray = np.eye(4)
        self.intrinsics: tuple[float, float, float, float] = MINWM_DEFAULT_INTRINSICS
        # Per-frame w2c / intrinsics history, appended chunk-by-chunk. The
        # recache path replays the last `local_attn_size` frames, so the
        # history is trimmed by the denoising stage together with the latent
        # history.
        self.viewmats_history: torch.Tensor | None = None  # (1, F_hist, 4, 4)
        self.ks_history: torch.Tensor | None = None  # (1, F_hist, 3, 3)

    def reset_camera(self):
        self.current_c2w = np.eye(4)
        self.intrinsics = MINWM_DEFAULT_INTRINSICS
        self.viewmats_history = None
        self.ks_history = None

    def append_history(self, viewmats: torch.Tensor, ks: torch.Tensor) -> None:
        if self.viewmats_history is None:
            self.viewmats_history = viewmats
            self.ks_history = ks
        else:
            self.viewmats_history = torch.cat([self.viewmats_history, viewmats], dim=1)
            self.ks_history = torch.cat([self.ks_history, ks], dim=1)

    def trim_history(self, max_frames: int) -> None:
        if self.viewmats_history is None:
            return
        if self.viewmats_history.shape[1] > max_frames:
            self.viewmats_history = self.viewmats_history[:, -max_frames:]
            self.ks_history = self.ks_history[:, -max_frames:]

    def dispose(self):
        super().dispose()
        self.reset_camera()


def _validate_camera_actions(actions) -> list[list[str]]:
    if not isinstance(actions, list):
        raise TypeError("minwm_camera_actions must be a list[list[str]]")
    result: list[list[str]] = []
    for frame_actions in actions:
        if not isinstance(frame_actions, list):
            raise TypeError("minwm_camera_actions must be a list[list[str]]")
        result.append([str(key) for key in frame_actions])
    return result


def _pad_actions_to_chunk(
    actions: list[list[str]], chunk_size: int
) -> list[list[str]]:
    if len(actions) >= chunk_size:
        return actions[:chunk_size]
    fill_item = actions[-1] if actions else []
    return actions + [list(fill_item) for _ in range(chunk_size - len(actions))]


@dataclass
class MinWMCausalDMDConfig(SelfForcingWanT2V480PConfig):
    """MinWM Wan2.1-1.3B causal DMD realtime pipeline config.

    Inherits the Self-Forcing DMD setup that exactly matches minWM stage4:
    flow_shift=5.0, dmd_denoising_steps=[1000, 750, 500, 250],
    warp_denoising_step=True.
    """

    dit_config: DiTConfig = field(default_factory=MinWMVideoConfig)

    # minWM runs every component in bf16 (PipelineBase.load_and_place).
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    # KV refill timestep for clean-context passes (minWM `context_noise`).
    context_noise: int = 0

    # Request-level overridable cache geometry (CausalDMDDenoisingStage).
    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None

    # Prompt-switch recache semantics (minWM stage4 eval:
    # global_sink=true, allow_sink_write_on_recache=false).
    recache_on_prompt_switch: bool = True
    global_sink: bool = True
    allow_sink_write_on_recache: bool = False

    def prepare_minwm_camera_chunk(
        self,
        batch,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Integrate this chunk's key states into (viewmats, Ks) tensors.

        Camera pose accumulates across chunks in the session's
        ``MinWMCameraState``; frame ``i`` uses the pose before motion ``i``
        applies (private-stack serving semantics). Returns ``None`` when the
        request carries no camera actions (camera-free operation).
        """
        actions = batch.condition_inputs.get(MINWM_CAMERA_ACTIONS_CONDITION)
        if actions is None:
            return None
        chunk_size = batch.realtime_chunk_size or int(
            self.dit_config.arch_config.num_frames_per_block
        )
        actions = _pad_actions_to_chunk(_validate_camera_actions(actions), chunk_size)

        if batch.session is None:
            state = MinWMCameraState()
        else:
            state = batch.session.get_or_create_state(MinWMCameraState)
            if batch.block_idx == 0:
                state.reset_camera()

        intrinsics = batch.condition_inputs.get(MINWM_CAMERA_INTRINSICS_CONDITION)
        if intrinsics is not None:
            state.intrinsics = tuple(float(v) for v in intrinsics)

        new_c2w, viewmats, ks = advance_camera_chunk(
            state.current_c2w,
            actions,
            intrinsics=state.intrinsics,
            device=device,
            dtype=dtype,
        )
        state.current_c2w = new_c2w
        state.append_history(viewmats, ks)
        state.trim_history(
            max(
                int(self.dit_config.arch_config.local_attn_size),
                int(self.dit_config.arch_config.sliding_window_num_frames),
            )
        )
        logger.debug(
            "MinWM camera chunk prepared: session_id=%s block_idx=%s frames=%s",
            batch.realtime_session_id,
            batch.block_idx,
            viewmats.shape[1],
        )
        return viewmats, ks
