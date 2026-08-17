# SPDX-License-Identifier: Apache-2.0

"""Terminal stage that hands MinWM latents to a remote realtime VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _align_reference_latent_for_handoff(
    reference_latent: torch.Tensor, generated_latents: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    if reference_latent.ndim != generated_latents.ndim:
        raise ValueError(
            "Realtime reference latent and generated latent must have the same rank"
        )
    if reference_latent.shape[:2] != generated_latents.shape[:2]:
        raise ValueError(
            "Realtime reference latent and generated latent must match batch/channels"
        )
    if reference_latent.shape[-2:] == generated_latents.shape[-2:]:
        return reference_latent, False

    target_h, target_w = generated_latents.shape[-2:]
    aligned = reference_latent

    height, width = aligned.shape[-2:]
    if height > target_h:
        top = (height - target_h) // 2
        aligned = aligned[..., top : top + target_h, :]
    if width > target_w:
        left = (width - target_w) // 2
        aligned = aligned[..., :, left : left + target_w]

    height, width = aligned.shape[-2:]
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        aligned = F.pad(aligned, (left, right, top, bottom))

    return aligned, True


def _prepare_reference_latent_for_handoff(
    reference_latent: torch.Tensor, generated_latents: torch.Tensor
) -> tuple[torch.Tensor, bool, bool, bool]:
    if reference_latent.ndim != generated_latents.ndim:
        raise ValueError(
            "Realtime reference latent and generated latent must have the same rank"
        )
    if reference_latent.shape[0] != generated_latents.shape[0]:
        raise ValueError(
            "Realtime reference latent and generated latent must match batch"
        )

    channel_sliced = False
    temporal_sliced = False
    if reference_latent.shape[1] != generated_latents.shape[1]:
        target_channels = generated_latents.shape[1]
        if reference_latent.shape[1] <= target_channels:
            raise ValueError(
                "Realtime reference latent has fewer channels than generated latent"
            )
        # LingBot I2V stores condition tensors as [mask, latent]. Remote VAE only
        # accepts the real latent channels, which are appended after the mask.
        reference_latent = reference_latent[:, -target_channels:, ...]
        channel_sliced = True
        # LingBot's condition tensor contains one real reference latent followed
        # by temporal padding. Decoding that padding produces the gray frames
        # seen before the first generated chunk reaches the browser.
        if reference_latent.shape[2] > 1:
            reference_latent = reference_latent[:, :, :1, ...].contiguous()
            temporal_sliced = True

    reference_latent, spatial_aligned = _align_reference_latent_for_handoff(
        reference_latent, generated_latents
    )
    return reference_latent, channel_sliced, temporal_sliced, spatial_aligned


class RealtimeLatentHandoffStage(PipelineStage):
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if not isinstance(batch.latents, torch.Tensor):
            raise ValueError("Realtime latent handoff requires tensor latents")
        if not batch.realtime_session_id or not batch.realtime_generation_id:
            raise ValueError(
                "Realtime latent handoff requires session generation identity"
            )

        generated_latents = batch.latents
        has_reference = isinstance(batch.image_latent, torch.Tensor)
        handoff_latents = generated_latents
        reference_latent_aligned = False
        reference_latent_channel_sliced = False
        reference_latent_temporal_sliced = False
        if batch.block_idx == 0 and has_reference:
            (
                reference_latent,
                reference_latent_channel_sliced,
                reference_latent_temporal_sliced,
                reference_latent_aligned,
            ) = _prepare_reference_latent_for_handoff(
                batch.image_latent, generated_latents
            )
            handoff_latents = torch.cat([reference_latent, generated_latents], dim=2)

        handoff_latents = handoff_latents.detach()
        if getattr(server_args, "realtime_vae_backend", None) != "exact_remote":
            handoff_latents = handoff_latents.to(dtype=torch.bfloat16)
        handoff_latents = handoff_latents.contiguous()
        handoff = {
            "session_id": batch.realtime_session_id,
            "generation_id": batch.realtime_generation_id,
            "request_id": batch.request_id,
            "chunk_index": batch.block_idx,
            "event_id": batch.realtime_event_id,
            "action_version": batch.realtime_action_version,
            "prompt_version": batch.realtime_prompt_version,
            "has_reference": has_reference,
            "is_final_chunk": bool(batch.extra.get("realtime_is_final_chunk", False)),
            "generated_latent_frames": int(generated_latents.shape[2]),
            "output_format": batch.realtime_output_format,
            "preview_max_width": batch.realtime_preview_max_width,
        }
        if reference_latent_channel_sliced:
            handoff["reference_latent_channel_slice"] = (
                f"last_{int(generated_latents.shape[1])}"
            )
        if reference_latent_temporal_sliced:
            handoff["reference_latent_temporal_slice"] = "first_1"
        if reference_latent_aligned:
            handoff["reference_latent_aligned"] = True
        if reference_latent_channel_sliced or reference_latent_aligned:
            handoff["reference_latent_shape"] = list(batch.image_latent.shape)
            handoff["generated_latent_shape"] = list(generated_latents.shape)
        return OutputBatch(
            realtime_latents=handoff_latents,
            realtime_handoff=handoff,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            metrics=batch.metrics,
        )
