# SPDX-License-Identifier: Apache-2.0
# Adapted from minWM (Wan2.1-1.3B causal DMD world model)

"""MinWM realtime causal DMD pipeline (pure T2V + PRoPE camera control)."""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DMDTimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm import (
    MinWMCausalDMDDenoisingStage,
    MinWMChunkNoisePreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    CausalVaeDecodingStage,
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MinWMCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MinWMCausalDMDPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # Byte-identical to minWM's FlowMatchScheduler(num_train_timesteps=1000,
        # shift=timestep_shift, sigma_min=0.0, extra_one_step=True) followed by
        # set_timesteps(1000): same sigma grid, same warp lookup table.
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(DMDTimestepPreparationStage(self.get_module("scheduler")))
        self.add_stage(
            MinWMChunkNoisePreparationStage(
                transformer=self.get_module("transformer"),
                vae_config=server_args.pipeline_config.vae_config,
            )
        )
        self.add_stage(
            MinWMCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            CausalVaeDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = MinWMCausalDMDPipeline
