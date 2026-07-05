# SPDX-License-Identifier: Apache-2.0
"""Assemble an SGLang-loadable diffusers-layout checkpoint for MinWM.

The MinWM stage4 checkpoint is a single ``model.pt`` holding the generator
state dict (``generator_ema`` preferred, then ``generator``, then ``model``)
with original-Wan-repo key naming under a ``model.`` prefix. Text encoder,
tokenizer and VAE come from the Wan2.1-T2V-1.3B diffusers checkpoint the model
was trained from.

This tool:
1. extracts the generator state dict, strips the ``model.`` prefix, drops any
   FSDP wrapper prefixes, and writes ``transformer/`` safetensors +
   ``config.json`` (``_class_name: MinWMCausalTransformer3DModel``);
2. links/copies ``text_encoder``, ``tokenizer``, ``vae`` and ``scheduler``
   from the donor diffusers directory;
3. writes ``model_index.json`` with
   ``_class_name: MinWMCausalDMDPipeline``.

Example:

    python -m sglang.multimodal_gen.tools.convert_minwm_checkpoint \
        --minwm-checkpoint /ckpt/wan-step-3400/model.pt \
        --donor-diffusers-dir /ckpt/Wan2.1-T2V-1.3B-from-diffusers \
        --output-dir /ckpt/minwm-stage4-diffusers \
        --link-donor
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

GENERATOR_KEYS = ("generator_ema", "generator", "model")
FSDP_PREFIX = "model._fsdp_wrapped_module."
MODEL_PREFIX = "model."

DONOR_COMPONENTS = ("text_encoder", "tokenizer", "vae", "scheduler")

TRANSFORMER_CONFIG = {
    "_class_name": "MinWMCausalTransformer3DModel",
    "_diffusers_version": "0.30.0",
    "model_type": "t2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 16,
    "out_dim": 16,
    "dim": 1536,
    "num_heads": 12,
    "num_layers": 30,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
    "local_attn_size": 16,
    "sink_size": 4,
    "num_frame_per_block": 4,
    "rope_position_mode": "block_relative",
}

MODEL_INDEX = {
    "_class_name": "MinWMCausalDMDPipeline",
    "_diffusers_version": "0.30.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "MinWMCausalTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def extract_generator_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict):
        for key in GENERATOR_KEYS:
            if key in state_dict:
                state_dict = state_dict[key]
                break

    cleaned: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.startswith(FSDP_PREFIX):
            name = name.replace(FSDP_PREFIX, MODEL_PREFIX, 1)
        if name.startswith(MODEL_PREFIX):
            name = name[len(MODEL_PREFIX) :]
        cleaned[name] = tensor.contiguous()
    return cleaned


def summarize_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    num_blocks = (
        max(
            (
                int(name.split(".")[1])
                for name in state_dict
                if name.startswith("blocks.")
            ),
            default=-1,
        )
        + 1
    )
    has_prope = any(".prope_o." in name for name in state_dict)
    total_params = sum(t.numel() for t in state_dict.values())
    print(
        f"generator state dict: {len(state_dict)} tensors, "
        f"{total_params / 1e9:.2f}B params, blocks={num_blocks}, prope_o={has_prope}"
    )
    if not has_prope:
        raise SystemExit(
            "checkpoint has no prope_o weights; this does not look like a "
            "camera-conditioned MinWM stage checkpoint"
        )


def link_or_copy(src: str, dst: str, link: bool) -> None:
    if os.path.exists(dst):
        return
    if link:
        os.symlink(os.path.abspath(src), dst)
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minwm-checkpoint", required=True, help="path to model.pt")
    parser.add_argument(
        "--donor-diffusers-dir",
        required=True,
        help="Wan2.1-T2V-1.3B diffusers checkpoint dir (text_encoder/tokenizer/vae/scheduler donor)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--link-donor",
        action="store_true",
        help="symlink donor components instead of copying",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="transformer weight dtype to store (bf16 matches minWM serving)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    state_dict = extract_generator_state_dict(args.minwm_checkpoint)
    summarize_state_dict(state_dict)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    transformer_dir = os.path.join(args.output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    save_file(
        state_dict,
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
    )
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(TRANSFORMER_CONFIG, f, indent=2)

    for component in DONOR_COMPONENTS:
        src = os.path.join(args.donor_diffusers_dir, component)
        if not os.path.exists(src):
            raise SystemExit(f"donor component missing: {src}")
        link_or_copy(src, os.path.join(args.output_dir, component), args.link_donor)

    with open(os.path.join(args.output_dir, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2)

    print(f"wrote MinWM diffusers-layout checkpoint to {args.output_dir}")
    print(
        "serve with: sglang serve --model-path "
        f"{args.output_dir} --pipeline-class-name MinWMCausalDMDPipeline"
    )


if __name__ == "__main__":
    main()
