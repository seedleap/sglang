# SPDX-License-Identifier: Apache-2.0
"""Single-GPU BF16 parity and timing probe for H3 causal attention."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_causal import (
    MiniMaxH3CausalAttentionSpec,
    minimax_h3_build_causal_attention_plan,
    minimax_h3_flex_causal_attention,
    minimax_h3_reference_causal_attention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-frames", type=int, default=3)
    parser.add_argument("--sink-frames", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--latent-t", type=int, default=9)
    parser.add_argument("--latent-h", type=int, default=8)
    parser.add_argument("--latent-w", type=int, default=8)
    parser.add_argument("--audio-t", type=int, default=24)
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--mask-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("attention_probe requires CUDA")
    packed = minimax_h3_packed_sequence(
        text_len=args.text_len,
        latent_t=args.latent_t,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        audio_t=args.audio_t,
        include_keyframe_cond=False,
    )
    spec = MiniMaxH3CausalAttentionSpec(
        mode="flex",
        block_frames=args.block_frames,
        sink_frames=args.sink_frames,
        window_frames=args.window_frames,
        cache_block_mask=args.mask_cache,
    )
    plan = minimax_h3_build_causal_attention_plan(packed, spec)
    assert plan is not None
    seq_len = int(packed["seq_len"])
    if seq_len > 4096:
        raise ValueError(
            f"parity probe seq_len={seq_len} exceeds dense reference limit 4096"
        )

    torch.manual_seed(17)
    shape = (seq_len, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = 1.0 / math.sqrt(args.head_dim)
    torch.cuda.synchronize()
    mask_started = time.perf_counter()
    first_block_mask = plan.get_flex_block_mask(
        device=query.device,
        length=plan.padded_length(),
    )
    torch.cuda.synchronize()
    mask_build_ms = (time.perf_counter() - mask_started) * 1000

    second_plan = minimax_h3_build_causal_attention_plan(packed, spec)
    assert second_plan is not None
    torch.cuda.synchronize()
    lookup_started = time.perf_counter()
    second_block_mask = second_plan.get_flex_block_mask(
        device=query.device,
        length=second_plan.padded_length(),
    )
    torch.cuda.synchronize()
    mask_lookup_ms = (time.perf_counter() - lookup_started) * 1000
    reference = minimax_h3_reference_causal_attention(
        query,
        key,
        value,
        plan=plan,
        softmax_scale=scale,
    )

    for _ in range(args.warmup):
        minimax_h3_flex_causal_attention(
            query,
            key,
            value,
            plan=plan,
            softmax_scale=scale,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.repeats):
        actual = minimax_h3_flex_causal_attention(
            query,
            key,
            value,
            plan=plan,
            softmax_scale=scale,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000 / args.repeats
    error = (actual.float() - reference.float()).abs()
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "seq_len": seq_len,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "target_frames": plan.num_target_frames,
                "target_blocks": plan.num_target_blocks,
                "configured_sink_frames": spec.sink_frames,
                "effective_sink_frames": spec.effective_sink_frames,
                "configured_window_frames": spec.window_frames,
                "effective_window_frames": spec.effective_window_frames,
                "mask_cache_enabled": spec.cache_block_mask,
                "mask_cache_reused_object": first_block_mask is second_block_mask,
                "mask_build_ms": mask_build_ms,
                "mask_second_plan_lookup_ms": mask_lookup_ms,
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
                "flex_latency_ms": elapsed_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
