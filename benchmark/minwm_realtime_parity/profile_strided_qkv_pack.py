#!/usr/bin/env python3
"""Validate and microbenchmark MinWM's strided peer-first QKV pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ulysses_qkv_pack import (
    fused_pack_peer_first_qkv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=429)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--profile-mode", choices=("none", "baseline", "candidate"), default="none"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.heads % args.world_size:
        raise ValueError("heads must be divisible by world size")
    shape = (1, args.sequence, args.heads, args.head_dim)
    torch.manual_seed(17)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    qkv_storage = torch.randn(
        (*shape[:-1], 3 * shape[-1]), dtype=torch.bfloat16, device="cuda"
    )
    value = qkv_storage[..., 2 * shape[-1] :]
    if value.is_contiguous():
        raise RuntimeError("benchmark value must be a strided fused-QKV view")
    output = torch.empty(3 * query.numel(), dtype=query.dtype, device=query.device)

    def run(mode: str) -> torch.Tensor:
        pack_value = value.contiguous() if mode == "baseline" else value
        return fused_pack_peer_first_qkv(
            query, key, pack_value, args.world_size, output
        )

    baseline = run("baseline").clone()
    candidate = run("candidate").clone()
    torch.cuda.synchronize()
    if not torch.equal(baseline, candidate):
        raise RuntimeError("strided candidate differs from contiguous baseline")

    if args.profile_mode != "none":
        for _ in range(args.warmup):
            run(args.profile_mode)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"qkv_pack_{args.profile_mode}")
        run(args.profile_mode)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    def measure(mode: str) -> tuple[float, int]:
        for _ in range(args.warmup):
            run(mode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            run(mode)
        end.record()
        end.synchronize()
        elapsed_us = start.elapsed_time(end) * 1000 / args.iterations
        peak_delta = torch.cuda.max_memory_allocated() - allocated_before
        return elapsed_us, peak_delta

    baseline_us, baseline_peak = measure("baseline")
    candidate_us, candidate_peak = measure("candidate")
    result = {
        "schema_version": "minwm-strided-qkv-pack-micro/v1",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(query.dtype),
        "shape": list(shape),
        "world_size": args.world_size,
        "value_stride": list(value.stride()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "bitwise_equal": True,
        "baseline_contiguous_copy_and_pack_us": baseline_us,
        "candidate_direct_strided_pack_us": candidate_us,
        "candidate_speedup_percent": (baseline_us / candidate_us - 1) * 100,
        "baseline_peak_delta_bytes": baseline_peak,
        "candidate_peak_delta_bytes": candidate_peak,
        "profile_mode": args.profile_mode,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
