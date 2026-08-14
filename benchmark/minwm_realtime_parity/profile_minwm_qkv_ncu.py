#!/usr/bin/env python3
"""Single-GPU NCU target for MinWM's local-shard QKV boundary.

The profiled region includes the QKV projection, the eager Q/K RMSNorms, and
the fused path's V materialization required by the existing peer-first pack.
It deliberately excludes communication and therefore remains a kernel-level
diagnostic rather than an SP end-to-end claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


K = 3072
N = 3072
NUM_HEADS = 24
EPS = 1e-5
MODES = ("baseline", "fused")


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Preserve MinWM's FP32 normalize -> BF16 round -> BF16 weight boundary."""
    hidden_states_float = hidden_states.float()
    normalized = hidden_states_float * torch.rsqrt(
        hidden_states_float.pow(2).mean(dim=-1, keepdim=True) + EPS
    )
    return normalized.type_as(hidden_states) * weight


def _build_workloads(m: int):
    torch.manual_seed(20260814)
    x = torch.randn((1, m, K), device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn((N, K), device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    biases = [torch.randn(N, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    packed_weight = torch.cat(weights, dim=0).contiguous()
    packed_bias = torch.cat(biases, dim=0).contiguous()
    query_weight = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    def consume(query, key, value, *, materialize_value: bool):
        query = _rms_norm(query, query_weight).reshape(1, m, NUM_HEADS, N // NUM_HEADS)
        key = _rms_norm(key, key_weight).reshape(1, m, NUM_HEADS, N // NUM_HEADS)
        if materialize_value:
            value = value.contiguous()
        return query, key, value.reshape(1, m, NUM_HEADS, N // NUM_HEADS)

    def baseline():
        query, key, value = (
            F.linear(x, weight, bias) for weight, bias in zip(weights, biases)
        )
        return consume(query, key, value, materialize_value=False)

    def fused():
        query, key, value = F.linear(x, packed_weight, packed_bias).chunk(3, dim=-1)
        return consume(query, key, value, materialize_value=True)

    keepalive = [
        x,
        *weights,
        *biases,
        packed_weight,
        packed_bias,
        query_weight,
        key_weight,
    ]
    return {"baseline": baseline, "fused": fused}, keepalive


def _time_cuda(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[max(0, int(len(ordered) * 0.10) - 1)],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--m", type=int, choices=(215, 429, 858), required=True)
    parser.add_argument("--order", choices=("baseline-first", "fused-first"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    workloads, keepalive = _build_workloads(args.m)
    order = MODES if args.order == "baseline-first" else tuple(reversed(MODES))
    samples_by_mode = {name: [] for name in MODES}
    for _ in range(2):
        for name in order:
            samples_by_mode[name].extend(
                _time_cuda(workloads[name], args.warmup, args.repeats // 2)
            )
        order = tuple(reversed(order))

    profile_fn = workloads[args.mode]
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        output = profile_fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        output = profile_fn()
        torch.cuda.synchronize()

    assert keepalive
    record = {
        "schema_version": "minwm-qkv-local-boundary-ncu/v1",
        "mode": args.mode,
        "order": args.order,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "shape": {"batch": 1, "M": args.m, "K": K, "N_each": N},
        "timing": {name: _summary(samples) for name, samples in samples_by_mode.items()},
        "output_shapes": [list(tensor.shape) for tensor in output],
        "output_strides": [list(tensor.stride()) for tensor in output],
        "checksum": float(sum(tensor.float().mean() for tensor in output)),
        "candidate_extra_operation": "value.contiguous()",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
