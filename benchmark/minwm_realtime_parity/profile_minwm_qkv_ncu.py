#!/usr/bin/env python3
"""Single-GPU NCU target for the MinWM 5B 720p QKV fast lane.

This deliberately profiles only the projection geometry used by the real
720p/SP1 layer probe: [1, 858, 3072].  It is a kernel diagnosis tool, not an
end-to-end performance claim and must not be used to infer SP2/SP4 latency.
"""

import argparse

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "fused"), required=True)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((1, 858, 3072), device=device, dtype=torch.bfloat16)
    weights = [
        torch.randn((3072, 3072), device=device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    packed = torch.cat(weights, dim=0).contiguous()

    # Warmup forces cuBLASLt selection before NCU replay begins.
    for _ in range(10):
        if args.mode == "baseline":
            outputs = [F.linear(x, weight) for weight in weights]
            sink = outputs[0].float().sum()
        else:
            sink = F.linear(x, packed).float().sum()
    sink.item()
    torch.cuda.synchronize()

    for _ in range(args.iterations):
        if args.mode == "baseline":
            outputs = [F.linear(x, weight) for weight in weights]
            sink = outputs[0].float().sum()
        else:
            sink = F.linear(x, packed).float().sum()
    sink.item()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
