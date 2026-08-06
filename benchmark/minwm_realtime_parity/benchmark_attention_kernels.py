#!/usr/bin/env python3
"""Benchmark the attention APIs used by MinWM at production tensor shapes.

The benchmark intentionally separates packed FA4 from the dense Torch SDPA
lane.  Historical MinWM reports called the latter "dense FA", but on SM120 the
resolved implementation was PyTorch SDPA.  NVTX ranges make preprocessing,
layout conversion, quantization, and the attention kernels visible in Nsight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MinWMAttentionShape:
    name: str
    width: int
    height: int
    sink_frames: int = 4
    window_frames: int = 20
    key_length_override: int | None = None
    chunk_frames: int = 4
    batch_size: int = 1
    num_heads: int = 24
    head_dim: int = 128
    vae_spatial_compression: int = 8
    patch_size: int = 2

    @property
    def tokens_per_frame(self) -> int:
        divisor = self.vae_spatial_compression * self.patch_size
        if self.width % divisor or self.height % divisor:
            raise ValueError(
                f"{self.name}: width/height must be divisible by {divisor}"
            )
        return (self.width // divisor) * (self.height // divisor)

    @property
    def query_length(self) -> int:
        return self.chunk_frames * self.tokens_per_frame

    @property
    def key_length(self) -> int:
        if self.key_length_override is not None:
            return self.key_length_override
        return self.window_frames * self.tokens_per_frame

    def validate(self) -> None:
        if not 0 <= self.sink_frames < self.window_frames:
            raise ValueError("sink_frames must be in [0, window_frames)")
        if self.chunk_frames <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("chunk/head geometry must be positive")
        if self.key_length_override is not None and self.key_length_override <= 0:
            raise ValueError("key_length_override must be positive")
        _ = self.tokens_per_frame


PRESETS = {
    "smoke": MinWMAttentionShape(name="smoke", width=256, height=256, window_frames=8),
    "480p": MinWMAttentionShape(name="480p", width=832, height=480),
    "480p-cross": MinWMAttentionShape(
        name="480p-cross", width=832, height=480, key_length_override=512
    ),
    "704p": MinWMAttentionShape(name="704p", width=1248, height=704),
    "704p-cross": MinWMAttentionShape(
        name="704p-cross", width=1248, height=704, key_length_override=512
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="smoke,480p,480p-cross,704p,704p-cross")
    parser.add_argument("--backends", default="sdpa,fa4_dense,fa4,sage2,sage3")
    parser.add_argument("--sink-frames", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def backend_availability() -> dict[str, dict[str, object]]:
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    )
    return {
        "sdpa": {"available": torch.cuda.is_available(), "precision": "BF16"},
        "fa4": {
            "available": torch.cuda.is_available()
            and _module_available("flash_attn.cute"),
            "precision": "BF16",
        },
        "fa4_dense": {
            "available": torch.cuda.is_available()
            and _module_available("flash_attn.cute"),
            "precision": "BF16",
        },
        "sage2": {
            "available": torch.cuda.is_available()
            and _module_available("sageattention"),
            "precision": "INT8 QK + FP8 PV on SM120",
        },
        "sage3": {
            "available": capability in {(12, 0), (12, 1)}
            and _module_available("sageattn3"),
            "precision": "block-wise NVFP4",
        },
    }


def make_tensors(shape: MinWMAttentionShape) -> tuple[torch.Tensor, ...]:
    geometry = (shape.batch_size, shape.query_length, shape.num_heads, shape.head_dim)
    kv_geometry = (shape.batch_size, shape.key_length, shape.num_heads, shape.head_dim)
    query = torch.randn(geometry, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(kv_geometry, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(kv_geometry, device="cuda", dtype=torch.bfloat16)
    return query, key, value


def make_backend(
    name: str,
    shape: MinWMAttentionShape,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    if name == "sdpa":

        def sdpa() -> torch.Tensor:
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            return output.transpose(1, 2)

        return sdpa

    if name == "fa4":
        from flash_attn.cute import flash_attn_varlen_func

        cu_query = torch.tensor(
            [0, shape.query_length], dtype=torch.int32, device=query.device
        )
        cu_key = torch.tensor(
            [0, shape.key_length], dtype=torch.int32, device=query.device
        )

        def fa4() -> torch.Tensor:
            output = flash_attn_varlen_func(
                q=query.flatten(0, 1),
                k=key.flatten(0, 1),
                v=value.flatten(0, 1),
                cu_seqlens_q=cu_query,
                cu_seqlens_k=cu_key,
                max_seqlen_q=shape.query_length,
                max_seqlen_k=shape.key_length,
                softmax_scale=None,
                causal=False,
                deterministic=False,
                window_size=(None, None),
                return_lse=False,
            )
            if isinstance(output, tuple):
                output = output[0]
            return output.reshape_as(query)

        return fa4

    if name == "fa4_dense":
        from flash_attn.cute import flash_attn_func

        def fa4_dense() -> torch.Tensor:
            output = flash_attn_func(
                q=query,
                k=key,
                v=value,
                softmax_scale=None,
                causal=False,
                deterministic=False,
                window_size=(None, None),
                return_lse=False,
            )
            if isinstance(output, tuple):
                output = output[0]
            return output

        return fa4_dense

    if name == "sage2":
        from sageattention import sageattn

        def sage2() -> torch.Tensor:
            return sageattn(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=False,
                sm_scale=1.0 / math.sqrt(shape.head_dim),
                return_lse=False,
            )

        return sage2

    if name == "sage3":
        from sageattn3 import sageattn3_blackwell

        def sage3() -> torch.Tensor:
            query_hnd = query.transpose(1, 2)
            # This clone is part of the production SGLang boundary because the
            # upstream Sage3 implementation centers K in place.
            key_hnd = key.transpose(1, 2).clone(memory_format=torch.contiguous_format)
            value_hnd = value.transpose(1, 2)
            output = sageattn3_blackwell(query_hnd, key_hnd, value_hnd, is_causal=False)
            return output.transpose(1, 2)

        return sage3

    raise ValueError(f"unknown backend {name!r}")


def summarize_timings(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "p20_ms": ordered[max(0, math.ceil(0.20 * len(ordered)) - 1)],
        "p80_ms": ordered[max(0, math.ceil(0.80 * len(ordered)) - 1)],
        "max_ms": ordered[-1],
    }


def benchmark_backend(
    name: str,
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, object], torch.Tensor]:
    with torch.cuda.nvtx.range(f"{name}/warmup"):
        for _ in range(warmup):
            output = function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    timings = []
    with torch.cuda.nvtx.range(f"{name}/measured"):
        for iteration in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.nvtx.range(f"{name}/iteration_{iteration:03d}"):
                start.record()
                output = function()
                end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    peak_allocated = torch.cuda.max_memory_allocated()
    return (
        {
            "timing": summarize_timings(timings),
            "iterations": iterations,
            "incremental_peak_allocated_bytes": max(
                0, peak_allocated - allocated_before
            ),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
        },
        output.detach(),
    )


def compare_output(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    reference_norm = torch.linalg.vector_norm(reference_float)
    candidate_norm = torch.linalg.vector_norm(candidate_float)
    cosine = torch.sum(reference_float * candidate_float) / (
        reference_norm * candidate_norm
    )
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_abs": difference.abs().max().item(),
        "cosine_similarity": cosine.item(),
    }


def runtime_metadata() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    return {
        "created_unix_seconds": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": properties.total_memory,
        "sdpa_flags": {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    torch.manual_seed(args.seed)
    requested_presets = [item.strip() for item in args.presets.split(",") if item]
    requested_backends = [item.strip() for item in args.backends.split(",") if item]
    unknown_presets = set(requested_presets) - PRESETS.keys()
    unknown_backends = set(requested_backends) - {
        "sdpa",
        "fa4",
        "fa4_dense",
        "sage2",
        "sage3",
    }
    if unknown_presets or unknown_backends:
        raise ValueError(
            f"unknown presets/backends: {sorted(unknown_presets)}, "
            f"{sorted(unknown_backends)}"
        )

    availability = backend_availability()
    report: dict[str, object] = {
        "schema_version": "minwm-attention-kernel-benchmark/v1",
        "runtime": runtime_metadata(),
        "kv_contract": {
            "sink_frames": args.sink_frames,
            "window_frames": args.window_frames,
            "note": "sink changes cache composition, not dense tensor length",
        },
        "availability": availability,
        "shapes": {},
    }
    for preset in requested_presets:
        base = PRESETS[preset]
        shape = MinWMAttentionShape(
            **{
                **asdict(base),
                "sink_frames": args.sink_frames,
                "window_frames": args.window_frames,
            }
        )
        shape.validate()
        shape_report: dict[str, object] = {
            "contract": {
                **asdict(shape),
                "tokens_per_frame": shape.tokens_per_frame,
                "query_length": shape.query_length,
                "key_length": shape.key_length,
            },
            "backends": {},
        }
        report["shapes"][preset] = shape_report
        query, key, value = make_tensors(shape)
        reference = None
        for name in requested_backends:
            if not availability[name]["available"]:
                shape_report["backends"][name] = {
                    "status": "unavailable",
                    "precision": availability[name]["precision"],
                }
                continue
            try:
                function = make_backend(name, shape, query, key, value)
                with torch.cuda.nvtx.range(f"shape/{preset}/backend/{name}"):
                    result, output = benchmark_backend(
                        name,
                        function,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                result["status"] = "ok"
                result["precision"] = availability[name]["precision"]
                if name == "sdpa":
                    reference = output
                    result["comparison_to_sdpa"] = {
                        "mae": 0.0,
                        "rmse": 0.0,
                        "max_abs": 0.0,
                        "cosine_similarity": 1.0,
                    }
                elif reference is not None:
                    result["comparison_to_sdpa"] = compare_output(reference, output)
                shape_report["backends"][name] = result
                del output
            except Exception as error:  # Preserve backend failures as evidence.
                torch.cuda.synchronize()
                shape_report["backends"][name] = {
                    "status": "error",
                    "precision": availability[name]["precision"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            torch.cuda.empty_cache()
        del query, key, value, reference
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
