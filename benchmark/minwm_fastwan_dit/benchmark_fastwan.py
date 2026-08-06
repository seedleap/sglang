#!/usr/bin/env python3
"""Benchmark warmed FastWan requests with an explicit DiT-forward schedule."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one timestep is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--timesteps",
        type=parse_int_list,
        default=parse_int_list("1000,757,522,300,100"),
    )
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-requests", type=int, default=5)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument(
        "--prompt",
        default=(
            "Inside a dim, rustic pottery workshop, wet clay-covered hands "
            "shape a small vessel on a spinning wheel."
        ),
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def request_record(result: Any, request_index: int) -> dict[str, Any]:
    if result is None:
        raise RuntimeError(f"FastWan request {request_index} returned no result")
    metrics = result.metrics or {}
    step_ms = [float(value) for value in metrics.get("steps", [])]
    return {
        "request_index": request_index,
        "generation_time_ms": float(result.generation_time) * 1000.0,
        "metrics_total_duration_ms": float(metrics.get("total_duration_ms", 0.0)),
        "stages_ms": {
            str(name): float(value)
            for name, value in (metrics.get("stages", {}) or {}).items()
        },
        "step_ms": step_ms,
        "step_sum_ms": sum(step_ms),
        "peak_memory_mb": (
            float(result.peak_memory_mb) if result.peak_memory_mb is not None else None
        ),
        "size": list(result.size),
    }


def main() -> None:
    args = parse_args()
    if args.warmup_requests < 1 or args.measured_requests < 1:
        raise ValueError("warmup and measured request counts must be positive")
    if not Path(args.image_path).is_file():
        raise FileNotFoundError(args.image_path)

    # Import after the caller has fixed profiling-related environment variables.
    from sglang.multimodal_gen import DiffGenerator

    server_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "dmd_denoising_steps": args.timesteps,
        "num_gpus": 1,
        "tp_size": 1,
        "performance_mode": "speed",
        "dit_precision": "bf16",
        "enable_torch_compile": args.enable_torch_compile,
        "warmup": False,
        "warmup_mode": "off",
    }
    if args.attention_backend != "auto":
        server_kwargs["attention_backend"] = args.attention_backend

    sampling = {
        "prompt": args.prompt,
        "image_path": args.image_path,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": 24,
        "guidance_scale": 0.0,
        "num_inference_steps": len(args.timesteps),
        "seed": args.seed,
        "save_output": False,
    }

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        total_requests = args.warmup_requests + args.measured_requests
        for request_index in range(total_requests):
            result = generator.generate(sampling_params_kwargs=sampling)
            record = request_record(result, request_index)
            if len(record["step_ms"]) != len(args.timesteps):
                raise RuntimeError(
                    f"request {request_index} recorded {len(record['step_ms'])} "
                    f"DiT steps, expected {len(args.timesteps)}; set "
                    "SGLANG_DIFFUSION_STAGE_LOGGING=1"
                )
            records.append(record)
            del result
            gc.collect()
    process_wall_s = time.perf_counter() - started

    measured = records[args.warmup_requests :]
    step_sums = [item["step_sum_ms"] for item in measured]
    generation_times = [item["generation_time_ms"] for item in measured]
    output_new_frames = args.num_frames - 1
    result = {
        "schema_version": "minwm-fastwan-dit/v1",
        "contract": {
            "model_path": args.model_path,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "new_pixel_frames": output_new_frames,
            "timesteps": args.timesteps,
            "dit_forward_count": len(args.timesteps),
            "dtype": "bf16",
            "attention_backend_requested": args.attention_backend,
            "torch_compile": args.enable_torch_compile,
            "seed": args.seed,
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.measured_requests,
            "sync_step_profiling": os.environ.get(
                "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"
            ),
        },
        "summary": {
            "dit_step_sum_ms": summarize(step_sums),
            "generation_time_ms": summarize(generation_times),
            "dit_ms_per_forward_p50": statistics.median(step_sums)
            / len(args.timesteps),
            "dit_ms_per_new_frame_per_forward_p50": statistics.median(step_sums)
            / (output_new_frames * len(args.timesteps)),
            "e2e_fps_ratio_of_sums": len(measured)
            * output_new_frames
            / (sum(generation_times) / 1000.0),
            "process_wall_s_including_load": process_wall_s,
        },
        "requests": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
