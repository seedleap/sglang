#!/usr/bin/env python3
"""Measure FastWan DiT efficiency as temporal request size changes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from pathlib import Path
from typing import Any


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--num-frames", type=parse_int_list, default=parse_int_list("17,33,49,65,81")
    )
    parser.add_argument(
        "--timesteps", type=parse_int_list, default=parse_int_list("1000,757,522,300,100")
    )
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-requests", type=int, default=5)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
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


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def record_request(result: Any, request_index: int, expected_steps: int) -> dict[str, Any]:
    if result is None:
        raise RuntimeError(f"request {request_index} returned no result")
    metrics = result.metrics or {}
    steps = [float(value) for value in metrics.get("steps", [])]
    if len(steps) != expected_steps:
        raise RuntimeError(
            f"request {request_index} recorded {len(steps)} DiT steps; "
            f"expected {expected_steps}"
        )
    return {
        "request_index": request_index,
        "generation_time_ms": float(result.generation_time) * 1000.0,
        "step_ms": steps,
        "step_sum_ms": sum(steps),
        "stages_ms": {
            str(name): float(value)
            for name, value in (metrics.get("stages", {}) or {}).items()
        },
        "size": list(result.size),
    }


def main() -> None:
    args = parse_args()
    if not Path(args.image_path).is_file():
        raise FileNotFoundError(args.image_path)
    if args.warmup_requests < 1 or args.measured_requests < 1:
        raise ValueError("warmup and measured request counts must be positive")
    for frames in args.num_frames:
        if frames < 5 or (frames - 1) % 4:
            raise ValueError(
                f"num_frames must satisfy (frames - 1) % 4 == 0, got {frames}"
            )

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

    frame_results: list[dict[str, Any]] = []
    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        for num_frames in args.num_frames:
            sampling = {
                "prompt": args.prompt,
                "image_path": args.image_path,
                "height": args.height,
                "width": args.width,
                "num_frames": num_frames,
                "fps": 24,
                "guidance_scale": 0.0,
                "num_inference_steps": len(args.timesteps),
                "seed": args.seed,
                "save_output": False,
            }
            records = []
            total = args.warmup_requests + args.measured_requests
            for request_index in range(total):
                result = generator.generate(sampling_params_kwargs=sampling)
                records.append(record_request(result, request_index, len(args.timesteps)))
                del result
                gc.collect()

            measured = records[args.warmup_requests :]
            step_sums = [item["step_sum_ms"] for item in measured]
            generation_times = [item["generation_time_ms"] for item in measured]
            new_pixel_frames = num_frames - 1
            new_latent_frames = new_pixel_frames // 4
            latent_frames = new_latent_frames + 1
            spatial_tokens_per_latent_frame = (args.height // 16) * (args.width // 16)
            sequence_tokens = latent_frames * spatial_tokens_per_latent_frame
            new_sequence_tokens = new_latent_frames * spatial_tokens_per_latent_frame
            p50_step_sum_ms = statistics.median(step_sums)
            frame_result = {
                "num_frames": num_frames,
                "new_pixel_frames": new_pixel_frames,
                "latent_frames": latent_frames,
                "new_latent_frames": new_latent_frames,
                "spatial_tokens_per_latent_frame": spatial_tokens_per_latent_frame,
                "sequence_tokens": sequence_tokens,
                "new_sequence_tokens": new_sequence_tokens,
                "summary": {
                    "dit_step_sum_ms": summarize(step_sums),
                    "generation_time_ms": summarize(generation_times),
                    "dit_ms_per_forward_p50": p50_step_sum_ms / len(args.timesteps),
                    "dit_ms_per_new_pixel_frame_per_forward_p50": (
                        p50_step_sum_ms / (new_pixel_frames * len(args.timesteps))
                    ),
                    "dit_new_tokens_per_ms_per_forward_p50": (
                        new_sequence_tokens / (p50_step_sum_ms / len(args.timesteps))
                    ),
                },
                "requests": records,
            }
            frame_results.append(frame_result)
            print(json.dumps({"num_frames": num_frames, **frame_result["summary"]}, sort_keys=True))

    payload = {
        "schema_version": "minwm-fastwan-shape-sweep/v1",
        "contract": {
            "model_path": args.model_path,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "timesteps": args.timesteps,
            "dit_forward_count": len(args.timesteps),
            "dtype": "bf16",
            "attention_backend_requested": args.attention_backend,
            "torch_compile": args.enable_torch_compile,
            "seed": args.seed,
            "warmup_requests_per_shape": args.warmup_requests,
            "measured_requests_per_shape": args.measured_requests,
            "sync_step_profiling": os.environ.get(
                "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"
            ),
        },
        "frames": frame_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
