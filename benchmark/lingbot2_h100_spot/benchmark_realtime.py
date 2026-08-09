#!/usr/bin/env python3
"""Measure steady-state LingBot generated and delivered FPS over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import msgspec.msgpack
import websockets


DEFAULT_FIRST_FRAME = (
    "https://raw.githubusercontent.com/robbyant/lingbot-world/main/"
    "examples/00/image.jpg"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[max(index, 0)]


def summarize(
    stats: list[dict[str, Any]],
    receive_times: list[float],
    warmup_chunks: int,
) -> dict[str, Any]:
    measured = stats[warmup_chunks:]
    measured_receive_times = receive_times[warmup_chunks:]
    if not measured:
        raise ValueError("warmup_chunks must leave at least one measured chunk")

    frames = sum(int(item["num_frames"]) for item in measured)
    scheduler_ms = [float(item["scheduler_forward_ms"]) for item in measured]
    chunk_total_ms = [float(item["chunk_total_ms"]) for item in measured]
    if len(measured_receive_times) >= 2:
        receive_seconds = measured_receive_times[-1] - measured_receive_times[0]
        receive_frames = sum(int(item["num_frames"]) for item in measured[1:])
        delivered_fps = receive_frames / receive_seconds
    else:
        delivered_fps = math.nan

    return {
        "total_chunks": len(stats),
        "warmup_chunks": warmup_chunks,
        "measured_chunks": len(measured),
        "measured_frames": frames,
        "generated_fps_scheduler_sum": frames / (sum(scheduler_ms) / 1000.0),
        "delivered_fps_chunk_stats": delivered_fps,
        "scheduler_forward_ms": {
            "mean": statistics.fmean(scheduler_ms),
            "p50": statistics.median(scheduler_ms),
            "p95": percentile(scheduler_ms, 0.95),
            "p99": percentile(scheduler_ms, 0.99),
            "max": max(scheduler_ms),
        },
        "chunk_total_ms": {
            "mean": statistics.fmean(chunk_total_ms),
            "p50": statistics.median(chunk_total_ms),
            "p95": percentile(chunk_total_ms, 0.95),
            "p99": percentile(chunk_total_ms, 0.99),
            "max": max(chunk_total_ms),
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    # Bootstrap with four moving chunks so that chunk 0 is moving even if the
    # event listener has not consumed the state event yet.  Do not encode the
    # whole run as a script: LingBot intentionally caps script mode at 512
    # entries, which silently switches a long benchmark back to still mode.
    bootstrap_actions = [[args.action]] * 12
    init_payload = {
        "type": "init",
        "prompt": args.prompt,
        "first_frame": args.first_frame,
        "size": args.size,
        "fps": args.fps,
        "num_frames": args.frames_per_chunk,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": args.seed,
        "max_chunks": args.chunks,
        "realtime_output_format": args.output_format,
        "output_compression": args.output_quality,
        "realtime_output_pacing": False,
        "enable_upscaling": False,
        "enable_frame_interpolation": False,
        "profile": args.profile,
        "profile_all_stages": args.profile_all_stages,
        "num_profiled_timesteps": args.num_profiled_timesteps,
        "condition_inputs": {"camera_actions": bootstrap_actions},
    }

    stats: list[dict[str, Any]] = []
    receive_times: list[float] = []
    frame_chunks: set[int] = set()
    captured_chunks: dict[str, dict[str, Any]] = {}
    capture_indices = set(args.capture_chunk)
    async with websockets.connect(
        args.url,
        max_size=None,
        ping_interval=None,
        open_timeout=args.timeout,
        close_timeout=args.timeout,
    ) as ws:
        await ws.send(msgspec.msgpack.encode(init_payload))
        # State mode is level-triggered: one transition holds the requested
        # action for the complete session, independent of benchmark length.
        await ws.send(
            msgspec.msgpack.encode(
                {
                    "type": "event",
                    "kind": "camera_actions",
                    "payload": {
                        "mode": "state",
                        "transitions": [
                            {
                                "actions": [args.action],
                                "client_ts_ms": int(time.monotonic() * 1000),
                            }
                        ],
                    },
                    "event_id": 1,
                }
            )
        )
        while len(stats) < args.chunks:
            raw_message = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            message = msgspec.msgpack.decode(raw_message)
            message_type = message.get("type")
            if message_type == "error":
                raise RuntimeError(str(message.get("content")))
            if message_type == "chunk_stats":
                stats.append(dict(message))
                receive_times.append(time.monotonic())
                continue
            if message_type == "frame_batch":
                frame_chunks.add(int(message["chunk_index"]))
                continue
            if message_type == "frame_batch_header":
                frame_payload = await asyncio.wait_for(
                    ws.recv(), timeout=args.timeout
                )
                chunk_index = int(message["chunk_index"])
                frame_chunks.add(chunk_index)
                if chunk_index in capture_indices:
                    if not isinstance(frame_payload, bytes):
                        raise RuntimeError(
                            "raw frame payload must be bytes for capture"
                        )
                    capture_dir = args.capture_dir or args.output.parent
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    capture_path = (
                        capture_dir
                        / f"{args.output.stem}.chunk{chunk_index}.raw"
                    )
                    capture_path.write_bytes(frame_payload)
                    captured_chunks[str(chunk_index)] = {
                        "path": str(capture_path),
                        "bytes": len(frame_payload),
                        "sha256": hashlib.sha256(frame_payload).hexdigest(),
                    }
                continue
            raise RuntimeError(f"unexpected realtime message: {message_type!r}")

    result = {
        "config": {
            "url": args.url,
            "size": args.size,
            "frames_per_chunk": args.frames_per_chunk,
            "chunks": args.chunks,
            "warmup_chunks": args.warmup_chunks,
            "steps": 4,
            "guidance_scale": 1.0,
            "seed": args.seed,
            "action": args.action,
            "action_control_mode": "persistent_state",
            "action_event_id": 1,
            "output_format": args.output_format,
            "output_quality": args.output_quality,
            "realtime_output_pacing": False,
            "frame_interpolation": False,
            "upscaling": False,
            "profile": args.profile,
            "profile_all_stages": args.profile_all_stages,
            "num_profiled_timesteps": args.num_profiled_timesteps,
        },
        "summary": summarize(stats, receive_times, args.warmup_chunks),
        "received_frame_chunks": len(frame_chunks),
        "captured_chunks": captured_chunks,
        "chunk_stats": stats,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="ws://codex-lingbot2-h100-perf:30000/v1/realtime_video/generate",
    )
    parser.add_argument("--chunks", type=int, default=220)
    parser.add_argument("--warmup-chunks", type=int, default=20)
    parser.add_argument("--frames-per-chunk", type=int, default=9)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action", default="w")
    parser.add_argument("--output-format", choices=("raw", "webp", "jpeg"), default="raw")
    parser.add_argument("--output-quality", type=int, default=95)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-all-stages", action="store_true")
    parser.add_argument("--num-profiled-timesteps", type=int)
    parser.add_argument("--first-frame", default=DEFAULT_FIRST_FRAME)
    parser.add_argument(
        "--capture-chunk",
        action="append",
        type=int,
        default=[],
        help="Save the raw RGB payload for this chunk index; repeat as needed.",
    )
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic first-person flight above a lush tropical valley, "
            "stable geometry, natural parallax, detailed landscape."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_chunks >= args.chunks:
        raise SystemExit("--warmup-chunks must be smaller than --chunks")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
