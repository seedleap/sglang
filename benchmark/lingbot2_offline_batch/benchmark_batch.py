#!/usr/bin/env python3
"""Benchmark production-like offline LingBot video batches over realtime WS.

Each request generates a configurable number of chunks, persists the raw RGB
stream as H.264 MP4, and reports request and aggregate throughput metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msgspec.msgpack
import websockets


DEFAULT_FIRST_FRAME = (
    "https://raw.githubusercontent.com/robbyant/lingbot-world/main/"
    "examples/00/image.jpg"
)

PROMPTS = {
    "first_person": (
        "A cinematic first-person egocentric flight above a lush tropical "
        "valley, moving steadily forward, stable world geometry, natural "
        "parallax, detailed landscape, continuous motion."
    ),
    "third_person": (
        "A cinematic third-person wide tracking shot following two astronauts "
        "exploring the surface of Mars beside a large pressurized rover. The "
        "astronauts and rover remain visible as the camera moves steadily forward, "
        "with stable world geometry, natural parallax, and continuous motion."
    ),
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[max(index, 0)]


@dataclass(frozen=True)
class WorkItem:
    index: int
    perspective: str
    seed: int
    action: str


async def start_encoder(
    output: Path, width: int, height: int, fps: float
) -> asyncio.subprocess.Process:
    output.parent.mkdir(parents=True, exist_ok=True)
    return await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def probe_mp4(path: Path) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,duration,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {stderr.decode()}")
    stream = json.loads(stdout)["streams"][0]
    return {
        "frames": int(stream["nb_read_frames"]),
        "duration_sec": float(stream["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "bytes": path.stat().st_size,
    }


async def generate_video(
    *,
    url: str,
    item: WorkItem,
    output: Path | None,
    chunks: int,
    frames_per_chunk: int,
    size: str,
    fps: float,
    timeout: float,
    first_frame: str,
) -> dict[str, Any]:
    width, height = (int(value) for value in size.split("x", 1))
    init_payload = {
        "type": "init",
        "prompt": PROMPTS[item.perspective],
        "first_frame": first_frame,
        "size": size,
        "fps": fps,
        "num_frames": frames_per_chunk,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": item.seed,
        "max_chunks": chunks,
        "realtime_output_format": "raw",
        "output_compression": 95,
        "realtime_output_pacing": False,
        "enable_upscaling": False,
        "enable_frame_interpolation": False,
        "profile": False,
        "profile_all_stages": False,
        "condition_inputs": {"camera_actions": [[item.action]] * 12},
    }

    encoder = await start_encoder(output, width, height, fps) if output else None
    stats: dict[int, dict[str, Any]] = {}
    frame_chunks: set[int] = set()
    start = time.perf_counter()
    try:
        async with websockets.connect(
            url,
            max_size=None,
            ping_interval=None,
            open_timeout=timeout,
            close_timeout=timeout,
        ) as ws:
            await ws.send(msgspec.msgpack.encode(init_payload))
            await ws.send(
                msgspec.msgpack.encode(
                    {
                        "type": "event",
                        "kind": "camera_actions",
                        "payload": {
                            "mode": "state",
                            "transitions": [
                                {
                                    "actions": [item.action],
                                    "client_ts_ms": int(time.monotonic() * 1000),
                                }
                            ],
                        },
                        "event_id": 1,
                    }
                )
            )

            while len(stats) < chunks or len(frame_chunks) < chunks:
                raw_message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                message = msgspec.msgpack.decode(raw_message)
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(str(message.get("content")))
                if message_type == "chunk_stats":
                    stats[int(message["chunk_index"])] = dict(message)
                    continue
                if message_type == "frame_batch_header":
                    frame_payload = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not isinstance(frame_payload, bytes):
                        raise RuntimeError("raw frame payload must be bytes")
                    chunk_index = int(message["chunk_index"])
                    if encoder is not None:
                        assert encoder.stdin is not None
                        encoder.stdin.write(frame_payload)
                        await encoder.stdin.drain()
                    frame_chunks.add(chunk_index)
                    continue
                if message_type == "frame_batch":
                    raise RuntimeError("expected raw frame_batch_header, got frame_batch")
                raise RuntimeError(f"unexpected realtime message: {message_type!r}")

        delivery_end = time.perf_counter()
        ordered_stats = [stats[index] for index in sorted(stats)]
        frame_count = sum(int(stat["num_frames"]) for stat in ordered_stats)
        scheduler_seconds = sum(
            float(stat["scheduler_forward_ms"]) for stat in ordered_stats
        ) / 1000.0

        media: dict[str, Any] | None = None
        if encoder is not None:
            assert encoder.stdin is not None
            encoder.stdin.close()
            await encoder.stdin.wait_closed()
            _, stderr = await encoder.communicate()
            if encoder.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
            assert output is not None
            media = await probe_mp4(output)
            if media["frames"] != frame_count:
                raise RuntimeError(
                    f"MP4 has {media['frames']} frames, expected {frame_count}"
                )
        persist_end = time.perf_counter()

        return {
            "work_item": asdict(item),
            "url": url,
            "output": str(output) if output else None,
            "chunks": chunks,
            "frames": frame_count,
            "video_seconds": frame_count / fps,
            "generation_delivery_sec": delivery_end - start,
            "persisted_end_to_end_sec": persist_end - start,
            "scheduler_generated_fps": frame_count / scheduler_seconds,
            "delivered_and_persisted_fps": frame_count / (persist_end - start),
            "realtime_factor": (frame_count / fps) / (persist_end - start),
            "media": media,
            "success": True,
        }
    finally:
        if encoder is not None and encoder.returncode is None:
            encoder.kill()
            await encoder.wait()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not urls:
        raise ValueError("at least one URL is required")
    perspective_first_frames = {
        "first_person": args.first_frame_first_person or args.first_frame,
        "third_person": args.first_frame_third_person or args.first_frame,
    }
    if perspective_first_frames["first_person"] == perspective_first_frames["third_person"]:
        print(
            "warning: first_person and third_person share one conditioning frame; "
            "latency remains valid, but perspective semantics require distinct inputs",
            file=sys.stderr,
            flush=True,
        )

    warmup_start = time.perf_counter()
    warmup_perspective = (
        "third_person"
        if args.perspective_mode == "third_person"
        else "first_person"
    )
    warmup_items = [
        WorkItem(
            index=-(index + 1),
            perspective=warmup_perspective,
            seed=1,
            action="w",
        )
        for index in range(len(urls))
    ]
    await asyncio.gather(
        *[
            generate_video(
                url=url,
                item=item,
                output=None,
                chunks=args.warmup_chunks,
                frames_per_chunk=args.frames_per_chunk,
                size=args.size,
                fps=args.fps,
                timeout=args.timeout,
                first_frame=perspective_first_frames[item.perspective],
            )
            for url, item in zip(urls, warmup_items, strict=True)
        ]
    )
    warmup_wall_sec = time.perf_counter() - warmup_start

    items = [
        WorkItem(
            index=index,
            perspective=(
                "first_person" if index % 2 == 0 else "third_person"
            )
            if args.perspective_mode == "mixed"
            else args.perspective_mode,
            seed=1000 + index,
            action=("w", "a", "d", "s")[index % 4],
        )
        for index in range(args.videos)
    ]
    queue: asyncio.Queue[WorkItem] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    results: list[dict[str, Any]] = []

    async def worker(url: str) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            output = args.output_dir / f"video-{item.index:02d}-{item.perspective}.mp4"
            try:
                result = await generate_video(
                    url=url,
                    item=item,
                    output=output,
                    chunks=args.chunks,
                    frames_per_chunk=args.frames_per_chunk,
                    size=args.size,
                    fps=args.fps,
                    timeout=args.timeout,
                    first_frame=perspective_first_frames[item.perspective],
                )
            except Exception as exc:
                result = {
                    "work_item": asdict(item),
                    "url": url,
                    "output": str(output),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            queue.task_done()

    measured_start = time.perf_counter()
    await asyncio.gather(*[worker(url) for url in urls])
    measured_wall_sec = time.perf_counter() - measured_start

    successful = [result for result in results if result["success"]]
    latencies = [result["persisted_end_to_end_sec"] for result in successful]
    total_video_seconds = sum(result["video_seconds"] for result in successful)
    node_videos_per_hour = len(successful) * 3600.0 / measured_wall_sec
    gpu_count = args.gpu_count
    summary = {
        "server_startup_sec": args.server_startup_sec,
        "warmup_wall_sec": warmup_wall_sec,
        "measured_wall_sec": measured_wall_sec,
        "requested_videos": args.videos,
        "successful_videos": len(successful),
        "failed_videos": args.videos - len(successful),
        "failure_rate": (args.videos - len(successful)) / args.videos,
        "total_generated_video_seconds": total_video_seconds,
        "node_videos_per_hour": node_videos_per_hour,
        "videos_per_gpu_hour": node_videos_per_hour / gpu_count,
        "generated_seconds_per_gpu_hour": (
            total_video_seconds * 3600.0 / (measured_wall_sec * gpu_count)
        ),
        "aggregate_realtime_factor": total_video_seconds / measured_wall_sec,
        "request_persisted_end_to_end_sec": {
            "mean": statistics.fmean(latencies) if latencies else math.nan,
            "p50": statistics.median(latencies) if latencies else math.nan,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else math.nan,
        },
        "perspective_p50_sec": {
            perspective: statistics.median(
                result["persisted_end_to_end_sec"]
                for result in successful
                if result["work_item"]["perspective"] == perspective
            )
            for perspective in PROMPTS
            if any(
                result["work_item"]["perspective"] == perspective
                for result in successful
            )
        },
    }
    return {
        "config": {
            "urls": urls,
            "gpu_count": gpu_count,
            "size": args.size,
            "fps": args.fps,
            "chunks": args.chunks,
            "frames_per_chunk": args.frames_per_chunk,
            "nominal_video_seconds": (9 + (args.chunks - 1) * 12) / args.fps,
            "videos": args.videos,
            "perspective_mode": args.perspective_mode,
            "warmup_chunks_per_server": args.warmup_chunks,
            "steps": 4,
            "output_format": "raw-to-H.264-MP4",
            "upscaling": False,
            "frame_interpolation": False,
            "distinct_perspective_conditioning_frames": (
                perspective_first_frames["first_person"]
                != perspective_first_frames["third_person"]
            ),
        },
        "summary": summary,
        "results": sorted(results, key=lambda result: result["work_item"]["index"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True, help="Comma-separated WS endpoints")
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--frames-per-chunk", type=int, default=9)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--perspective-mode",
        choices=("mixed", "first_person", "third_person"),
        default="mixed",
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--first-frame",
        default=DEFAULT_FIRST_FRAME,
        help="Fallback conditioning frame used when a perspective-specific frame is absent",
    )
    parser.add_argument("--first-frame-first-person")
    parser.add_argument("--first-frame-third-person")
    parser.add_argument("--server-startup-sec", type=float, default=math.nan)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    if result["summary"]["failed_videos"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
