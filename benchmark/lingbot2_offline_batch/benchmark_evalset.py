#!/usr/bin/env python3
"""Run a minWM messages.jsonl evaluation set through LingBot realtime serving.

The minWM fixture stores camera controls at decoded-video-frame granularity.
LingBot consumes one camera control per latent frame, so this client verifies
that each four-frame control block is constant and downsamples it exactly once.
Outputs are truncated to the fixture's requested frame count when that count is
not divisible by LingBot's three-latent-frame realtime chunk size.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import msgspec.msgpack
import websockets


TEMPORAL_COMPRESSION = 4
LINGBOT_LATENTS_PER_CHUNK = 3
FIRST_CHUNK_VIDEO_FRAMES = 9
LATER_CHUNK_VIDEO_FRAMES = 12


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[max(index, 0)]


@dataclass(frozen=True)
class PromptUpdate:
    event_id: int
    requested_generated_latent: int
    target_chunk: int
    trigger_after_chunk: int
    prompt: str


@dataclass(frozen=True)
class EvalItem:
    index: int
    sample_id: str
    group: str
    image_id: str
    image_url: str
    prompt: str
    negative_prompt: str | None
    seed: int
    trajectory: str
    view: dict[str, Any]
    generated_latent_frames: int
    target_video_frames: int
    chunks: int
    camera_actions: list[list[str]]
    prompt_updates: list[PromptUpdate]

    @property
    def relative_output(self) -> Path:
        parts = self.sample_id.split("/")
        return Path(*parts[1:]).with_suffix(".mp4")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def _control_by_type(target: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for control in target.get("controls", []):
        if control.get("type") == kind:
            return control
    return None


def _latent_camera_actions(control: dict[str, Any], sample_id: str) -> list[list[str]]:
    keys = control["action_keys"]
    video_actions = control["actions"]
    if len(video_actions) % TEMPORAL_COMPRESSION:
        raise ValueError(
            f"{sample_id}: {len(video_actions)} controls are not divisible by "
            f"{TEMPORAL_COMPRESSION}"
        )

    latent_actions = []
    for offset in range(0, len(video_actions), TEMPORAL_COMPRESSION):
        block = video_actions[offset : offset + TEMPORAL_COMPRESSION]
        if any(frame != block[0] for frame in block[1:]):
            raise ValueError(
                f"{sample_id}: camera control changes inside four-frame block {offset // 4}"
            )
        if len(block[0]) != len(keys):
            raise ValueError(f"{sample_id}: control width does not match action_keys")
        latent_actions.append([key for key, enabled in zip(keys, block[0]) if enabled])
    return latent_actions


def _prompt_updates(
    row: dict[str, Any], target: dict[str, Any]
) -> list[PromptUpdate]:
    switch_frames = row["metadata"].get("condition_switch_frame_indices", [])
    prompt_control = _control_by_type(target, "text_prompt_interval")
    if not switch_frames:
        if prompt_control is not None:
            raise ValueError(f"{row['sample_id']}: prompt segments exist without switches")
        return []
    if prompt_control is None:
        raise ValueError(f"{row['sample_id']}: prompt switches lack text_prompt_interval")

    segments = prompt_control["segments"]
    if len(segments) != len(switch_frames) + 1:
        raise ValueError(
            f"{row['sample_id']}: {len(switch_frames)} switches but "
            f"{len(segments)} prompt segments"
        )

    updates = []
    for update_index, (generated_latent, segment) in enumerate(
        zip(switch_frames, segments[1:]), 1
    ):
        # The fixture switch is a generated-latent boundary. LingBot can only
        # update on a 3-latent chunk boundary, so choose the first boundary at
        # or after it. The server pipelines one output send with the following
        # generation, hence the event is sent after target_chunk - 2 arrives.
        target_chunk = math.ceil((int(generated_latent) + 1) / LINGBOT_LATENTS_PER_CHUNK)
        updates.append(
            PromptUpdate(
                event_id=update_index,
                requested_generated_latent=int(generated_latent),
                target_chunk=target_chunk,
                trigger_after_chunk=max(0, target_chunk - 2),
                prompt=str(segment["text"]),
            )
        )
    return updates


def load_items(messages: Path, image_urls: Path) -> list[EvalItem]:
    url_map = json.loads(image_urls.read_text())
    items = []
    for index, row in enumerate(read_jsonl(messages)):
        sample_id = row["sample_id"]
        target_messages = [
            message
            for message in row["messages"]
            if message.get("role") == "target" and message.get("type") == "video"
        ]
        user_messages = [
            message
            for message in row["messages"]
            if message.get("role") == "user" and message.get("type") == "text"
        ]
        if len(target_messages) != 1 or len(user_messages) != 1:
            raise ValueError(f"{sample_id}: expected one user text and one target video")
        target = target_messages[0]
        video_metadata = target["metadata"]
        generated_latents = int(video_metadata["generated_latent_frames"])
        target_frames = int(video_metadata["output_video_frames"])
        expected_frames = generated_latents * TEMPORAL_COMPRESSION + 1
        if target_frames != expected_frames:
            raise ValueError(
                f"{sample_id}: output_video_frames={target_frames}, expected {expected_frames}"
            )

        camera_control = _control_by_type(target, "keyboard_direction_frame_interval")
        if camera_control is None:
            raise ValueError(f"{sample_id}: missing keyboard control")
        latent_actions = _latent_camera_actions(camera_control, sample_id)
        if len(latent_actions) != generated_latents:
            raise ValueError(
                f"{sample_id}: {len(latent_actions)} latent controls for "
                f"{generated_latents} generated latents"
            )

        chunks = math.ceil(
            (generated_latents + 1) / LINGBOT_LATENTS_PER_CHUNK
        )
        # The first output latent is fixed by the reference image. Add a
        # neutral camera pose for it, then pad only the final partial chunk.
        model_actions = [[]] + latent_actions
        model_actions.extend(
            [] for _ in range(chunks * LINGBOT_LATENTS_PER_CHUNK - len(model_actions))
        )

        image_id = row["metadata"]["image_id"]
        image_url = url_map.get(image_id)
        if not image_url:
            raise ValueError(f"{sample_id}: no signed image URL for {image_id}")
        items.append(
            EvalItem(
                index=index,
                sample_id=sample_id,
                group=row["metadata"]["group"],
                image_id=image_id,
                image_url=image_url,
                prompt=user_messages[0]["content"],
                negative_prompt=video_metadata.get("negative_prompt"),
                seed=int(row["metadata"].get("seed", 0)),
                trajectory=row["metadata"]["trajectory"],
                view=row["metadata"]["view"],
                generated_latent_frames=generated_latents,
                target_video_frames=target_frames,
                chunks=chunks,
                camera_actions=model_actions,
                prompt_updates=_prompt_updates(row, target),
            )
        )
    return items


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
        "-f",
        "mp4",
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


async def existing_output_is_valid(
    output: Path, *, frames: int, width: int, height: int
) -> bool:
    if not output.exists():
        return False
    try:
        media = await probe_mp4(output)
    except Exception:
        return False
    return (
        media["frames"] == frames
        and media["width"] == width
        and media["height"] == height
    )


async def generate_video(
    *,
    url: str,
    item: EvalItem,
    phase: str,
    output: Path | None,
    width: int,
    height: int,
    fps: float,
    timeout: float,
    close_timeout: float,
) -> dict[str, Any]:
    temp_output = (
        output.with_name(f"{output.stem}.partial.mp4") if output is not None else None
    )
    if temp_output is not None:
        temp_output.unlink(missing_ok=True)
    encoder = (
        await start_encoder(temp_output, width, height, fps)
        if temp_output is not None
        else None
    )
    init_payload = {
        "type": "init",
        "prompt": item.prompt,
        "negative_prompt": item.negative_prompt,
        "first_frame": item.image_url,
        "size": f"{width}x{height}",
        "fps": fps,
        "num_frames": FIRST_CHUNK_VIDEO_FRAMES,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": item.seed,
        "max_chunks": item.chunks,
        "realtime_output_format": "raw",
        "output_compression": 95,
        "realtime_output_pacing": False,
        "enable_upscaling": False,
        "enable_frame_interpolation": False,
        "profile": False,
        "profile_all_stages": False,
        "condition_inputs": {"camera_actions": item.camera_actions},
    }

    stats: dict[int, dict[str, Any]] = {}
    frame_chunks: set[int] = set()
    sent_prompt_events: list[dict[str, Any]] = []
    updates_by_trigger: dict[int, list[PromptUpdate]] = {}
    for update in item.prompt_updates:
        updates_by_trigger.setdefault(update.trigger_after_chunk, []).append(update)
    received_service_frames = 0
    persisted_frames = 0
    frame_bytes = width * height * 3
    start = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "realtime_client_connect",
                "phase": phase,
                "sample_id": item.sample_id,
                "url": url,
                "chunks": item.chunks,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        async with websockets.connect(
            url,
            max_size=None,
            ping_interval=None,
            open_timeout=timeout,
            close_timeout=close_timeout,
        ) as ws:
            await ws.send(msgspec.msgpack.encode(init_payload))
            while len(stats) < item.chunks or len(frame_chunks) < item.chunks:
                raw_message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                message = msgspec.msgpack.decode(raw_message)
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(str(message.get("content")))
                if message_type == "chunk_stats":
                    chunk_index = int(message["chunk_index"])
                    stats[chunk_index] = dict(message)
                    for update in updates_by_trigger.pop(chunk_index, []):
                        await ws.send(
                            msgspec.msgpack.encode(
                                {
                                    "type": "event",
                                    "kind": "prompt",
                                    "payload": update.prompt,
                                    "event_id": update.event_id,
                                }
                            )
                        )
                        sent_prompt_events.append(
                            {
                                "event_id": update.event_id,
                                "requested_generated_latent": update.requested_generated_latent,
                                "target_chunk": update.target_chunk,
                                "sent_after_chunk": chunk_index,
                            }
                        )
                    continue
                if message_type == "frame_batch_header":
                    frame_payload = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not isinstance(frame_payload, bytes):
                        raise RuntimeError("raw frame payload must be bytes")
                    payload_frames = int(message["num_frames"])
                    expected_bytes = payload_frames * frame_bytes
                    if len(frame_payload) != expected_bytes:
                        raise RuntimeError(
                            f"raw payload has {len(frame_payload)} bytes, expected {expected_bytes}"
                        )
                    received_service_frames += payload_frames
                    keep_frames = min(
                        payload_frames, item.target_video_frames - persisted_frames
                    )
                    if encoder is not None and keep_frames > 0:
                        assert encoder.stdin is not None
                        encoder.stdin.write(frame_payload[: keep_frames * frame_bytes])
                        await encoder.stdin.drain()
                    persisted_frames += keep_frames
                    if message.get("is_final_frame_batch", True):
                        frame_chunks.add(int(message["chunk_index"]))
                    continue
                if message_type == "frame_batch":
                    raise RuntimeError("expected raw frame_batch_header, got frame_batch")
                raise RuntimeError(f"unexpected realtime message: {message_type!r}")

        delivery_end = time.perf_counter()
        expected_service_frames = FIRST_CHUNK_VIDEO_FRAMES + (
            item.chunks - 1
        ) * LATER_CHUNK_VIDEO_FRAMES
        if received_service_frames != expected_service_frames:
            raise RuntimeError(
                f"received {received_service_frames} service frames, "
                f"expected {expected_service_frames}"
            )
        if persisted_frames != item.target_video_frames:
            raise RuntimeError(
                f"persisted {persisted_frames} frames, expected {item.target_video_frames}"
            )
        if updates_by_trigger:
            raise RuntimeError(
                f"prompt events were not sent for triggers {sorted(updates_by_trigger)}"
            )

        media = None
        if encoder is not None:
            assert encoder.stdin is not None
            encoder.stdin.close()
            await encoder.stdin.wait_closed()
            _, stderr = await encoder.communicate()
            if encoder.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
            assert temp_output is not None and output is not None
            media = await probe_mp4(temp_output)
            if media["frames"] != item.target_video_frames:
                raise RuntimeError(
                    f"MP4 has {media['frames']} frames, expected {item.target_video_frames}"
                )
            temp_output.replace(output)
            media["bytes"] = output.stat().st_size
        persist_end = time.perf_counter()
        scheduler_seconds = sum(
            float(chunk_stats["scheduler_forward_ms"])
            for chunk_stats in stats.values()
        ) / 1000.0
        return {
            "sample_id": item.sample_id,
            "group": item.group,
            "image_id": item.image_id,
            "view": item.view,
            "trajectory": item.trajectory,
            "url": url,
            "output": str(output) if output else None,
            "chunks": item.chunks,
            "generated_latent_frames": item.generated_latent_frames,
            "service_frames": received_service_frames,
            "persisted_frames": persisted_frames,
            "trimmed_service_frames": received_service_frames - persisted_frames,
            "video_seconds": persisted_frames / fps,
            "generation_delivery_sec": delivery_end - start,
            "persisted_end_to_end_sec": persist_end - start,
            "scheduler_generated_fps": received_service_frames / scheduler_seconds,
            "delivered_and_persisted_fps": persisted_frames / (persist_end - start),
            "realtime_factor": (persisted_frames / fps) / (persist_end - start),
            "prompt_events": sent_prompt_events,
            "media": media,
            "resumed": False,
            "success": True,
        }
    finally:
        if encoder is not None and encoder.returncode is None:
            encoder.kill()
            await encoder.wait()
        if temp_output is not None and temp_output.exists():
            temp_output.unlink()


def warmup_item(item: EvalItem, chunks: int) -> EvalItem:
    service_frames = FIRST_CHUNK_VIDEO_FRAMES + (chunks - 1) * LATER_CHUNK_VIDEO_FRAMES
    return EvalItem(
        index=-1,
        sample_id=f"warmup/{item.image_id}",
        group="warmup",
        image_id=item.image_id,
        image_url=item.image_url,
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        seed=1,
        trajectory="idle",
        view=item.view,
        generated_latent_frames=chunks * LINGBOT_LATENTS_PER_CHUNK - 1,
        target_video_frames=service_frames,
        chunks=chunks,
        camera_actions=[[] for _ in range(chunks * LINGBOT_LATENTS_PER_CHUNK)],
        prompt_updates=[],
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not urls:
        raise ValueError("at least one URL is required")
    all_items = load_items(args.messages, args.image_urls)
    items = all_items
    if args.sample_id_regex:
        pattern = re.compile(args.sample_id_regex)
        items = [item for item in items if pattern.search(item.sample_id)]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("sample selection is empty")

    if args.dry_run:
        return {
            "config": {
                "total_fixture_samples": len(all_items),
                "selected_samples": len(items),
                "groups": dict(Counter(item.group for item in items)),
                "generated_latent_frames": dict(
                    Counter(item.generated_latent_frames for item in items)
                ),
                "prompt_switch_samples": sum(bool(item.prompt_updates) for item in items),
            }
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    progress_lock = asyncio.Lock()

    warmup_wall_sec = 0.0
    if args.warmup_chunks > 0:
        warmup_start = time.perf_counter()
        await asyncio.gather(
            *[
                generate_video(
                    url=url,
                    item=warmup_item(items[index % len(items)], args.warmup_chunks),
                    phase="warmup",
                    output=None,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    timeout=args.timeout,
                    close_timeout=args.close_timeout,
                )
                for index, url in enumerate(urls)
            ]
        )
        warmup_wall_sec = time.perf_counter() - warmup_start

    queue: asyncio.Queue[EvalItem] = asyncio.Queue()
    resumed_results = []
    for item in items:
        output = args.output_dir / "videos" / item.relative_output
        if args.resume and await existing_output_is_valid(
            output,
            frames=item.target_video_frames,
            width=args.width,
            height=args.height,
        ):
            media = await probe_mp4(output)
            resumed_results.append(
                {
                    "sample_id": item.sample_id,
                    "group": item.group,
                    "output": str(output),
                    "persisted_frames": item.target_video_frames,
                    "video_seconds": item.target_video_frames / args.fps,
                    "media": media,
                    "resumed": True,
                    "success": True,
                }
            )
        else:
            queue.put_nowait(item)

    new_results: list[dict[str, Any]] = []

    async def record_progress(result: dict[str, Any]) -> None:
        async with progress_lock:
            with progress_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()

    async def worker(url: str) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            output = args.output_dir / "videos" / item.relative_output
            try:
                result = await generate_video(
                    url=url,
                    item=item,
                    phase="measurement",
                    output=output,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    timeout=args.timeout,
                    close_timeout=args.close_timeout,
                )
            except Exception as exc:
                result = {
                    "sample_id": item.sample_id,
                    "group": item.group,
                    "image_id": item.image_id,
                    "url": url,
                    "output": str(output),
                    "resumed": False,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            new_results.append(result)
            await record_progress(result)
            print(
                json.dumps(
                    {
                        "completed": len(new_results),
                        "queued_this_run": len(items) - len(resumed_results),
                        "sample_id": item.sample_id,
                        "success": result["success"],
                        "latency_sec": result.get("persisted_end_to_end_sec"),
                        "error": result.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            queue.task_done()

    measured_start = time.perf_counter()
    await asyncio.gather(*[worker(url) for url in urls])
    measured_wall_sec = time.perf_counter() - measured_start
    results = resumed_results + new_results
    successful = [result for result in results if result["success"]]
    successful_new = [
        result for result in new_results if result["success"] and not result["resumed"]
    ]
    failed = [result for result in results if not result["success"]]
    latencies = [result["persisted_end_to_end_sec"] for result in successful_new]
    total_new_video_seconds = sum(
        result["video_seconds"] for result in successful_new
    )
    node_videos_per_hour = (
        len(successful_new) * 3600.0 / measured_wall_sec
        if measured_wall_sec > 0
        else math.nan
    )
    summary = {
        "warmup_wall_sec": warmup_wall_sec,
        "measured_wall_sec": measured_wall_sec,
        "selected_samples": len(items),
        "resumed_samples": len(resumed_results),
        "new_successful_samples": len(successful_new),
        "successful_samples": len(successful),
        "failed_samples": len(failed),
        "failure_rate": len(failed) / len(items),
        "total_persisted_video_seconds": sum(
            result["video_seconds"] for result in successful
        ),
        "new_persisted_video_seconds": total_new_video_seconds,
        "node_videos_per_hour_this_run": node_videos_per_hour,
        "videos_per_gpu_hour_this_run": node_videos_per_hour / args.gpu_count,
        "generated_seconds_per_gpu_hour_this_run": (
            total_new_video_seconds * 3600.0 / (measured_wall_sec * args.gpu_count)
            if measured_wall_sec > 0
            else math.nan
        ),
        "aggregate_realtime_factor_this_run": (
            total_new_video_seconds / measured_wall_sec
            if measured_wall_sec > 0
            else math.nan
        ),
        "request_persisted_end_to_end_sec": {
            "mean": statistics.fmean(latencies) if latencies else math.nan,
            "p50": statistics.median(latencies) if latencies else math.nan,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else math.nan,
        },
        "successful_by_group": dict(
            Counter(result["group"] for result in successful)
        ),
        "failed_sample_ids": [result["sample_id"] for result in failed],
    }
    return {
        "config": {
            "source_messages": str(args.messages),
            "fixture_samples": len(all_items),
            "urls": urls,
            "request_concurrency": len(urls),
            "gpu_count": args.gpu_count,
            "gpus_per_request": args.gpu_count // len(urls),
            "size": f"{args.width}x{args.height}",
            "fps": args.fps,
            "steps": 4,
            "output_format": "raw-to-H.264-MP4",
            "temporal_control_downsample": TEMPORAL_COMPRESSION,
            "lingbot_latents_per_chunk": LINGBOT_LATENTS_PER_CHUNK,
            "prompt_switch_quantization": "first 3-latent chunk boundary at or after fixture switch",
            "sample_id_regex": args.sample_id_regex,
        },
        "summary": summary,
        "results": sorted(results, key=lambda result: result["sample_id"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--image-urls", type=Path, required=True)
    parser.add_argument("--urls", required=True, help="Comma-separated WS endpoints")
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--close-timeout", type=float, default=10.0)
    parser.add_argument("--sample-id-regex")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result.get("summary", result["config"]), indent=2), flush=True)
    if result.get("summary", {}).get("failed_samples"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
