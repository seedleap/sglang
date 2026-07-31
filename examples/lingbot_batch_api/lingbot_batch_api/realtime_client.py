"""One-job client for SGLang's internal LingBot realtime WebSocket API."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import msgspec.msgpack
import websockets

from .actions import ActionPair, FPS, MAX_CHUNKS, VIDEO_FRAMES, realtime_latent_actions


WIDTH = 1280
HEIGHT = 720
FIRST_CHUNK_VIDEO_FRAMES = 9
LATER_CHUNK_VIDEO_FRAMES = 12


async def _start_encoder(output: Path) -> asyncio.subprocess.Process:
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
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
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


async def generate_mp4(
    *,
    ws_url: str,
    prompt: str,
    negative_prompt: str | None,
    first_frame: str,
    pair: ActionPair,
    seed: int,
    output: Path,
    timeout: float = 1200.0,
) -> dict[str, Any]:
    """Generate exactly one 129-frame 720p MP4."""

    partial = output.with_name(f"{output.stem}.partial.mp4")
    partial.unlink(missing_ok=True)
    encoder = await _start_encoder(partial)
    init_payload = {
        "type": "init",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "first_frame": first_frame,
        "size": f"{WIDTH}x{HEIGHT}",
        "fps": FPS,
        "num_frames": FIRST_CHUNK_VIDEO_FRAMES,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": seed,
        "max_chunks": MAX_CHUNKS,
        "realtime_output_format": "raw",
        "output_compression": 95,
        "realtime_output_pacing": False,
        "enable_upscaling": False,
        "enable_frame_interpolation": False,
        "profile": False,
        "profile_all_stages": False,
        "condition_inputs": {"camera_actions": realtime_latent_actions(pair)},
    }
    frame_bytes = WIDTH * HEIGHT * 3
    chunks_with_stats: set[int] = set()
    chunks_with_frames: set[int] = set()
    persisted_frames = 0
    start = time.perf_counter()
    try:
        async with websockets.connect(
            ws_url,
            max_size=None,
            ping_interval=None,
            open_timeout=timeout,
            close_timeout=timeout,
        ) as websocket:
            await websocket.send(msgspec.msgpack.encode(init_payload))
            while (
                len(chunks_with_stats) < MAX_CHUNKS
                or len(chunks_with_frames) < MAX_CHUNKS
            ):
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                message = msgspec.msgpack.decode(raw)
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(str(message.get("content")))
                if message_type == "chunk_stats":
                    chunks_with_stats.add(int(message["chunk_index"]))
                    continue
                if message_type != "frame_batch_header":
                    raise RuntimeError(f"unexpected realtime message: {message_type!r}")
                payload = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                if not isinstance(payload, bytes):
                    raise RuntimeError("raw frame payload must be bytes")
                count = int(message["num_frames"])
                if len(payload) != count * frame_bytes:
                    raise RuntimeError("raw frame payload size mismatch")
                keep = min(count, VIDEO_FRAMES - persisted_frames)
                if keep:
                    assert encoder.stdin is not None
                    encoder.stdin.write(payload[: keep * frame_bytes])
                    await encoder.stdin.drain()
                    persisted_frames += keep
                if message.get("is_final_frame_batch", True):
                    chunks_with_frames.add(int(message["chunk_index"]))

        if persisted_frames != VIDEO_FRAMES:
            raise RuntimeError(
                f"received {persisted_frames} frames, expected {VIDEO_FRAMES}"
            )
        assert encoder.stdin is not None
        encoder.stdin.close()
        await encoder.stdin.wait_closed()
        _, stderr = await encoder.communicate()
        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
        partial.replace(output)
        return {
            "frames": persisted_frames,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_sec": VIDEO_FRAMES / FPS,
            "latency_sec": time.perf_counter() - start,
            "bytes": output.stat().st_size,
        }
    finally:
        if encoder.returncode is None:
            encoder.kill()
            await encoder.wait()
        partial.unlink(missing_ok=True)
