#!/usr/bin/env python3
"""Warm a realtime video shape before a worker is advertised as ready."""

from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
from pathlib import Path
import re
import time
from uuid import uuid4

import msgspec.msgpack
from PIL import Image
import websockets
from websockets.exceptions import ConnectionClosedOK


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value)
    if match is None:
        raise ValueError(f"invalid warmup size {value!r}; expected WIDTHxHEIGHT")
    return int(match.group(1)), int(match.group(2))


def build_warmup_request(
    *,
    model: str,
    first_frame: bytes,
    trace_id: str,
    size: str = "1280x720",
    fps: int = 24,
    guidance_scale: float = 1.0,
    sink_size: int = 3,
    kv_cache_num_frames: int = 12,
) -> dict:
    parse_size(size)
    if fps <= 0:
        raise ValueError("warmup fps must be positive")
    if sink_size < 0 or kv_cache_num_frames <= sink_size:
        raise ValueError("warmup KV window must be positive and larger than sink size")
    return {
        "type": "init",
        "generation_mode": "i2v",
        "model": model,
        "prompt": (
            "A stable forward camera view through a bright mountain valley, "
            "with consistent geometry and gentle motion."
        ),
        "size": size,
        "fps": fps,
        "first_frame": first_frame,
        "seed": 42,
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": guidance_scale,
        "max_chunks": 1,
        "realtime_output_format": "webp",
        "realtime_preview_max_width": 320,
        "realtime_output_pacing": False,
        "output_compression": 45,
        "realtime_causal_sink_size": sink_size,
        "realtime_causal_kv_cache_num_frames": kv_cache_num_frames,
        "trace_id": trace_id,
    }


def create_reference_frame(size: str = "1280x720") -> bytes:
    image = Image.new("RGB", parse_size(size), (82, 126, 91))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def _is_generation_complete_close(exc: ConnectionClosedOK) -> bool:
    close = exc.rcvd or exc.sent
    return (
        close is not None
        and int(close.code) == 1000
        and close.reason == "generation complete"
    )


async def wait_for_first_frame(
    websocket, *, timeout_s: float, allow_empty_complete: bool = False
) -> dict:
    deadline = time.monotonic() + timeout_s
    pending_header: dict | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("realtime warmup timed out before the first frame")
        try:
            packed = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except ConnectionClosedOK as exc:
            if allow_empty_complete and _is_generation_complete_close(exc):
                return {"chunk_index": 0, "empty_complete": True}
            raise
        if pending_header is not None:
            if not isinstance(packed, bytes) or not packed:
                raise RuntimeError("realtime warmup received an empty frame payload")
            return pending_header
        if not isinstance(packed, bytes):
            continue
        message = msgspec.msgpack.decode(packed)
        message_type = message.get("type")
        if message_type == "error":
            raise RuntimeError(message.get("content") or "realtime warmup failed")
        if message_type == "frame_batch" and int(message.get("num_frames") or 0) > 0:
            if message.get("payload"):
                return message
        if (
            message_type == "frame_batch_header"
            and int(message.get("num_frames") or 0) > 0
        ):
            pending_header = message


async def warmup(
    *,
    url: str,
    model: str,
    timeout_s: float,
    size: str = "1280x720",
    fps: int = 24,
    guidance_scale: float = 1.0,
    sink_size: int = 3,
    kv_cache_num_frames: int = 12,
    allow_empty_complete: bool = False,
) -> dict:
    request = build_warmup_request(
        model=model,
        first_frame=create_reference_frame(size),
        trace_id=f"startup-warmup-{uuid4().hex}",
        size=size,
        fps=fps,
        guidance_scale=guidance_scale,
        sink_size=sink_size,
        kv_cache_num_frames=kv_cache_num_frames,
    )
    async with websockets.connect(
        url,
        max_size=None,
        open_timeout=min(timeout_s, 30.0),
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        return await wait_for_first_frame(
            websocket,
            timeout_s=timeout_s,
            allow_empty_complete=allow_empty_complete,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:30000/v1/realtime_video/generate",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--sink-size", type=int, default=3)
    parser.add_argument("--kv-cache-num-frames", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--allow-empty-complete",
        action="store_true",
        help="Accept a normal generation-complete close from a remote-VAE denoiser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ready_file.unlink(missing_ok=True)
    started_at = time.monotonic()
    result = asyncio.run(
        warmup(
            url=args.url,
            model=args.model,
            timeout_s=args.timeout_s,
            size=args.size,
            fps=args.fps,
            guidance_scale=args.guidance_scale,
            sink_size=args.sink_size,
            kv_cache_num_frames=args.kv_cache_num_frames,
            allow_empty_complete=args.allow_empty_complete,
        )
    )
    elapsed_s = time.monotonic() - started_at
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix(".tmp")
    temporary.write_text(
        f"chunk={int(result.get('chunk_index') or 0)} elapsed_s={elapsed_s:.3f}\n",
        encoding="utf-8",
    )
    temporary.replace(args.ready_file)
    print(
        "realtime startup warmup completed: "
        f"size={args.size} chunk={int(result.get('chunk_index') or 0)} "
        f"elapsed_s={elapsed_s:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
