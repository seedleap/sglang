# SPDX-License-Identifier: Apache-2.0
"""Run warmed MiniMax H3 causal T2VA/first-frame FL2VA latency probes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _json_request(url: str, *, method: str = "GET", body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def _submit_and_wait(base_url: str, payload: dict[str, Any]) -> tuple[str, float]:
    started = time.perf_counter()
    created = _json_request(f"{base_url}/v1/videos", method="POST", body=payload)
    video_id = created["id"]
    while True:
        status = _json_request(f"{base_url}/v1/videos/{video_id}")
        state = status["status"]
        if state == "completed":
            return video_id, time.perf_counter() - started
        if state == "failed":
            raise RuntimeError(f"video job {video_id} failed: {status}")
        time.sleep(0.2)


def _download(base_url: str, video_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(
            f"{base_url}/v1/videos/{video_id}/content", timeout=3600
        ) as response,
        path.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _payload(args, *, task: str, nfe: int, seed: int) -> dict[str, Any]:
    conditions = []
    prompt = args.prompt
    if task == "fl2va":
        if not args.first_frame_uri:
            raise ValueError("--first-frame-uri is required for FL2VA")
        conditions = [
            {
                "type": "image",
                "uri": args.first_frame_uri,
                "role": "keyframe",
                "frame_index": 0,
            }
        ]
        prompt = f"{prompt} Continue naturally from the supplied first frame."

    return {
        "model": args.model,
        "prompt": prompt,
        "seconds": args.seconds,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": args.short_edge,
            "aspect_ratio": args.aspect_ratio,
            "duration_seconds": args.seconds,
        },
        "num_outputs_per_prompt": 1,
        # H3 currently stores N sigma points and executes N-1 DiT forwards.
        "num_inference_steps": nfe + 1,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30010")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument(
        "--model-revision",
        default="bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
        help="Pinned checkpoint revision recorded with every result row",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=("t2va", "fl2va"), default=["t2va", "fl2va"]
    )
    parser.add_argument("--nfe", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--first-frame-uri")
    parser.add_argument(
        "--prompt",
        default="A quiet cinematic street scene with coherent natural motion and synchronized ambient audio.",
    )
    parser.add_argument("--topology", required=True, help="Label such as tp8-u1-b200")
    parser.add_argument(
        "--variant",
        default="causal-flex",
        help="Experiment stage recorded in every row, such as noncausal or causal-flex",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(nfe <= 0 for nfe in args.nfe):
        raise ValueError("--nfe values must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be non-negative and --repeats must be positive")
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    base_url = args.base_url.rstrip("/")
    requested_frames = max(1, int(round(args.seconds * 24)))
    aligned_frames = requested_frames + (5 - requested_frames) % 17
    playback_seconds = aligned_frames / 24
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output:
        for task in args.tasks:
            for nfe in args.nfe:
                for warmup_index in range(args.warmup):
                    _submit_and_wait(
                        base_url,
                        _payload(
                            args,
                            task=task,
                            nfe=nfe,
                            seed=1000 + warmup_index,
                        ),
                    )
                for repeat in range(args.repeats):
                    seed = 2000 + repeat
                    payload = _payload(args, task=task, nfe=nfe, seed=seed)
                    video_id, latency_s = _submit_and_wait(base_url, payload)
                    record = {
                        "topology": args.topology,
                        "variant": args.variant,
                        "model_revision": args.model_revision,
                        "task": task,
                        "nfe": nfe,
                        "api_num_inference_steps": nfe + 1,
                        "repeat": repeat,
                        "seed": seed,
                        "latency_s": latency_s,
                        "playback_s": playback_seconds,
                        "aligned_frames": aligned_frames,
                        "rtf": latency_s / playback_seconds,
                        "realtime": latency_s <= playback_seconds,
                        "video_id": video_id,
                        "payload": payload,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps(record, ensure_ascii=False))
                    if args.video_dir is not None:
                        _download(
                            base_url,
                            video_id,
                            args.video_dir
                            / f"{args.topology}-{task}-nfe{nfe}-r{repeat}.mp4",
                        )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
