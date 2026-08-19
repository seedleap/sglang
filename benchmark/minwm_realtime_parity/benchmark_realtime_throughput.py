#!/usr/bin/env python3
"""Measure one persistent MinWM realtime session with LingBot's timing window.

Run this client against two separately launched servers to compare execution
profiles.  Keep the checkpoint, hardware, request, and action contract fixed;
only the server implementation/profile may change.  The client can retain one
measured RGB frame for numerical-parity diagnostics without keeping the full
multi-gigabyte stream in host memory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import msgspec.msgpack
from common import (
    action_weights,
    is_realtime_trace_event,
    load_cases,
    materialize_first_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ws-url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--model", default="minwm")
    parser.add_argument("--case", default="00_forward_pottery")
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--warmup-chunks", type=int, default=20)
    parser.add_argument("--measured-chunks", type=int, default=200)
    parser.add_argument("--kv-cache-num-frames", type=int)
    parser.add_argument("--sink-size", type=int)
    parser.add_argument("--save-first-measured-frame", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


_CHUNK_TELEMETRY_TIMING_ALIASES = {
    "output_pace_ms": "pace_wait_ms",
    "transport_encode_ms": "raw_payload_build_ms",
    "transport_write_ms": "ws_write_ms",
}


def record_server_chunk_timing(
    stats_by_chunk: dict[int, dict[str, Any]], message: dict[str, Any]
) -> dict[str, Any]:
    message_type = message.get("type")
    if message_type not in {"chunk_stats", "chunk_telemetry"}:
        raise ValueError(f"not a server chunk timing message: {message_type!r}")
    normalized = dict(message)
    if message_type == "chunk_telemetry":
        for source, destination in _CHUNK_TELEMETRY_TIMING_ALIASES.items():
            if source in normalized:
                normalized[destination] = normalized[source]
    chunk_index = int(normalized["chunk_index"])
    if chunk_index in stats_by_chunk:
        raise AssertionError(
            f"chunk {chunk_index} received duplicate server timing messages"
        )
    stats_by_chunk[chunk_index] = normalized
    return normalized


def validate_contract(manifest: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    contract = manifest["contract"]
    if contract.get("action_type") != "primitive_token_residual":
        raise ValueError("throughput comparison requires primitive_token_residual")
    if int(contract["latent_frames_per_chunk"]) != 4:
        raise ValueError("MinWM throughput contract requires four latent frames/chunk")
    if int(contract["generated_pixel_frames"]) % int(
        contract["generated_latent_frames"]
    ):
        raise ValueError("pixel/latent frame ratio must be integral")
    if args.warmup_chunks < 1 or args.measured_chunks < 1:
        raise ValueError("warmup-chunks and measured-chunks must be positive")
    if args.kv_cache_num_frames is not None and args.kv_cache_num_frames < 1:
        raise ValueError("kv-cache-num-frames must be positive")
    if args.sink_size is not None and args.sink_size < 0:
        raise ValueError("sink-size must be non-negative")
    if (
        args.sink_size is not None
        and args.kv_cache_num_frames is not None
        and args.sink_size >= args.kv_cache_num_frames
    ):
        raise ValueError("sink-size must be smaller than kv-cache-num-frames")
    cases = {case["id"]: case for case in manifest["cases"]}
    if args.case not in cases:
        raise ValueError(f"unknown case {args.case!r}; choose from {sorted(cases)}")
    return contract, cases[args.case]


def validate_frame_batch(
    header: dict[str, Any],
    payload: bytes,
    *,
    chunk_index: int,
    expected_width: int,
    expected_height: int,
    expected_channels: int = 3,
) -> tuple[int, int, int]:
    batch_frames = int(header["num_frames"])
    expected_bytes = (
        batch_frames
        * int(header["height"])
        * int(header["width"])
        * int(header["channels"])
    )
    batch_index = int(header.get("frame_batch_index", 0))
    num_batches = int(header.get("num_frame_batches", 1))
    expected_final = batch_index == num_batches - 1
    is_final = bool(header.get("is_final_frame_batch", expected_final))
    checks = {
        "chunk_index": int(header["chunk_index"]) == chunk_index,
        "positive_num_frames": batch_frames > 0,
        "content_type": header["content_type"] == "application/x-raw-rgb",
        "width": int(header["width"]) == expected_width,
        "height": int(header["height"]) == expected_height,
        "channels": int(header["channels"]) == expected_channels,
        "payload_bytes": len(payload) == expected_bytes,
        "bytes_per_frame": int(header["bytes_per_frame"])
        == expected_bytes // batch_frames,
        "raw_size": int(header.get("raw_size", len(payload))) == len(payload),
        "total_size": int(header.get("total_size", len(payload))) == len(payload),
        # Streamed remote decoding does not know the total until the final batch.
        # It uses zero as the documented unknown-count sentinel.
        "batch_index": batch_index >= 0
        and (num_batches == 0 or batch_index < num_batches),
        "num_frame_batches": num_batches >= 0,
        "is_final_frame_batch": (
            not is_final if num_batches == 0 else is_final == expected_final
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            f"chunk {chunk_index} raw frame batch contract failed: {failed}; "
            f"header={header}"
        )
    return batch_index, num_batches, batch_frames


def record_frame_batch(
    state: dict[str, Any],
    *,
    chunk_index: int,
    batch_index: int,
    num_batches: int,
    batch_frames: int,
    expected_frames: int,
) -> bool:
    if state["complete"]:
        raise AssertionError(f"chunk {chunk_index} received a batch after completion")
    known_num_batches = state["num_batches"]
    if num_batches:
        if known_num_batches not in (None, num_batches):
            raise AssertionError(
                f"chunk {chunk_index} changed num_frame_batches from "
                f"{known_num_batches} to {num_batches}"
            )
        state["num_batches"] = num_batches
    elif known_num_batches is not None:
        raise AssertionError(
            f"chunk {chunk_index} changed num_frame_batches from "
            f"{known_num_batches} to unknown"
        )
    if batch_index in state["seen"]:
        raise AssertionError(f"chunk {chunk_index} repeated frame batch {batch_index}")
    state["seen"].add(batch_index)
    state["frames"] += batch_frames
    if state["frames"] > expected_frames:
        raise AssertionError(
            f"chunk {chunk_index} produced more than {expected_frames} frames"
        )
    if not num_batches:
        state["complete"] = state["frames"] == expected_frames
        return state["complete"]
    if batch_index != num_batches - 1:
        return False
    expected_batch_indices = set(range(num_batches))
    if state["seen"] != expected_batch_indices:
        raise AssertionError(
            f"chunk {chunk_index} frame batches are incomplete: "
            f"seen={sorted(state['seen'])} expected="
            f"{sorted(expected_batch_indices)}"
        )
    if state["frames"] != expected_frames:
        raise AssertionError(
            f"chunk {chunk_index} produced {state['frames']} frames, "
            f"expected {expected_frames}"
        )
    state["complete"] = True
    return True


async def receive_run(args: argparse.Namespace, contract: dict, case: dict) -> dict:
    import websockets

    total_chunks = args.warmup_chunks + args.measured_chunks
    latent_frames_per_chunk = int(contract["latent_frames_per_chunk"])
    expected_width = int(contract["width"])
    expected_height = int(contract["height"])
    expected_channels = 3
    pixel_frames_per_latent = int(contract["generated_pixel_frames"]) // int(
        contract["generated_latent_frames"]
    )
    first_frame = materialize_first_frame(case, Path(args.output).parent / "inputs")
    first_frame_sha256 = hashlib.sha256(first_frame.read_bytes()).hexdigest()
    if contract.get("action_output_format") == "primitive_float":
        action_condition = {
            "action_weights": [action_weights(case)]
            * total_chunks
            * latent_frames_per_chunk
            * pixel_frames_per_latent
        }
    else:
        action_condition = {
            "action_labels": [int(case["action_label"])]
            * total_chunks
            * latent_frames_per_chunk
        }
    canonical_condition = json.dumps(
        action_condition, separators=(",", ":"), sort_keys=True
    ).encode()
    request = {
        "type": "init",
        "model": args.model,
        "prompt": case["prompt"],
        "first_frame": first_frame.read_bytes(),
        "size": f"{contract['width']}x{contract['height']}",
        "fps": int(contract["fps"]),
        "seed": int(contract["seed"]),
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": total_chunks,
        "realtime_output_format": "raw",
        "condition_inputs": action_condition,
    }
    if args.kv_cache_num_frames is not None:
        request["realtime_causal_kv_cache_num_frames"] = args.kv_cache_num_frames
    if args.sink_size is not None:
        request["realtime_causal_sink_size"] = args.sink_size
    stats_by_chunk: dict[int, dict[str, Any]] = {}
    payload_complete_ns: dict[int, int] = {}
    frame_batches_by_chunk: dict[int, dict[str, Any]] = {}
    measured_payload_sha256 = hashlib.sha256()
    measured_frame_sha256: dict[str, str] = {}
    measured_frame_samples: dict[str, str] = {}
    measured_payload_samples: dict[str, str] = {}
    first_measured_frame_saved = False
    init_started_ns = time.perf_counter_ns()
    async with websockets.connect(
        args.ws_url, max_size=None, ping_interval=None, open_timeout=args.timeout
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        init_completed_ns = time.perf_counter_ns()
        while (
            len(payload_complete_ns) < total_chunks
            or len(stats_by_chunk) < total_chunks
        ):
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            if not isinstance(packed, bytes):
                raise TypeError(
                    f"expected binary MessagePack, got {type(packed).__name__}"
                )
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if is_realtime_trace_event(header):
                continue
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown realtime error"))
            if message_type in {"chunk_stats", "chunk_telemetry"}:
                record_server_chunk_timing(stats_by_chunk, header)
                continue
            if message_type == "frame_batch":
                payload = header.pop("payload")
            elif message_type == "frame_batch_header":
                payload = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
                if not isinstance(payload, bytes):
                    raise TypeError("raw frame payload must be bytes")
            else:
                raise ValueError(f"unexpected realtime message: {header}")
            chunk_index = int(header["chunk_index"])
            expected_frames = (
                int(contract["reference_pixel_frames"])
                + pixel_frames_per_latent * latent_frames_per_chunk
                if chunk_index == 0
                else pixel_frames_per_latent * latent_frames_per_chunk
            )
            batch_index, num_batches, batch_frames = validate_frame_batch(
                header,
                payload,
                chunk_index=chunk_index,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_channels=expected_channels,
            )
            state = frame_batches_by_chunk.setdefault(
                chunk_index,
                {
                    "num_batches": None,
                    "seen": set(),
                    "frames": 0,
                    "complete": False,
                },
            )
            if chunk_index >= args.warmup_chunks:
                measured_payload_sha256.update(payload)
                bytes_per_frame = int(header["bytes_per_frame"])
                first_frame_index = int(state["frames"])
                for frame_offset in range(batch_frames):
                    start = frame_offset * bytes_per_frame
                    frame = payload[start : start + bytes_per_frame]
                    frame_key = f"{chunk_index}:{first_frame_index + frame_offset}"
                    measured_frame_sha256[frame_key] = hashlib.sha256(frame).hexdigest()
                    frame_sample_stride = max(1, len(frame) // 1024)
                    measured_frame_samples[frame_key] = base64.b64encode(
                        frame[::frame_sample_stride][:1024]
                    ).decode("ascii")
                    if (
                        args.save_first_measured_frame
                        and not first_measured_frame_saved
                    ):
                        Path(args.output).with_name(
                            "first-measured-frame.rgb"
                        ).write_bytes(frame)
                        first_measured_frame_saved = True
                sample_stride = max(1, len(payload) // 4096)
                sample = payload[::sample_stride][:4096]
                measured_payload_samples[f"{chunk_index}:{batch_index}"] = (
                    base64.b64encode(sample).decode("ascii")
                )
            if record_frame_batch(
                state,
                chunk_index=chunk_index,
                batch_index=batch_index,
                num_batches=num_batches,
                batch_frames=batch_frames,
                expected_frames=expected_frames,
            ):
                payload_complete_ns[chunk_index] = time.perf_counter_ns()

    expected_indices = list(range(total_chunks))
    if sorted(stats_by_chunk) != expected_indices:
        raise AssertionError("server chunk timing indices are not contiguous")
    if sorted(payload_complete_ns) != expected_indices:
        raise AssertionError("frame payload indices are not contiguous")

    measured_indices = list(range(args.warmup_chunks, total_chunks))
    measured_stats = [
        stats_by_chunk[index] for index in measured_indices if index in stats_by_chunk
    ]
    measured_frames = sum(
        int(frame_batches_by_chunk[index]["frames"]) for index in measured_indices
    )
    expected_measured_frames = (
        args.measured_chunks * pixel_frames_per_latent * latent_frames_per_chunk
    )
    if measured_frames != expected_measured_frames:
        raise AssertionError(
            f"measured {measured_frames} frames, expected {expected_measured_frames}"
        )

    timing_fields = (
        "request_prepare_ms",
        "scheduler_forward_ms",
        "video_serialize_ms",
        "raw_payload_build_ms",
        "pace_wait_ms",
        "ws_write_ms",
        "chunk_total_ms",
        "model_vae_encode_ms",
        "model_denoise_ms",
        "model_vae_decode_ms",
        "model_post_decode_ms",
        "raw_frame_async_enqueue_ms",
    )
    server = {}
    for field in timing_fields:
        values = [float(stat[field]) for stat in measured_stats if field in stat]
        server[field] = {
            "missing_count": args.measured_chunks - len(values),
            "sample_count": len(values),
        }
        if values:
            server[field].update(latency_summary(values))
            if field in {"scheduler_forward_ms", "chunk_total_ms"}:
                server[field.replace("_ms", "_fps_ratio_of_sums")] = measured_frames / (
                    sum(values) / 1000.0
                )

    previous_ns = payload_complete_ns[args.warmup_chunks - 1]
    interarrival_ms = []
    for index in measured_indices:
        completion_ns = payload_complete_ns[index]
        interarrival_ms.append((completion_ns - previous_ns) / 1e6)
        previous_ns = completion_ns
    client_window_s = sum(interarrival_ms) / 1000.0
    client_fps = measured_frames / client_window_s
    interarrival_summary = latency_summary(interarrival_ms)
    target_fps = 24.0
    target_ms_per_chunk = (
        1000.0 * (pixel_frames_per_latent * latent_frames_per_chunk) / target_fps
    )
    return {
        "schema_version": "minwm-realtime-throughput/v1",
        "profile_name": args.profile_name,
        "comparison_contract": {
            "case": case["id"],
            "action_type": contract["action_type"],
            "action_label": int(case["action_label"]),
            "seed": int(contract["seed"]),
            "size": f"{contract['width']}x{contract['height']}",
            "steps": 4,
            "guidance_scale": 0.0,
            "latent_frames_per_chunk": latent_frames_per_chunk,
            "generated_pixel_frames_per_steady_chunk": pixel_frames_per_latent
            * latent_frames_per_chunk,
            "kv_cache_num_frames": args.kv_cache_num_frames,
            "sink_size": args.sink_size,
            "required_fixed_between_profiles": [
                "checkpoint bytes",
                "GPU model and count",
                "software image",
                "attention backend",
                "request payload",
            ],
        },
        "warmup_chunks": args.warmup_chunks,
        "measured_chunks": args.measured_chunks,
        "received_payload_chunks": len(payload_complete_ns),
        "received_server_timing_chunks": len(stats_by_chunk),
        "measured_frames": measured_frames,
        "received_frame_contract": {
            "bytes_per_frame": expected_width * expected_height * expected_channels,
            "channels": expected_channels,
            "content_type": "application/x-raw-rgb",
            "height": expected_height,
            "width": expected_width,
        },
        "request_evidence": {
            "case_id": case["id"],
            "condition_inputs_sha256": hashlib.sha256(canonical_condition).hexdigest(),
            "first_frame_sha256": first_frame_sha256,
            "first_frame_uri": case["first_frame"],
            "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        },
        "target_fps": target_fps,
        "target_ms_per_chunk": target_ms_per_chunk,
        "target_24fps_pass": client_fps >= target_fps,
        "target_chunk_p50_pass": interarrival_summary["p50"] <= target_ms_per_chunk,
        "target_chunk_p95_pass": interarrival_summary["p95"] <= target_ms_per_chunk,
        "measured_payload_sha256": measured_payload_sha256.hexdigest(),
        "measured_frame_sha256": measured_frame_sha256,
        "measured_frame_samples_base64": measured_frame_samples,
        "measured_payload_samples_base64": measured_payload_samples,
        "server": server,
        "client": {
            "init_send_start_to_first_payload_complete_ms": (
                payload_complete_ns[0] - init_started_ns
            )
            / 1e6,
            "init_send_complete_to_first_payload_complete_ms": (
                payload_complete_ns[0] - init_completed_ns
            )
            / 1e6,
            "steady_payload_interarrival_ms": interarrival_summary,
            "steady_received_fps_ratio_of_sums": client_fps,
            "steady_window_seconds": client_window_s,
        },
    }


async def async_main(args: argparse.Namespace) -> None:
    manifest = load_cases(args.cases)
    contract, case = validate_contract(manifest, args)
    result = await receive_run(args, contract, case)
    write_json(args.output, result)
    print(json.dumps(result["server"], indent=2, sort_keys=True))
    print(json.dumps(result["client"], indent=2, sort_keys=True))


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
