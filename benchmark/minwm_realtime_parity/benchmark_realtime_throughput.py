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
import os
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
from measurement import (
    MeasurementValidationError,
    available,
    build_measurement,
    latency_summary,
    stage_trace_values,
    unavailable,
)

TraceSelector = tuple[str, str, str]
_TRACE_LOG_MARKER = "realtime_trace "

CHUNK_STATS_TRACE_FIELDS = {
    "chunk_index": "chunk_index",
    "request_prepare_ms": "request_prepare_ms",
    "scheduler_forward_ms": "scheduler_forward_ms",
    "output_pace_ms": "pace_wait_ms",
    "header_write_ms": "header_write_ms",
    "raw_payload_build_ms": "raw_payload_build_ms",
    "raw_write_ms": "raw_write_ms",
    "ws_write_ms": "ws_write_ms",
    "chunk_total_ms": "chunk_total_ms",
    "num_batches": "num_batches",
    "num_frames": "num_frames",
    "raw_bytes": "raw_bytes",
    "ws_payload_bytes": "ws_payload_bytes",
    "content_type": "content_type",
}


def chunk_stats_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Restore the removed chunk_stats message from its authoritative log event."""
    if trace.get("event") != "server.chunk_complete":
        return None
    missing = [name for name in CHUNK_STATS_TRACE_FIELDS if name not in trace]
    if missing:
        raise MeasurementValidationError(
            f"server.chunk_complete trace missing required fields: {missing}"
        )
    stats = {
        target: trace[source] for source, target in CHUNK_STATS_TRACE_FIELDS.items()
    }
    stats["type"] = "chunk_stats"
    for name in ("trace_id", "session_id", "request_id", "event_id"):
        if name in trace:
            stats[name] = trace[name]
    return stats


async def send_realtime_heartbeats(
    websocket: Any, *, trace_id: str | None, interval_s: float
) -> None:
    event_id = 1
    while True:
        await asyncio.sleep(interval_s)
        await websocket.send(
            msgspec.msgpack.encode(
                {
                    "type": "event",
                    "kind": "heartbeat",
                    "payload": {},
                    "event_id": event_id,
                    "trace_id": trace_id,
                }
            )
        )
        event_id += 1


async def cancel_task(task: asyncio.Task[Any]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def realtime_heartbeat(
    websocket: Any, *, trace_id: str | None, interval_s: float
):
    task = asyncio.create_task(
        send_realtime_heartbeats(websocket, trace_id=trace_id, interval_s=interval_s)
    )
    try:
        yield
    finally:
        await cancel_task(task)


def required_stage_trace_chunks(mode: str) -> dict[TraceSelector, set[int]]:
    required = {
        ("server.model_denoise_complete", "source", "scheduler_result_metrics"): set(),
        ("server.vae_decode_complete", "source", "scheduler_result_metrics"): set(),
    }
    if mode == "profiler_on":
        required.update(
            {
                (
                    "server.model_denoise_complete",
                    "component",
                    "minwm_denoising",
                ): set(),
                ("server.vae_decode_complete", "component", "vae_decoder"): set(),
            }
        )
    return required


def record_required_stage_trace(
    required: dict[TraceSelector, set[int]], trace: dict[str, Any]
) -> None:
    for (event, selector, value), chunk_indices in required.items():
        if trace.get("event") != event or trace.get(selector) != value:
            continue
        try:
            chunk_indices.add(int(trace["chunk_index"]))
        except (KeyError, TypeError, ValueError):
            pass


def missing_required_stage_trace(
    required: dict[TraceSelector, set[int]], expected_indices: set[int]
) -> dict[str, dict[str, list[int]]]:
    return {
        "/".join(selector): {
            "missing": sorted(expected_indices - observed),
            "unexpected": sorted(observed - expected_indices),
        }
        for selector, observed in required.items()
        if not expected_indices.issubset(observed)
    }


def required_stage_trace_is_complete(
    required: dict[TraceSelector, set[int]], expected_indices: set[int]
) -> bool:
    return all(expected_indices.issubset(observed) for observed in required.values())


def incomplete_measurement_diagnostic(
    required: dict[TraceSelector, set[int]],
    expected_indices: set[int],
    stats_by_chunk: dict[int, Any],
    payload_complete_ns: dict[int, Any],
) -> dict[str, Any]:
    return {
        "missing_stats": sorted(expected_indices - set(stats_by_chunk)),
        "missing_payloads": sorted(expected_indices - set(payload_complete_ns)),
        "stage_trace": missing_required_stage_trace(required, expected_indices),
    }


def load_realtime_trace_log(path: str | Path, trace_id: str) -> list[dict[str, Any]]:
    """Load one request's structured trace events from the server log plane."""
    events = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        marker_index = line.find(_TRACE_LOG_MARKER)
        if marker_index < 0:
            continue
        encoded = line[marker_index + len(_TRACE_LOG_MARKER) :].strip()
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise MeasurementValidationError(
                f"malformed realtime trace JSON at {path}:{line_number}: {exc}"
            ) from exc
        if isinstance(payload, dict) and payload.get("trace_id") == trace_id:
            events.append(payload)
    return events


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
    parser.add_argument(
        "--measurement-mode",
        choices=("profiler_off", "profiler_on"),
        default="profiler_off",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--timestamp-utc")
    parser.add_argument("--sglang-commit")
    parser.add_argument("--minwm-commit")
    parser.add_argument("--container-image")
    parser.add_argument("--gpu-model")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--allocated-gpu-count", type=int)
    parser.add_argument("--sp-degree", type=int, default=1)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--fast-lane", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--checkpoint-id", default="step-3200")
    parser.add_argument("--checkpoint-step", type=int, default=3200)
    parser.add_argument("--precondition-warmup-chunks", type=int, default=0)
    parser.add_argument("--warmup-chunks", type=int, default=20)
    parser.add_argument("--measured-chunks", type=int, default=200)
    parser.add_argument("--kv-cache-num-frames", type=int)
    parser.add_argument(
        "--require-complete-stage-trace",
        action="store_true",
        help="require every DiT/VAE wall (and profiler-on CUDA) trace in --trace-log",
    )
    parser.add_argument(
        "--trace-log",
        help="server log containing structured 'realtime_trace' JSON events",
    )
    parser.add_argument("--heartbeat-interval-s", type=float, default=15.0)
    parser.add_argument("--save-first-measured-frame", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


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
    if args.measurement_mode == "profiler_on" and args.measured_chunks < 10:
        raise ValueError("profiler-on capture requires at least 10 measured chunks")
    if args.precondition_warmup_chunks < 0:
        raise ValueError("precondition-warmup-chunks must be non-negative")
    if args.sp_degree < 1 or (args.gpu_count is not None and args.gpu_count < 1):
        raise ValueError("sp-degree and gpu-count must be positive")
    if args.allocated_gpu_count is not None and args.allocated_gpu_count < 1:
        raise ValueError("allocated-gpu-count must be positive")
    active_gpu_count = args.gpu_count or args.sp_degree
    if (
        args.allocated_gpu_count is not None
        and args.allocated_gpu_count < active_gpu_count
    ):
        raise ValueError("allocated-gpu-count cannot be smaller than active gpu-count")
    if args.kv_cache_num_frames is not None and args.kv_cache_num_frames < 1:
        raise ValueError("kv-cache-num-frames must be positive")
    if args.heartbeat_interval_s <= 0:
        raise ValueError("heartbeat-interval-s must be positive")
    if args.require_complete_stage_trace and not args.trace_log:
        raise ValueError("--require-complete-stage-trace requires --trace-log")
    if args.require_complete_stage_trace and not args.run_id:
        raise ValueError("--require-complete-stage-trace requires --run-id")
    cases = {case["id"]: case for case in manifest["cases"]}
    if args.case not in cases:
        raise ValueError(f"unknown case {args.case!r}; choose from {sorted(cases)}")
    return contract, cases[args.case]


def validate_frame_batch(
    header: dict[str, Any], payload: bytes, *, chunk_index: int
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
    pixel_frames_per_latent = int(contract["generated_pixel_frames"]) // int(
        contract["generated_latent_frames"]
    )
    first_frame = materialize_first_frame(case, Path(args.output).parent / "inputs")
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
    request = {
        "type": "init",
        "model": args.model,
        "prompt": case["prompt"],
        "first_frame": first_frame.read_bytes(),
        "size": f"{contract['width']}x{contract['height']}",
        "fps": int(contract["fps"]),
        "seed": int(contract["seed"]),
        "trace_id": args.run_id,
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": total_chunks,
        "realtime_output_format": "raw",
        "condition_inputs": action_condition,
    }
    if args.kv_cache_num_frames is not None:
        request["realtime_causal_kv_cache_num_frames"] = args.kv_cache_num_frames
    stats_by_chunk: dict[int, dict[str, Any]] = {}
    payload_complete_ns: dict[int, int] = {}
    frame_batches_by_chunk: dict[int, dict[str, Any]] = {}
    required_trace_chunks = (
        required_stage_trace_chunks(args.measurement_mode)
        if args.require_complete_stage_trace
        else {}
    )
    expected_trace_indices = set(range(total_chunks))
    measured_payload_sha256 = hashlib.sha256()
    measured_frame_sha256: dict[str, str] = {}
    measured_frame_samples: dict[str, str] = {}
    measured_payload_samples: dict[str, str] = {}
    first_measured_frame_saved = False
    init_started_ns = time.perf_counter_ns()
    async with (
        websockets.connect(
            args.ws_url, max_size=None, ping_interval=None, open_timeout=args.timeout
        ) as websocket,
        realtime_heartbeat(
            websocket,
            trace_id=args.run_id,
            interval_s=args.heartbeat_interval_s,
        ),
    ):
        await websocket.send(msgspec.msgpack.encode(request))
        init_completed_ns = time.perf_counter_ns()
        while len(payload_complete_ns) < total_chunks:
            try:
                packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            except websockets.exceptions.ConnectionClosedOK as exc:
                diagnostic = incomplete_measurement_diagnostic(
                    {},
                    expected_trace_indices,
                    stats_by_chunk,
                    payload_complete_ns,
                )
                raise MeasurementValidationError(
                    "realtime stream closed normally before the measurement "
                    f"contract was complete: close={exc}; "
                    f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                ) from exc
            except TimeoutError as exc:
                diagnostic = incomplete_measurement_diagnostic(
                    {},
                    expected_trace_indices,
                    stats_by_chunk,
                    payload_complete_ns,
                )
                raise TimeoutError(
                    "timed out waiting for complete realtime measurement: "
                    f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
                ) from exc
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
            if message_type == "chunk_stats":
                stats_by_chunk[int(header["chunk_index"])] = header
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
                header, payload, chunk_index=chunk_index
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

    trace_events = (
        load_realtime_trace_log(args.trace_log, args.run_id)
        if args.trace_log and args.run_id
        else []
    )
    for trace in trace_events:
        record_required_stage_trace(required_trace_chunks, trace)
        chunk_stats = chunk_stats_from_trace(trace)
        if chunk_stats is not None:
            stats_by_chunk[int(chunk_stats["chunk_index"])] = chunk_stats
    if args.require_complete_stage_trace and not required_stage_trace_is_complete(
        required_trace_chunks, expected_trace_indices
    ):
        diagnostic = incomplete_measurement_diagnostic(
            required_trace_chunks,
            expected_trace_indices,
            stats_by_chunk,
            payload_complete_ns,
        )
        raise MeasurementValidationError(
            "server trace log did not satisfy the measurement contract: "
            f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
        )

    expected_indices = list(range(total_chunks))
    if sorted(stats_by_chunk) != expected_indices:
        raise AssertionError("chunk_stats indices are not contiguous")
    if sorted(payload_complete_ns) != expected_indices:
        raise AssertionError("frame payload indices are not contiguous")

    measured_indices = list(range(args.warmup_chunks, total_chunks))
    measured_stats = [stats_by_chunk[index] for index in measured_indices]
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
    )
    server = {}
    for field in timing_fields:
        values = [float(stat[field]) for stat in measured_stats if field in stat]
        if values:
            server[field] = latency_summary(values)
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
    comparison_contract = {
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
        "required_fixed_between_profiles": [
            "checkpoint bytes",
            "GPU model and count",
            "software image",
            "attention backend",
            "request payload",
        ],
    }
    client = {
        "init_send_start_to_first_payload_complete_ms": (
            payload_complete_ns[0] - init_started_ns
        )
        / 1e6,
        "init_send_complete_to_first_payload_complete_ms": (
            payload_complete_ns[0] - init_completed_ns
        )
        / 1e6,
        "steady_payload_interarrival_ms": latency_summary(interarrival_ms),
        "steady_received_fps_ratio_of_sums": measured_frames / client_window_s,
        "steady_window_seconds": client_window_s,
    }

    measured_index_set = set(measured_indices)
    dit_wall = stage_trace_values(
        trace_events,
        event="server.model_denoise_complete",
        field="duration_ms",
        measured_indices=measured_index_set,
        source="scheduler_result_metrics",
    )
    vae_wall = stage_trace_values(
        trace_events,
        event="server.vae_decode_complete",
        field="duration_ms",
        measured_indices=measured_index_set,
        source="scheduler_result_metrics",
    )
    dit_cuda = stage_trace_values(
        trace_events,
        event="server.model_denoise_complete",
        field="cuda_ms",
        measured_indices=measured_index_set,
        component="minwm_denoising",
    )
    vae_cuda = stage_trace_values(
        trace_events,
        event="server.vae_decode_complete",
        field="cuda_ms",
        measured_indices=measured_index_set,
        component="vae_decoder",
    )

    def complete_latency_metric(
        values: list[float], name: str, source: str
    ) -> dict[str, Any]:
        if len(values) != args.measured_chunks:
            return unavailable(
                "incomplete_trace_metric",
                f"{name}: expected {args.measured_chunks} chunks, observed {len(values)}",
            )
        return available(latency_summary(values), "ms_per_chunk", source)

    scheduler_values = [float(stat["scheduler_forward_ms"]) for stat in measured_stats]
    chunk_wall_values = [float(stat["chunk_total_ms"]) for stat in measured_stats]
    wall_metrics = {
        "client_fps": available(
            client["steady_received_fps_ratio_of_sums"],
            "frames_per_second",
            "client payload-complete monotonic window",
        ),
        "scheduler_fps": available(
            measured_frames / (sum(scheduler_values) / 1000.0),
            "frames_per_second",
            "ratio of measured frames to scheduler_forward_ms sum",
        ),
        "scheduler_chunk_wall_ms": complete_latency_metric(
            chunk_wall_values,
            "scheduler_chunk_wall_ms",
            "server chunk_total_ms",
        ),
        "dit_wall_ms": complete_latency_metric(
            dit_wall,
            "dit_wall_ms",
            "scheduler result MinWMCausalDMDDenoisingStage monotonic wall",
        ),
        "vae_wall_ms": complete_latency_metric(
            vae_wall,
            "vae_wall_ms",
            "scheduler result causal VAE decode stage monotonic wall",
        ),
    }
    cuda_metrics = {
        "dit_cuda_ms": complete_latency_metric(
            dit_cuda, "dit_cuda_ms", "worker structured trace log CUDA events"
        ),
        "vae_cuda_ms": complete_latency_metric(
            vae_cuda, "vae_cuda_ms", "worker structured trace log CUDA events"
        ),
    }
    timestamp_utc = args.timestamp_utc or datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or f"{args.profile_name}-{timestamp_utc}"
    sglang_commit = args.sglang_commit or _detect_sglang_commit()
    minwm_commit = _metadata_value(
        args.minwm_commit or os.environ.get("MINWM_GIT_REF"),
        "git_commit",
        "--minwm-commit or MINWM_GIT_REF",
    )
    container_image = _metadata_value(
        args.container_image or os.environ.get("MINWM_CONTAINER_IMAGE"),
        "image_reference",
        "--container-image or MINWM_CONTAINER_IMAGE",
    )
    gpu_model = _metadata_value(
        args.gpu_model or _detect_gpu_model(),
        "model_name",
        "--gpu-model or nvidia-smi",
    )
    result = build_measurement(
        mode=args.measurement_mode,
        run_id=run_id,
        profile_name=args.profile_name,
        timestamp_utc=timestamp_utc,
        sglang_commit=sglang_commit,
        minwm_commit=minwm_commit,
        container_image=container_image,
        gpu_model=gpu_model,
        gpu_count=args.gpu_count or args.sp_degree,
        allocated_gpu_count=args.allocated_gpu_count
        or args.gpu_count
        or args.sp_degree,
        sp_degree=args.sp_degree,
        checkpoint_id=args.checkpoint_id,
        checkpoint_step=args.checkpoint_step,
        width=int(contract["width"]),
        height=int(contract["height"]),
        warmup_chunks=args.warmup_chunks,
        measured_chunks=args.measured_chunks,
        precondition_warmup_chunks=args.precondition_warmup_chunks,
        precision=args.precision,
        fast_lane=args.fast_lane,
        comparison_contract=comparison_contract,
        profiler_off_metrics=wall_metrics,
        profiler_on_cuda_metrics=cuda_metrics,
        artifacts={"client_result": str(Path(args.output).resolve())},
    )
    # Compatibility fields remain during migration of the existing throughput
    # summary scripts. New consumers should read metrics.* and timing_domains.
    result.update(
        {
            "warmup_chunks": args.warmup_chunks,
            "measured_chunks": args.measured_chunks,
            "measured_frames": measured_frames,
            "measured_payload_sha256": measured_payload_sha256.hexdigest(),
            "measured_frame_sha256": measured_frame_sha256,
            "measured_frame_samples_base64": measured_frame_samples,
            "measured_payload_samples_base64": measured_payload_samples,
            "server": server,
            "client": client,
        }
    )
    return result


def _detect_sglang_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable:not-a-git-checkout"


def _detect_gpu_model() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    models = sorted({line.strip() for line in output.splitlines() if line.strip()})
    return ", ".join(models) if models else None


def _metadata_value(value: str | None, unit: str, source: str) -> dict[str, Any]:
    if value:
        return available(value, unit, source)
    return unavailable("not_recorded", f"No value from {source}")


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
