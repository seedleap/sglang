#!/usr/bin/env python3
"""Reproduce the 2026-08-01 RTX 6000 realtime request contract.

Current realtime servers publish per-chunk timing in structured server trace
logs instead of sending the legacy ``chunk_stats`` websocket message.  This
client keeps the original payload contract and joins those trace events back to
each request using its exact wall-clock interval and trace id.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import msgspec.msgpack
import websockets

from tianpeng_alignment import DEFAULT_ALIGNMENT_URL, load_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--sglang-git-ref", required=True)
    parser.add_argument(
        "--ws-url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--measured-chunks", type=int, default=69)
    parser.add_argument("--steady-start-chunk", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def request_for_size(base_request: dict, size: str, chunks: int) -> dict:
    request = copy.deepcopy(base_request)
    request["generation_mode"] = "t2v"
    request["size"] = size
    request["max_chunks"] = chunks
    request["num_frames"] = 1 + 16 * (chunks - 1)
    request["realtime_output_format"] = "raw"
    conditions = request["condition_inputs"]
    conditions["action_weights"] = conditions["action_weights"][
        : request["num_frames"] - 1
    ]
    conditions["minwm_chunk_seeds"] = conditions["minwm_chunk_seeds"][:chunks]
    conditions["minwm_prompt_schedule"] = [
        item
        for item in conditions["minwm_prompt_schedule"]
        if int(item["target_chunk"]) < chunks
    ]
    return request


async def stream_request(args: argparse.Namespace, request: dict) -> dict:
    completed: dict[int, int] = {}
    frame_count = 0
    payload_bytes = 0
    payload_sha256 = hashlib.sha256()
    started_ns = time.perf_counter_ns()
    started_epoch_ms = time.time_ns() / 1e6
    first_payload_ns = None
    async with websockets.connect(
        args.ws_url, max_size=None, ping_interval=None, open_timeout=args.timeout
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        while len(completed) < int(request["max_chunks"]):
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown error"))
            if message_type == "trace_event" or message_type == "chunk_stats":
                continue
            if message_type == "frame_batch":
                payload = header.get("payload", b"")
            elif message_type == "frame_batch_header":
                payload = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            else:
                raise RuntimeError(f"unexpected message: {header}")
            now_ns = time.perf_counter_ns()
            if first_payload_ns is None:
                first_payload_ns = now_ns
            frame_count += int(header["num_frames"])
            payload_bytes += len(payload)
            payload_sha256.update(payload)
            if header.get("is_final_frame_batch", True):
                completed[int(header["chunk_index"])] = now_ns
    ended_epoch_ms = time.time_ns() / 1e6
    ended_ns = time.perf_counter_ns()
    expected_indices = list(range(int(request["max_chunks"])))
    if sorted(completed) != expected_indices:
        raise AssertionError("frame payload indices are not contiguous")
    return {
        "frame_count": frame_count,
        "payload_bytes": payload_bytes,
        "payload_sha256": payload_sha256.hexdigest(),
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "started_epoch_ms": started_epoch_ms,
        "ended_epoch_ms": ended_epoch_ms,
        "first_payload_ns": first_payload_ns,
        "payload_complete_ns": completed,
    }


def trace_events(
    path: Path, started_epoch_ms: float, ended_epoch_ms: float
) -> list[dict]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        marker = "realtime_trace {"
        if marker not in line:
            continue
        try:
            event = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        epoch_ms = float(event.get("server_epoch_ms", -1))
        if started_epoch_ms - 1000 <= epoch_ms <= ended_epoch_ms + 1000:
            events.append(event)
    return events


def select_trace(events: list[dict], chunks: int) -> tuple[str, list[dict]]:
    by_trace: dict[str, list[dict]] = {}
    for event in events:
        trace_id = event.get("trace_id")
        if trace_id:
            by_trace.setdefault(str(trace_id), []).append(event)
    candidates = []
    for trace_id, rows in by_trace.items():
        indices = {
            int(row["chunk_index"])
            for row in rows
            if row.get("event") == "server.chunk_complete" and "chunk_index" in row
        }
        if indices == set(range(chunks)):
            candidates.append((trace_id, rows))
    if len(candidates) != 1:
        raise AssertionError(
            f"expected one complete trace, found {[item[0] for item in candidates]}"
        )
    return candidates[0]


def ratio_of_sums(rows: list[dict], field: str) -> float:
    frames = sum(int(row["num_frames"]) for row in rows)
    milliseconds = sum(float(row[field]) for row in rows)
    return frames / (milliseconds / 1000.0)


def stage_summary(events: list[dict], stage: str, steady_start: int) -> dict | None:
    rows = [
        event
        for event in events
        if event.get("event") == "server.pipeline_stage_complete"
        and event.get("stage") == stage
        and "source" not in event
        and int(event.get("chunk_index", -1)) >= steady_start
    ]
    if not rows:
        return None
    values = [float(row["duration_ms"]) for row in rows]
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
    }


def summarize(args: argparse.Namespace, size: str, result: dict) -> dict:
    events = trace_events(
        Path(args.server_log), result["started_epoch_ms"], result["ended_epoch_ms"]
    )
    trace_id, events = select_trace(events, args.measured_chunks)
    chunks = sorted(
        (event for event in events if event.get("event") == "server.chunk_complete"),
        key=lambda event: int(event["chunk_index"]),
    )
    steady = [
        event
        for event in chunks
        if int(event["chunk_index"]) >= args.steady_start_chunk
    ]
    completion_ns = result["payload_complete_ns"]
    client_started_ns = completion_ns[args.steady_start_chunk - 1]
    client_ended_ns = completion_ns[args.measured_chunks - 1]
    steady_frames = sum(int(event["num_frames"]) for event in steady)
    client_window_s = (client_ended_ns - client_started_ns) / 1e9
    chunk_totals = [float(event["chunk_total_ms"]) for event in steady]
    return {
        "size": size,
        "frames": result["frame_count"],
        "chunks": len(chunks),
        "wall_seconds": (result["ended_ns"] - result["started_ns"]) / 1e9,
        "client_fps": result["frame_count"]
        / ((result["ended_ns"] - result["started_ns"]) / 1e9),
        "steady_client_fps": steady_frames / client_window_s,
        "steady_source_fps": ratio_of_sums(steady, "chunk_total_ms"),
        "steady_scheduler_fps": ratio_of_sums(steady, "scheduler_forward_ms"),
        "steady_chunk_total_ms_p50": statistics.median(chunk_totals),
        "steady_chunk_total_ms_min": min(chunk_totals),
        "steady_chunk_total_ms_max": max(chunk_totals),
        "ttff_ms": (result["first_payload_ns"] - result["started_ns"]) / 1e6,
        "payload_bytes": result["payload_bytes"],
        "payload_sha256": result["payload_sha256"],
        "trace_id": trace_id,
        "steady_start_chunk": args.steady_start_chunk,
        "dit_wall": stage_summary(
            events, "MinWMCausalDMDDenoisingStage", args.steady_start_chunk
        ),
        "vae_wall": stage_summary(
            events, "MinWMCausalVaeDecodingStage", args.steady_start_chunk
        ),
    }


async def main() -> None:
    args = parse_args()
    contract = load_contract(DEFAULT_ALIGNMENT_URL)
    base_request = contract["request"]
    summaries = []
    warmups = []
    for size in ("832x480", "1248x704"):
        warm = await stream_request(
            args, request_for_size(base_request, size, args.warmup_chunks)
        )
        warmups.append(
            {
                "size": size,
                "frames": warm["frame_count"],
                "wall_seconds": (warm["ended_ns"] - warm["started_ns"]) / 1e9,
            }
        )
        measured = await stream_request(
            args, request_for_size(base_request, size, args.measured_chunks)
        )
        summary = summarize(args, size, measured)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    output = {
        "schema_version": "minwm-rtx6000-20260801-contract/v2",
        "profile_name": args.profile_name,
        "contract": {
            "checkpoint_sha256": contract["expected"]["checkpoint_sha256"],
            "local_attn_size": 32,
            "sink_size": 8,
            "rope_position_mode": "block_relative",
            "rope_max_frame_gap": 12,
            "dmd_steps": 4,
            "sglang_git_ref": args.sglang_git_ref,
        },
        "warmups": warmups,
        "results": summaries,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
