#!/usr/bin/env python3
"""Concurrent WebSocket load generator for MinWM realtime sync/async profiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import msgspec.msgpack
from summarize import latency_summary

MEDIA_PROFILE_MULTIPLIERS = {
    "native_v1": 1,
    "rife2x_v1": 2,
    "rife3x_v1": 3,
}
NORMAL_FINITE_CLOSE_REASONS = frozenset({"generation complete", "normal"})


def media_profile_multiplier(media_profile: str) -> int:
    try:
        return MEDIA_PROFILE_MULTIPLIERS[media_profile]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported realtime media profile {media_profile!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--profile", choices=("sync", "async"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--measured-chunks", type=int, default=6)
    parser.add_argument("--model", default="/work/model")
    parser.add_argument("--prompt", default="A cinematic forward-moving landscape")
    parser.add_argument(
        "--generation-mode",
        choices=("i2v", "t2v"),
        default="t2v",
    )
    parser.add_argument(
        "--first-frame",
        type=Path,
        help="Reference image used when --generation-mode=i2v",
    )
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--preview-max-width", type=int, default=560)
    parser.add_argument("--preview-quality", type=int, default=55)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--realtime-media-profile",
        choices=tuple(MEDIA_PROFILE_MULTIPLIERS),
        default="native_v1",
        help="Request native frames or an exact negotiated remote-VAE RIFE profile.",
    )
    parser.add_argument(
        "--expected-media-weights-sha256",
        help="Required exact 64-hex RIFE weights digest for interpolated runs.",
    )
    parser.add_argument("--sink", type=int, default=9)
    parser.add_argument("--window", type=int, default=18)
    parser.add_argument(
        "--completion-signal",
        choices=("final-frame", "chunk-stats"),
        default="final-frame",
        help=(
            "Use chunk-stats only for monolithic backends that send frames and "
            "stats in-order on the same WebSocket."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--trace-http-url",
        help="Gateway HTTP origin; defaults to the public WebSocket origin",
    )
    parser.add_argument("--trace-timeout-s", type=float, default=75.0)
    parser.add_argument("--skip-trace-query", action="store_true")
    parser.add_argument("--hardware-json", type=Path)
    return parser.parse_args()


def with_identity(url: str, *, user_id: str, trace_id: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(user_id=user_id, trace_id=trace_id)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def derive_trace_http_url(ws_url: str) -> str:
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise ValueError("ws_url must use ws:// or wss://")
    return urlunsplit((scheme, parts.netloc, "", "", ""))


def websocket_close_receipt(exc) -> dict[str, int | str]:
    """Return the peer close code across supported websockets versions."""

    received = getattr(exc, "rcvd", None)
    code = getattr(received, "code", None)
    reason = getattr(received, "reason", None)
    if code is None:
        code = getattr(exc, "code", None)
    if reason is None:
        reason = getattr(exc, "reason", "")
    return {"code": int(code or 0), "reason": str(reason or "")}


async def collect_trace_events(
    http_origin: str,
    trace_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float = 2.0,
    stable_polls: int = 2,
    expected_chunks: int | None = None,
    client=None,
) -> list[dict]:
    """Collect an incrementally published Trace without touching the video WS."""

    import httpx

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if stable_polls < 1:
        raise ValueError("stable_polls must be positive")
    if expected_chunks is not None and expected_chunks < 1:
        raise ValueError("expected_chunks must be positive")
    endpoint = f"{http_origin.rstrip('/')}/v1/realtime_video/traces/{trace_id}"
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=min(10.0, timeout_s))
    unchanged = 0
    by_cursor: dict[int, dict] = {}
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    endpoint,
                    # CloudWatch may publish an older trace_seq after a newer
                    # event. Re-read the full snapshot so late events cannot
                    # fall behind an incremental cursor permanently.
                    params={"after": 0, "limit": 500},
                )
                response.raise_for_status()
            except (TimeoutError, httpx.TimeoutException, httpx.TransportError):
                if time.monotonic() >= deadline:
                    raise
                if poll_interval_s > 0:
                    await asyncio.sleep(poll_interval_s)
                continue
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                if time.monotonic() >= deadline:
                    raise
                if poll_interval_s > 0:
                    await asyncio.sleep(poll_interval_s)
                continue
            payload = response.json()
            added = 0
            for raw_event in payload.get("events") or []:
                if not isinstance(raw_event, dict):
                    continue
                event_cursor = int(raw_event.get("trace_seq") or 0)
                if event_cursor <= 0 or event_cursor in by_cursor:
                    continue
                by_cursor[event_cursor] = dict(raw_event)
                added += 1
            unchanged = 0 if added else unchanged + 1
            completed_chunks = {
                int(event["chunk_index"])
                for event in by_cursor.values()
                if event.get("event") == "server.chunk_complete"
                and event.get("chunk_index") is not None
            }
            terminal_seen = any(
                event.get("event") == "gateway.session_closed"
                for event in by_cursor.values()
            )
            complete = expected_chunks is None or (
                len(completed_chunks) >= expected_chunks and terminal_seen
            )
            if by_cursor and complete and unchanged >= stable_polls:
                break
            if poll_interval_s > 0:
                await asyncio.sleep(poll_interval_s)
        if not by_cursor:
            raise RuntimeError(f"no Trace events published for {trace_id}")
        return [by_cursor[key] for key in sorted(by_cursor)]
    finally:
        if owns_client:
            await client.aclose()


def action_event(event_id: int, actions: list[str]) -> bytes:
    now_ms = time.time() * 1000.0
    return msgspec.msgpack.encode(
        {
            "type": "event",
            "kind": "camera_actions",
            "event_id": event_id,
            "client_sent_epoch_ms": now_ms,
            "payload": {
                "mode": "state",
                "transitions": [{"actions": actions, "client_ts_ms": int(now_ms)}],
            },
        }
    )


def init_request(args: argparse.Namespace, *, total_chunks: int, trace_id: str) -> dict:
    generation_mode = getattr(args, "generation_mode", "t2v")
    request = {
        "type": "init",
        "generation_mode": generation_mode,
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "fps": args.fps,
        "seed": 42,
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": total_chunks,
        "realtime_output_format": "webp",
        "realtime_preview_max_width": getattr(args, "preview_max_width", 560),
        "realtime_output_pacing": False,
        "output_compression": getattr(args, "preview_quality", 55),
        "realtime_causal_sink_size": getattr(args, "sink", 9),
        "realtime_causal_kv_cache_num_frames": getattr(args, "window", 18),
        "trace_id": trace_id,
    }
    media_profile = getattr(args, "realtime_media_profile", "native_v1")
    if media_profile != "native_v1":
        request["realtime_media_profile"] = media_profile
    if generation_mode == "i2v":
        first_frame = getattr(args, "first_frame_bytes", None)
        if not first_frame:
            raise ValueError("i2v benchmark requires first_frame_bytes")
        request["first_frame"] = first_frame
    else:
        request["num_frames"] = 1 + (total_chunks - 1) * 16
    return request


def stage_values(
    trace_events: list[dict], *, min_chunk_index: int = 0
) -> dict[str, list[float]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for event in trace_events:
        name = event.get("event")
        chunk_index_value = event.get("chunk_index")
        if chunk_index_value is None:
            continue
        chunk_index = int(chunk_index_value)
        if chunk_index < min_chunk_index:
            continue
        if name == "server.model_denoise_complete":
            values["denoise_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
        elif name == "server.vae_encode_complete":
            values["vae_encode_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
        elif name == "server.vae_decode_complete":
            values["vae_decode_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
        elif name == "server.remote_vae_complete":
            for field in (
                "vae_queue_wait_ms",
                "vae_decode_ms",
                "vae_post_decode_ms",
                "frame_encode_ms",
                "latent_serialize_ms",
                "latent_send_ms",
                "vae_credit_wait_ms",
                "first_frame_ms",
                "actor_wait_ms",
                "rife_interpolation_ms",
                "overlap_with_next_denoise_ms",
                "overlap_ratio",
            ):
                if event.get(field) is not None:
                    values[field][chunk_index] = float(event[field])
        elif name == "server.vae_denoise_overlap_complete":
            for field in (
                "overlap_with_next_denoise_ms",
                "overlap_ratio",
            ):
                if event.get(field) is not None:
                    values[field][chunk_index] = float(event[field])
    return {
        name: [by_chunk[index] for index in sorted(by_chunk)]
        for name, by_chunk in values.items()
    }


def stage_values_from_chunk_messages(
    stats: dict[int, dict], *, min_chunk_index: int = 0
) -> dict[str, list[float]]:
    """Read stage values from in-band chunk telemetry when Trace is disabled."""

    fields = (
        "request_prepare_ms",
        "scheduler_forward_ms",
        "chunk_total_ms",
        "output_pace_ms",
        "transport_encode_ms",
        "transport_write_ms",
        "vae_queue_wait_ms",
        "vae_decode_ms",
        "vae_post_decode_ms",
        "vae_encode_ms",
        "vae_transfer_ms",
        "latent_serialize_ms",
        "latent_send_ms",
        "vae_credit_wait_ms",
        "actor_wait_ms",
        "rife_interpolation_ms",
        "source_frames_per_chunk_wall_second",
        "output_frames_per_chunk_wall_second",
        "source_realtime_factor",
        "output_realtime_factor",
    )
    values: dict[str, list[float]] = defaultdict(list)
    for chunk_index in sorted(stats):
        if chunk_index < min_chunk_index:
            continue
        message = stats[chunk_index]
        for field in fields:
            if message.get(field) is not None:
                values[field].append(float(message[field]))
    return dict(values)


def merged_stage_values(
    trace_events: list[dict],
    chunk_messages: dict[int, dict],
    *,
    min_chunk_index: int = 0,
) -> dict[str, list[float]]:
    """Prefer complete Trace columns without dropping chunk telemetry evidence."""

    telemetry_values = stage_values_from_chunk_messages(
        chunk_messages, min_chunk_index=min_chunk_index
    )
    trace_values = stage_values(trace_events, min_chunk_index=min_chunk_index)
    merged = dict(telemetry_values)
    for field, values in trace_values.items():
        telemetry_samples = telemetry_values.get(field, [])
        if len(values) >= len(telemetry_samples):
            merged[field] = values
    return merged


def trace_contract_summary(trace_events: list[dict]) -> dict:
    event_names = sorted(
        {
            str(event["event"])
            for event in trace_events
            if isinstance(event.get("event"), str)
        }
    )
    direct_batches = sum(
        1
        for event in trace_events
        if event.get("event") == "server.vae_frame_batch_sent"
        and event.get("output_direct") is True
    )
    return {
        "event_names": event_names,
        "direct_vae_frame_batches": direct_batches,
    }


def record_action_latency(
    message: dict,
    *,
    first_frame_at: dict[int, float],
    action_sent_at: dict[int, float],
    action_latencies: list[float],
    min_chunk_index: int,
) -> None:
    chunk = int(message.get("chunk_index") or 0)
    sampled_event = int(message.get("event_id") or 0)
    if sampled_event <= 0 or chunk not in first_frame_at:
        return
    eligible = [event for event in action_sent_at if event <= sampled_event]
    if not eligible:
        return
    latest = max(eligible)
    if chunk >= min_chunk_index:
        action_latencies.append(
            round((first_frame_at[chunk] - action_sent_at[latest]) * 1000.0, 3)
        )
    for event_id in eligible:
        action_sent_at.pop(event_id, None)


def record_frame_batch(message: dict, *, frame_counts: dict[int, int]) -> None:
    chunk_index = int(message.get("chunk_index") or 0)
    num_frames = int(message.get("num_frames") or 0)
    if chunk_index < 0 or num_frames <= 0:
        return
    frame_counts[chunk_index] = frame_counts.get(chunk_index, 0) + num_frames


def frame_batch_contract_metadata(message: dict) -> dict:
    """Retain contract fields without holding encoded payloads during a soak."""

    fields = (
        "media_profile",
        "source_timeline_fps",
        "output_timeline_fps",
    )
    return {field: message[field] for field in fields if field in message}


def validate_media_profile_contract(
    *,
    media_profile: str,
    requested_fps: float,
    expected_media_weights_sha256: str | None,
    session_ready: dict | None,
    completions: dict[int, dict],
    frame_counts: dict[int, int],
    frame_messages: dict[int, list[dict]],
    expected_chunks: set[int],
) -> dict:
    """Validate negotiated RIFE timing and exact source/output frame counts."""

    multiplier = media_profile_multiplier(media_profile)
    if multiplier == 1:
        total = sum(frame_counts.get(chunk, 0) for chunk in expected_chunks)
        return {
            "source_frames": total,
            "output_frames": total,
            "acceptance": None,
        }
    if session_ready is None:
        raise RuntimeError("RIFE session omitted the required session_ready receipt")
    if (
        session_ready.get("requested_media_profile") != media_profile
        or session_ready.get("effective_media_profile") != media_profile
    ):
        raise RuntimeError(f"RIFE profile was not accepted: {session_ready!r}")
    source_timeline_fps = float(session_ready.get("source_timeline_fps") or 0)
    output_timeline_fps = float(session_ready.get("output_timeline_fps") or 0)
    if not math.isfinite(source_timeline_fps) or not math.isfinite(output_timeline_fps):
        raise RuntimeError("RIFE session_ready returned a non-finite timeline")
    if abs(source_timeline_fps - requested_fps) > 1e-6:
        raise RuntimeError(
            "RIFE source timeline does not match the request: "
            f"{source_timeline_fps} != {requested_fps}"
        )
    expected_output_timeline_fps = requested_fps * multiplier
    if abs(output_timeline_fps - expected_output_timeline_fps) > 1e-6:
        raise RuntimeError(
            f"RIFE output timeline is not {multiplier}x the source: "
            f"{output_timeline_fps} != {expected_output_timeline_fps}"
        )
    expected_digest = (expected_media_weights_sha256 or "").lower()
    actual_digest = str(session_ready.get("media_weights_sha256") or "").lower()
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise RuntimeError("RIFE validation requires an exact 64-hex weights digest")
    if len(actual_digest) != 64 or any(
        character not in "0123456789abcdef" for character in actual_digest
    ):
        raise RuntimeError("RIFE session_ready returned an invalid weights digest")
    if actual_digest != expected_digest:
        raise RuntimeError(
            "RIFE weights digest mismatch: "
            f"expected={expected_digest} actual={actual_digest}"
        )

    missing = sorted(expected_chunks - set(completions))
    if missing:
        raise RuntimeError(f"RIFE media completion metadata omitted chunks {missing}")

    source_total = 0
    output_total = 0
    has_source_history = False
    for chunk in sorted(expected_chunks):
        completion = completions[chunk]
        if completion.get("media_profile") != media_profile:
            raise RuntimeError(f"chunk {chunk} omitted the effective RIFE profile")
        source_frames = int(completion.get("source_num_frames", -1))
        output_frames = int(completion.get("output_num_frames", -1))
        completion_frames = int(completion.get("num_frames", -1))
        actual_frames = frame_counts.get(chunk, 0)
        if source_frames < 0 or output_frames < 0:
            raise RuntimeError(f"chunk {chunk} omitted RIFE frame counts")
        completion_source_fps = float(completion.get("source_timeline_fps") or 0)
        completion_output_fps = float(completion.get("output_timeline_fps") or 0)
        if (
            not math.isfinite(completion_source_fps)
            or not math.isfinite(completion_output_fps)
            or abs(completion_source_fps - source_timeline_fps) > 1e-6
            or abs(completion_output_fps - output_timeline_fps) > 1e-6
        ):
            raise RuntimeError(f"chunk {chunk} completion timeline drifted")
        expected_output = 0
        if source_frames > 0:
            expected_output = source_frames * multiplier - (
                0 if has_source_history else multiplier - 1
            )
            has_source_history = True
        if output_frames != expected_output:
            raise RuntimeError(
                f"chunk {chunk} violated RIFE {multiplier}x cadence: "
                f"source={source_frames} output={output_frames} "
                f"expected={expected_output}"
            )
        if completion_frames != output_frames or actual_frames != output_frames:
            raise RuntimeError(
                f"chunk {chunk} frame count mismatch: received={actual_frames} "
                f"num_frames={completion_frames} output_num_frames={output_frames}"
            )
        if output_frames > 0 and not frame_messages.get(chunk):
            raise RuntimeError(f"chunk {chunk} omitted RIFE frame metadata")
        for frame_message in frame_messages.get(chunk, []):
            frame_source_fps = float(frame_message.get("source_timeline_fps") or 0)
            frame_output_fps = float(frame_message.get("output_timeline_fps") or 0)
            if (
                frame_message.get("media_profile") != media_profile
                or not math.isfinite(frame_source_fps)
                or not math.isfinite(frame_output_fps)
                or abs(frame_source_fps - source_timeline_fps) > 1e-6
                or abs(frame_output_fps - output_timeline_fps) > 1e-6
            ):
                raise RuntimeError(f"chunk {chunk} frame metadata drifted")
        source_total += source_frames
        output_total += output_frames

    return {
        "source_frames": source_total,
        "output_frames": output_total,
        "acceptance": {
            "requested_media_profile": session_ready["requested_media_profile"],
            "effective_media_profile": session_ready["effective_media_profile"],
            "source_timeline_fps": source_timeline_fps,
            "output_timeline_fps": output_timeline_fps,
            "media_weights_sha256": actual_digest,
        },
    }


def final_frame_batch_chunk(message: dict) -> int | None:
    message_type = message.get("type")
    if message_type == "media_chunk_complete":
        chunk_index = int(message.get("chunk_index", -1))
        return chunk_index if chunk_index >= 0 else None
    if message_type not in {"frame_batch", "frame_batch_header"} or (
        message.get("is_final_frame_batch") is not True
    ):
        return None
    chunk_index = int(message.get("chunk_index", -1))
    return chunk_index if chunk_index >= 0 else None


def completion_chunk(
    message: dict,
    *,
    completion_signal: str,
    frame_counts: dict[int, int],
) -> int | None:
    chunk_index = final_frame_batch_chunk(message)
    if chunk_index is not None or completion_signal != "chunk-stats":
        return chunk_index
    if message.get("type") != "chunk_stats":
        return None
    chunk_index = int(message.get("chunk_index", -1))
    if chunk_index < 0 or frame_counts.get(chunk_index, 0) <= 0:
        return None
    return chunk_index


def chunk_stats_from_trace(trace_events: list[dict]) -> dict[int, dict]:
    """Read authoritative chunk timings from the out-of-band Trace API."""

    stats: dict[int, dict] = {}
    for event in trace_events:
        if event.get("event") != "server.chunk_complete":
            continue
        chunk_value = event.get("chunk_index")
        if chunk_value is None or event.get("chunk_total_ms") is None:
            continue
        chunk_index = int(chunk_value)
        if chunk_index >= 0:
            stats[chunk_index] = dict(event)
    return stats


def server_action_latencies(
    trace_events: list[dict], *, min_chunk_index: int = 0
) -> dict[str, list[float]]:
    received: dict[int, tuple[float, float]] = {}
    markers: dict[int, dict] = {}
    for event in trace_events:
        name = event.get("event")
        event_id_value = event.get("event_id")
        if name == "server.event_received" and event_id_value is not None:
            client_epoch_ms = event.get("client_sent_epoch_ms")
            server_elapsed_ms = event.get("server_elapsed_ms")
            if client_epoch_ms is not None and server_elapsed_ms is not None:
                received[int(event_id_value)] = (
                    float(client_epoch_ms),
                    float(server_elapsed_ms),
                )
            continue
        if name not in {
            "server.remote_first_frame_received",
            "server.output_send_start",
        }:
            continue
        chunk_value = event.get("chunk_index")
        if chunk_value is None:
            continue
        chunk_index = int(chunk_value)
        if chunk_index >= min_chunk_index:
            markers.setdefault(chunk_index, event)

    client_to_first_frame: list[float] = []
    ingress_to_first_frame: list[float] = []
    for marker in (markers[index] for index in sorted(markers)):
        marker_event_id = marker.get("event_id")
        marker_epoch_ms = marker.get("server_epoch_ms")
        marker_elapsed_ms = marker.get("server_elapsed_ms")
        if (
            marker_event_id is None
            or marker_epoch_ms is None
            or marker_elapsed_ms is None
        ):
            continue
        eligible = [
            event_id for event_id in received if event_id <= int(marker_event_id)
        ]
        if not eligible:
            continue
        client_epoch_ms, received_elapsed_ms = received[max(eligible)]
        client_delta = float(marker_epoch_ms) - client_epoch_ms
        ingress_delta = float(marker_elapsed_ms) - received_elapsed_ms
        if client_delta >= 0:
            client_to_first_frame.append(round(client_delta, 3))
        if ingress_delta >= 0:
            ingress_to_first_frame.append(round(ingress_delta, 3))

    return {
        "action_to_server_first_frame_ms": client_to_first_frame,
        "action_ingress_to_server_first_frame_ms": ingress_to_first_frame,
    }


def aggregate_measurement_seconds(sessions: list[dict]) -> float:
    starts = [
        float(session["measured_started_at"])
        for session in sessions
        if session.get("measured_started_at") is not None
    ]
    completions = [
        float(session["measured_completed_at"])
        for session in sessions
        if session.get("measured_completed_at") is not None
    ]
    if not starts or not completions:
        return 0.0
    return max(0.0, max(completions) - min(starts))


def measurement_window_start(
    *,
    chunk_index: int,
    observed_at: float,
    warmup_chunks: int,
    current: float | None,
) -> float | None:
    """Start timing after the final warmup chunk has fully completed."""

    if current is not None:
        return current
    if warmup_chunks > 0 and chunk_index == warmup_chunks - 1:
        return observed_at
    return None


async def stream_actions(
    websocket,
    *,
    action_sent_at: dict[int, float],
    stop: asyncio.Event,
    interval_s: float = 0.1,
) -> None:
    event_id = 1
    while not stop.is_set():
        sent_at = time.perf_counter()
        actions = ["w"] if event_id % 2 else ["a", "w"]
        try:
            await websocket.send(action_event(event_id, actions))
        except Exception:
            return
        action_sent_at[event_id] = sent_at
        event_id += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass


async def run_session(args: argparse.Namespace, concurrency: int, index: int) -> dict:
    import websockets

    total_chunks = args.warmup_chunks + args.measured_chunks
    media_profile = getattr(args, "realtime_media_profile", "native_v1")
    trace_id = uuid4().hex
    url = with_identity(
        args.ws_url,
        user_id=f"load-{concurrency}-{index}-{uuid4().hex}",
        trace_id=trace_id,
    )
    stats: dict[int, dict] = {}
    first_frame_at: dict[int, float] = {}
    frame_counts: dict[int, int] = {}
    frame_messages: dict[int, list[dict]] = defaultdict(list)
    trace_events: list[dict] = []
    action_sent_at: dict[int, float] = {}
    action_latencies: list[float] = []
    pending_raw_header = None
    completed_chunk_at: dict[int, float] = {}
    completion_metadata: dict[int, dict] = {}
    media_acceptance: dict | None = None
    measured_started_at: float | None = None
    measured_completed_at: float | None = None
    websocket_close: dict[str, int | str] | None = None
    session_started_at = time.perf_counter()

    def mark_chunk_complete(message: dict, observed_at: float) -> None:
        nonlocal measured_started_at, measured_completed_at
        chunk = completion_chunk(
            message,
            completion_signal=getattr(args, "completion_signal", "final-frame"),
            frame_counts=frame_counts,
        )
        if chunk is None or chunk in completed_chunk_at:
            return
        completed_chunk_at[chunk] = observed_at
        measured_started_at = measurement_window_start(
            chunk_index=chunk,
            observed_at=observed_at,
            warmup_chunks=args.warmup_chunks,
            current=measured_started_at,
        )
        if chunk >= args.warmup_chunks:
            measured_completed_at = observed_at

    async with websockets.connect(
        url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=args.timeout_s,
    ) as websocket:
        if args.warmup_chunks == 0:
            measured_started_at = time.perf_counter()
        await websocket.send(
            msgspec.msgpack.encode(
                init_request(args, total_chunks=total_chunks, trace_id=trace_id)
            )
        )
        action_stop = asyncio.Event()
        action_task = asyncio.create_task(
            stream_actions(
                websocket,
                action_sent_at=action_sent_at,
                stop=action_stop,
            )
        )
        try:
            while True:
                try:
                    packed = await asyncio.wait_for(
                        websocket.recv(), timeout=args.timeout_s
                    )
                except websockets.ConnectionClosed as exc:
                    websocket_close = websocket_close_receipt(exc)
                    break
                if not isinstance(packed, bytes):
                    continue
                if pending_raw_header is not None:
                    mark_chunk_complete(pending_raw_header, time.perf_counter())
                    pending_raw_header = None
                    continue
                message = msgspec.msgpack.decode(packed)
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(
                        message.get("content") or "realtime server error"
                    )
                if message_type in {"trace_event", "trace_events"}:
                    raise RuntimeError(
                        "production video WebSocket carried forbidden Trace data"
                    )
                if message_type == "media_chunk_complete":
                    chunk_value = int(message.get("chunk_index", -1))
                    if chunk_value >= 0:
                        completion_metadata[chunk_value] = dict(message)
                    mark_chunk_complete(message, time.perf_counter())
                    continue
                if message_type == "session_ready":
                    if media_acceptance is not None:
                        raise RuntimeError(
                            "session emitted duplicate session_ready receipts"
                        )
                    media_acceptance = dict(message)
                    continue
                if message_type == "frame_batch_header":
                    pending_raw_header = message
                if message_type in {"frame_batch", "frame_batch_header"}:
                    chunk = int(message.get("chunk_index") or 0)
                    observed_at = time.perf_counter()
                    record_frame_batch(message, frame_counts=frame_counts)
                    if media_profile_multiplier(media_profile) > 1:
                        frame_messages[chunk].append(
                            frame_batch_contract_metadata(message)
                        )
                    first_frame_at.setdefault(chunk, observed_at)
                    record_action_latency(
                        message,
                        first_frame_at=first_frame_at,
                        action_sent_at=action_sent_at,
                        action_latencies=action_latencies,
                        min_chunk_index=args.warmup_chunks,
                    )
                    if message_type == "frame_batch":
                        mark_chunk_complete(message, observed_at)
                    continue
                if message_type not in {"chunk_stats", "chunk_telemetry"}:
                    continue

                chunk = int(message["chunk_index"])
                observed_at = time.perf_counter()
                stats[chunk] = dict(message)
                mark_chunk_complete(message, observed_at)
                record_action_latency(
                    message,
                    first_frame_at=first_frame_at,
                    action_sent_at=action_sent_at,
                    action_latencies=action_latencies,
                    min_chunk_index=args.warmup_chunks,
                )
        finally:
            action_stop.set()
            await action_task

    if websocket_close is None:
        raise RuntimeError("finite session ended without a WebSocket close receipt")
    if (
        websocket_close["code"] != 1000
        or websocket_close["reason"] not in NORMAL_FINITE_CLOSE_REASONS
    ):
        raise RuntimeError(
            "finite session did not close normally: "
            f"code={websocket_close['code']} reason={websocket_close['reason']!r}"
        )

    if not args.skip_trace_query:
        trace_events = await collect_trace_events(
            args.trace_http_url or derive_trace_http_url(args.ws_url),
            trace_id,
            timeout_s=args.trace_timeout_s,
            expected_chunks=total_chunks,
        )

    expected_chunks = set(range(total_chunks))
    completed_chunks = set(completed_chunk_at)
    if completed_chunks != expected_chunks:
        missing = sorted(expected_chunks - completed_chunks)
        raise RuntimeError(
            "session media stream closed before final frame batches for "
            f"chunks {missing}"
        )

    media_contract = validate_media_profile_contract(
        media_profile=media_profile,
        requested_fps=float(args.fps),
        expected_media_weights_sha256=getattr(
            args, "expected_media_weights_sha256", None
        ),
        session_ready=media_acceptance,
        completions=completion_metadata,
        frame_counts=frame_counts,
        frame_messages=frame_messages,
        expected_chunks=expected_chunks,
    )

    websocket_stats = dict(stats)
    trace_stats = chunk_stats_from_trace(trace_events)
    stats.update(trace_stats)
    if trace_stats and len(trace_stats) == total_chunks:
        timing_source = "trace_http"
    elif len(stats) == total_chunks:
        timing_source = (
            "video_ws_chunk_telemetry"
            if any(
                message.get("type") == "chunk_telemetry"
                for message in websocket_stats.values()
            )
            else "video_ws_chunk_stats"
        )
    elif args.skip_trace_query:
        timing_source = "client_frame_completion_interval"
        previous = session_started_at
        for chunk in range(total_chunks):
            completed_at = completed_chunk_at[chunk]
            stats[chunk] = {
                "chunk_index": chunk,
                "chunk_total_ms": max(0.0, (completed_at - previous) * 1000.0),
            }
            previous = completed_at
    else:
        missing = sorted(expected_chunks - set(stats))
        raise RuntimeError(f"Trace API omitted chunk timings for chunks {missing}")

    if len(stats) != total_chunks:
        raise RuntimeError(
            f"session closed after {len(stats)} of {total_chunks} chunks"
        )

    measured = [stats[index] for index in range(args.warmup_chunks, total_chunks)]
    chunk_total = [float(item["chunk_total_ms"]) for item in measured]
    measured_chunk_indexes = set(range(args.warmup_chunks, total_chunks))
    frame_count = sum(frame_counts.get(index, 0) for index in measured_chunk_indexes)
    if media_profile_multiplier(media_profile) > 1:
        source_frame_count = sum(
            int(completion_metadata[index]["source_num_frames"])
            for index in measured_chunk_indexes
        )
    else:
        source_frame_count = frame_count
    measured_seconds = (
        max(0.0, measured_completed_at - measured_started_at)
        if measured_started_at is not None and measured_completed_at is not None
        else 0.0
    )
    server_action = server_action_latencies(
        trace_events, min_chunk_index=args.warmup_chunks
    )
    trace_contract = trace_contract_summary(trace_events)
    return {
        "session_index": index,
        "trace_id": trace_id,
        "timing_source": timing_source,
        "chunk_total_ms": chunk_total,
        "action_to_first_frame_ms": server_action["action_to_server_first_frame_ms"],
        "action_ingress_to_first_frame_ms": server_action[
            "action_ingress_to_server_first_frame_ms"
        ],
        "client_observed_action_to_first_frame_ms": action_latencies,
        "frames": frame_count,
        "source_frames": source_frame_count,
        "output_frames": frame_count,
        "media_profile_acceptance": media_contract["acceptance"],
        "measured_seconds": measured_seconds,
        "measured_started_at": measured_started_at,
        "measured_completed_at": measured_completed_at,
        "stage_values": merged_stage_values(
            trace_events,
            websocket_stats,
            min_chunk_index=args.warmup_chunks,
        ),
        "trace_event_names": trace_contract["event_names"],
        "direct_vae_frame_batches": trace_contract["direct_vae_frame_batches"],
        "websocket_close": websocket_close,
    }


async def run_level(args: argparse.Namespace, concurrency: int) -> dict:
    results = await asyncio.gather(
        *(run_session(args, concurrency, index) for index in range(concurrency)),
        return_exceptions=True,
    )
    sessions = [result for result in results if isinstance(result, dict)]
    errors = [str(result) for result in results if isinstance(result, BaseException)]
    chunks = [value for session in sessions for value in session["chunk_total_ms"]]
    action = [
        value for session in sessions for value in session["action_to_first_frame_ms"]
    ]
    action_ingress = [
        value
        for session in sessions
        for value in session["action_ingress_to_first_frame_ms"]
    ]
    client_observed_action = [
        value
        for session in sessions
        for value in session["client_observed_action_to_first_frame_ms"]
    ]
    stages: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        for name, values in session["stage_values"].items():
            stages[name].extend(values)
    trace_event_names = sorted(
        {name for session in sessions for name in session["trace_event_names"]}
    )
    total_frames = sum(session["frames"] for session in sessions)
    total_source_frames = sum(session["source_frames"] for session in sessions)
    wall_seconds = aggregate_measurement_seconds(sessions)
    session_fps = [
        session["frames"] / session["measured_seconds"]
        for session in sessions
        if session["measured_seconds"]
    ]
    source_session_fps = [
        session["source_frames"] / session["measured_seconds"]
        for session in sessions
        if session["measured_seconds"]
    ]
    aggregate_output_wall_fps = total_frames / wall_seconds if wall_seconds else 0.0
    aggregate_source_wall_fps = (
        total_source_frames / wall_seconds if wall_seconds else 0.0
    )
    output_wall_summary = latency_summary(session_fps)
    source_wall_summary = latency_summary(source_session_fps)
    min_output_wall_fps = min(session_fps, default=0.0)
    min_source_wall_fps = min(source_session_fps, default=0.0)
    return {
        "concurrency": concurrency,
        "successful_sessions": len(sessions),
        "errors": errors,
        "error_rate": len(errors) / concurrency,
        "timing_sources": sorted({session["timing_source"] for session in sessions}),
        "chunk_total_ms": latency_summary(chunks),
        "action_to_first_frame_ms": latency_summary(action),
        "action_ingress_to_first_frame_ms": latency_summary(action_ingress),
        "client_observed_action_to_first_frame_ms": latency_summary(
            client_observed_action
        ),
        # Keep the historical keys for result-reader compatibility. The
        # explicit aliases make clear these are measured wall-clock delivery
        # rates, never the negotiated output_timeline_fps timebase.
        "aggregate_fps": aggregate_output_wall_fps,
        "aggregate_output_wall_fps": aggregate_output_wall_fps,
        "aggregate_source_fps": aggregate_source_wall_fps,
        "aggregate_source_wall_fps": aggregate_source_wall_fps,
        "measurement_wall_seconds": wall_seconds,
        "per_session_fps": output_wall_summary,
        "per_session_output_wall_fps": output_wall_summary,
        "per_session_source_fps": source_wall_summary,
        "per_session_source_wall_fps": source_wall_summary,
        "min_session_fps": min_output_wall_fps,
        "min_session_output_wall_fps": min_output_wall_fps,
        "min_session_source_fps": min_source_wall_fps,
        "min_session_source_wall_fps": min_source_wall_fps,
        "media_profile_acceptance": [
            session["media_profile_acceptance"] for session in sessions
        ],
        "stage_ms": {name: latency_summary(values) for name, values in stages.items()},
        "trace_event_names": trace_event_names,
        "direct_vae_frame_batches": sum(
            session["direct_vae_frame_batches"] for session in sessions
        ),
    }


async def async_main(args: argparse.Namespace) -> None:
    levels = [int(value) for value in args.concurrency.split(",") if value.strip()]
    if not levels or any(value < 1 for value in levels):
        raise ValueError("concurrency levels must be positive")
    if len(levels) != len(set(levels)):
        raise ValueError("concurrency levels must be unique")
    if media_profile_multiplier(args.realtime_media_profile) > 1:
        expected_digest = (args.expected_media_weights_sha256 or "").lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError(
                "--expected-media-weights-sha256 must be an exact 64-hex digest "
                "for an interpolated media profile"
            )
        args.expected_media_weights_sha256 = expected_digest
    if args.generation_mode == "i2v":
        if args.first_frame is None or not args.first_frame.is_file():
            raise ValueError("--first-frame must name an existing image for i2v")
        args.first_frame_bytes = args.first_frame.read_bytes()
    elif args.first_frame is not None:
        raise ValueError("--first-frame is only valid with --generation-mode=i2v")
    runs = []
    for concurrency in levels:
        run = await run_level(args, concurrency)
        runs.append(run)
        print(json.dumps(run, indent=2, sort_keys=True))
    output = {
        "schema_version": "minwm-realtime-load/v1",
        "profile": args.profile,
        "requested_concurrency_levels": levels,
        "request": {
            "model": args.model,
            "generation_mode": args.generation_mode,
            "first_frame": str(args.first_frame) if args.first_frame else None,
            "size": args.size,
            "fps": args.fps,
            "realtime_media_profile": args.realtime_media_profile,
            "expected_media_weights_sha256": args.expected_media_weights_sha256,
            "sink": args.sink,
            "window": args.window,
            "completion_signal": args.completion_signal,
            "steps": 4,
            "warmup_chunks": args.warmup_chunks,
            "measured_chunks": args.measured_chunks,
        },
        "hardware": (
            json.loads(args.hardware_json.read_text()) if args.hardware_json else {}
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
