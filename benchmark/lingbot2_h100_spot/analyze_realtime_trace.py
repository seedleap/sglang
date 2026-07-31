#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Estimate realtime denoiser/VAE overlap potential from realtime_trace logs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

TRACE_MARKER = "realtime_trace "
DENOISE_EVENT = "server.model_denoise_complete"
VAE_DECODE_EVENT = "server.vae_decode_complete"
VAE_ENCODE_EVENT = "server.vae_encode_complete"
PIPELINE_STAGE_EVENT = "server.pipeline_stage_complete"


def parse_realtime_trace_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        _, sep, payload_text = line.partition(TRACE_MARKER)
        if not sep:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def summarize_async_potential(
    events: list[dict[str, Any]],
    *,
    warmup_chunks: int,
    transfer_ms: float = 0.0,
    trace_id: str | None = None,
    use_cuda_ms: bool = True,
) -> dict[str, Any]:
    chunks = _chunk_records(events, trace_id=trace_id, use_cuda_ms=use_cuda_ms)
    measured = [
        record
        for record in chunks
        if record["chunk_index"] >= warmup_chunks
        and record.get("denoise_ms") is not None
        and record.get("vae_decode_ms") is not None
    ]

    for record in measured:
        denoise_ms = float(record["denoise_ms"])
        decode_path_ms = transfer_ms + float(record["vae_decode_ms"])
        sync_compute_ms = denoise_ms + float(record["vae_decode_ms"])
        async_critical_ms = max(denoise_ms, decode_path_ms)
        record["transfer_ms"] = transfer_ms
        record["sync_compute_ms"] = sync_compute_ms
        record["async_critical_ms"] = async_critical_ms
        record["saved_ms"] = sync_compute_ms - async_critical_ms
        record["speedup"] = (
            sync_compute_ms / async_critical_ms if async_critical_ms > 0 else math.nan
        )

    first = [
        record
        for record in chunks
        if record["chunk_index"] == 0
        and record.get("denoise_ms") is not None
        and record.get("vae_decode_ms") is not None
    ]

    return {
        "trace_id": trace_id or _single_trace_id(events),
        "warmup_chunks": warmup_chunks,
        "transfer_ms": transfer_ms,
        "total_chunks_with_trace": len(chunks),
        "measured_chunks": len(measured),
        "first_chunk": first[0] if first else None,
        "denoise_ms": _stats([float(item["denoise_ms"]) for item in measured]),
        "vae_decode_ms": _stats([float(item["vae_decode_ms"]) for item in measured]),
        "vae_encode_ms": _stats(
            [float(item["vae_encode_ms"]) for item in measured if "vae_encode_ms" in item]
        ),
        "sync_compute_ms": _stats([item["sync_compute_ms"] for item in measured]),
        "async_critical_ms": _stats(
            [item["async_critical_ms"] for item in measured]
        ),
        "saved_ms": _stats([item["saved_ms"] for item in measured]),
        "speedup": _ratio_stats(measured),
        "chunks": measured,
    }


def _chunk_records(
    events: list[dict[str, Any]],
    *,
    trace_id: str | None,
    use_cuda_ms: bool,
) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for event in events:
        if trace_id is not None and event.get("trace_id") != trace_id:
            continue
        chunk_index = event.get("chunk_index")
        if chunk_index is None:
            continue
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            continue
        record = records.setdefault(chunk_index, {"chunk_index": chunk_index})
        duration = _event_duration_ms(event, use_cuda_ms=use_cuda_ms)
        if duration is None:
            continue
        if event.get("event") == DENOISE_EVENT:
            record["denoise_ms"] = duration
        elif event.get("event") == PIPELINE_STAGE_EVENT and _is_denoising_stage(event):
            record.setdefault("denoise_ms", duration)
        elif event.get("event") == VAE_DECODE_EVENT:
            record["vae_decode_ms"] = duration
            record["decoder_backend"] = event.get("decoder_backend")
        elif event.get("event") == VAE_ENCODE_EVENT:
            record["vae_encode_ms"] = duration
    return [records[key] for key in sorted(records)]


def _event_duration_ms(event: dict[str, Any], *, use_cuda_ms: bool) -> float | None:
    if use_cuda_ms and event.get("cuda_ms") is not None:
        return _float_or_none(event.get("cuda_ms"))
    return _float_or_none(event.get("duration_ms"))


def _is_denoising_stage(event: dict[str, Any]) -> bool:
    stage = str(event.get("stage") or "")
    return "denois" in stage.lower()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
    }


def _ratio_stats(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    ratios = [float(item["speedup"]) for item in records]
    stats = _stats(ratios)
    total_sync = sum(float(item["sync_compute_ms"]) for item in records)
    total_async = sum(float(item["async_critical_ms"]) for item in records)
    stats["mean"] = round(total_sync / total_async, 3) if total_async > 0 else None
    return stats


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _single_trace_id(events: list[dict[str, Any]]) -> str | None:
    trace_ids = sorted({str(item["trace_id"]) for item in events if item.get("trace_id")})
    return trace_ids[0] if len(trace_ids) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="server log path; defaults to stdin")
    parser.add_argument("--trace-id", help="only analyze one trace_id")
    parser.add_argument("--warmup-chunks", type=int, default=1)
    parser.add_argument(
        "--transfer-ms",
        type=float,
        default=0.0,
        help="modeled queue+transfer cost added to VAE decode in async path",
    )
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        help="use duration_ms even when cuda_ms exists",
    )
    args = parser.parse_args()

    if args.log:
        lines = Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        lines = sys.stdin
    events = parse_realtime_trace_lines(lines)
    summary = summarize_async_potential(
        events,
        warmup_chunks=args.warmup_chunks,
        transfer_ms=args.transfer_ms,
        trace_id=args.trace_id,
        use_cuda_ms=not args.wall_clock,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
