#!/usr/bin/env python3
"""Summarize MinWM 720p attention/FFN throughput lanes and sampled quality."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path


def sampled_error(reference: dict, candidate: dict) -> dict | None:
    ref = reference.get("measured_frame_samples_base64", {})
    other = candidate.get("measured_frame_samples_base64", {})
    keys = sorted(ref.keys() & other.keys())
    if not keys:
        return None
    absolute = squared = equal = maximum = count = 0
    for key in keys:
        left = base64.b64decode(ref[key])
        right = base64.b64decode(other[key])
        if len(left) != len(right):
            raise ValueError(f"sample length differs for {key}")
        for a, b in zip(left, right):
            error = abs(a - b)
            absolute += error
            squared += error * error
            equal += error == 0
            maximum = max(maximum, error)
            count += 1
    mse = squared / count
    return {
        "bytes": count,
        "lossless": mse == 0,
        "equal_fraction": equal / count,
        "mae": absolute / count,
        "rmse": math.sqrt(mse),
        # Keep strict JSON: lossless samples are represented explicitly instead
        # of serializing the non-standard Infinity token.
        "psnr_db": None if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        "max_abs": maximum,
    }


def peak_memory(path: Path) -> float | None:
    values = []
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            values.append(float(fields[2]))
        except ValueError:
            continue
    return max(values) if values else None


def backend_evidence(path: Path) -> list[str]:
    if not path.is_file():
        return []
    pattern = re.compile(
        r"(packed attention backend|attention backend|attention_impl=|Sage Attention|Flash Attention)",
        re.IGNORECASE,
    )
    found = []
    for line in path.read_text(errors="replace").splitlines():
        if pattern.search(line):
            cleaned = line[-500:]
            if cleaned not in found:
                found.append(cleaned)
    return found[-12:]


def scheduler_fps_from_log(path: Path, throughput: dict) -> tuple[float | None, int]:
    """Recover scheduler FPS when this server does not return chunk_stats."""
    if not path.is_file():
        return None, 0
    first = int(throughput["warmup_chunks"])
    count = int(throughput["measured_chunks"])
    measured = range(first, first + count)
    scheduler_ms_by_chunk = {}
    for line in path.read_text(errors="replace").splitlines():
        marker = "realtime_trace "
        offset = line.find(marker)
        if offset < 0:
            continue
        try:
            event = json.loads(line[offset + len(marker) :])
        except json.JSONDecodeError:
            continue
        if event.get("event") != "server.chunk_complete":
            continue
        chunk_index = event.get("chunk_index")
        scheduler_ms = event.get("scheduler_forward_ms")
        if chunk_index in measured and scheduler_ms is not None:
            scheduler_ms_by_chunk[int(chunk_index)] = float(scheduler_ms)
    if len(scheduler_ms_by_chunk) != count:
        return None, len(scheduler_ms_by_chunk)
    measured_frames = int(throughput["measured_frames"])
    return measured_frames / (sum(scheduler_ms_by_chunk.values()) / 1000.0), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--performance-reference",
        default="packed-fast-bf16",
        help="Lane used as the speedup denominator.",
    )
    parser.add_argument(
        "--quality-reference",
        default="packed-det-bf16",
        help="Lane used only for the sampled byte-level corruption screen.",
    )
    parser.add_argument(
        "--min-sampled-psnr-db",
        type=float,
        default=40.0,
        help="Screen raw-speed results using sampled frame bytes (not a perceptual gate).",
    )
    args = parser.parse_args()

    lanes = {}
    for lane_path in sorted((args.root / "lanes").glob("*")):
        if not lane_path.is_dir():
            continue
        metadata_path = lane_path / "lane.json"
        throughput_path = lane_path / "throughput.json"
        status_path = lane_path / "status.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        )
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        lane = {"metadata": metadata, "status": status}
        if throughput_path.is_file():
            throughput = json.loads(throughput_path.read_text())
            lane["throughput"] = throughput
            lane["scheduler_fps"] = throughput["server"].get(
                "scheduler_forward_fps_ratio_of_sums"
            )
            lane["scheduler_fps_source"] = "client_chunk_stats"
            if lane["scheduler_fps"] is None:
                recovered, recovered_chunks = scheduler_fps_from_log(
                    lane_path / "server.log", throughput
                )
                lane["scheduler_fps"] = recovered
                lane["scheduler_fps_source"] = "server.chunk_complete log"
                lane["scheduler_fps_recovered_chunks"] = recovered_chunks
            lane["client_fps"] = throughput["client"].get(
                "steady_received_fps_ratio_of_sums"
            )
        lane["peak_gpu_memory_mib"] = peak_memory(lane_path / "gpu.csv")
        lane["backend_evidence"] = backend_evidence(lane_path / "server.log")
        lanes[lane_path.name] = lane

    if (
        args.performance_reference not in lanes
        or "throughput" not in lanes[args.performance_reference]
    ):
        raise SystemExit(
            "missing successful performance reference lane "
            f"{args.performance_reference!r}"
        )
    if (
        args.quality_reference not in lanes
        or "throughput" not in lanes[args.quality_reference]
    ):
        raise SystemExit(
            f"missing successful quality reference lane {args.quality_reference!r}"
        )
    successful = [lane for lane in lanes.values() if "throughput" in lane]
    use_scheduler = all(lane.get("scheduler_fps") is not None for lane in successful)
    ranking_metric = "scheduler_fps" if use_scheduler else "client_fps"
    quality_reference = lanes[args.quality_reference]["throughput"]
    reference_fps = lanes[args.performance_reference][ranking_metric]
    for name, lane in lanes.items():
        if "throughput" not in lane:
            continue
        lane["sampled_error_vs_quality_reference"] = sampled_error(
            quality_reference, lane["throughput"]
        )
        lane["ranking_fps"] = lane[ranking_metric]
        if reference_fps and lane["ranking_fps"]:
            lane["ranking_speedup_vs_performance_reference_pct"] = 100 * (
                lane["ranking_fps"] / reference_fps - 1
            )

    ranked = sorted(
        (
            {
                "lane": name,
                "ranking_fps": lane.get("ranking_fps"),
                "scheduler_fps": lane.get("scheduler_fps"),
                "client_fps": lane.get("client_fps"),
                "ranking_speedup_vs_performance_reference_pct": lane.get(
                    "ranking_speedup_vs_performance_reference_pct"
                ),
                "peak_gpu_memory_mib": lane.get("peak_gpu_memory_mib"),
                "sampled_error_vs_quality_reference": lane.get(
                    "sampled_error_vs_quality_reference"
                ),
            }
            for name, lane in lanes.items()
            if lane.get("ranking_fps") is not None
        ),
        key=lambda item: item["ranking_fps"],
        reverse=True,
    )
    for item in ranked:
        error = item["sampled_error_vs_quality_reference"]
        item["passes_sampled_quality_screen"] = bool(
            error
            and (
                error.get("lossless")
                or (
                    error.get("psnr_db") is not None
                    and error["psnr_db"] >= args.min_sampled_psnr_db
                )
            )
        )
    quality_screened = [
        item for item in ranked if item["passes_sampled_quality_screen"]
    ]
    eager_ranked = [
        item
        for item in ranked
        if not lanes[item["lane"]]["metadata"].get("whole_dit_compile", False)
    ]
    eager_quality_screened = [
        item for item in eager_ranked if item["passes_sampled_quality_screen"]
    ]
    result = {
        "schema_version": "minwm-720p-attn-ffn-matrix/v2",
        "performance_reference_lane": args.performance_reference,
        "quality_reference_lane": args.quality_reference,
        "ranking_metric": ranking_metric,
        "contract": (
            json.loads((args.root / "contract.json").read_text())
            if (args.root / "contract.json").is_file()
            else {}
        ),
        "ranking": ranked,
        "raw_speed_ranking": ranked,
        "quality_screen": {
            "metric": (
                f"sampled output-frame byte PSNR versus {args.quality_reference}"
            ),
            "minimum_psnr_db": args.min_sampled_psnr_db,
            "warning": "This is a corruption screen, not a perceptual or production quality guarantee.",
        },
        "quality_screened_ranking": quality_screened,
        "best_eager_lane_raw_speed": eager_ranked[0]["lane"] if eager_ranked else None,
        "best_eager_lane_quality_screened": (
            eager_quality_screened[0]["lane"] if eager_quality_screened else None
        ),
        "lanes": lanes,
    }
    output_json = args.root / "summary.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# MinWM 720p attention / FFN Spot matrix",
        "",
        f"Performance reference: `{args.performance_reference}`",
        "",
        f"Quality byte reference: `{args.quality_reference}`",
        "",
        "Raw-speed ranking (the quality column is only a sampled corruption screen):",
        "",
        f"Ranking metric: `{ranking_metric}`",
        "",
        "| rank | lane | scheduler FPS | client FPS | speedup | peak MiB | sampled PSNR | screen |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, item in enumerate(ranked, start=1):
        error = item["sampled_error_vs_quality_reference"] or {}
        psnr = error.get("psnr_db")
        lines.append(
            "| {rank} | `{lane}` | {scheduler} | {client} | {speedup:+.2f}% | {memory} | {psnr} | {screen} |".format(
                rank=index,
                lane=item["lane"],
                scheduler=(
                    f"{item['scheduler_fps']:.3f}"
                    if item["scheduler_fps"] is not None
                    else "n/a"
                ),
                client=(
                    f"{item['client_fps']:.3f}"
                    if item["client_fps"] is not None
                    else "n/a"
                ),
                speedup=item["ranking_speedup_vs_performance_reference_pct"] or 0.0,
                memory=(
                    f"{item['peak_gpu_memory_mib']:.0f}"
                    if item["peak_gpu_memory_mib"] is not None
                    else "n/a"
                ),
                psnr=(
                    "lossless"
                    if error.get("lossless")
                    else f"{psnr:.2f}"
                    if psnr is not None
                    else "n/a"
                ),
                screen="yes" if item["passes_sampled_quality_screen"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Quality-screened order: "
            + (
                ", ".join(f"`{item['lane']}`" for item in quality_screened)
                if quality_screened
                else "none"
            ),
        ]
    )
    failed = [name for name, lane in lanes.items() if "throughput" not in lane]
    if failed:
        lines.extend(
            ["", "Failed or skipped lanes: " + ", ".join(f"`{x}`" for x in failed)]
        )
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "raw_speed_ranking": ranked,
                "ranking_metric": ranking_metric,
                "quality_screened_ranking": quality_screened,
                "best_eager_lane_raw_speed": result["best_eager_lane_raw_speed"],
                "best_eager_lane_quality_screened": result[
                    "best_eager_lane_quality_screened"
                ],
                "failed_or_skipped": failed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
