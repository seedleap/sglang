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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", default="packed-det-bf16")
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
            lane["client_fps"] = throughput["client"].get(
                "steady_received_fps_ratio_of_sums"
            )
        lane["peak_gpu_memory_mib"] = peak_memory(lane_path / "gpu.csv")
        lane["backend_evidence"] = backend_evidence(lane_path / "server.log")
        lanes[lane_path.name] = lane

    if args.reference not in lanes or "throughput" not in lanes[args.reference]:
        raise SystemExit(f"missing successful reference lane {args.reference!r}")
    reference = lanes[args.reference]["throughput"]
    reference_scheduler = lanes[args.reference]["scheduler_fps"]
    for name, lane in lanes.items():
        if "throughput" not in lane:
            continue
        lane["sampled_error_vs_reference"] = sampled_error(
            reference, lane["throughput"]
        )
        if reference_scheduler and lane["scheduler_fps"]:
            lane["scheduler_speedup_vs_reference_pct"] = 100 * (
                lane["scheduler_fps"] / reference_scheduler - 1
            )

    ranked = sorted(
        (
            {
                "lane": name,
                "scheduler_fps": lane.get("scheduler_fps"),
                "client_fps": lane.get("client_fps"),
                "scheduler_speedup_vs_reference_pct": lane.get(
                    "scheduler_speedup_vs_reference_pct"
                ),
                "peak_gpu_memory_mib": lane.get("peak_gpu_memory_mib"),
                "sampled_error_vs_reference": lane.get("sampled_error_vs_reference"),
            }
            for name, lane in lanes.items()
            if lane.get("scheduler_fps") is not None
        ),
        key=lambda item: item["scheduler_fps"],
        reverse=True,
    )
    for item in ranked:
        error = item["sampled_error_vs_reference"]
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
        "schema_version": "minwm-720p-attn-ffn-matrix/v1",
        "reference_lane": args.reference,
        "contract": (
            json.loads((args.root / "contract.json").read_text())
            if (args.root / "contract.json").is_file()
            else {}
        ),
        "ranking": ranked,
        "raw_speed_ranking": ranked,
        "quality_screen": {
            "metric": "sampled output-frame byte PSNR versus packed deterministic BF16",
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
        f"Reference: `{args.reference}`",
        "",
        "Raw-speed ranking (the quality column is only a sampled corruption screen):",
        "",
        "| rank | lane | scheduler FPS | client FPS | speedup | peak MiB | sampled PSNR | screen |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, item in enumerate(ranked, start=1):
        error = item["sampled_error_vs_reference"] or {}
        psnr = error.get("psnr_db")
        lines.append(
            "| {rank} | `{lane}` | {scheduler:.3f} | {client:.3f} | {speedup:+.2f}% | {memory} | {psnr} | {screen} |".format(
                rank=index,
                lane=item["lane"],
                scheduler=item["scheduler_fps"],
                client=item["client_fps"],
                speedup=item["scheduler_speedup_vs_reference_pct"] or 0.0,
                memory=(
                    f"{item['peak_gpu_memory_mib']:.0f}"
                    if item["peak_gpu_memory_mib"] is not None
                    else "n/a"
                ),
                psnr=(
                    "lossless"
                    if error.get("lossless")
                    else f"{psnr:.2f}" if psnr is not None else "n/a"
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
