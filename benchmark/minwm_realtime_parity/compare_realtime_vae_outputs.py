#!/usr/bin/env python3
"""Compare local and remote exact-VAE RGB outputs under an explicit gate."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path


def _byte_error_summary(local: bytes, remote: bytes) -> dict[str, float | int | None]:
    if len(local) != len(remote):
        raise ValueError("local and remote RGB byte lengths differ")
    if not local:
        raise ValueError("RGB comparison requires at least one byte")
    different = 0
    absolute_error_sum = 0
    squared_error_sum = 0
    max_absolute_error = 0
    for local_value, remote_value in zip(local, remote):
        error = abs(local_value - remote_value)
        different += error != 0
        absolute_error_sum += error
        squared_error_sum += error * error
        max_absolute_error = max(max_absolute_error, error)
    mean_squared_error = squared_error_sum / len(local)
    return {
        "num_bytes": len(local),
        "different_byte_fraction": different / len(local),
        "mean_absolute_error": absolute_error_sum / len(local),
        "max_absolute_error": max_absolute_error,
        "psnr_db": (
            None
            if mean_squared_error == 0
            else 20 * math.log10(255) - 10 * math.log10(mean_squared_error)
        ),
    }


def compare_results(
    local_path: Path,
    remote_path: Path,
    *,
    max_absolute_error_threshold: int = 1,
    psnr_threshold_db: float = 60.0,
) -> dict:
    local = json.loads(local_path.read_text())
    remote = json.loads(remote_path.read_text())
    local_frames = local.get("measured_frame_sha256", {})
    remote_frames = remote.get("measured_frame_sha256", {})
    all_frame_keys = sorted(set(local_frames) | set(remote_frames))
    first_differing_frame = next(
        (
            key
            for key in all_frame_keys
            if local_frames.get(key) != remote_frames.get(key)
        ),
        None,
    )

    first_frame = _byte_error_summary(
        local_path.with_name("first-measured-frame.rgb").read_bytes(),
        remote_path.with_name("first-measured-frame.rgb").read_bytes(),
    )
    local_samples = local.get("measured_frame_samples_base64", {})
    remote_samples = remote.get("measured_frame_samples_base64", {})
    if set(local_samples) != set(remote_samples):
        raise ValueError("local and remote measured RGB frame sample keys differ")
    sampled_local = bytearray()
    sampled_remote = bytearray()
    for frame_key in sorted(local_samples):
        sampled_local.extend(base64.b64decode(local_samples[frame_key]))
        sampled_remote.extend(base64.b64decode(remote_samples[frame_key]))
    sampled = _byte_error_summary(bytes(sampled_local), bytes(sampled_remote))

    numerical_parity = (
        first_frame["max_absolute_error"] <= max_absolute_error_threshold
        and (
            first_frame["psnr_db"] is None
            or first_frame["psnr_db"] >= psnr_threshold_db
        )
        and sampled["max_absolute_error"] <= max_absolute_error_threshold
        and (sampled["psnr_db"] is None or sampled["psnr_db"] >= psnr_threshold_db)
    )
    return {
        "local_payload_sha256": local["measured_payload_sha256"],
        "remote_payload_sha256": remote["measured_payload_sha256"],
        "bitwise_equal": local["measured_payload_sha256"]
        == remote["measured_payload_sha256"],
        "frame_hashes_equal": local_frames == remote_frames,
        "first_differing_frame": first_differing_frame,
        **{f"first_frame_{key}": value for key, value in first_frame.items()},
        **{f"sampled_all_frames_{key}": value for key, value in sampled.items()},
        "max_absolute_error_threshold": max_absolute_error_threshold,
        "psnr_threshold_db": psnr_threshold_db,
        "numerical_parity": numerical_parity,
        "local_client_fps": local["client"]["steady_received_fps_ratio_of_sums"],
        "remote_client_fps": remote["client"]["steady_received_fps_ratio_of_sums"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local", type=Path)
    parser.add_argument("remote", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-absolute-error", type=int, default=1)
    parser.add_argument("--min-psnr-db", type=float, default=60.0)
    args = parser.parse_args()
    summary = compare_results(
        args.local,
        args.remote,
        max_absolute_error_threshold=args.max_absolute_error,
        psnr_threshold_db=args.min_psnr_db,
    )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
