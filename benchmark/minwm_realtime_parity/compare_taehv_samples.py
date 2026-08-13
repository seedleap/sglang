#!/usr/bin/env python3
"""Compare fixed-frame RGB samples from exact causal VAE and local TAEHV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--taehv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"P6"
        width, height = (int(value) for value in stream.readline().split())
        assert stream.readline().strip() == b"255"
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    assert pixels.size == width * height * 3
    return pixels.reshape(height, width, 3)


def ssim(exact: np.ndarray, candidate: np.ndarray) -> float:
    left = exact.astype(np.float64)
    right = candidate.astype(np.float64)
    sigma = 1.5
    mu_left = gaussian_filter(left, sigma=(sigma, sigma, 0))
    mu_right = gaussian_filter(right, sigma=(sigma, sigma, 0))
    var_left = gaussian_filter(left * left, sigma=(sigma, sigma, 0)) - mu_left**2
    var_right = gaussian_filter(right * right, sigma=(sigma, sigma, 0)) - mu_right**2
    covariance = (
        gaussian_filter(left * right, sigma=(sigma, sigma, 0))
        - mu_left * mu_right
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    score = ((2 * mu_left * mu_right + c1) * (2 * covariance + c2)) / (
        (mu_left**2 + mu_right**2 + c1) * (var_left + var_right + c2)
    )
    return float(np.mean(score))


def frame_stats(frame: np.ndarray) -> dict:
    return {
        "black_pixel_ratio": float(np.mean(np.max(frame, axis=2) <= 4)),
        "channel_mean": [float(value) for value in frame.mean(axis=(0, 1))],
        "channel_std": [float(value) for value in frame.std(axis=(0, 1))],
        "max": int(frame.max()),
        "min": int(frame.min()),
    }


def main() -> None:
    args = parse_args()
    exact_root = Path(args.exact)
    taehv_root = Path(args.taehv)
    names = sorted(path.name for path in exact_root.glob("*.ppm"))
    if not names or names != sorted(path.name for path in taehv_root.glob("*.ppm")):
        raise RuntimeError("exact and TAEHV sample sets do not match")
    rows = []
    for name in names:
        exact = read_ppm(exact_root / name)
        taehv = read_ppm(taehv_root / name)
        if exact.shape != taehv.shape:
            raise RuntimeError(f"shape mismatch for {name}: {exact.shape} != {taehv.shape}")
        difference = exact.astype(np.float64) - taehv.astype(np.float64)
        mse = float(np.mean(difference**2))
        rows.append(
            {
                "exact": frame_stats(exact),
                "file": name,
                "height": int(exact.shape[0]),
                "max_abs": int(np.max(np.abs(difference))),
                "psnr_db": math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
                "rmse": math.sqrt(mse),
                "ssim": ssim(exact, taehv),
                "taehv": frame_stats(taehv),
                "width": int(exact.shape[1]),
            }
        )
    finite_psnr = [row["psnr_db"] for row in rows if math.isfinite(row["psnr_db"])]
    payload = {
        "approximate_decoder": True,
        "bitwise_parity_expected": False,
        "frames": rows,
        "lpips": {"status": "not_computed", "reason": "lpips package not installed"},
        "mean_psnr_db": float(np.mean(finite_psnr)) if finite_psnr else math.inf,
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "sample_count": len(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
