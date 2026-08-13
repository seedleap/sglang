#!/usr/bin/env python3
"""Fail closed before same-GPU exact-VAE overlap can OOM a long request."""

import argparse
import json
import subprocess


def mib(value: str) -> int:
    return int(float(value.strip()))


p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, required=True)
p.add_argument("--denoiser-peak-mib", type=int, required=True)
p.add_argument("--vae-peak-mib", type=int, required=True)
p.add_argument("--transient-mib", type=int, default=516)
p.add_argument("--safety-mib", type=int, default=2048)
p.add_argument("--output", required=True)
a = p.parse_args()

row = (
    subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={a.gpu}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    .strip()
    .split(",")
)
total, free = map(mib, row)
required = a.denoiser_peak_mib + a.vae_peak_mib + a.transient_mib + a.safety_mib
result = {
    "gpu": a.gpu,
    "total_mib": total,
    "free_mib_at_gate": free,
    "denoiser_peak_mib": a.denoiser_peak_mib,
    "vae_peak_mib": a.vae_peak_mib,
    "transient_mib": a.transient_mib,
    "safety_mib": a.safety_mib,
    "required_mib": required,
    "admitted": required <= total,
    "fallback": (
        "serial VAE on the same GPU, or exact-remote VAE on a second GPU; "
        "do not shorten/chunk the 1089-frame request"
    ),
}
open(a.output, "w").write(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["admitted"] else 3)
