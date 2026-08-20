"""Launch Tianpeng-aligned MinWM with a validated hardware profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


TAEHV_SHA256 = "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"
PROFILE_AUTO = "auto"
PROFILE_SM120_32G_SPEED = "sm120-32g-speed"
PROFILE_SM120_HIGHMEM_SPEED = "sm120-highmem-speed"
PROFILE_RESIDENT_SPEED = "resident-speed"
PROFILE_CHOICES = (
    PROFILE_AUTO,
    PROFILE_SM120_32G_SPEED,
    PROFILE_SM120_HIGHMEM_SPEED,
)

COMMON_ENV = {
    "MINWM_ATTENTION_IMPL": "packed",
    "MINWM_PACKED_ATTENTION_DETERMINISTIC": "false",
    "MINWM_NATIVE_COMPONENTS": "",
    "MINWM_SEGMENT_COMPILE": "true",
    "MINWM_CACHE_ROTATED_K": "true",
    "MINWM_PRECOMPUTE_CACHE_ROPE": "true",
    "MINWM_CACHE_PACKED_METADATA": "true",
    "MINWM_RUNTIME_ALIGNMENT_LOG": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "false",
}

COMMON_SERVER_ARGS = (
    ("--pipeline-class-name", "MinWMCausalDMDPipeline"),
    ("--attention-backend", "fa"),
    ("--performance-mode", "speed"),
    ("--num-gpus", "1"),
    ("--tp-size", "1"),
    ("--sp-degree", "1"),
    ("--ulysses-degree", "1"),
    ("--ring-degree", "1"),
    ("--enable-cfg-parallel", "false"),
    ("--enable-torch-compile", "false"),
    ("--enable-cuda-graph", "false"),
    ("--warmup-mode", "off"),
    ("--realtime-causal-sink-size", "8"),
    ("--realtime-causal-kv-cache-num-frames", "32"),
)

PROFILE_SERVER_ARGS = {
    PROFILE_SM120_32G_SPEED: (
        ("--text-encoder-cpu-offload", "true"),
        ("--vae-cpu-offload", "false"),
        ("--dit-cpu-offload", "false"),
        ("--dit-layerwise-offload", "false"),
        ("--pin-cpu-memory", "true"),
    ),
    PROFILE_SM120_HIGHMEM_SPEED: (
        ("--text-encoder-cpu-offload", "false"),
        ("--vae-cpu-offload", "false"),
        ("--dit-cpu-offload", "false"),
        ("--dit-layerwise-offload", "false"),
    ),
    PROFILE_RESIDENT_SPEED: (
        ("--text-encoder-cpu-offload", "false"),
        ("--vae-cpu-offload", "false"),
        ("--dit-cpu-offload", "false"),
        ("--dit-layerwise-offload", "false"),
    ),
}

EXPECTED_MODEL_CONFIG = {
    "local_attn_size": 32,
    "sink_size": 8,
    "sliding_window_num_frames": 32,
    "rope_position_mode": "block_relative",
    "rope_max_frame_gap": 12,
    "prompt_first_frame_pin_enabled": True,
}


@dataclass(frozen=True)
class GPUInfo:
    name: str
    capability: tuple[int, int]
    total_memory_mib: int


def resolve_profile(requested: str, gpu: GPUInfo) -> str:
    if requested not in PROFILE_CHOICES:
        raise ValueError(f"unknown MinWM hardware profile: {requested}")

    major = gpu.capability[0]
    if requested == PROFILE_AUTO:
        if gpu.capability == (9, 0) or major == 10:
            return PROFILE_RESIDENT_SPEED
        if major != 12:
            raise ValueError(
                "unsupported MinWM inference GPU capability: "
                f"{gpu.capability}; expected SM90, SM10x, or SM12x"
            )
        if 28 * 1024 <= gpu.total_memory_mib < 40 * 1024:
            return PROFILE_SM120_32G_SPEED
        if gpu.total_memory_mib >= 64 * 1024:
            return PROFILE_SM120_HIGHMEM_SPEED
        raise ValueError(
            "SM120 GPU memory does not match a validated MinWM profile: "
            f"{gpu.total_memory_mib} MiB"
        )

    if major != 12:
        raise ValueError(
            f"{requested} requires SM120, got compute capability {gpu.capability}"
        )
    if requested == PROFILE_SM120_32G_SPEED and gpu.total_memory_mib < 28 * 1024:
        raise ValueError(
            f"{requested} requires at least 28 GiB, got {gpu.total_memory_mib} MiB"
        )
    if requested == PROFILE_SM120_HIGHMEM_SPEED and gpu.total_memory_mib < 64 * 1024:
        raise ValueError(
            f"{requested} requires at least 64 GiB, got {gpu.total_memory_mib} MiB"
        )
    return requested


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"missing value for {option}")
            return arguments[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def _validate_artifacts(server_args: list[str], taehv_path: Path) -> None:
    if not taehv_path.is_file():
        raise ValueError(f"TAEHV checkpoint is missing: {taehv_path}")
    actual_taehv_sha = _sha256(taehv_path)
    if actual_taehv_sha != TAEHV_SHA256:
        raise ValueError(
            "TAEHV checkpoint SHA-256 mismatch: "
            f"expected {TAEHV_SHA256}, got {actual_taehv_sha}"
        )

    model_path_value = _option_value(server_args, "--model-path")
    if model_path_value is None:
        raise ValueError("profile launcher requires --model-path")
    config_path = Path(model_path_value) / "transformer" / "config.json"
    if not config_path.is_file():
        raise ValueError(f"MinWM transformer config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_MODEL_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "model is not Tianpeng gap12 aligned: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _reject_managed_overrides(server_args: list[str]) -> None:
    managed = {
        option
        for option, _ in COMMON_SERVER_ARGS
        + PROFILE_SERVER_ARGS[PROFILE_SM120_32G_SPEED]
        + PROFILE_SERVER_ARGS[PROFILE_SM120_HIGHMEM_SPEED]
    }
    managed.add("--vae-config.taehv-checkpoint-path")
    conflicts = sorted(
        option
        for option in managed
        if any(arg == option or arg.startswith(f"{option}=") for arg in server_args)
    )
    if conflicts:
        raise ValueError(
            "profile-managed server args cannot be overridden: " + ", ".join(conflicts)
        )


def build_launch_contract(
    *,
    requested_profile: str,
    gpu: GPUInfo,
    taehv_path: Path,
    server_args: list[str],
    validate_artifacts: bool = True,
) -> dict:
    resolved_profile = resolve_profile(requested_profile, gpu)
    _reject_managed_overrides(server_args)
    if validate_artifacts:
        _validate_artifacts(server_args, taehv_path)

    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
    ]
    for option, value in COMMON_SERVER_ARGS:
        command.extend((option, value))
    command.extend(("--vae-config.taehv-checkpoint-path", str(taehv_path)))
    for option, value in PROFILE_SERVER_ARGS[resolved_profile]:
        command.extend((option, value))
    command.extend(server_args)
    return {
        "schema_version": "minwm-hardware-profile/v1",
        "requested_profile": requested_profile,
        "resolved_profile": resolved_profile,
        "gpu": asdict(gpu),
        "environment": dict(COMMON_ENV),
        "command": command,
    }


def _detect_gpu() -> GPUInfo:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("MinWM hardware profiles require exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    return GPUInfo(
        name=torch.cuda.get_device_name(0),
        capability=tuple(torch.cuda.get_device_capability(0)),
        total_memory_mib=int(properties.total_memory // (1024**2)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=PROFILE_AUTO)
    parser.add_argument("--taehv-checkpoint-path", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.server_args
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    contract = build_launch_contract(
        requested_profile=args.profile,
        gpu=_detect_gpu(),
        taehv_path=args.taehv_checkpoint_path,
        server_args=server_args,
    )
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    environment.update(contract["environment"])
    os.execvpe(contract["command"][0], contract["command"], environment)


if __name__ == "__main__":
    main()
