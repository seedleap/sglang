# SPDX-License-Identifier: Apache-2.0

"""Strictly local RIFE runtime for the remote realtime VAE worker."""

from __future__ import annotations

import hashlib
import hmac
import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

_RIFE_TRAINING_ONLY_PREFIXES = ("teacher.", "caltime.")


def resolve_rife_weight_file(model_path: str | Path) -> Path:
    """Resolve only an explicit local directory/file; never consult a registry."""

    raw_path = Path(model_path).expanduser()
    if not raw_path.is_absolute():
        raise ValueError("RIFE model path must be an absolute local path")
    resolved = raw_path.resolve(strict=True)
    weight_file = resolved / "flownet.pkl" if resolved.is_dir() else resolved
    if weight_file.name != "flownet.pkl" or not weight_file.is_file():
        raise ValueError(
            "RIFE model path must be flownet.pkl or a directory containing it"
        )
    return weight_file


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rife_weights(
    model_path: str | Path,
    expected_sha256: str,
) -> tuple[Path, str]:
    weight_file, actual, _ = _read_validated_rife_weights(
        model_path,
        expected_sha256,
    )
    return weight_file, actual


def _read_validated_rife_weights(
    model_path: str | Path,
    expected_sha256: str,
) -> tuple[Path, str, bytes]:
    """Read once so the state dict is loaded from the bytes that were hashed."""

    expected = str(expected_sha256 or "").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("RIFE weights SHA256 must contain exactly 64 hex characters")
    weight_file = resolve_rife_weight_file(model_path)
    payload = weight_file.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"RIFE weights SHA256 mismatch for {weight_file}: "
            f"expected {expected}, got {actual}"
        )
    return weight_file, actual, payload


def _load_strict_rife_state(
    model: Any,
    weight_payload: bytes,
    weight_file: Path,
) -> None:
    """Load a complete IFNet state dict or fail before advertising capability.

    The vendored legacy loader intentionally uses ``strict=False`` and filters
    for ``module.`` keys.  That is useful for permissive offline conversion,
    but a realtime capability must not become healthy with an empty or partial
    state dict.  Accept exactly one of the two known layouts: every key has the
    DataParallel prefix, or no key has it.  The official RIFE 4.22.lite
    checkpoint also contains ``teacher.`` and ``caltime.`` training-only
    branches.  Strip only those confirmed auxiliary branches, then require the
    remaining inference state to match IFNet exactly.
    """

    try:
        state = torch.load(
            io.BytesIO(weight_payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"unable to read RIFE state dict: {weight_file}") from exc
    if not isinstance(state, Mapping) or not state:
        raise ValueError("RIFE weights must contain a non-empty state dict")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("RIFE state-dict keys must be strings")

    module_prefixed = [key.startswith("module.") for key in state]
    if any(module_prefixed) and not all(module_prefixed):
        raise ValueError("RIFE state dict cannot mix prefixed and unprefixed keys")
    normalized = {
        (key.removeprefix("module.") if all(module_prefixed) else key): value
        for key, value in state.items()
    }
    inference_state = {
        key: value
        for key, value in normalized.items()
        if not key.startswith(_RIFE_TRAINING_ONLY_PREFIXES)
    }
    try:
        incompatible = model.flownet.load_state_dict(inference_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("RIFE state dict does not exactly match IFNet") from exc
    # ``strict=True`` raises on current PyTorch releases.  Keep this explicit
    # check so a future/custom module cannot silently relax the contract.
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "RIFE state dict is incomplete: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


class RIFE2xMediaProcessor:
    """Preloaded RIFE 4.22.lite realtime interpolation engine.

    The caller supplies a local checkpoint and an exact digest at process
    startup.  No code path in this class downloads weights at runtime.  The
    historical class name remains wire/source compatible, while the engine
    now serves both the exact 2x midpoint profile and the exact 3x profile.
    """

    # Keep the singular attribute for callers that introspect the historical
    # 2x-only processor while advertising the complete negotiated set through
    # ``profiles``.
    profile = "rife2x_v1"
    profiles = (profile, "rife3x_v1")

    def __init__(
        self,
        model_path: str | Path,
        expected_sha256: str,
        *,
        device: str | torch.device,
        max_batch_pairs: int = 16,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        if max_batch_pairs < 1:
            raise ValueError("RIFE max_batch_pairs must be positive")
        (
            self.weight_file,
            self.weights_sha256,
            weight_payload,
        ) = _read_validated_rife_weights(
            model_path,
            expected_sha256,
        )
        self.device = torch.device(device)
        self.max_batch_pairs = max_batch_pairs
        self.ready = False
        if model_factory is None:
            from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
                Model,
            )

            model_factory = Model
        self.model = model_factory()
        if not hasattr(self.model, "flownet"):
            raise ValueError("RIFE model factory must expose flownet")
        _load_strict_rife_state(self.model, weight_payload, self.weight_file)
        self.model.eval()
        self.model.flownet = self.model.flownet.to(self.device)

    @torch.inference_mode()
    def warmup(self, *, height: int = 64, width: int = 64) -> None:
        self.ready = False
        if height < 32 or width < 32:
            raise ValueError("RIFE warmup dimensions must be at least 32x32")
        first = torch.zeros((1, 3, height, width), device=self.device)
        second = torch.ones_like(first)
        self.model.inference(first, second, scale=1.0)
        self.model.inference(
            first.repeat(2, 1, 1, 1),
            second.repeat(2, 1, 1, 1),
            scale=1.0,
            timestep=torch.tensor(
                (1.0 / 3.0, 2.0 / 3.0),
                device=self.device,
                dtype=first.dtype,
            )[:, None, None, None],
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.ready = True

    @torch.inference_mode()
    def interpolate_midpoints(self, source_frames: torch.Tensor) -> torch.Tensor:
        """Return one midpoint for every adjacent BCTHW source-frame pair."""

        intermediates = self.interpolate_intermediates(source_frames, multiplier=2)
        return intermediates[:, :, :, 0].contiguous()

    @torch.inference_mode()
    def interpolate_intermediates(
        self,
        source_frames: torch.Tensor,
        *,
        multiplier: int,
    ) -> torch.Tensor:
        """Return ordered intermediate frames for every adjacent source pair.

        The result is ``BCPKHW`` where ``P`` is the adjacent-pair count and
        ``K = multiplier - 1``.  For 3x, each pair is evaluated directly at
        timesteps 1/3 and 2/3.  Direct arbitrary-time inference avoids the
        recursive-quarter semantics of RIFE 4x and gives the wire profile an
        exact, uniform 3x cadence.
        """

        if not isinstance(source_frames, torch.Tensor) or source_frames.ndim != 5:
            raise ValueError("RIFE source frames must be a BCTHW tensor")
        if source_frames.shape[0] != 1 or source_frames.shape[1] < 3:
            raise ValueError("RIFE source frames require one RGB sample")
        if multiplier not in (2, 3):
            raise ValueError("realtime RIFE multiplier must be 2 or 3")
        intermediate_count = multiplier - 1
        pair_count = int(source_frames.shape[2]) - 1
        if pair_count < 1:
            height, width = source_frames.shape[-2:]
            return torch.empty(
                (1, 3, 0, intermediate_count, height, width),
                dtype=torch.float32,
            )

        source = (
            source_frames[0, :3]
            .permute(1, 0, 2, 3)
            .contiguous()
            .to(device=self.device, dtype=torch.float32, non_blocking=True)
        )
        # Keep the number of inference samples bounded by the existing batch
        # gate.  A 3x pair contributes two arbitrary-time samples.
        pairs_per_batch = max(1, self.max_batch_pairs // intermediate_count)
        intermediate_groups: list[torch.Tensor] = []
        for start in range(0, pair_count, pairs_per_batch):
            end = min(start + pairs_per_batch, pair_count)
            left = source[start:end]
            right = source[start + 1 : end + 1]
            group_pairs = end - start
            if multiplier == 2:
                inferred = self.model.inference(left, right, scale=1.0)
            else:
                left = left.repeat_interleave(intermediate_count, dim=0)
                right = right.repeat_interleave(intermediate_count, dim=0)
                timestep = torch.tensor(
                    (1.0 / 3.0, 2.0 / 3.0),
                    device=self.device,
                    dtype=left.dtype,
                ).repeat(group_pairs)
                inferred = self.model.inference(
                    left,
                    right,
                    scale=1.0,
                    timestep=timestep[:, None, None, None],
                )
            intermediate_groups.append(
                inferred.detach().reshape(
                    group_pairs,
                    intermediate_count,
                    3,
                    int(source.shape[-2]),
                    int(source.shape[-1]),
                )
            )
        intermediates = torch.cat(intermediate_groups, dim=0)
        return (
            intermediates.permute(2, 0, 1, 3, 4)
            .unsqueeze(0)
            .contiguous()
            .to(device="cpu", dtype=torch.float32)
            .clamp_(0.0, 1.0)
        )
