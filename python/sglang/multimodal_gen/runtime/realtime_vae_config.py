# SPDX-License-Identifier: Apache-2.0

"""Explicit realtime VAE deployment modes."""

from __future__ import annotations

LOCAL_VAE_BACKEND = "local"
EXACT_REMOTE_VAE_BACKEND = "exact_remote"
TAEHV_REMOTE_VAE_BACKEND = "taehv_remote"

REALTIME_VAE_BACKENDS = (
    LOCAL_VAE_BACKEND,
    EXACT_REMOTE_VAE_BACKEND,
    TAEHV_REMOTE_VAE_BACKEND,
)
REALTIME_VAE_TRANSPORTS = ("auto", "websocket", "shared_memory")


def worker_decoder_backend(deployment_backend: str) -> str | None:
    if deployment_backend == LOCAL_VAE_BACKEND:
        return None
    if deployment_backend == EXACT_REMOTE_VAE_BACKEND:
        return "exact"
    if deployment_backend == TAEHV_REMOTE_VAE_BACKEND:
        return "taehv"
    raise ValueError(
        "realtime_vae_backend must be one of: " + ", ".join(REALTIME_VAE_BACKENDS)
    )


def uses_remote_vae(deployment_backend: str) -> bool:
    return worker_decoder_backend(deployment_backend) is not None
