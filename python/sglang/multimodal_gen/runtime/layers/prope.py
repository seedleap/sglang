# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Projective RoPE (PRoPE) transforms for camera-conditioned self-attention.

Numerics must stay aligned with the minWM reference (``wan/modules/prope.py``):
the MinWM causal world model applies these transforms as a second attention
path whose output is fused through a learned zero-init projection, so any
change here breaks cross-stack parity with minWM checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

PropeApplyFn = Callable[[torch.Tensor], torch.Tensor]


def prope_qkv(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4) world-to-camera extrinsics
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3) intrinsics
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PropeApplyFn]:
    """Apply PRoPE projection transforms to q/k/v.

    The sequence length must be divisible by ``cameras`` and token ordering must
    allow reshaping ``(seqlen,)`` into ``(cameras, patches_per_camera)``.

    Returns the transformed ``(q, k, v)`` plus ``apply_fn_o`` which must be
    applied to the attention output (same ``(B, H, L, D)`` layout).
    """
    batch, _, _, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)

    apply_fn_q, apply_fn_kv, apply_fn_o = prope_prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
    )
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o


def prope_prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
) -> tuple[PropeApplyFn, PropeApplyFn, PropeApplyFn]:
    """Prepare transforms for PRoPE-style positional encoding.

    The transforms depend only on the camera parameters, so callers with many
    attention layers can build them once per forward and reuse across layers.
    """
    batch, cameras = viewmats.shape[:2]

    if Ks is not None:
        # Normalize camera intrinsics (zero out the principal point).
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)

        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)
    else:
        # GTA formula. P is a `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
    assert head_dim % 4 == 0

    apply_fn_q = partial(_apply_tiled_projmat, matrix=P_T)
    apply_fn_kv = partial(_apply_tiled_projmat, matrix=P_inv)
    apply_fn_o = partial(_apply_tiled_projmat, matrix=P)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply the projection matrix block-diagonally over 4-dim feature tiles."""
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out.to(dtype=Ks.dtype)


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def expand_camera_params_to_tokens(
    viewmats: torch.Tensor,  # (B, F, 4, 4)
    Ks: torch.Tensor,  # (B, F, 3, 3)
    *,
    frame_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast per-frame camera params to per-token, matching minWM's
    ``CausalWanModel._forward_inference`` expansion (frame-major token order).
    """
    b, f = viewmats.shape[:2]
    viewmats_tok = (
        viewmats[:, :, None]
        .expand(b, f, frame_seqlen, 4, 4)
        .reshape(b, f * frame_seqlen, 4, 4)
    )
    ks_tok = (
        Ks[:, :, None]
        .expand(b, f, frame_seqlen, 3, 3)
        .reshape(b, f * frame_seqlen, 3, 3)
    )
    return viewmats_tok, ks_tok
