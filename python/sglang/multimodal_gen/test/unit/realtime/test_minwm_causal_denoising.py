# SPDX-License-Identifier: Apache-2.0

"""MinWM numerics tests against verbatim minWM reference implementations.

The reference functions in this file are copied verbatim from the minWM repo
(wan/modules/prope.py, wan/modules/causal_model.py, wan_utils/camera_trajectory.py)
and serve as the ground truth for cross-stack parity: every ported component
must match them exactly (zero tolerance) on CPU.
"""

import math
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.minwm_rope import (
    minwm_cache_rope_frame_indices,
    minwm_query_rope_frame_indices,
    minwm_rope_apply,
    minwm_rope_params,
)
from sglang.multimodal_gen.runtime.layers.prope import (
    expand_camera_params_to_tokens,
    prope_qkv,
)
from sglang.multimodal_gen.runtime.utils.minwm_camera import (
    MOTION_PRIMITIVES,
    ROTATION_STEP_RAD,
    TRANSLATION_STEP,
    advance_camera_chunk,
    keys_to_motion,
    step_c2w,
)

# ---------------------------------------------------------------------------
# Verbatim minWM reference: prope.py
# ---------------------------------------------------------------------------


def _ref_invert_SE3(transforms):
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _ref_lift_K(Ks):
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out.to(dtype=Ks.dtype)


def _ref_invert_K(Ks):
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def _ref_apply_tiled_projmat(feats, matrix):
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _ref_prope_qkv(q, k, v, *, viewmats, Ks):
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0]
    Ks_norm[..., 1, 1] = Ks[..., 1, 1]
    Ks_norm[..., 2, 2] = 1.0
    Ks_norm = Ks_norm.to(dtype=Ks.dtype)
    P = torch.einsum("...ij,...jk->...ik", _ref_lift_K(Ks_norm), viewmats)
    P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        _ref_invert_SE3(viewmats),
        _ref_lift_K(_ref_invert_K(Ks_norm)),
    ).to(dtype=viewmats.dtype)
    fq = partial(_ref_apply_tiled_projmat, matrix=P_T)
    fkv = partial(_ref_apply_tiled_projmat, matrix=P_inv)
    fo = partial(_ref_apply_tiled_projmat, matrix=P)
    return fq(q), fkv(k), fkv(v), fo


# ---------------------------------------------------------------------------
# Verbatim minWM reference: rope (model.py rope_params + causal_rope_apply)
# ---------------------------------------------------------------------------


def _ref_rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _ref_causal_rope_apply(
    x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None
):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        if relative_frame_indices is not None:
            frame_indices = relative_frame_indices.long()
            freqs_temporal = (
                freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
            )
        else:
            freqs_temporal = (
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1)
            )
        freqs_i = torch.cat(
            [
                freqs_temporal,
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ---------------------------------------------------------------------------
# Verbatim minWM reference: KV cache semantics
# (causal_model.py prepare_cache_attention + _apply_cache_updates, standalone)
# ---------------------------------------------------------------------------


class _RefMinWMCache:
    def __init__(self, batch, cache_tokens, heads, dim, sink_tokens):
        self.cache = {
            "k": torch.zeros(batch, cache_tokens, heads, dim),
            "v": torch.zeros(batch, cache_tokens, heads, dim),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
        self.sink_tokens = sink_tokens
        self.max_attention_size = cache_tokens

    def _window(self, cache_k, cache_v, local_end_index):
        sink_tokens = self.sink_tokens
        max_attention_size = self.max_attention_size
        if sink_tokens > 0 and local_end_index > max_attention_size:
            tail_tokens = max_attention_size - sink_tokens
            return (
                torch.cat(
                    [
                        cache_k[:, :sink_tokens],
                        cache_k[:, local_end_index - tail_tokens : local_end_index],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        cache_v[:, :sink_tokens],
                        cache_v[:, local_end_index - tail_tokens : local_end_index],
                    ],
                    dim=1,
                ),
            )
        start = max(0, local_end_index - max_attention_size)
        return cache_k[:, start:local_end_index], cache_v[:, start:local_end_index]

    def forward(self, new_k, new_v, current_start, allow_sink_write_on_recache=False):
        cache = self.cache
        sink_tokens = self.sink_tokens
        current_end = current_start + new_k.shape[1]
        cache_size = cache["k"].shape[1]
        num_new_tokens = new_k.shape[1]
        global_end_index = cache["global_end_index"].item()
        previous_local_end_index = cache["local_end_index"].item()
        is_recompute = current_end <= global_end_index and current_start > 0

        if (current_end > global_end_index) and (
            num_new_tokens + previous_local_end_index > cache_size
        ):
            num_evicted_tokens = num_new_tokens + previous_local_end_index - cache_size
            num_rolled_tokens = (
                previous_local_end_index - num_evicted_tokens - sink_tokens
            )
            temp_k = cache["k"].clone()
            temp_v = cache["v"].clone()
            temp_k[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_k[
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            temp_v[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_v[
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            local_end_index = (
                previous_local_end_index
                + current_end
                - global_end_index
                - num_evicted_tokens
            )
            roll_info = (num_evicted_tokens, num_rolled_tokens)
        else:
            temp_k = cache["k"].clone()
            temp_v = cache["v"].clone()
            local_end_index = previous_local_end_index + current_end - global_end_index
            roll_info = None

        local_start_index = local_end_index - num_new_tokens
        write_start_index = (
            max(local_start_index, sink_tokens) if is_recompute else local_start_index
        )
        if allow_sink_write_on_recache:
            write_start_index = local_start_index

        new_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = new_k[
                :, new_offset : new_offset + write_len
            ]
            temp_v[:, write_start_index:local_end_index] = new_v[
                :, new_offset : new_offset + write_len
            ]

        attn_k, attn_v = self._window(temp_k, temp_v, local_end_index)

        # _apply_cache_updates (immediate commit form)
        if roll_info is not None:
            num_evicted_tokens, num_rolled_tokens = roll_info
            cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["k"][
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["v"][
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        if write_len > 0:
            cache["k"][:, write_start_index:local_end_index] = new_k[
                :, new_offset : new_offset + write_len
            ]
            cache["v"][:, write_start_index:local_end_index] = new_v[
                :, new_offset : new_offset + write_len
            ]
        if not is_recompute:
            cache["global_end_index"].fill_(current_end)
            cache["local_end_index"].fill_(local_end_index)
        return attn_k, attn_v, local_start_index, local_end_index


def _make_sglang_cache(batch, cache_tokens, heads, dim, sink_tokens):
    return CausalSelfAttentionKVCache(
        k=torch.zeros(batch, cache_tokens, heads, dim),
        v=torch.zeros(batch, cache_tokens, heads, dim),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=cache_tokens,
        sink_tokens=sink_tokens,
        attention_window_size=cache_tokens,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _rand_se3(n, generator):
    mats = []
    for _ in range(n):
        theta = torch.rand(1, generator=generator).item()
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        mat = torch.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = torch.randn(3, generator=generator) * 0.1
        mats.append(mat)
    return torch.stack(mats)


def test_prope_matches_minwm_reference_exactly():
    g = torch.Generator().manual_seed(0)
    batch, heads, frames, frame_seqlen, head_dim = 2, 3, 5, 4, 16
    seqlen = frames * frame_seqlen
    q = torch.randn(batch, heads, seqlen, head_dim, generator=g)
    k = torch.randn(batch, heads, seqlen, head_dim, generator=g)
    v = torch.randn(batch, heads, seqlen, head_dim, generator=g)

    vm_frames = _rand_se3(frames, g)
    ks_frame = torch.tensor([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    vm_tok, ks_tok = expand_camera_params_to_tokens(
        vm_frames[None].expand(batch, frames, 4, 4),
        ks_frame[None, None].expand(batch, frames, 3, 3),
        frame_seqlen=frame_seqlen,
    )
    vm_tok = vm_tok.contiguous()
    ks_tok = ks_tok.contiguous()

    q1, k1, v1, fo1 = prope_qkv(q, k, v, viewmats=vm_tok, Ks=ks_tok)
    q2, k2, v2, fo2 = _ref_prope_qkv(q, k, v, viewmats=vm_tok, Ks=ks_tok)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)
    out = torch.randn(batch, heads, seqlen, head_dim, generator=g)
    assert torch.equal(fo1(out), fo2(out))


def test_minwm_rope_matches_reference_exactly():
    head_dim = 24
    freqs_dims = [head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)]
    ref_freqs = torch.cat([_ref_rope_params(64, d) for d in freqs_dims], dim=1)
    freqs = torch.cat([minwm_rope_params(64, d) for d in freqs_dims], dim=1)
    assert torch.equal(torch.view_as_real(ref_freqs), torch.view_as_real(freqs))

    g = torch.Generator().manual_seed(1)
    frames, height, width, heads = 4, 2, 3, 2
    x = torch.randn(
        2, frames * height * width, heads, head_dim, generator=g
    ).to(torch.bfloat16)
    grid = torch.tensor([[frames, height, width]] * 2)

    for start in (0, 3, 17):
        ref = _ref_causal_rope_apply(x, grid, ref_freqs, start_frame=start)
        mine = minwm_rope_apply(
            x, torch.arange(start, start + frames), height, width, freqs
        )
        assert torch.equal(ref, mine)

    rel = torch.tensor([0, 1, 5, 9])
    ref = _ref_causal_rope_apply(x, grid, ref_freqs, relative_frame_indices=rel)
    mine = minwm_rope_apply(x, rel, height, width, freqs)
    assert torch.equal(ref, mine)


def test_block_relative_indices_saturated_window():
    # cache == window == local_attn: block_relative reduces to buffer-slot
    # positions with the query at the window tail.
    idx = minwm_cache_rope_frame_indices(
        num_cache_frames=16,
        global_end_frame=40,
        sink_size=4,
        local_attn_size=16,
        rope_position_mode="block_relative",
        device=torch.device("cpu"),
    )
    assert torch.equal(idx, torch.arange(0, 16))

    qidx = minwm_query_rope_frame_indices(
        local_start_frame=12,
        local_end_frame=16,
        num_new_frames=4,
        current_start_frame=36,
        local_attn_size=16,
        rope_position_mode="block_relative",
        device=torch.device("cpu"),
    )
    assert torch.equal(qidx, torch.arange(12, 16))


def test_block_relative_indices_oversized_cache_remap():
    # Cache larger than the attention window: sink keeps [0, sink), tail maps
    # to [sink, local_attn).
    idx = minwm_cache_rope_frame_indices(
        num_cache_frames=20,
        global_end_frame=20,
        sink_size=4,
        local_attn_size=16,
        rope_position_mode="block_relative",
        device=torch.device("cpu"),
    )
    assert torch.equal(idx[:4], torch.arange(0, 4))
    assert torch.equal(idx[8:], torch.arange(4, 16))


def test_kv_cache_session_matches_minwm_reference():
    batch, heads, dim = 1, 2, 8
    frame_seqlen = 4
    block_frames = 2
    window_frames = 8
    sink_tokens = 2 * frame_seqlen
    cache_tokens = window_frames * frame_seqlen

    for allow_sink_write in (False, True):
        ref = _RefMinWMCache(batch, cache_tokens, heads, dim, sink_tokens)
        mine = _make_sglang_cache(batch, cache_tokens, heads, dim, sink_tokens)
        g = torch.Generator().manual_seed(7)

        num_blocks = 7  # 14 frames > 8-frame window: rolling active
        for blk in range(num_blocks):
            cur_start = blk * block_frames * frame_seqlen
            for _step in range(5):  # 4 denoise passes + 1 clean-context refill
                new_k = torch.randn(
                    batch, block_frames * frame_seqlen, heads, dim, generator=g
                )
                new_v = torch.randn(
                    batch, block_frames * frame_seqlen, heads, dim, generator=g
                )
                ref_k, ref_v, ref_ls, ref_le = ref.forward(new_k, new_v, cur_start)
                view = mine.update_and_get_attention_kv(
                    key=new_k,
                    value=new_v,
                    current_chunk_start=cur_start,
                    sink_protected_rewrite=True,
                )
                assert torch.equal(ref_k, view.k)
                assert torch.equal(ref_v, view.v)
                assert ref_ls == view.local_start_index
                assert ref_le == view.visible_local_end
            assert torch.equal(ref.cache["k"], mine.k)
            assert ref.cache["global_end_index"].item() == mine.global_end_index.item()
            assert ref.cache["local_end_index"].item() == mine.local_end_index.item()

        # Prompt-switch recache: replay the last window in one forward.
        total_frames = num_blocks * block_frames
        recache_frames = min(window_frames, total_frames)
        recache_start = (total_frames - recache_frames) * frame_seqlen
        new_k = torch.randn(
            batch, recache_frames * frame_seqlen, heads, dim, generator=g
        )
        new_v = torch.randn(
            batch, recache_frames * frame_seqlen, heads, dim, generator=g
        )
        ref_k, _, _, _ = ref.forward(
            new_k, new_v, recache_start, allow_sink_write_on_recache=allow_sink_write
        )
        view = mine.update_and_get_attention_kv(
            key=new_k,
            value=new_v,
            current_chunk_start=recache_start,
            sink_protected_rewrite=not allow_sink_write,
        )
        assert torch.equal(ref_k, view.k)
        assert torch.equal(ref.cache["k"], mine.k)
        assert torch.equal(ref.cache["v"], mine.v)
        assert ref.cache["global_end_index"].item() == mine.global_end_index.item()

        # Generation continues normally after the recache.
        cur_start = total_frames * frame_seqlen
        new_k = torch.randn(batch, block_frames * frame_seqlen, heads, dim, generator=g)
        new_v = torch.randn(batch, block_frames * frame_seqlen, heads, dim, generator=g)
        ref_k, _, _, _ = ref.forward(new_k, new_v, cur_start)
        view = mine.update_and_get_attention_kv(
            key=new_k,
            value=new_v,
            current_chunk_start=cur_start,
            sink_protected_rewrite=True,
        )
        assert torch.equal(ref_k, view.k)
        assert torch.equal(ref.cache["k"], mine.k)


def test_minwm_camera_constants_and_step_c2w():
    assert TRANSLATION_STEP == 0.08
    assert ROTATION_STEP_RAD == np.radians(3.0)
    assert MOTION_PRIMITIVES["w"] == {"forward": 0.08}
    assert MOTION_PRIMITIVES["j"] == {"yaw": -ROTATION_STEP_RAD}

    # Opposing keys cancel.
    assert keys_to_motion(["w", "s"]) == {"forward": 0.0}
    # Unknown keys ignored.
    assert keys_to_motion(["x"]) == {}

    # Forward dolly: 3 frames of "w" from identity.
    new_c2w, viewmats, ks = advance_camera_chunk(
        np.eye(4),
        [["w"], ["w"], ["w"]],
        intrinsics=(0.5, 0.5, 0.5, 0.5),
        device="cpu",
        dtype=torch.float32,
    )
    # Frame 0 pose is identity (pose before first motion applies).
    assert torch.allclose(viewmats[0, 0], torch.eye(4))
    # Frame 2 pose has translated 2 steps forward (+Z).
    assert abs(viewmats[0, 2][2, 3].item() + 2 * TRANSLATION_STEP) < 1e-6
    # Cumulative pose advanced 3 steps.
    assert abs(new_c2w[2, 3] - 3 * TRANSLATION_STEP) < 1e-9
    assert ks.shape == (1, 3, 3, 3)
    assert ks[0, 0, 0, 0].item() == 0.5

    # step_c2w matches manual composition for a yaw+forward combo.
    motion = keys_to_motion(["w", "l"])
    T, poses = step_c2w(np.eye(4), [motion])
    assert len(poses) == 1
    # Yaw applied before translation: forward is along the rotated +Z.
    expected_forward = T[:3, :3] @ np.array([0, 0, TRANSLATION_STEP])
    np.testing.assert_allclose(T[:3, 3], expected_forward, atol=1e-12)


def test_minwm_stage_cache_geometry():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising import (
        MinWMCausalDMDDenoisingStage,
    )

    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=4,
                sliding_window_num_frames=16,
                local_attn_size=16,
            )
        )
    )
    stage.local_attn_size = -1  # stage-level fallback; window drives sizing
    stage.sink_size = 4
    stage.sliding_window_num_frames = 16
    stage.num_token_per_frame = 1560

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=None,
            realtime_causal_kv_cache_num_frames=None,
        )
    )
    stage._apply_causal_cache_overrides(SimpleNamespace(), server_args)
    assert stage._get_causal_kv_cache_size() == 16 * 1560
    assert stage._get_causal_sink_tokens() == 4 * 1560

    # Request-level override wins and does not leak (defaults restored on the
    # next call).
    batch = SimpleNamespace(
        realtime_causal_sink_size=2,
        realtime_causal_kv_cache_num_frames=24,
    )
    stage._apply_causal_cache_overrides(batch, server_args)
    assert stage._get_causal_sink_tokens() == 2 * 1560
    assert stage._get_causal_kv_cache_size() == 24 * 1560

    stage._apply_causal_cache_overrides(SimpleNamespace(), server_args)
    assert stage._get_causal_sink_tokens() == 4 * 1560
    assert stage._get_causal_kv_cache_size() == 16 * 1560
