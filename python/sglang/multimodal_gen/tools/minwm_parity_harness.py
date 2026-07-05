# SPDX-License-Identifier: Apache-2.0
"""Cross-stack parity harness: SGLang MinWM DiT vs the minWM reference model.

Loads BOTH implementations in one process, syncs weights (via
``param_names_mapping``), and drives the exact minWM block loop — 4-step DMD
denoising with shared noise, clean-context KV refill, optional prompt-switch
recache — comparing the raw model outputs (flow predictions) after every
forward and the clean latents after every block.

Modes:
  tiny        random-weight small model, runs on CPU (SDPA shim), fp32.
              Gate: maxabs < 5e-4 per forward.
  checkpoint  real stage4 checkpoint + 1.3B config on GPU, bf16.
              Gate: maxabs < 1e-2 (minWM parity_harness tolerance).

Examples:

  # CPU tiny-model parity (no checkpoint needed)
  python -m sglang.multimodal_gen.tools.minwm_parity_harness \
      --minwm-repo ~/workspace/minWM/.../Wan21 --mode tiny --blocks 6 \
      --prompt-switch-block 4

  # GPU checkpoint parity
  python -m sglang.multimodal_gen.tools.minwm_parity_harness \
      --minwm-repo /work/minWM/Wan21 --mode checkpoint \
      --checkpoint /ckpt/model.pt --device cuda --dtype bf16 --blocks 10

  # Camera-builder probe (my advance_camera_chunk vs minWM parse_trajectory)
  python -m sglang.multimodal_gen.tools.minwm_parity_harness \
      --minwm-repo /work/minWM/Wan21 --camera-probe
"""

from __future__ import annotations

import argparse
import re
import sys

import torch


def _cpu_sdpa_shim(q, k, v, *args, **kwargs):
    """SDPA in the input dtype, [B, L, H, D] layout (minWM fallback semantics,
    minus the hardcoded bf16 cast which breaks fp32 CPU parity runs)."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


def _import_minwm(minwm_repo: str, patch_cpu_attention: bool):
    sys.path.insert(0, minwm_repo)
    import wan.modules.causal_model as causal_model_mod  # noqa: PLC0415
    import wan.modules.model as model_mod  # noqa: PLC0415
    from wan.modules.causal_model import CausalWanModel  # noqa: PLC0415
    from wan.modules.prope import add_prope_parameters  # noqa: PLC0415
    from wan_utils import camera_trajectory  # noqa: PLC0415

    if patch_cpu_attention:
        # minWM's attention() casts to bf16 and flash_attention() is CUDA-only;
        # neither works for fp32 CPU parity runs.
        causal_model_mod.attention = _cpu_sdpa_shim
        model_mod.flash_attention = _cpu_sdpa_shim

    return CausalWanModel, add_prope_parameters, camera_trajectory


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

TINY = dict(
    dim=64,
    ffn_dim=128,
    num_heads=4,
    num_layers=2,
    text_len=16,
    text_dim=32,
    freq_dim=32,
    in_dim=16,
    out_dim=16,
    height=32,  # pixel-equivalent: latent 4x6 -> post-patch 2x3
    width=48,
    local_attn_size=8,
    sink_size=2,
    num_frame_per_block=2,
)

FULL = dict(
    dim=1536,
    ffn_dim=8960,
    num_heads=12,
    num_layers=30,
    text_len=512,
    text_dim=4096,
    freq_dim=256,
    in_dim=16,
    out_dim=16,
    height=480,
    width=832,
    local_attn_size=16,
    sink_size=4,
    num_frame_per_block=4,
)


def build_minwm_model(CausalWanModel, add_prope_parameters, cfg: dict, seed: int):
    from omegaconf import OmegaConf

    top = OmegaConf.create(
        {
            "action_config": None,
            "deterministic_attention": False,
            "generator_config": {
                "model_type": "t2v",
                "patch_size": [1, 2, 2],
                "text_len": cfg["text_len"],
                "in_dim": cfg["in_dim"],
                "out_dim": cfg["out_dim"],
                "dim": cfg["dim"],
                "num_heads": cfg["num_heads"],
                "num_layers": cfg["num_layers"],
                "ffn_dim": cfg["ffn_dim"],
                "freq_dim": cfg["freq_dim"],
                "text_dim": cfg["text_dim"],
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
                "rope_max_seq_len": 1024,
                "local_attn_size": cfg["local_attn_size"],
                "sink_size": cfg["sink_size"],
                "rope_position_mode": "block_relative",
            },
        }
    )
    torch.manual_seed(seed)
    model = CausalWanModel(top)
    add_prope_parameters(model, zero_init=True)
    model.num_frame_per_block = cfg["num_frame_per_block"]
    # Randomize the zero-initialized layers so the comparison is not vacuous:
    # minWM init_weights() zeroes head.head (making random-init flow outputs
    # identically zero) and add_prope_parameters zeroes prope_o (masking the
    # PRoPE fusion path). A trained checkpoint has both non-zero.
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.head.head.weight.normal_(0, 0.02, generator=g)
        model.head.head.bias.normal_(0, 0.02, generator=g)
        for _name, module in model.named_modules():
            if hasattr(module, "prope_o"):
                module.prope_o.weight.normal_(0, 0.02, generator=g)
                module.prope_o.bias.normal_(0, 0.02, generator=g)
    return model.eval()


def build_sglang_model(cfg: dict):
    import os

    from sglang.multimodal_gen.configs.models.dits.minwm import (
        MinWMVideoArchConfig,
        MinWMVideoConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.dits.minwm import (
        MinWMCausalTransformer3DModel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        global_force_attn_backend,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if not torch.cuda.is_available():
        # Same math as minWM's non-flash SDPA fallback.
        global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
        # @torch.compile'd layer ops need triton; run them eagerly off-GPU.
        from torch import _dynamo as _torch_dynamo

        _torch_dynamo.config.disable = True

    arch = MinWMVideoArchConfig(
        num_attention_heads=cfg["num_heads"],
        attention_head_dim=cfg["dim"] // cfg["num_heads"],
        in_channels=cfg["in_dim"],
        out_channels=cfg["out_dim"],
        text_dim=cfg["text_dim"],
        text_len=cfg["text_len"],
        freq_dim=cfg["freq_dim"],
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        local_attn_size=cfg["local_attn_size"],
        sink_size=cfg["sink_size"],
        num_frames_per_block=cfg["num_frame_per_block"],
        sliding_window_num_frames=cfg["local_attn_size"],
    )
    config = MinWMVideoConfig(arch_config=arch)
    model = MinWMCausalTransformer3DModel(config=config, hf_config={})
    return model.eval()


def sync_weights(minwm_model, sglang_model) -> None:
    """minWM state dict -> SGLang module names via param_names_mapping."""
    mapping = sglang_model.param_names_mapping
    compiled = [(re.compile(pat), repl) for pat, repl in mapping.items()]
    converted = {}
    for name, tensor in minwm_model.state_dict().items():
        new_name = name
        for pat, repl in compiled:
            if pat.match(name):
                new_name = pat.sub(repl, name)
                break
        converted[new_name] = tensor
    missing, unexpected = sglang_model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"weight sync mismatch:\n  missing: {missing}\n  unexpected: {unexpected}"
        )


# ---------------------------------------------------------------------------
# Cache allocation
# ---------------------------------------------------------------------------


def make_minwm_caches(cfg, batch, frame_seqlen, dtype, device):
    heads, head_dim = cfg["num_heads"], cfg["dim"] // cfg["num_heads"]
    cache_tokens = cfg["local_attn_size"] * frame_seqlen

    def kv_list():
        return [
            {
                "k": torch.zeros(batch, cache_tokens, heads, head_dim, dtype=dtype, device=device),
                "v": torch.zeros(batch, cache_tokens, heads, head_dim, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(cfg["num_layers"])
        ]

    crossattn = [
        {
            "k": torch.zeros(batch, cfg["text_len"], heads, head_dim, dtype=dtype, device=device),
            "v": torch.zeros(batch, cfg["text_len"], heads, head_dim, dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(cfg["num_layers"])
    ]
    return kv_list(), crossattn, kv_list()


def make_sglang_caches(cfg, batch, frame_seqlen, dtype, device):
    from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
        CausalSelfAttentionKVCache,
        CrossAttentionKVCache,
    )

    heads, head_dim = cfg["num_heads"], cfg["dim"] // cfg["num_heads"]
    cache_tokens = cfg["local_attn_size"] * frame_seqlen
    sink_tokens = cfg["sink_size"] * frame_seqlen

    def kv_list():
        return [
            CausalSelfAttentionKVCache(
                k=torch.zeros(batch, cache_tokens, heads, head_dim, dtype=dtype, device=device),
                v=torch.zeros(batch, cache_tokens, heads, head_dim, dtype=dtype, device=device),
                global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                local_end_index=torch.zeros(1, dtype=torch.long, device=device),
                cache_size=cache_tokens,
                sink_tokens=sink_tokens,
                attention_window_size=cache_tokens,
            )
            for _ in range(cfg["num_layers"])
        ]

    crossattn = [
        CrossAttentionKVCache(
            k=torch.zeros(batch, cfg["text_len"], heads, head_dim, dtype=dtype, device=device),
            v=torch.zeros(batch, cfg["text_len"], heads, head_dim, dtype=dtype, device=device),
        )
        for _ in range(cfg["num_layers"])
    ]
    return kv_list(), crossattn, kv_list()


# ---------------------------------------------------------------------------
# Shared scheduler math (Self-Forcing FlowMatchScheduler, shift warp)
# ---------------------------------------------------------------------------


def warped_timesteps(steps, shift=5.0, num_train=1000):
    sigmas = torch.linspace(1.0, 0.0, num_train + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    ts = sigmas * num_train
    table = torch.cat([ts, torch.tensor([0.0])])
    warped = table[num_train - torch.tensor(steps, dtype=torch.long)]
    return warped, sigmas, ts


def sigma_for(timestep_value, sigmas, ts):
    tid = torch.argmin((ts - timestep_value).abs())
    return sigmas[tid].item()


# ---------------------------------------------------------------------------
# Parity run
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_parity(args):
    CausalWanModel, add_prope_parameters, camera_trajectory = _import_minwm(
        args.minwm_repo, patch_cpu_attention=not torch.cuda.is_available()
    )
    cfg = TINY if args.mode == "tiny" else FULL
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    minwm_model = build_minwm_model(CausalWanModel, add_prope_parameters, cfg, seed=0)
    sglang_model = build_sglang_model(cfg)

    if args.mode == "checkpoint":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for key in ("generator_ema", "generator", "model"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        state = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in state.items()
        }
        minwm_model.load_state_dict(state)

    sync_weights(minwm_model, sglang_model)
    minwm_model = minwm_model.to(device=device, dtype=dtype)
    sglang_model = sglang_model.to(device=device, dtype=dtype)

    fpb = cfg["num_frame_per_block"]
    lat_h = cfg["height"] // 8
    lat_w = cfg["width"] // 8
    frame_seqlen = (lat_h // 2) * (lat_w // 2)
    seq_len = fpb * frame_seqlen
    batch = 1

    m_kv, m_x, m_prope = make_minwm_caches(cfg, batch, frame_seqlen, dtype, device)
    s_kv, s_x, s_prope = make_sglang_caches(cfg, batch, frame_seqlen, dtype, device)

    g = torch.Generator().manual_seed(args.seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    def make_text():
        n_tok = max(4, cfg["text_len"] // 2)
        text = rand(n_tok, cfg["text_dim"])
        padded = torch.cat(
            [text, text.new_zeros(cfg["text_len"] - n_tok, cfg["text_dim"])]
        )[None]
        return text, padded

    text_unpadded, text_padded = make_text()

    total_frames = args.blocks * fpb
    traj = camera_trajectory.make_camera_tensors(
        f"w*{max(1, total_frames // 2)},l*{total_frames - max(1, total_frames // 2) - 1}",
        fx=0.5, fy=0.5, cx=0.5, cy=0.5,
    )
    viewmats_all = traj[0].to(device=device, dtype=dtype)
    ks_all = traj[1].to(device=device, dtype=dtype)
    assert viewmats_all.shape[1] >= total_frames

    steps, sigmas, ts_grid = warped_timesteps([1000, 750, 500, 250])
    print(f"warped timesteps: {[round(float(t), 2) for t in steps]}")

    tol = args.tolerance
    worst = 0.0
    failures = 0
    output_history_m = []
    output_history_s = []
    vm_hist = []
    ks_hist = []

    def minwm_forward(x_bfchw, t_value, text, cur_start_frame, vm, ks, allow_sink=False):
        t = torch.full((batch, x_bfchw.shape[1]), 0.0, device=device, dtype=torch.float32) + t_value
        out = minwm_model(
            [x_bfchw[0].permute(1, 0, 2, 3)],  # [C, F, H, W]
            t=t,
            context=[text],
            seq_len=seq_len if x_bfchw.shape[1] == fpb else x_bfchw.shape[1] * frame_seqlen,
            kv_cache=m_kv,
            crossattn_cache=m_x,
            current_start=cur_start_frame * frame_seqlen,
            viewmats=vm,
            Ks=ks,
            prope_kv_cache=m_prope,
            allow_sink_write_on_recache=allow_sink,
        )
        return out  # [B, C, F, H, W]

    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    def sglang_forward(x_bcfhw, t_value, cur_start_frame, vm, ks, allow_sink=False):
        t = torch.full((batch, 1), 0.0, device=device, dtype=torch.float32) + t_value
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=None
        ):
            return sglang_model(
                x_bcfhw,
                [text_padded],
                t,
                kv_cache=s_kv,
                crossattn_cache=s_x,
                current_start=cur_start_frame * frame_seqlen,
                start_frame=cur_start_frame,
                viewmats=vm,
                Ks=ks,
                prope_kv_cache=s_prope if vm is not None else None,
                sink_protected_rewrite=not allow_sink,
            )

    def compare(tag, a, b):
        nonlocal worst, failures
        diff = (a.float() - b.float()).abs().max().item()
        worst = max(worst, diff)
        status = "ok" if diff < tol else "FAIL"
        if diff >= tol:
            failures += 1
        print(f"  {tag}: maxabs={diff:.3e} [{status}]")

    switch_done = False
    for blk in range(args.blocks):
        cur = blk * fpb
        vm = None if args.no_camera else viewmats_all[:, cur : cur + fpb]
        ks = None if args.no_camera else ks_all[:, cur : cur + fpb]

        # optional prompt switch with LongLive recache before this block
        if args.prompt_switch_block == blk and cur > 0 and not switch_done:
            switch_done = True
            new_text_unpadded, new_text_padded = make_text()
            # switch the active prompt BEFORE the recache forwards: the recache
            # must replay the window under the NEW prompt on both stacks.
            text_unpadded, text_padded = new_text_unpadded, new_text_padded
            n_rec = min(cfg["local_attn_size"], cur)
            rec_start = cur - n_rec
            frames_m = torch.cat(output_history_m, dim=1)[:, -n_rec:]  # [B,F,C,H,W]
            frames_s = torch.cat(output_history_s, dim=2)[:, :, -n_rec:]  # [B,C,F,H,W]
            if vm_hist:
                vm_rec = torch.cat(vm_hist, dim=1)[:, -n_rec:]
                ks_rec = torch.cat(ks_hist, dim=1)[:, -n_rec:]
            else:
                vm_rec = ks_rec = None
            for cache in m_x:
                cache["k"].zero_(); cache["v"].zero_(); cache["is_init"] = False
            for cache in s_x:
                cache.reset()
            minwm_forward(frames_m, 0.0, new_text_unpadded, rec_start, vm_rec, ks_rec)
            sglang_forward(frames_s, 0.0, rec_start, vm_rec, ks_rec)
            for cache in m_x:
                cache["k"].zero_(); cache["v"].zero_(); cache["is_init"] = False
            for cache in s_x:
                cache.reset()
            compare("post-recache kv k", m_kv[0]["k"], s_kv[0].k)
            compare("post-recache kv v", m_kv[0]["v"], s_kv[0].v)
            print(f"block {blk}: prompt switch applied (recache {n_rec} frames)")

        noise = rand(batch, fpb, cfg["in_dim"], lat_h, lat_w)  # minWM layout
        x_m = noise.clone()
        x_s = noise.permute(0, 2, 1, 3, 4).clone()

        print(f"block {blk} (frames {cur}..{cur + fpb}):")
        for i, t_val in enumerate(steps.tolist()):
            flow_m = minwm_forward(x_m, t_val, text_unpadded, cur, vm, ks)
            flow_s = sglang_forward(x_s, t_val, cur, vm, ks)
            compare(f"step{i} t={t_val:.1f} flow", flow_m, flow_s)
            # x0 = x_t - sigma * flow  (both in fp64, shared formula)
            sig = sigma_for(t_val, sigmas, ts_grid)
            # x_m is [B,F,C,H,W]; flow_m is [B,C,F,H,W]
            x0_m = x_m.double() - sig * flow_m.double().permute(0, 2, 1, 3, 4)
            x0_s = x_s.double() - sig * flow_s.double()
            if i < len(steps) - 1:
                next_t = steps[i + 1].item()
                next_sig = sigma_for(next_t, sigmas, ts_grid)
                renoise = rand(batch, fpb, cfg["in_dim"], lat_h, lat_w).double()
                x_m = ((1 - next_sig) * x0_m + next_sig * renoise).to(dtype)
                x_s = x_m.permute(0, 2, 1, 3, 4).clone()
                # NOTE: both stacks are re-noised from the SAME x0 (minwm's) so
                # divergence does not compound across steps; per-step flow
                # comparison above is the real signal.
            else:
                clean_m = x0_m.to(dtype)
                clean_s = x0_s.to(dtype)
                compare("block clean x0", clean_m, clean_s.permute(0, 2, 1, 3, 4))

        # context refill with clean latents (shared: minwm's clean)
        minwm_forward(clean_m, 0.0, text_unpadded, cur, vm, ks)
        sglang_forward(clean_m.permute(0, 2, 1, 3, 4), 0.0, cur, vm, ks)

        # cache parity after each block
        compare("kv cache k", m_kv[0]["k"], s_kv[0].k)
        if not args.no_camera:
            compare("prope cache k", m_prope[0]["k"], s_prope[0].k)

        output_history_m.append(clean_m)
        output_history_s.append(clean_m.permute(0, 2, 1, 3, 4))
        if vm is not None:
            vm_hist.append(vm)
            ks_hist.append(ks)

    print(f"\nworst maxabs diff: {worst:.3e} (tolerance {tol})")
    if failures:
        print(f"{failures} comparisons exceeded tolerance")
        sys.exit(1)
    print("PARITY OK")


# ---------------------------------------------------------------------------
# Camera probe: my chunked integrator vs minWM parse_trajectory
# ---------------------------------------------------------------------------


def run_camera_probe(args):
    import numpy as np

    _, _, camera_trajectory = _import_minwm(args.minwm_repo, patch_cpu_attention=False)
    from sglang.multimodal_gen.runtime.utils.minwm_camera import advance_camera_chunk

    cases = ["w*12", "w*6,l*6", "i*3,w*4,k*3,d*2", "j*5,s*4,u*3"]
    worst = 0.0
    for traj_str in cases:
        ref = camera_trajectory.parse_trajectory(traj_str)  # (T, 4, 4) w2c
        # replay through my chunked integrator, 4 frames per chunk
        keys: list[list[str]] = []
        for seg in traj_str.split(","):
            key, n = seg.split("*")
            keys.extend([[key]] * int(n))
        c2w = np.eye(4)
        mine = []
        chunk = 4
        for i in range(0, len(keys), chunk):
            frame_keys = keys[i : i + chunk]
            c2w, vm, _ks = advance_camera_chunk(
                c2w, frame_keys, intrinsics=(0.5, 0.5, 0.5, 0.5),
                device="cpu", dtype=torch.float64,
            )
            mine.append(vm[0])
        mine = torch.cat(mine, dim=0)[: len(ref)].numpy()
        # my frame i pose = pose BEFORE motion i; minWM parse_trajectory frame 0
        # is identity then poses after each motion -> shift by one neutral frame
        diff = np.abs(mine - ref[: mine.shape[0]]).max()
        worst = max(worst, diff)
        print(f"camera probe {traj_str}: maxabs={diff:.3e}")
    print(f"worst camera diff: {worst:.3e}")
    if worst > 1e-6:
        sys.exit(1)
    print("CAMERA PROBE OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minwm-repo", required=True, help="path to minWM Wan21 dir")
    parser.add_argument("--mode", choices=["tiny", "checkpoint"], default="tiny")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--prompt-switch-block", type=int, default=-1)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="camera-free run (PRoPE path disabled on both stacks)",
    )
    parser.add_argument("--camera-probe", action="store_true")
    args = parser.parse_args()

    if args.camera_probe:
        run_camera_probe(args)
        return

    if args.tolerance is None:
        args.tolerance = 5e-4 if args.dtype == "fp32" else 1e-2
    if args.mode == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint required in checkpoint mode")
    run_parity(args)


if __name__ == "__main__":
    main()
