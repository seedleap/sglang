# Hopper FA3 quality validation

This gate validates the mandatory Hopper FA3 path against an FA2 reference without
adding an FA2 runtime switch to the product. The `fa3-quality` Job creates a detached
worktree at the exact candidate commit, applies the archived
`fa2_reference_hopper_validation.patch` only to that worktree, and then runs FA2 and
FA3 serially on the same visible GPU.

## Fixed contract

- One H100 or H200 GPU; the renderer rejects B200/B300.
- Same immutable image, candidate commit, 5B checkpoint, TAEHV checkpoint, prompt,
  first frame, seed, and raw RGB transport.
- 1248x704, four DMD steps, local TAEHV, no CPU offload.
- Tianpeng cache settings: sink 8, window 32, block-relative RoPE, gap 12.
- Deterministic packed attention and deterministic inference enabled for both lanes.
- Each backend first passes an 8-warmup + 2-measured no-offload protocol smoke.
- Server logs must explicitly announce `backend=fa2` or `backend=fa3`; the FA3 lane
  rejects any FA2 announcement.

## Evidence matrix

Each backend runs every case twice. Replay output must be bitwise identical within
the backend.

1. Six 720p action cases, 128 generated frames each: idle, W, S, J, L, and W+L.
2. One 720p 60-second rollout, 1,440 generated frames, with scheduled
   idle/W/S/J/L/idle controls.
3. Lossless `.npy` frames and reviewable `.mp4` videos from all four lanes are
   archived with SHA-256 hashes.

The analyzer reports max absolute error, mean absolute error, RMSE, PSNR, cosine
similarity, sampled SSIM, and sampled LPIPS. It also reports per-15-second long-run
windows, temporal activity/freeze statistics, action-effect onset, action-effect
delta cosine, and FA3/FA2 action-effect norm ratio.

## Predeclared pass conditions

- FA2 and FA3 replay are each bitwise identical.
- Short action clips: max abs <= 8, RMSE <= 1, sampled SSIM >= 0.995, sampled
  LPIPS <= 0.05.
- Each action has the same first effect frame; effect-delta cosine >= 0.95 and
  FA3/FA2 effect-norm ratio is within [0.8, 1.25].
- The long rollout has no frozen transitions and its FA3/FA2 temporal-activity ratio
  is within [0.5, 2.0]. Long-run cross-backend metrics are reported per 15-second
  window rather than being hidden behind a single aggregate.

The Job publishes `FA3_QUALITY_PASS` only after all checks pass. A failed threshold
still archives the report and artifacts, but cannot be presented as approval
evidence.
