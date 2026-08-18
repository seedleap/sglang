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

The analyzer separates numerical alignment from autoregressive trajectory stability.
It reports max absolute error, mean absolute error, RMSE, PSNR, cosine similarity,
sampled SSIM, and sampled LPIPS both for the first generated chunk and for the full
trajectory. Full-trajectory pixel deltas are diagnostic: after the first chunk,
small backend rounding differences can choose a different but valid camera path.
It also reports per-15-second long-run windows, temporal activity/freeze statistics,
action-effect onset, and optical-flow direction/magnitude for each action.

## Predeclared pass conditions

- FA2 and FA3 replay are each bitwise identical.
- First generated chunk (16 frames): max abs <= 96, RMSE <= 2, cosine >= 0.9998,
  sampled SSIM >= 0.99, and sampled LPIPS <= 0.05. Full 128-frame pixel metrics
  remain in the report but do not require two autoregressive trajectories to stay
  pixel aligned.
- Each action has the same first effect frame. The FA3/FA2 steady optical-flow
  direction cosine must be >= 0.95 and magnitude ratio within [0.7, 1.4]. This
  checks the requested camera response without subtracting already-diverged idle
  trajectories pixel by pixel.
- The long rollout has no frozen transitions and its FA3/FA2 temporal-activity ratio
  is within [0.5, 2.0]. Long-run cross-backend metrics are reported per 15-second
  window rather than being hidden behind a single aggregate.

The Job publishes `FA3_QUALITY_PASS` only after all checks pass. A failed threshold
still archives the report and artifacts, but cannot be presented as approval
evidence.
