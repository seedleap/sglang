# Result storage

The full local result set is intentionally not committed. It is about 487 MB
and includes raw RGB chunks, Nsight reports, traces, client JSON, stage JSONL,
and GPU telemetry. The files remain in this directory in the experiment
workspace and are ignored by Git.

The durable result summary, protocol, deviations, and exact ceiling argument
are maintained in:

`docs/diffusion/lingbot_world_v2_h100_40fps_experiment_log_zh.md`

Headline results:

- best correctness-preserving 200-measured-chunk run: 23.238 generated FPS;
- paired real A2A versus zero-A2A scheduler cost: 35.75 ms/chunk;
- fastest measured zero-A2A DiT mean across the tuned split sweep:
  310.541 ms per 12-frame chunk;
- empirical optimistic current-kernel ceiling: 38.642 FPS.
