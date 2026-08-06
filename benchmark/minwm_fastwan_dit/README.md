# minWM vs FastWan DiT benchmark

This benchmark isolates the speed question from the models' advertised DMD
step counts.  The comparable contract is:

- one GPU, one warmed process, BF16 DiT;
- 704-class output;
- exactly five DiT forwards per output unit;
- minWM: four active DMD forwards plus its required clean-cache commit, 16 new
  pixel frames per steady chunk, `sink=4/window=20`;
- FastWan: an explicit five-entry DMD schedule, 81 total / 80 new pixel frames.

The 81-frame FastWan case is intentional.  At 704p it has 21 latent temporal
positions, so its full-attention work per newly generated latent frame is close
to minWM's four-query-frame by twenty-key-frame steady-cache contract.  Results
also report time per new pixel frame per DiT forward so the remaining 1280 vs
1248 width difference is visible rather than hidden.

`benchmark_fastwan.py` enables synchronized per-step profiling.  Those timings
are diagnostic-only and are collected identically across all FastWan lanes;
headline end-to-end timing remains the warmed client-side generation time.

The Kubernetes job runs minWM first and FastWan second on the same single-GPU
Spot node.  It publishes immutable source revisions, runtime details, raw
request records, server logs, and GPU telemetry beneath the run-specific result
directory.
