# Single-GPU local-TAEHV 720p profiling

Use the tracked renderer; generated, run-specific YAML stays under the ignored
`k8s/generated/` directory.

```bash
python3 benchmark/minwm_realtime_parity/k8s/render_single_gpu_taehv_24fps.py \
  --sku b200 --mode baseline --run-tag 20260817-a1 \
  --sglang-git-ref 54bdfea9cd52ac1cd79896e1a7275e18a0257b79 \
  --harness-git-ref <40-character-harness-commit>

python3 benchmark/minwm_realtime_parity/k8s/render_single_gpu_taehv_24fps.py \
  --sku b300 --mode baseline --run-tag 20260817-a2 \
  --sglang-git-ref 54bdfea9cd52ac1cd79896e1a7275e18a0257b79 \
  --harness-git-ref <40-character-harness-commit>
```

Render candidate A/B jobs with the same entry point, full commit hashes, and
distinct run tags. This keeps Job names and S3 result prefixes disjoint.
The renderer normally verifies the embedded harness against
`--harness-git-ref`. Before that harness is committed, the explicit
`--allow-uncommitted-harness-for-dry-run` escape hatch emits a manifest whose
runtime safety gate refuses execution; use it only for Kubernetes dry-runs.

```bash
python3 benchmark/minwm_realtime_parity/k8s/render_single_gpu_taehv_24fps.py \
  --sku b200 --mode baseline --run-tag 20260817-candidate-a \
  --sglang-git-ref <40-character-commit-a> \
  --harness-git-ref <same-40-character-harness-commit> --require-24fps
python3 benchmark/minwm_realtime_parity/k8s/render_single_gpu_taehv_24fps.py \
  --sku b200 --mode baseline --run-tag 20260817-candidate-b \
  --sglang-git-ref <40-character-commit-b> \
  --harness-git-ref <same-40-character-harness-commit> --require-24fps

python3 benchmark/minwm_realtime_parity/k8s/render_single_gpu_taehv_24fps.py \
  --sku b200 --mode nsys --run-tag 20260817-candidate-nsys \
  --sglang-git-ref <40-character-candidate-commit> \
  --harness-git-ref <same-40-character-harness-commit> --candidate-evidence
```

Before any apply, run both Kubernetes dry-runs against the hardware's context.
The renderer emits one ConfigMap and one Job per YAML file.

```bash
kubectl --context <context> apply --dry-run=client -f <generated-yaml>
kubectl --context <context> apply --dry-run=server -f <generated-yaml>
```

## Load-bearing contract

- One requested and limited GPU; Spot-only selectors. B200 uses the
  `minwm-spot/ray` Auto Mode pool on `p6-b200.48xlarge`. B300 uses the
  `aws03-usw2/default` managed Spot node group
  `minwm-spot-p6-b300-0703` on `p6-b300.48xlarge` in `us-west-2a`.
- Immutable image digest, full SGLang commit, checkpoint version/size/SHA-256,
  first-frame URI/version/size/SHA-256, MinWM provenance commit, TAEHV revision,
  and `taew2_2.pth` SHA-256 are embedded and asserted at runtime. The
  first-frame ETag and CRC64NVME are also recorded.
- Request: `1248x704`, four denoise steps, four latent frames / 16 pixel frames
  per chunk, 24 FPS target, local StreamingTAEHV, no VAE CPU offload.
- Tianpeng: local attention 32, sliding window 32, sink 8,
  `block_relative`, max frame gap 12, first-frame pin enabled.
- Baseline uses 20 warmup and 200 measured chunks. NSYS is diagnostic, not a
  headline: capture starts before its formal request, then records eight warmup
  plus eight measured chunks (16 total). Chunk 7 is the first request chunk to
  cross the bounded 32-latent-frame window, so measured chunks 8-15 are steady.
  This larger trace deliberately replaces the racy historical claim that a
  client-observed chunk-19 boundary could guarantee a 10-chunk server capture.
- After server readiness, the same formal Job first runs a protocol smoke with
  the telemetry-aware profile client. Baseline requires one warmup plus two
  measured chunks (three complete payload/timing pairs and 32 measured frames);
  NSYS requires one warmup plus one measured chunk before tracing. These smoke
  chunks are recorded separately in `protocol-smoke.json` and are excluded from
  headline throughput and the NSYS capture.
- Realtime session idle and maximum-lifetime watchdogs are both fixed at 900
  seconds. `SGLANG_REALTIME_TRACE_SYNC_CUDA=0` and
  `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0` prevent timing instrumentation from
  injecting device synchronizations into the headline path.
- The client requires one server timing message per chunk. It accepts both the
  legacy `chunk_stats` schema and current `chunk_telemetry`, normalizing the
  transport fields while retaining model encode/denoise/decode timings. A
  duplicate timing message is fatal; every field records measured sample and
  missing counts. Raw payloads must be exactly 1248x704 RGB, not merely
  self-consistent with their own headers.
- Current main emits a tiny `model_vae_encode_ms` sample for the steady no-op
  image-encode stage wrapper on every chunk. The harness requires complete
  samples but does not interpret that host wrapper time as native VAE
  re-encoding; GPU attribution still comes from Nsight.
- Reference main remains compatible with its legacy layer-0 alignment line and
  may report zero samples for the candidate-only async-enqueue timing. A
  candidate rendered with `--require-24fps` (or an NSYS candidate rendered with
  `--candidate-evidence`) additionally requires the
  structured 30-layer `MINWM_RUNTIME_ALIGNMENT_JSON` all-match assertion and
  one `raw_frame_async_enqueue_ms` sample for every measured chunk.
- Baseline has no `SYS_ADMIN`. Only NSYS gets `SYS_ADMIN`; it uses launch/start/
  stop, CUDA+NVTX, graph-node tracing, and never runs `torch.profiler`.
- NSYS export is a hard gate, not a best-effort name search. The analyzer dumps
  the actual SQLite schema, resolves registered NVTX names through `StringIds`,
  and requires 16 complete, non-overlapping
  `stage_MinWMCausalDMDDenoisingStage` ranges with at least one contained
  kernel each and no partial/invalid target ranges. Candidate NSYS also
  requires exactly one nested
  `minwm_action_residual_prepare_once_per_chunk` range per chunk.
- Storage is an explicit deployment-profile contract. The supported B200 and
  aws03 B300 profiles both use the verified RWX `s3-claim` once, mounted as
  read-only `/s3-input` and writable `/s3-results`. The phx2 B300 cluster's
  `s3-claim` is ROX and is intentionally not a renderer target. A split layout
  fails closed unless it names a distinct, verified RWX S3 results PVC.
  Before any expensive setup, runtime writes and reads back a non-empty
  `STORAGE_WRITE_PROBE` in the unique result prefix, then recursively copies
  and reads back a nested `ARCHIVE_COPY_PROBE` without POSIX metadata. Runtime
  exports `MINWM_S3_MOUNT=/s3-input`, derives the mount key from the immutable
  case URI, and refuses to launch the server unless that object is readable and
  matches the recorded byte count and SHA-256. The complete provenance is
  recorded in `first-frame-source.json`.
- Runtime
  artifacts include the alignment contract, complete `pip freeze --all`, and a
  non-gating `pip check` result/status. Known unrelated base-image extras are
  `decord`, `open-clip-torch`, and `wandb`.
- The ConfigMap owns the runner, profile client, `common.py`, and cases JSON.
  Runtime verifies and archives all four SHA-256 values and requires a full
  harness commit whose contents match those files. Main/candidate jobs therefore
  use an identical request builder and payload contract; only the explicit
  server checkout changes.
- `SUCCESS` means the run completed its correctness contract. Baseline output
  separately writes `PERFORMANCE_PASS` or `PERFORMANCE_FAIL` from client steady
  ratio-of-sums FPS against 24. Candidate jobs rendered with `--require-24fps`
  exit nonzero after preserving results when that performance gate fails.
  The runner copies a non-empty local `RUN_COMPLETE` payload to the remote
  `SUCCESS` key only after all artifacts have been archived; it does not rely
  on S3 CSI metadata-only `touch` semantics. Artifact archival creates
  directories with `mkdir` and copies regular files individually without
  requesting POSIX metadata preservation; directory trees are never passed to
  `cp` because Mountpoint S3 rejects its permission and timestamp finalization.
  Recursive archival is attempted at most once; if it or a later completion
  step fails, cleanup only publishes the non-empty `FAILED` marker and never
  retries a partially uploaded tree.
