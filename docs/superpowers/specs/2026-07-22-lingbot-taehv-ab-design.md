# LingBot TAEHV A/B Test Design

## Goal

Adapt the optional TAEHV realtime decoder from upstream SGLang PR #31921 to
`codex/lingbot-action-override`, then run a controlled 100-video A/B test.
The production SGLang video path must continue to use the existing VAE until a
human explicitly approves a later rollout.

## Scope and Safety Boundary

- The test corpus is the existing fixed `testset100-v2` set.
- Both arms use the same input image, prompt, action, seed, resolution, frame
  count, and topology.
- The baseline arm uses the current VAE decode path without a TAEHV checkpoint
  argument.
- The candidate arm adds only
  `--vae-config.taehv-checkpoint-path <checkpoint>`.
- Test Jobs use dedicated names, immutable candidate image tags, and separate
  S3/FSx result roots. They do not update the production controller Deployment,
  production image reference, or controller defaults.
- Baseline and candidate run sequentially on B300. This removes simultaneous
  node contention from the primary performance comparison and leaves the online
  controller's behavior unchanged.

## Implementation Design

1. Bring in the upstream opt-in TAEHV realtime decoder implementation:
   `VAEConfig.taehv_checkpoint_path`, lazy `taehv` import, cached model, and a
   per-session streaming decoder with frame carry-over.
2. Add `taehv` to the immutable offline video-runner image and expose a
   test-only `TAEHV_CHECKPOINT_PATH` environment variable in the benchmark
   launcher. If the variable is empty, the launcher emits no TAEHV CLI option
   and retains the original decoder.
3. Add an early test-only preflight which verifies both the Python package and
   checkpoint before GPUs start model warmup. A missing dependency or checkpoint
   fails the candidate test clearly rather than silently falling back.
4. Produce a comparison report from both result roots. Each of the 100 rows
   contains the original image, prompt, action metadata, baseline video, and
   TAEHV video. The report also states failures, first-video time, warmup,
   steady-state throughput, S3 upload lag, total wall time, tail shutdown time,
   and GPU metrics when available.

## Acceptance Criteria

- Existing unit tests cover the new opt-in configuration and prove that an empty
  checkpoint keeps the existing decode implementation.
- A TAEHV-specific unit test verifies streaming frame carry-over and the
  NCTHW/NTCHW conversion without requiring a real GPU.
- The candidate image contains `taehv`; the baseline test does not pass the
  checkpoint argument.
- Both 100-video arms use identical case IDs and input metadata.
- HTML exposes all required side-by-side artifacts and clearly labels decoder
  type, image revision, checkpoint checksum, and test topology.
- No production configuration changes until a separate human approval after
  review of the A/B result.

## Test Data Writes Requiring Separate Confirmation

The final A/B execution will write only to new, test-specific locations:

- an SQS message or test-only Kubernetes Job for each arm;
- two new FSx result directories; and
- two new S3 test prefixes plus one comparison HTML/report prefix.

Before those writes, the exact account, Job manifests, S3 paths, expected
object counts, and recovery procedure will be shown for confirmation.
