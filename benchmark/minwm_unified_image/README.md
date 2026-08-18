# Unified MinWM Hopper / Blackwell image

This directory owns the release procedure for one CUDA 13 / Torch 2.11 MinWM
runtime image that supports both Hopper and Blackwell. It deliberately uses the
repository's full `docker/Dockerfile` build instead of layering a new Torch ABI
over the historical `bedc07...` image.

The current H200/B200-validated immutable release and its exact gate results are
recorded in [RELEASES.zh-CN.md](RELEASES.zh-CN.md).  H100 and B300 share the
family dispatch paths but still require a SKU-specific smoke before production
use.

## Runtime contract

The release target is:

| GPU | Compute capability | dense `fa` | MinWM packed |
|---|---:|---:|---:|
| H100 / H200 | 9.0 | FA3 | FA3 |
| B200 / B300 | 10.x | FA4 | FA4 |

The core versions are sourced from `python/pyproject.toml`: Torch
`2.11.0+cu130`, `flash-attn-4==4.0.0b15`, `sglang-kernel==0.4.4`,
`nvidia-cutlass-dsl[cu13]==4.5.2`, and the locked
`kernels-community/sgl-flash-attn3` kernel. Do not add the classic
`flash-attn` distribution or a separate top-level `flash_attn_interface`; they
are not needed and can create an ambiguous `flash_attn` namespace or Torch ABI.
The MinWM runtime also excludes MoviePy 2.2.1: it requires Pillow `<12`, while
the image keeps a security-fixed Pillow `>=12.2`; MinWM video encoding uses
PyAV/imageio-ffmpeg and has no MoviePy runtime import. This exclusion is a
specialized build argument and does not change the default SGLang image.
The same specialized build omits NIXL: MinWM does not use disaggregated
transfer, while the NIXL 1.4 meta package intentionally pulls both CUDA 12 and
CUDA 13 backends. The core SGLang runtime keeps its default NIXL behavior.
The release build also fixes `SGLANG_USE_SGL_FA3_KERNEL=0`, so Hopper selects
that locked kernels-community artifact instead of the compatible sgl-kernel
fallback. The runtime contract rejects an image if this selector drifts.

The image must not clone the repository or install Python packages when a Pod
starts. Source, native extensions, the FA3 kernel lock, and package versions are
all fixed while building the image.

Release images and build cache intentionally use separate ECR repositories.
Release tags are immutable; the cache tag is mutable and has a lifecycle rule.
Bootstrap or verify that contract once from the target AWS account:

```bash
AWS_PROFILE=spot \
AWS_REGION=us-east-2 \
bash benchmark/minwm_unified_image/bootstrap_ecr.sh
```

Use an account-829 administrator identity if the local `spot` profile is not
available. The EC2 build role only needs pull/push access and does not manage
repository settings.

## Build a release candidate

Build on a Linux x86_64 Docker host with enough disk for the CUDA build. The
script refuses tracked local changes, builds from `git archive HEAD`, and runs
`pip check` plus the software contract before creating a release tag. It then
pushes a unique immutable tag, resolves the ECR digest, repeats the checks from
that digest, and verifies the OCI provenance and SBOM attestations. The CUDA
13.0.1 base image is digest-pinned by the script.

```bash
AWS_REGION=us-east-2 \
ECR_REPOSITORY=leap-world/minwm-runtime \
ECR_CACHE_REPOSITORY=leap-world/minwm-runtime-buildcache \
bash benchmark/minwm_unified_image/build_and_push.sh
```

The default candidate tag is
`minwm-cu130-torch211-<12-character-source-sha>-<UTC-build-id>`. Consumers must
use the printed `repository@sha256:...` reference, never a tag. Build evidence
contains `build.json`, pre/post-push software contracts and `pip check`, the
resolved package set, raw build metadata, ECR metadata, the OCI index,
attestation manifest, and `image.env`.

If the immutable tag was pushed but a later pull or attestation check failed
transiently, resume validation without rebuilding or creating another tag:

```bash
RESUME_IMAGE_TAG=minwm-cu130-torch211-<sha>-<build-id> \
bash benchmark/minwm_unified_image/build_and_push.sh
```

Resume mode requires that exact tag to exist, resolves it to an immutable
digest, and repeats all post-push software, lock, OCI, provenance, and SBOM
checks against the current source SHA. It never guesses or deletes a release
candidate.

## GPU release gates

Run the same immutable digest in a one-visible-GPU Pod on each required family:

```bash
python3 -m sglang.multimodal_gen.tools.minwm_image_runtime_probe \
  --expected-source-commit <40-character-source-sha> \
  --expected-family hopper \
  --expected-image-digest sha256:...
```

Use `--expected-family blackwell` on B200/B300. The probe fails unless the
image sees exactly one GPU, resolves dense and packed attention to the expected
FA version, imports the active implementation, and executes finite BF16
self/cross attention plus production-path online/static FP8 FFN kernel
smokes against references. Hopper must load the locked kernels-community FA3
artifact rather than its compatible fallback. Externally record the Pod's actual
`status.containerStatuses[].imageID`; the digest passed into the container is
evidence metadata, not proof of the image that Kubernetes pulled.

Render one Job at a time from the checked-in template, client-dry-run it, then
submit it to the matching cluster:

```bash
bash benchmark/minwm_unified_image/render_gpu_probe_job.sh \
  hopper <source-sha40> sha256:<image-digest> > /tmp/minwm-image-gate-h200.yaml
kubectl --context codex-minwm-test-phx2 apply \
  --dry-run=client -f /tmp/minwm-image-gate-h200.yaml

bash benchmark/minwm_unified_image/render_gpu_probe_job.sh \
  blackwell <source-sha40> sha256:<image-digest> > /tmp/minwm-image-gate-b200.yaml
kubectl --context leap-world-use2 apply \
  --dry-run=client -f /tmp/minwm-image-gate-b200.yaml
```

The Jobs do not mount S3/PVCs or install anything at startup. Archive their raw
logs, complete Pod JSON, requested top-level image digest, and kubelet `imageID`.
Do not submit the B200 Job while the warm p6 node is full unless provisioning a
second Spot node is intentional.

Before the 720p performance gate, verify that the current S3 objects still
match the pinned checkpoint, first frame, and 12 donor-component versions. The
preflight compares the control-plane VersionId, size, ETag, and available S3
checksum and should be archived with the release evidence:

```bash
python3 benchmark/minwm_unified_image/verify_720p_inputs.py \
  --profile spot \
  --output /tmp/minwm-720p-input-preflight.json
```

The in-Pod runner independently checks the staged checkpoint and first-frame
SHA256 plus every donor file's path, size, and SHA256 from `inputs_720p.json`. Render
the matching performance Job only after the preflight passes:

```bash
bash benchmark/minwm_unified_image/render_gpu_performance_job.sh \
  hopper <source-sha40> sha256:<image-digest> \
  > /tmp/minwm-image-perf-h200.yaml
kubectl --context codex-minwm-test-phx2 apply \
  --dry-run=server -f /tmp/minwm-image-perf-h200.yaml

bash benchmark/minwm_unified_image/render_gpu_performance_job.sh \
  blackwell <source-sha40> sha256:<image-digest> \
  > /tmp/minwm-image-perf-b200.yaml
kubectl --context leap-world-use2 apply \
  --dry-run=server -f /tmp/minwm-image-perf-b200.yaml
```

Each Job reserves all eight GPUs for isolation but exposes only device 0 to
Torch. Audit and archive every Pod on the selected Node at gate start/end so a
CPU-only co-tenant cannot silently contaminate the 3% threshold. The accepted
H200 floor (`7.8969868` scheduler FPS) came from the legacy packed FA2 image and
is only the bootstrap floor for the first FA3 release; after that release
passes, promote its 100-chunk FA3 result to the next image's H200 baseline.
The B200 floor (`14.3957956`) already used packed FA4.

An image is releasable only after:

1. Software contract and `pip check` pass without startup installation.
2. H200 (or H100) reports dense/packed FA3 and passes attention plus FP8 FFN
   kernel smokes.
3. B200 (or B300) reports dense/packed FA4 and passes the same kernel smokes.
4. A short 720p speed run stays within 3% of the accepted device baseline.
5. The immutable digest, source SHA, both GPU contracts, and throughput evidence
   are archived together.

Changing CUDA, Torch, FA, `sglang-kernel`, CUTLASS DSL, or the FA3 lock creates
a new candidate and requires all release gates again. SageAttention remains an
optional family-specific overlay and is not part of this core image.
