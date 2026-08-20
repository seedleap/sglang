# Unified MinWM CUDA inference image

This directory owns the release procedure for one CUDA 13 / Torch 2.11 MinWM
runtime image that supports Hopper, datacenter Blackwell, and SM120 RTX. It
deliberately uses the repository's full `docker/Dockerfile` build instead of
layering a new Torch ABI over the historical `bedc07...` image.

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
| RTX 5090 / RTX PRO 6000 Blackwell | 12.x | FA4 | FA4 |

The core contract uses Torch `2.11.0+cu130`, `sglang-kernel==0.4.4`, and the
locked `kernels-community/sgl-flash-attn3` kernel from the general build. The
MinWM release overlay replaces only the SM120 attention stack with
`flash-attn-4==4.0.0b21`, `nvidia-cutlass-dsl[cu13]==4.6.0.dev0`, and
`quack-kernels==0.5.3`. Do not add the classic
`flash-attn` distribution or a separate top-level `flash_attn_interface`; they
are not needed and can create an ambiguous `flash_attn` namespace or Torch ABI.
The same image pins `taehv==0.1.0` at source revision
`093b918971d59001a0bad6dfd6e0409b5e1752cf`; the software gate checks both the
distribution version and its PEP 610 source revision.
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

The SM120 overlay intentionally keeps the general image's TVM FFI 0.1.11 and
protobuf 7 runtime. Those versions passed the real SM120 FA4 kernel gate, but
the upstream beta21/CUTLASS wheel metadata declares narrower ranges. The build
runs `minwm_dependency_check`, which accepts only those five exact metadata
lines and fails on any additional dependency conflict; it does not hide an
unbounded `pip check` failure.

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
the bounded dependency check plus the software contract before creating a
release tag. It then
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

Use `--expected-family sm120` on RTX 5090/RTX PRO 6000 Blackwell. The FA4 dense
and packed kernel smokes must pass on the actual SM120 device; availability of
the Python module alone is not an acceptance gate. The promoted SM120 profiles
are BF16, so the probe does not apply the SM100/Hopper FP8 FFN gate to them.

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

bash benchmark/minwm_unified_image/render_gpu_probe_job.sh \
  sm120 <source-sha40> sha256:<image-digest> > /tmp/minwm-image-gate-sm120.yaml
kubectl --context leap-world-use2 apply \
  --dry-run=server -f /tmp/minwm-image-gate-sm120.yaml
```

## Tianpeng 480p hardware profiles

Profiles are executable runtime policy, not documentation-only presets. Start
the server through the image-bundled launcher and leave `--profile auto`; it
selects `sm120-32g-speed` for a 28-40 GiB SM120 GPU and
`sm120-highmem-speed` for an SM120 GPU with at least 64 GiB. H100/H200 and
B200/B300 use the resident speed policy. Unsupported capabilities and SM120
memory sizes without a validated profile fail closed.

```bash
python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
  --profile auto \
  --taehv-checkpoint-path /models/taehv/taew2_2.pth \
  -- \
  --model-path /models/minwm-tianpeng-gap12 \
  --host 0.0.0.0 \
  --port 30000
```

The launcher checks the TAEHV SHA-256 and the converted model's Tianpeng gap12
fields, fixes packed-fast/segment-compile/runtime flags, and rejects conflicting
managed arguments. The 32 GiB profile offloads only the text encoder; DiT and
TAEHV stay resident. Promotion of that profile requires a same-digest 832x480
gate with client FPS at least 24 and peak process memory below 32,000 MiB.
The explicit `sm120-32g-speed` policy may be selected on a larger SM120 GPU to
qualify the exact low-memory policy; `auto` still chooses the high-memory policy
on RTX PRO 6000.

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
H200 locked-FA3 scheduler/client baselines are `9.501709058` / `9.490211946`
FPS, with 97% minima `9.216657786` / `9.205505587`. The accepted B200 packed-FA4
scheduler/client baselines remain `14.395795593` / `14.376812178` FPS, with 97%
minima `13.963921725` / `13.945507813`. Baselines are promoted only when a
validated release establishes a higher same-contract high-water mark.

An image is releasable only after:

1. Software contract and `pip check` pass without startup installation.
2. H200 (or H100) reports dense/packed FA3 and passes attention plus FP8 FFN
   kernel smokes.
3. B200 (or B300) reports dense/packed FA4 and passes the same kernel smokes.
4. RTX 5090 and RTX PRO 6000 report dense/packed FA4 and pass the SM120 BF16
   attention smokes before their profiles are promoted.
5. A short speed run stays within the accepted device/profile baseline.
6. The immutable digest, source SHA, GPU contracts, and throughput evidence
   are archived together.

Changing CUDA, Torch, FA, `sglang-kernel`, CUTLASS DSL, or the FA3 lock creates
a new candidate and requires all release gates again. SageAttention remains an
optional family-specific overlay and is not part of this core image.
