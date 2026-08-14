# World Model Loopit platform application draft

This directory is a reviewed Loopit API payload draft and golden-test input. It
is not a Kubernetes/Argo deployment source. Runtime configuration remains owned
by the Loopit platform and must be created or updated through its API/UI only
after WM-01, WM-03, WM-04, WM-08, and WM-11 have supplied the required IDs,
service accounts, pools, and image digests.

## Frozen inputs

- Business line, cluster, and namespace: `world-model`.
- Application group: `world-studio`.
- Production code baseline:
  `origin/codex/minwm-lingbot2-dual-webui-opt-20260812@d8019542103c83047997cf6dc2e7014cba8565e3`.
- Image build source: WM-09 implementation commit
  `codex/wm09-sglang-platform-config-20260814@03afce6463bda488fbfc124fa0c8d8efd104a080`.
  WM-08 must freeze this exact branch and commit before any of the eight images
  is built; the production baseline remains the lineage base, not the image
  source.
- Selected canary source: `19692819b..e01efdb6e`. The branch was not used as a
  production base. WM-09 carried over only the CRT downloader and tests, the
  digest-bake files, and the root-EBS/containerd versus NVMe/kubelet split. The
  canary NodePool/device-plugin, Spot-only Pod selector, recovery measurement,
  and captured live evidence were deliberately excluded.
- minWM model release:
  `models/minwm-async-denoiser-0/wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2-gs3200-ema-student-v1/20260810T042157Z-c302d572`.
- LingBot2 revision:
  `59cccf49f2d2dd27418ae7a04b82b10868d455c2`.
- LingBot2 reviewed release spec: release ID
  `20260814T054118Z-e0650875`, 26 objects, 86,071,995,490 bytes, manifest
  SHA256 `e065087570bde5ae45cac0f678239d6da5dafb7c1af3a2a1be0ddd6ea8929fdd`.

`platform_config.py` is the single source for the eight payload goldens. It
accepts unresolved image placeholders only for review. The checked-in
`required-inputs.json` is machine-verifiable and deliberately says
`executionReady=false`: the platform cluster ID, approved typed NetworkPolicy
peers for external dependencies, and eight WM-08 success callbacks are not yet
available. These review goldens must not be described as executable payloads.

## Application mapping and release gates

| Order | Platform object | Workload | Service account | Readiness gate / dependency |
|---:|---|---|---|---|
| 0 | `world-model-artifact-publisher` | Task/Job | `wm-artifact-publisher` | Generate, review, copy, and verify the LingBot2 immutable release; `_READY` must be last |
| 1 | `world-realtime-coordinator` | Deployment, 2 replicas | `wm-coordinator` | DynamoDB `world-model-realtime`, Pod Identity, `/healthz` |
| 2 | `minwm-vae` | Deployment, 1 L4 | `wm-worker-discovery` | coordinator heartbeat and `/health` |
| 3 | `lingbot2-vae` | Deployment, 1 L4 | `wm-worker-discovery` | coordinator heartbeat and `/health`; required anti-affinity keeps the two VAE Pods on different nodes |
| 4 | `minwm-denoiser` | StatefulSet/OnDelete, 2 H100 | `wm-model-fetcher` | minWM release, CRT init, heartbeat, headless Service, VAE, `/health` |
| 5 | `lingbot2-denoiser` | StatefulSet/OnDelete, 4 H100 | `wm-model-fetcher` | verified LingBot2 release, CRT init, heartbeat, headless Service, VAE, startup warmup/`/health` |
| 6 | `world-realtime-gateway` | Deployment, 2 replicas | `wm-gateway` | coordinator plus both model worker groups healthy, `/readyz` |
| 7 | `world-studio-webui` | Deployment, 2 replicas | `wm-webui` | gateway, `world-studio-runtime` Secret references, HappyOyster, prompt rewrite, world image configuration |

Every payload uses only WM-01 `serviceCreateRequest` fields. Custom Pod labels,
annotations and NetworkPolicy are typed fields; platform ownership labels are
not submitted by WM-09 and remain injected by WM-01. NetworkPolicy rules use
only explicit `podSelector`/`namespaceSelector`/`ipBlock` peers and ports—there
is no raw YAML, FQDN pseudo-rule or unrestricted CIDR. Internal service and DNS
rules are present; external ingress/egress remains a hard required input because
Kubernetes NetworkPolicy cannot safely represent a hostname. WebUI keeps Secret values out of Git and
the platform database by using `secretKeyRef` and a read-only Secret volume.
Tianpeng is excluded. GPU Pods select only `loopit.me/gpu-pool=h100|l4`; no Pod
selects `karpenter.sh/capacity-type`, so the Spot primary and on-demand fallback
pools remain usable. GPU capacity scaling/interruption handling stays a shared
WM-04 foundation controller per D-03; it is not an eighth resident application
or a platform CronJob in this draft.

Both denoisers use the AWS CRT transfer manager with 128 concurrent transfers
and 16 MiB parts. The init resolves a release root from `MODEL_NAME`,
`MODEL_VERSION`, and `MODEL_RELEASE_ID`, or uses an explicit immutable
`MODEL_S3_URI` release-root override for rollback. Before CRT starts, it validates
`info.json`, `artifact-manifest.json`, and `_READY`, including the info/manifest
SHA256 chain and the raw source revision. It then validates every path, file
size, and file SHA256 while holding a file lock, activates a staging directory
atomically, and writes the local `_READY` last. `model-cache`
is an `emptyDir`; Karpenter/nodeadm places kubelet data on instance-store RAID0,
while the AMI wrapper intentionally leaves the preloaded containerd cache on the
200 GiB root EBS volume.

## Render and image handoff

Review-mode goldens contain one distinct placeholder per WM-08 Job:

```text
${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}
${WORLD_REALTIME_COORDINATOR_IMAGE_DIGEST}
${WORLD_REALTIME_GATEWAY_IMAGE_DIGEST}
${MINWM_DENOISER_IMAGE_DIGEST}
${LINGBOT2_DENOISER_IMAGE_DIGEST}
${MINWM_VAE_IMAGE_DIGEST}
${LINGBOT2_VAE_IMAGE_DIGEST}
${WORLD_STUDIO_WEBUI_IMAGE_DIGEST}
```

Regenerate or verify the checked-in drafts:

```bash
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py --check
```

Image input is not a free-form placeholder map. It must contain exactly eight
WM-08 callback records with `status=success`, the reviewed branch, exact
full-Git-SHA audit tag, and ECR `imageDigest=sha256:<64 lowercase hex>`. The tag
is checked and retained only as audit evidence; the renderer constructs every
runtime image as `<repository>@<callback.imageDigest>`. It rejects `latest`, a
branch tag, a short SHA, a missing service, a failed callback, or a digest/tag
mismatch. The callbacks must come from a WM-08 contract updated to build
`03afce6463bda488fbfc124fa0c8d8efd104a080`: the older frozen `d8019542...`
image cannot contain WM-09's CRT downloader compatibility, exact release spec,
or copy verifier. `required-inputs.json` records this WM-08 contract update as a
missing hard input, including both the lineage baseline and required image
source SHA.

Validate a collected callback document without writing payloads:

```bash
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py \
  --image-inputs /path/to/reviewed-image-digests.json \
  --check-image-inputs
```

Image resolution alone does not make the payloads executable: the cluster ID
and typed external NetworkPolicy peers remain gated by `required-inputs.json`.
While `executionReady=false`, the CLI refuses to write a resolved deployment
render even when all eight image callbacks validate. Neither validation nor
review rendering calls the Loopit API or deploys.

## LingBot2 versioned release: read-only plan, publish, and verify

The source currently referenced by the production Kustomize is:

```text
s3://leap-world-us-west-2/world-model/minwm/serving-artifacts/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/model/
```

The destination shares the existing minWM serving bucket and has this shape:

```text
s3://leap-world-model-serving-829115578968-us-east-2/
  models/lingbot2-denoiser/
  robbyant-lingbot-world-v2-14b-causal-fast-diffusers/
  <UTC>-<artifact-manifest-sha8>/
    info.json
    artifact-manifest.json
    _READY
    model/
```

The checked-in `release-spec.json` was generated with the existing read-only AWS
identity. It pins `_READY`, the legacy parent `manifest.json`, and all 26 model
objects by VersionId. The source uses `resolved_revision`; the publisher
normalizes that versioned source into the destination's
`artifact-manifest.json` contract without weakening SHA or revision checks.
Regeneration performs only `GetObject`/`HeadObject` and writes JSON to stdout:

```bash
python3 benchmark/minwm_realtime_async_vae/build_model_release_spec.py \
  --source-bucket leap-world-us-west-2 \
  --source-region us-west-2 \
  --source-prefix world-model/minwm/serving-artifacts/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/model \
  --source-manifest-key world-model/minwm/serving-artifacts/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/manifest.json \
  --destination-bucket leap-world-model-serving-829115578968-us-east-2 \
  --destination-region us-east-2 \
  --serving-model-name lingbot2-denoiser \
  --model-version robbyant-lingbot-world-v2-14b-causal-fast-diffusers \
  --model-family lingbot2 \
  --model-id robbyant-lingbot-world-v2-14b-causal-fast-diffusers \
  --revision 59cccf49f2d2dd27418ae7a04b82b10868d455c2 \
  --created-at 2026-08-14T05:41:18Z \
  > /tmp/lingbot2-release-spec.review.json
```

Omitting `--created-at` intentionally creates a new review release ID from the
current UTC time. Supplying the reviewed timestamp above makes the read-only
inventory output byte-for-byte reproducible against the checked-in spec.

Review the source/destination, every VersionId, object count, total bytes,
manifest SHA, release ID, and `rollback_release`. Then run the networked but
read-only plan (source validation only):

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release benchmark/minwm_realtime_async_vae/model_releases/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/release-spec.json
```

The state-changing command is deliberately guarded and was not run in WM-09.
It requires separate approval plus the exact reviewed release ID:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release benchmark/minwm_realtime_async_vae/model_releases/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/release-spec.json \
  --execute \
  --confirm-release-id 20260814T054118Z-e0650875
```

The publisher copies version-pinned model objects under `model/`, verifies
destination size and SHA256, writes `artifact-manifest.json`,
`release-manifest.json`, then `info.json`, and finally `_READY`. The release
manifest records model/revision metadata through
the reviewed spec, source location, source and destination VersionIds, object
count, total bytes, and rollback release. Verification can be repeated without
writing:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release benchmark/minwm_realtime_async_vae/model_releases/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/release-spec.json \
  --verify
```

No S3 copy, ECR/GitHub push, platform API mutation, Kubernetes deployment, or
production canary was performed while preparing these files.

## Rollout and rollback

Do not publish all applications at once. Complete step 0, then follow steps
1-7 in the table, waiting for readiness, worker registration, and a model smoke
request at each GPU gate. Stop immediately on failure.

Rollback in reverse order: WebUI, gateway, denoisers, VAEs, coordinator. For an
application regression select the previous platform publish record/image digest.
For a model regression set only the denoiser's `MODEL_S3_URI` to the reviewed
immutable release-root URI and restart the StatefulSet Pod under the platform
OnDelete workflow. The override still has to match the configured model identity.
Never overwrite or delete the failed release during incident response. Cloudflare
origin rollback belongs to WM-12 and is not part of this application rollback.
