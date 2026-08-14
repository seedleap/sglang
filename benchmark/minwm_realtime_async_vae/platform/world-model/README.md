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
  `codex/wm09-sglang-platform-config-20260814@5ded4b5de2702d063cb9421d5c7049c0570c013b`.
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
- LingBot2 canonical source inventory: release ID
  `20260814T054118Z-e0650875`, 23 payload objects, 86,068,529,220 bytes,
  manifest SHA256
  `6a790fd04daecfa66bede8cc71f18ed96dda617bc74cecda51e5ce72c4cf19af`.
  `_READY` uses the canonical `revision` field.
- Reviewed two-model S3 migration bundle SHA256:
  `944d828d3eb4c3db52f761847046c2910b8243a23579553fe6bee2defa8b29c7`;
  41 payload objects plus 6 controls, 47 destination objects and
  110,277,738,028 bytes in total (110,277,729,372 payload bytes and 8,656
  control bytes).

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
`MODEL_VERSION`, and `MODEL_RELEASE_ID`, or accepts an explicit immutable
`MODEL_S3_URI` only when it is the exact
`models/<model-name>/<model-version>/<release-id>` release root. The latter
derives and validates the rollback release ID rather than letting a legacy or
cross-model URI masquerade as the configured root. Before CRT starts, init
validates `info.json`, `artifact-manifest.json`, and `_READY`: schema and model
identity, the canonical `revision`, info/manifest hashes, and equal
`release_spec_sha256` and `publisher_bundle_sha256` bindings in both info and
`_READY` when a publisher-generated release carries those fields. It then
validates every path, size, and SHA256 while holding a file
lock. Cache reuse re-hashes every local payload, so same-size corruption is a
miss; a cache miss also fails before CRT creation when free disk is smaller
than the manifest payload total. The downloader activates a staging directory
atomically and writes the local `_READY` last. `model-cache`
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
mismatch. The callbacks must come from a WM-08 contract updated to build the
exact WM-09 implementation commit frozen above. The older `5ded4b5...`
implementation and production `d8019542...` baseline cannot contain this
complete publisher bundle, three-hash execution gate, canonical revision
enforcement, or hardened cache validation. `required-inputs.json` records this
WM-08 contract update as a missing hard input, including both the lineage
baseline and required image source SHA.

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

## Generic versioned release: read-only plan, publish, and verify

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

The checked-in LingBot2 `release-spec.json` is deliberately retained as a
legacy, fail-closed review fixture. It describes a superseded 26-object source
inventory and predates `publisher_scripts`; `--offline-plan` therefore reports
`release_spec_ready=false`. It is not an executable approval artifact and the
platform publisher policy contains no execute command for it.

Generate a fresh spec from the canonical 23-object source. Generation performs
only `GetObject`/`HeadObject`, pins both source controls and every payload by
VersionId, and records the exact size/SHA256 of all three runtime scripts plus
their canonical `publisher_bundle_sha256`:

```text
build_model_release_spec.py  # source/control inventory and spec builder
copy_model_release.py        # destination copy and exact verifier
download_model_artifact.py   # init and immutable rollback consumer
```

The builder is model-parameter driven; the following LingBot2 invocation is an
example of the same contract used for any model identity:

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
current UTC time. Do not overwrite the checked-in legacy fixture. Review and
store the new spec as a distinct approval artifact.

Review the source/destination, both source-control VersionIds and hashes, every
payload VersionId/size/SHA256, exact script inventory, release ID, revision,
destination root, and `rollback_release`. Then run the networked but read-only
plan against the newly reviewed file:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /path/to/reviewed-release-spec.json
```

The plan validates the fixed source control and payload versions. If S3 does
not expose a full-object SHA256 checksum, it streams that fixed source version
to recompute the content hash; plan time and network use can therefore approach
the complete payload size. It prints three values that must be reviewed
together: raw `release_spec_sha256`, runtime `publisher_bundle_sha256`, and an
`execution_bundle_sha256` binding the spec, all three script hashes, source
control key/VersionId/size/hash, every payload path/size/hash/VersionId,
destination root, and deterministic hashes of all three destination controls.

The state-changing command is deliberately guarded and was not run in WM-09.
Approval must name the exact release ID and both computed approval hashes; a
release-ID-only confirmation is insufficient:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /path/to/reviewed-release-spec.json \
  --execute \
  --confirm-release-id '<REVIEWED_RELEASE_ID>' \
  --confirm-release-spec-sha256 '<REVIEWED_RELEASE_SPEC_SHA256>' \
  --confirm-execution-bundle-sha256 '<REVIEWED_EXECUTION_BUNDLE_SHA256>'
```

Every copy pins the source VersionId. Before any control upload, publisher HEADs
every destination payload and requires exact `ContentLength`, `sha256` metadata,
and `source-version-id` metadata. It uploads and byte-readbacks
`artifact-manifest.json`, then `info.json`; `_READY` is conditionally created
last and is never rewritten. Both `info.json` and `_READY` contain equal
`release_spec_sha256` and `publisher_bundle_sha256`, and `_READY.revision` is
canonical. There is no fourth `release-manifest.json` control.

The final verifier checks all three control bodies and their identity/revision
hash chain, every payload size and metadata, aggregate bytes, and the exact key
set: only `model/**`, `artifact-manifest.json`, `info.json`, and `_READY`. Any
missing or additional object fails closed. Verification can be repeated without
writing:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /path/to/reviewed-release-spec.json \
  --verify
```

`publish_model_artifact.py` and `publish_model_artifact.sh` are the older
MinWM checkpoint conversion path and are not this generic release publisher.
WM-08 must compile/check the three-script bundle above, not use the legacy
converter as the common publisher contract gate.

## Digest-pinned model-init canary

Run the canary only inside an approved digest-pinned denoiser image with an
empty writable cache. It performs a cold AWS CRT download and then a warm
restart against the same immutable release, requiring the second run to
re-hash and reuse the cache:

```bash
MODEL_S3_URI='s3://<bucket>/models/<name>/<version>/<release-id>' \
MODEL_NAME='<name>' \
MODEL_VERSION='<version>' \
MODEL_RELEASE_ID='<release-id>' \
MODEL_SOURCE_REVISION='<canonical-revision>' \
AWS_REGION='<region>' \
CANARY_DESTINATION='/model-cache/model' \
CANARY_LOCK_PATH='/model-cache/.download.lock' \
/opt/sglang/benchmark/minwm_realtime_async_vae/run_model_release_canary.sh
```

Success requires `backend=awscrt`, `cache_hit=false` for cold start,
`cache_hit=true` for restart, exact model identity/release fields, and all three
local controls. Capture the image repository/digest, release root, both JSON
results, elapsed time, bytes, and Pod exit status. A canary approval authorizes
only the named image digest, immutable release root, namespace/task, and command;
it does not authorize a general platform rollout.

## Exact CI/CD handoff gates

The release sequence is dependency driven and every remote mutation is a
separate gate:

1. With explicit confirmation naming GitHub repository `seedleap/sglang`,
   branch `codex/wm09-sglang-platform-config-20260814`, and push, publish the
   reviewed local commits. Creating a PR is a different operation and requires
   separate confirmation.
2. Update WM-08 to freeze the exact implementation SHA above. Its publisher
   compile/contract gate must cover `build_model_release_spec.py`,
   `copy_model_release.py`, `download_model_artifact.py`, and
   `run_model_release_canary.sh`; checking only the legacy
   `publish_model_artifact.py` is insufficient. Validate the Jenkins Job DSL
   locally before separately approving a GitLab push, seed sync, or build.
3. After the WM-08 source gate is merged/synced, approve the eight immutable
   BuildKit/ECR builds individually or as one explicit eight-job scope. Each
   full-SHA tag must resolve to an ECR registry digest and a successful Loopit
   callback; no platform deployment is part of these builds.
4. Generate a fresh release spec and run the read-only network plan. Approve an
   S3 execution only after recording the exact release ID,
   `release_spec_sha256`, `publisher_bundle_sha256`, and
   `execution_bundle_sha256`. Re-run `--verify` before using the release.
5. Approve one canary task naming the image digest, model release root,
   namespace/application, and command above. Preserve the cold/warm evidence.
6. Only after the canary passes and every `required-inputs.json` hard gate is
   resolved may a separately approved Loopit platform render/publish proceed.
   Roll out in table order and stop at the first failed readiness or model smoke
   gate.

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
