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
- Selected canary source: `19692819b..e01efdb6e`. The branch was not used as a
  production base. WM-09 carried over only the CRT downloader and tests, the
  digest-bake files, and the root-EBS/containerd versus NVMe/kubelet split. The
  canary NodePool/device-plugin, Spot-only Pod selector, recovery measurement,
  and captured live evidence were deliberately excluded.
- Denoiser image:
  `829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime@sha256:77b975f6758e642462c984dec3e1e51ef806622eb9bf3b9304330f6e072c3209`.
- minWM model release:
  `models/minwm/wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2/gs3200-ema-student-v1/releases/20260810T042157Z-c302d572/model`.
- LingBot2 revision:
  `59cccf49f2d2dd27418ae7a04b82b10868d455c2`.

`platform_config.py` is the single source for the eight payload goldens. It
accepts unresolved image placeholders only for review. A deploy render must
provide the WM-08 coordinator, gateway, and VAE image references, all pinned by
`@sha256`; tags and missing inputs fail validation.

## Application mapping and release gates

| Order | Platform object | Workload | Service account | Readiness gate / dependency |
|---:|---|---|---|---|
| 0 | `world-model-artifact-publisher` | Task/Job | `wm-artifact-publisher` | Generate, review, copy, and verify the LingBot2 immutable release; `_READY` must be last |
| 1 | `world-realtime-coordinator` | Deployment, 2 replicas | `wm-coordinator` | DynamoDB `world-model-realtime`, Pod Identity, `/healthz` |
| 2 | `minwm-vae` | Deployment/Recreate, 1 L4 | `wm-worker-discovery` | coordinator heartbeat and `/health` |
| 3 | `lingbot2-vae` | Deployment/Recreate, 1 L4 | `wm-worker-discovery` | coordinator heartbeat and `/health`; required anti-affinity keeps the two VAE Pods on different nodes |
| 4 | `minwm-denoiser` | StatefulSet/OnDelete, 2 H100 | `wm-model-fetcher` | minWM release, CRT init, heartbeat, headless Service, VAE, `/health` |
| 5 | `lingbot2-denoiser` | StatefulSet/OnDelete, 4 H100 | `wm-model-fetcher` | verified LingBot2 release, CRT init, heartbeat, headless Service, VAE, startup warmup/`/health` |
| 6 | `world-realtime-gateway` | Deployment, 2 replicas | `wm-gateway` | coordinator plus both model worker groups healthy, `/readyz` |
| 7 | `world-studio-webui` | Deployment, 2 replicas | `wm-webui` | gateway, `world-studio-runtime` Secret references, HappyOyster, prompt rewrite, world image configuration |

Every payload includes the business-line/service/lane labels, opt-in logging,
default-deny NetworkPolicy intent, resources, probes, volumes, startup command,
environment names, and dependencies. WebUI keeps Secret values out of Git and
the platform database by using `secretKeyRef` and a read-only Secret volume.
Tianpeng is excluded. GPU Pods select only `loopit.me/gpu-pool=h100|l4`; no Pod
selects `karpenter.sh/capacity-type`, so the Spot primary and on-demand fallback
pools remain usable. GPU capacity scaling/interruption handling stays a shared
WM-04 foundation controller per D-03; it is not an eighth resident application
or a platform CronJob in this draft.

Both denoisers use the AWS CRT transfer manager with 128 concurrent transfers
and 16 MiB parts. The init validates `_READY`, the artifact-manifest SHA256,
every path, file size, and file SHA256 while holding a file lock. It activates a
staging directory atomically and writes the local `_READY` last. `model-cache`
is an `emptyDir`; Karpenter/nodeadm places kubelet data on instance-store RAID0,
while the AMI wrapper intentionally leaves the preloaded containerd cache on the
200 GiB root EBS volume.

## Render and image handoff

Review-mode goldens contain only the four explicit WM-08 image placeholders:

```text
${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}
${WORLD_REALTIME_COORDINATOR_IMAGE_DIGEST}
${WORLD_REALTIME_GATEWAY_IMAGE_DIGEST}
${WORLD_REALTIME_VAE_IMAGE_DIGEST}
```

Regenerate or verify the checked-in drafts:

```bash
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py --check
```

For a deploy-ready local render, create a non-secret JSON file whose keys are
the exact placeholders above and whose values are registry references ending in
`@sha256:<64 lowercase hex>`, then run:

```bash
python3 benchmark/minwm_realtime_async_vae/render_platform_config.py \
  --image-inputs /path/to/reviewed-image-digests.json \
  --output-dir /tmp/world-model-platform-payloads
```

The renderer rejects unresolved placeholders or tags in this mode. Rendering
does not call the Loopit API and does not deploy.

## LingBot2 versioned release: read-only plan, publish, and verify

The source currently referenced by the production Kustomize is:

```text
s3://leap-world-us-west-2/world-model/minwm/serving-artifacts/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/model/
```

The destination shares the existing minWM serving bucket and has this shape:

```text
s3://leap-world-model-serving-829115578968-us-east-2/
  models/lingbot2/robbyant-lingbot-world-v2-14b-causal-fast-diffusers/
  59cccf49f2d2dd27418ae7a04b82b10868d455c2/
  releases/<UTC>-<artifact-manifest-sha8>/model/
```

The current no-credential, zero-network dry-run is checked in as
`model_releases/lingbot2/.../offline-dry-run.golden.json`. It intentionally
reports `release_id`, source control-object VersionIds, object VersionIds,
object count, and total bytes as unresolved instead of fabricating them.

When a read-only AWS identity is available, generate the exact release spec.
This command performs only `GetObject`/`HeadObject` calls and writes JSON to
stdout; it does not copy or put any object:

```bash
python3 benchmark/minwm_realtime_async_vae/build_model_release_spec.py \
  --source-bucket leap-world-us-west-2 \
  --source-region us-west-2 \
  --source-prefix world-model/minwm/serving-artifacts/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2/model \
  --destination-bucket leap-world-model-serving-829115578968-us-east-2 \
  --destination-region us-east-2 \
  --model-family lingbot2 \
  --model-id robbyant-lingbot-world-v2-14b-causal-fast-diffusers \
  --revision 59cccf49f2d2dd27418ae7a04b82b10868d455c2 \
  > /tmp/lingbot2-release-spec.json
```

Review the source/destination, every VersionId, object count, total bytes,
manifest SHA, release ID, and `rollback_release`. Then run the networked but
read-only plan (source validation only):

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /tmp/lingbot2-release-spec.json
```

The state-changing command is deliberately guarded and was not run in WM-09.
It requires separate approval plus the exact reviewed release ID:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /tmp/lingbot2-release-spec.json \
  --execute \
  --confirm-release-id <UTC>-<artifact-manifest-sha8>
```

The publisher copies version-pinned model objects, verifies destination size
and SHA256, writes `artifact-manifest.json`, then `release-manifest.json`, and
finally `_READY`. The release manifest records model/revision metadata through
the reviewed spec, source location, source and destination VersionIds, object
count, total bytes, and rollback release. Verification can be repeated without
writing:

```bash
python3 benchmark/minwm_realtime_async_vae/copy_model_release.py \
  --release /tmp/lingbot2-release-spec.json \
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
For a model regression change only the denoiser's immutable `MODEL_PREFIX` to
the reviewed rollback release and restart the StatefulSet Pod under the platform
OnDelete workflow. Never overwrite or delete the failed release during incident
response. Cloudflare origin rollback belongs to WM-12 and is not part of this
application rollback.
