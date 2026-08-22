# Aliyun Beijing K8s Realtime Chain

This overlay keeps the Beijing chain Kubernetes-managed. It targets a
lightweight k3s control plane on the Beijing control ECS and joins RTX 5090 ECS
instances as GPU worker nodes.

## Current control-plane bootstrap

```bash
SSH_HOST=root@39.105.87.6 \
PUBLIC_WEB_HOST=39.105.87.6 \
HTTPS_PROXY_SECRET_VALUE='http://proxy_user:***@aws-proxy.loopit.com.cn:5566' \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-beijing/deploy_control_plane.sh
```

The script installs k3s if needed, keeps k3s/containerd state on `/data/k3s`
and kubelet state on `/data/kubelet`, creates image-pull and optional
webui/proxy secrets, then applies:

- `zing-coordinator`: in-memory coordinator, one replica for the isolated
  pre-GPU Beijing control plane.
- `zing-gateway`: gateway on host port `18080`.
- `zing-webui`: web UI on host port `80`.
- `zing-denoiser-5090-sp2`: one DaemonSet Pod per 5090 worker node. The name is
  historical; the current pod runs seven single-GPU denoiser containers, each
  pinned with `NVIDIA_VISIBLE_DEVICES` to GPUs 0-6 and launched with `sp=1`.
  Five healthy workers provide 35 denoiser slots.
- `zing-vae-5090-dual`: one DaemonSet Pod per 5090 worker node. The name is
  historical; the current pod runs one VAE container pinned to GPU 7 and
  registers capacity 16.

`manifests/nvidia-device-plugin.yaml.tpl` is available for clusters where GPU
workers have registry egress or the plugin image is preloaded. The default
deployment path pins GPU indices directly because these fixed 8-GPU workers are
isolated from public registries.

The k3s system add-ons are also pinned to mainland-accessible mirrors for
CoreDNS and metrics-server. The default local-storage add-on is disabled because
this chain does not use PVCs; GPU workers mount model artifacts from host paths.
Without the mirror pinning, CoreDNS tries Docker Hub first and
gateway-to-coordinator service discovery can fail even when both Pods are
running.

By default the script keeps the cached `webui-c7ae70a65a2d` image and then
mounts the current source-tree WebUI static assets through the
`zing-webui-static-patch` ConfigMap. This preserves the single-Zing
Wulanchabu UI behavior without relying on the control ECS pulling another large
WebUI image tag. Set `APPLY_WEBUI_STATIC_PATCH=false` only when the selected
`WEBUI_IMAGE` already contains the desired static assets.
The default Beijing UI config also sets `tabScopedUserIds=true`, so separate
browser tabs or hard refreshes do not collide on the coordinator's
one-session-per-user admission fence.

The same pattern is used for realtime runtime code. `APPLY_RUNTIME_SOURCE_PATCH`
defaults to `true` and creates ConfigMaps for the current `runtime/realtime`
package, realtime entrypoints, and small compatibility shims. This keeps the
gateway/coordinator/VAE protocol at v2 with direct H.264 output even when the
cached runtime image lacks those source files or Python packages. The VAE uses
the `imageio-ffmpeg` binary bundled in the runtime image through
`H264_FFMPEG_BIN`; override it only when the selected image provides another
working FFmpeg path.

The UI is reachable at:

```text
http://39.105.87.6/?mode=i2v&playback=smooth_timeline
```

If GPU workers or device-plugin are not ready, opening the page should work but
entering a world will fail capacity/admission.

## Beijing 5090 worker shape

`manifests/gpu-workers-5090.yaml.tpl` is rendered as two DaemonSets. For five
Beijing 5090 hosts:

1. Join each host to the same k3s cluster.
2. Install/verify the NVIDIA container runtime. Apply the device plugin only
   when its image is already reachable from worker containerd.
3. Prepare `/data/zing-realtime/model-cache/zing/model` and
   `/data/zing-realtime/taehv/taew2_2.pth` on each GPU node, or replace the
   hostPath volumes with an OSS staging init container.
4. Import or pre-pull `GPU_RUNTIME_IMAGE` into k3s/containerd when workers do
   not have direct registry egress.

The worker template preserves the current Zing constraints:

- 480p UI (`832x480`)
- default target FPS `24`
- causal sink/window `8/32`
- `--attention-backend fa`
- `MINWM_ATTENTION_IMPL=packed`
- `minwm_profile_launcher --profile auto` with single-GPU `sp=1` overrides for
  5090 denoisers
- fixed runtime images receive the launcher patch through `zing-runtime-tools-patch`
- 7+1 GPU split per worker: 7 GPUs for seven denoisers, 1 GPU for one VAE
- direct VAE-side H.264/fMP4 output for the browser playback path
