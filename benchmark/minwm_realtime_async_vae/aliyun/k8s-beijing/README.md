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
- `zing-denoiser-5090-sp2`: one DaemonSet Pod per 5090 worker node, with three
  denoiser containers. Each container is pinned with `NVIDIA_VISIBLE_DEVICES`
  to two GPUs and launches with `sp=2`, so five workers provide 15 denoiser
  slots.
- `zing-vae-5090-dual`: one DaemonSet Pod per 5090 worker node, with two VAE
  containers. Each container is pinned to one GPU and registers capacity 16.

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

- 720p UI (`1280x704`)
- default target FPS `24`
- causal sink/window `8/32`
- `--attention-backend fa`
- `MINWM_ATTENTION_IMPL=packed`
- `sp=2` worker profile for 5090 denoisers
- 6+2 GPU split per worker: 6 GPUs for three denoisers, 2 GPUs for two VAE
