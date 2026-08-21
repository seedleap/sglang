# Aliyun Beijing K8s Realtime Chain

This overlay keeps the Beijing chain Kubernetes-managed while GPU capacity is
not ready yet. It targets a lightweight k3s control plane on the Beijing control
ECS and later joins RTX 5090 ECS instances as worker nodes.

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
- `zing-vae-5090` and `zing-denoiser-5090`: GPU worker templates applied with
  zero replicas.

The k3s system add-ons are also pinned to mainland-accessible mirrors for
CoreDNS and metrics-server. The default local-storage add-on is disabled because
this chain does not use PVCs; GPU workers mount model artifacts from host paths.
Without the mirror pinning, CoreDNS tries Docker Hub first and
gateway-to-coordinator service discovery can fail even when both Pods are
running.

The UI is reachable at:

```text
http://39.105.87.6/?mode=i2v&playback=smooth_timeline
```

Until GPU workers are joined and scaled up, opening the page should work but
entering a world will fail capacity/admission.

## Future 5090 worker shape

`manifests/gpu-workers-5090.yaml.tpl` is intentionally rendered with zero
replicas. When five Beijing 5090 hosts are ready:

1. Join each host to the same k3s cluster.
2. Install/verify the NVIDIA container runtime and device plugin.
3. Prepare `/data/zing-realtime/model-cache/zing/model` and
   `/data/zing-realtime/taehv/taew2_2.pth` on each GPU node, or replace the
   hostPath volumes with an OSS staging init container.
4. Scale VAE to `1` and denoiser to `7` for the first 8-GPU node shape, or
   split replicas across nodes with labels.

The worker template preserves the current Zing constraints:

- 480p UI (`832x480`)
- default target FPS `24`
- causal sink/window `8/32`
- `--attention-backend fa`
- `MINWM_ATTENTION_IMPL=packed`
- `sp=1` worker profile for 5090 denoisers
