# Aliyun RTX5880 Zing Realtime Deployment

This directory contains the Aliyun Cloud Desktop deployment for the single-model
Zing validation service.

## Topology

- 1 RTX5880 GPU runs the async TAEHV VAE and direct H.264 encoder when GPU
  workers are enabled.
- Zing denoiser workers are started only when `START_GPU_WORKERS=true`.
- A memory coordinator keeps worker reservations on the same Cloud Desktop.
- A gateway exposes the public realtime WebSocket API on
  `PUBLIC_GATEWAY_PORT`.
- The WebUI is single-model Zing only, centered, with 480p defaults.
- Playback defaults to `REALTIME_TARGET_FPS=18`, matching the measured 480p
  steady compute rate to avoid 24 FPS underruns.
- H.264 playback buffers to a `H264_LIVE_EDGE_TARGET_MS=500` live edge by
  default, with `H264_LIVE_EDGE_SEEK_THRESHOLD_MS=900` for catch-up seeks.
- Realtime worker consumed-reservation watchdog defaults to 120s through
  `REALTIME_WORKER_MAX_CONSUMED_AGE_S`.

The browser WebUI opens the H.264 realtime WebSocket directly to the Gateway via
`h264WebSocketBaseUrl`. The VAE worker encodes H.264/fMP4 beside the GPU and
sends compressed media to the Gateway. The legacy WebUI H.264 bridge is not
started or registered by this deployment.

The deployment reuses the CUDA 12.8 ACR image and overlays the current branch's
`python/sglang/multimodal_gen` code at container startup. This keeps the runtime
on the requested branch without rebuilding the large GPU image.

## Required Inputs

Run from the repository root or this directory:

```bash
SSH_HOST=root@<new-aliyun-host> \
benchmark/minwm_realtime_async_vae/aliyun/4090-zing-single/deploy.sh
```

The script requires passwordless SSH access to the new Aliyun host, uploads the
code overlay, and starts the containers directly through SSH. The existing model
cache and runtime image are reused. The previously used `116.62.150.115` address
was released and is blocked by the deploy script.

To bring up only the non-GPU control plane before GPU capacity is attached:

```bash
START_GPU_WORKERS=false \
SSH_HOST=root@<new-aliyun-host> \
benchmark/minwm_realtime_async_vae/aliyun/4090-zing-single/deploy.sh
```

The default Gateway URL published to the browser is
`http://${PUBLIC_WEB_HOST}:18080`. Override it with `PUBLIC_GATEWAY_BASE_URL`
when a domain or load balancer is used.

## Public Endpoint

After deployment, the WebUI is expected at:

```text
http://<new-aliyun-host>/?mode=i2v&playback=smooth_timeline
```

The Gateway WebSocket endpoint is expected at:

```text
http://<new-aliyun-host>:18080
```

## Validation Report

The Chinese end-to-end and concurrency report is available in
[`ALIYUN_4090_BENCHMARK_REPORT.md`](./ALIYUN_4090_BENCHMARK_REPORT.md). Raw JSON
results are kept under `results/`.

## Runtime Image

`Dockerfile.rtx5880-cu128` documents the CUDA 12.8 compatibility layer and
validates that system FFmpeg contains `libx264`. The currently deployed image is
`gpu-cu128-rtx5880-fa2-sm80-20260820`.
