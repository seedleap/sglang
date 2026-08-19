# Aliyun RTX5880 Zing Realtime Deployment

This directory contains the four-GPU Aliyun Cloud Desktop deployment for the
single-model Zing validation service.

## Topology

- 1 RTX5880 GPU runs the async TAEHV VAE and direct H.264 encoder.
- 3 RTX5880 GPUs run independent Zing denoiser workers with `sp=1`.
- A memory coordinator keeps worker reservations on the same Cloud Desktop.
- A gateway exposes the public realtime WebSocket API.
- The WebUI is single-model Zing only, centered, with 480p defaults.

The WebUI opens the realtime WebSocket directly to the Gateway. The VAE worker
encodes H.264/fMP4 beside the GPU and sends compressed media to the Gateway, so
the legacy WebUI H.264 bridge is not in the production data path.

The deployment reuses the CUDA 12.8 ACR image and overlays the current branch's
`python/sglang/multimodal_gen` code at container startup. This keeps the runtime
on the requested branch without rebuilding the large GPU image.

## Required Inputs

Run from the repository root or this directory:

```bash
benchmark/minwm_realtime_async_vae/aliyun/4090-zing-single/deploy.sh
```

The script requires passwordless SSH access to `root@116.62.150.115`, uploads
the code overlay, and starts the containers directly through SSH. The existing
model cache and runtime image are reused.

## Public Endpoint

After deployment, the WebUI is expected at:

```text
http://116.62.150.115/?mode=i2v&playback=smooth_timeline
```

## Validation Report

The Chinese end-to-end and concurrency report is available in
[`ALIYUN_4090_BENCHMARK_REPORT.md`](./ALIYUN_4090_BENCHMARK_REPORT.md). Raw JSON
results are kept under `results/`.

## Runtime Image

`Dockerfile.rtx5880-cu128` documents the CUDA 12.8 compatibility layer and
validates that system FFmpeg contains `libx264`. The currently deployed image is
`gpu-cu128-rtx5880-20260819`.
