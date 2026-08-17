# Aliyun 4090 Zing Realtime Deployment

This directory contains the single-node Aliyun deployment used for the 8 x RTX 4090
validation service.

## Topology

- 1 GPU runs the async TAEHV VAE server.
- 7 GPUs run independent Zing denoiser workers with `sp=1`.
- A memory coordinator keeps worker reservations on the same ECS host.
- A gateway exposes the public realtime WebSocket API.
- The WebUI is single-model Zing only, centered, with 480p defaults.

The deployment reuses the prebuilt ACR images and overlays the current branch's
`python/sglang/multimodal_gen` code from OSS at container startup. This keeps the
runtime on the requested branch without rebuilding the large GPU image.

## Required Inputs

Run from the repository root or this directory:

```bash
export ALIYUN_ACCESS_KEY_ID=...
export ALIYUN_ACCESS_KEY_SECRET=...
benchmark/minwm_realtime_async_vae/aliyun/4090-zing-single/deploy.sh
```

The script creates or reuses an attached data disk for `/data`, opens port 80 on
the instance security group if needed, uploads the code overlay to OSS, then
starts the containers through Cloud Assistant.

## Public Endpoint

After deployment, the WebUI is expected at:

```text
http://8.147.109.68/?mode=i2v&playback=smooth_timeline
```

## Validation Report

The Chinese end-to-end and concurrency report is available in
[`ALIYUN_4090_BENCHMARK_REPORT.md`](./ALIYUN_4090_BENCHMARK_REPORT.md). Raw JSON
results are kept under `results/`.

## Cost Notes

The ECS instance is a pay-as-you-go GPU instance. The helper may create one
additional pay-as-you-go ESSD data disk for model and Docker cache. The OSS
bucket and ACR repository are reused.
