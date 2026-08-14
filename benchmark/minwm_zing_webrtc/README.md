# Isolated Zing dynamic-batch and WebRTC lab

This lab creates a complete, isolated `minwm-zing-webrtc-lab` stack in
`us-east-2`. It does not reuse Services, Pods, load balancers, or DNS names from
the production `minwm-realtime` namespace.

The inference topology mirrors production: one Zing denoiser worker uses two
same-node H100 GPUs with SP/Ulysses degree 2, while TAEHV runs as a separate L4
VAE worker behind the same Coordinator and Gateway boundaries. The model is
staged only from the provided us-east-2 model-serving bucket.

The standard page keeps the existing WebSocket WebP/JPEG path as a baseline.
`/webrtc-benchmark.html` adds concurrent sessions whose model leg is raw RGB
inside the cluster; FFmpeg encodes H.264 baseline and publishes RTSP to
MediaMTX, which serves the browser over WHEP WebRTC (RTP/SRTP media plus a
small WebSocket control channel).

Native scheduler batching is enabled with a maximum batch size of eight for
compatible stateless T2V requests. Realtime I2V sessions retain independent
causal KV state and are measured as concurrent sessions; they are intentionally
not fused into a single tensor batch by the scheduler.

Build and deploy:

```bash
bash benchmark/minwm_zing_webrtc/build_and_push.sh
bash benchmark/minwm_zing_webrtc/deploy.sh
```
