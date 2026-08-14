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

## Validated profile (2026-08-14)

The isolated stack sustained 1, 2, 4, 6, and 8 simultaneous H.264/WHEP
sessions. Eight sessions remained healthy for more than 45 seconds. Aggregate
generation throughput plateaued around 37--38 source frames per second: about
20 FPS per user at one session, 15.7 FPS at two, 9.3--9.6 FPS at four,
6.3--6.6 FPS at six, and 4.6--4.8 FPS at eight.

Scheduler telemetry showed `avg_size=1.00`, `merged_rate=0.0%`, and
`stop_reason=head:realtime_session` under the eight-session load. Therefore the
current realtime result measures concurrent session admission and time-slicing
on the two-H100 SP=2 worker, not tensor-fused realtime dynamic batching. The
configured admission ceiling is eight; raising it without changing the worker
topology will reduce per-session FPS further.

Build and deploy:

```bash
bash benchmark/minwm_zing_webrtc/build_and_push.sh
bash benchmark/minwm_zing_webrtc/deploy.sh
```
