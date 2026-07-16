# LingBot-World 2.0 offline batch benchmark

This benchmark measures production-like 15-second video synthesis on a single
8-GPU node. It compares one 8-GPU server, two 4-GPU servers, four 2-GPU
servers, and eight 1-GPU servers. The included manifest reproduces the H100
deployment used by the recorded `2026-07-15-h100-final` run. Live B200 Spot
capacity was unavailable during this test.

Fixed request contract:

- LingBot-World 2.0 14B causal-fast BF16, four DMD steps;
- 832x480 at 25 FPS;
- 32 chunks: 9 initial frames plus 31x12 steady frames = 381 frames = 15.24s;
- four first-person and four third-person requests per topology;
- no frame interpolation or upscaling;
- raw RGB response persisted to H.264 MP4 with ffmpeg;
- every fresh server receives a separate three-chunk warmup request;
- startup and warmup are reported separately from steady-state throughput.

Primary metrics are node videos/hour, videos/GPU-hour, generated video
seconds/GPU-hour, aggregate realtime factor, per-video end-to-end latency, and
failure rate. Semantic/visual quality is outside this performance benchmark.

For a real mixed-perspective dataset, pass both
`--first-frame-first-person` and `--first-frame-third-person`. A text prompt
alone did not override the perspective embedded in the conditioning frame in
the H100 run. If the two requests share one frame, the benchmark emits a
warning and the performance numbers remain valid, but the labels must not be
treated as visual-quality ground truth.

`run_topologies.sh` forwards the equivalent `FIRST_FRAME_FIRST_PERSON` and
`FIRST_FRAME_THIRD_PERSON` environment variables.

The pod requests all eight GPUs, forcing Karpenter to place it on an isolated
H100 node instead of sharing a partially occupied node. From this directory,
create the ConfigMap,
dry-run, and submit with:

```bash
kubectl create configmap codex-lingbot2-offline-batch-h100 \
  --from-file=benchmark_batch.py \
  --from-file=run_topologies.sh \
  --from-file=summarize_topologies.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --server-side --dry-run=server -f k8s.yaml
kubectl apply -f k8s.yaml
```

Copy `/results` before deleting the completed pod and ConfigMap.

For the lightweight, externally callable AWS service built from this benchmark
contract, see [`examples/lingbot_batch_api`](../../examples/lingbot_batch_api/README.md).
It pairs one synchronous HTTP adapter with each SGLang instance, returns 429
instead of building a server-side queue, and lets HPA/Karpenter resize B300 Spot
capacity while callers provide backpressure.
