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

## B300 720p indexed batch

`k8s-thirdperson-remaining3699x5-720p-b300.yaml` is the production manifest for
the 3,699-image third-person batch. It runs 75 indexed shards at parallelism 4;
each pod occupies one 8-GPU `p6-b300.48xlarge` host and launches four 2-GPU
SGLang servers. Outputs are written under the LingBot2 FSx/S3 namespace rather
than the generic platform-eval namespace.

The action contract is generated directly from each image ID; it does not read
`trajs.jsonl`:

- five videos per image;
- one movement key from `wasd` and one camera key from `ijkl`;
- five distinct combinations per image, sampled deterministically without
  replacement from the 16 possible pairs;
- 129 action frames: 57 movement, 15 noop, and 57 camera;
- metadata includes `movement_key`, `camera_key`, `action_seed`, and
  `action_pattern`;
- the global scheduler balances coverage of all 16 combinations.

`prepare_remaining_thirdperson_720p.py` builds the sharded request inputs and
`build_video_action_manifest.py` emits the video/action trajectory manifest.
`test_thirdperson_actions.py` verifies determinism, uniqueness, frame layout,
and global balance.

## Streaming S3 upload

Each completed MP4 is uploaded immediately through its per-case presigned PUT
URL; the upload does not wait for the shard to finish and does not require AWS
credentials inside the GPU pod. `upload_progress.py` tails `progress.jsonl`,
resumes from an existing upload summary, and retries failed PUTs.

The B300 shard runner defaults to 32 upload workers and a 50ms progress poll:

```text
SHARD_UPLOAD_WORKERS=32
SHARD_UPLOAD_POLL_SECONDS=0.05
```

These values can be overridden in a manifest. Keep the uploader on the
FSx-adjacent GPU host; do not route MP4 bytes through a developer laptop.
Generated presigned URL files, expanded requests, videos, and HTML galleries
are intentionally ignored by Git and should be distributed through S3.

`publish_actual_video_actions.py` publishes the strict actual-only manifest for
each of the three logical TPV batches. It intersects successful generation and
successful S3 upload records, enriches each row with the original action
trajectory and measured output metadata, and writes only under the LingBot2
`video_action_manifests/` namespace. For the active 3,699-image batch it merges
completed S3 shard summaries with live FSx progress; once no inference pod is
available it can still reproduce all completed shards from S3 summaries.
