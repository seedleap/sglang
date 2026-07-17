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

## aws03 SQS video controller

`aws03_video_controller.py` consumes one SQS message per SGLang video batch. It
does not delete a message when the Kubernetes Job is created. Instead, the
message is dynamically renewed until the Job succeeds, so H100 Spot interruption,
pod failure, or controller restart does not lose work.

Scheduling policy:

- Prefer a B300 capacity-block backend when a matching Ready node has at least
  one full 8-GPU slot free.
- If B300 exists but is full, fall back to the configured H100/B200 Spot backends.
- The fallback GPU backends are capped together by
  `SGLANG_VIDEO_H100_MAX_ACTIVE_GPUS`; production default should be `32`, which
  means at most four 8-GPU Jobs.
- If neither backend has capacity, the controller changes message visibility and
  leaves the message in SQS for a later attempt.

The controller tracks inflight messages in memory. While a Job is pending or
running it calls `ChangeMessageVisibility` every
`SGLANG_VIDEO_MESSAGE_RENEW_INTERVAL_SECONDS`, extending the lease to
`SGLANG_VIDEO_MESSAGE_VISIBILITY_SECONDS`. If the controller crashes, renewal
stops and the message becomes visible again after the last visibility timeout.
The replacement Job resumes from S3 checkpoints written by the runner under:

```text
<output_prefix>/video_state/cases/<case_id>.json
```

Important controller environment variables:

```text
SGLANG_VIDEO_SQS_MAX_MESSAGES=10
SGLANG_VIDEO_MESSAGE_VISIBILITY_SECONDS=900
SGLANG_VIDEO_MESSAGE_RENEW_INTERVAL_SECONDS=60
SGLANG_VIDEO_MESSAGE_MAX_LEASE_SECONDS=28800
SGLANG_VIDEO_MAX_JOB_ATTEMPTS=5
SGLANG_VIDEO_H100_MAX_ACTIVE_GPUS=32
SGLANG_VIDEO_H100_NODE_GPUS=8
SGLANG_VIDEO_H100_MAX_NODES=4
SGLANG_VIDEO_H100_EKS_CLUSTER_NAME=leap-world-aws03-usw2
SGLANG_VIDEO_ALLOW_H100_DEMAND=false
```

Use `SGLANG_VIDEO_BACKENDS_JSON` to make backend choice explicit. Example:

```json
[
  {
    "name": "b300-capacity-block",
    "node_selector": {
      "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
      "eks.amazonaws.com/nodegroup": "wan22-cb-p6b300-0715-20c",
      "node.kubernetes.io/instance-type": "p6-b300.48xlarge"
    }
  },
  {
    "name": "b200-spot",
    "scale_nodegroup": true,
    "max_nodes": 2,
    "node_selector": {
      "eks.amazonaws.com/capacityType": "SPOT",
      "eks.amazonaws.com/nodegroup": "minwm-spot-p6-b200-0703",
      "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
      "seedleap.ai/workload": "wan22-ti2v"
    }
  },
  {
    "name": "h100-spot",
    "scale_nodegroup": true,
    "max_nodes": 4,
    "node_selector": {
      "eks.amazonaws.com/capacityType": "SPOT",
      "eks.amazonaws.com/nodegroup": "sglang-spot-p5-h100",
      "node.kubernetes.io/instance-type": "p5.48xlarge",
      "seedleap.ai/workload": "wan22-ti2v"
    }
  }
]
```

Apply `k8s-aws03-video-controller-rbac.yaml` with the controller deployment.
The service account also needs AWS IAM permissions for SQS receive/delete/change
visibility and `eks:UpdateNodegroupConfig` on the configured fallback nodegroups.
