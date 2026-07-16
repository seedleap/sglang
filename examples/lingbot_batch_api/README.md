# LingBot lightweight batch API on AWS

This directory exposes the existing SGLang LingBot realtime engine through a
small synchronous HTTP adapter. It deliberately has no server-side queue,
database, scheduler, or job state service.

For the two-stage ownership transfer from consuming an existing SGLang service
to operating the complete Spot-backed stack, see
[`HANDOFF.zh-CN.md`](HANDOFF.zh-CN.md). Codex sessions should also read
[`CODEX_CONTEXT.zh-CN.md`](CODEX_CONTEXT.zh-CN.md) before operating the live
batch or serving environments.

## Architecture

```text
caller with concurrency control
          |
       HTTPS
          v
internet-facing AWS ALB
          |
Kubernetes Service (round robin)
          |
  +-------+-------+---------+
  |               |         |
API sidecar    API sidecar  ...  one request slot each
  | localhost     |
2-GPU SGLang   2-GPU SGLang ...
  |               |
  +-------> private S3 output
```

One `Deployment` replica is one API sidecar plus one 2-GPU SGLang instance.
A B300 host fits four replicas. HPA changes the replica count; pending GPU pods
cause Karpenter to add `p6-b300.48xlarge` Spot nodes. Empty nodes are later
consolidated. Spot capacity failure is exposed as temporary 429/503 responses,
not hidden in an unbounded service queue.

ALB is used instead of API Gateway because a steady 720p request takes roughly
26 seconds before upload and can exceed API Gateway's practical integration
timeout once tail latency is included. The chart sets the ALB idle timeout to
300 seconds.

## Backpressure contract

Each replica accepts one generation at a time:

- success: HTTP 200, an S3 URI, and `X-LingBot-Processing-Seconds`;
- busy: HTTP 429 and `Retry-After`;
- worker/Spot disruption: HTTP 502/503/504; retry with the same `request_id`;
- `/v1/capacity`: local worker state only, with `server_queue_depth: 0`.

The caller should cap in-flight requests, reduce concurrency on 429/503, and
increase it slowly after sustained success. Because ALB may first select a busy
replica while another is idle, a 429 should be retried with jitter. There is no
request backlog inside the service.

`request_id` is idempotent. It maps to a deterministic S3 object key. Before
using GPU time the adapter checks whether that object already exists, so a retry
after a lost connection returns the existing output.

## Request

```http
POST /v1/videos/generate
X-API-Key: <secret>
Content-Type: application/json
```

```json
{
  "request_id": "remaining-20260715/gvs2_00002102/variant-0",
  "source_id": "gvs2_00002102",
  "image_index": 0,
  "variant_slot": 0,
  "variants": 5,
  "prompt": "Landscape 16:9 third-person gameplay screenshot ...",
  "negative_prompt": "HUD, text, logo",
  "first_frame": "s3://input-bucket/images/gvs2_00002102.png",
  "action_seed": 20260715,
  "video_seed": 0
}
```

For one image the caller sends `variant_slot` 0 through 4. The API generates
five distinct `(wasd, ijkl)` pairs without reading `trajs.jsonl`. Across
sequential `image_index` values the 16 pairs differ in frequency by at most one.
The generated request uses the production schedule:

```text
129 video frames: 57 movement + 15 noop + 57 camera
33 model controls: 1 reference noop + 14 movement + 4 noop + 14 camera
```

Callers may explicitly send both `movement_key` and `camera_key`; otherwise both
must be omitted and the deterministic schedule is used.

## Response

```json
{
  "request_id": "remaining-20260715/gvs2_00002102/variant-0",
  "source_id": "gvs2_00002102",
  "status": "succeeded",
  "movement_key": "d",
  "camera_key": "j",
  "output_s3_uri": "s3://output-bucket/lingbot-api/ab/ab...mp4",
  "media": {
    "frames": 129,
    "width": 1280,
    "height": 720,
    "fps": 24,
    "duration_sec": 5.375,
    "latency_sec": 25.7
  },
  "idempotent_replay": false
}
```

## Deploy

1. Build and push this adapter image:

   ```bash
   docker build -t <account>.dkr.ecr.<region>.amazonaws.com/lingbot-batch-api:<tag> .
   docker push <account>.dkr.ecr.<region>.amazonaws.com/lingbot-batch-api:<tag>
   ```

2. Publish an immutable SGLang image containing the target repository commit.
   Do not use a floating `latest` or `dev` tag for production.

3. Create an IRSA role from `deploy/iam-policy.example.json`, restricted to the
   input and output prefixes. Create the API key secret without putting its
   value in Git:

   ```bash
   kubectl create secret generic lingbot-batch-api-key \
     --from-literal=api-key='<random-secret>'
   ```

4. Fill a private Helm values file with the two image references, IRSA role,
   output bucket, ACM certificate, and hostname. Validate before applying:

   ```bash
   helm template lingbot-api deploy/helm -f values.private.yaml
   helm upgrade --install lingbot-api deploy/helm -f values.private.yaml
   ```

5. Ensure a Karpenter NodePool accepts B300 Spot pods. The example NodePool
   references a cluster-specific `EC2NodeClass`; replace that reference rather
   than applying it verbatim.

The Helm default starts four replicas (one B300 host), scales up to 32 replicas,
and holds scale-down for ten minutes to avoid repeatedly reloading model weights.
At the measured 4x2-GPU topology, one B300 host produces about 556 videos/hour.

## Local tests

```bash
python -m pytest -q
```

The tests verify action uniqueness, global balance for 3699x5 cases, the exact
129-frame/33-control layout, and request validation.
