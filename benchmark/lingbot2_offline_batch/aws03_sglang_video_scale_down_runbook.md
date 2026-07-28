# AWS03 SGLang Video Scale-Down Runbook

Last inspected: 2026-07-28 17:15 CST

Minimal scale-down applied: 2026-07-28 17:26 CST

This runbook captures the key state needed to stop the SGLang video controller and
bring it back quickly. It is intentionally focused on reversible control-plane
changes. It does not delete Kubernetes Jobs, PVCs, PVs, FSx data, S3 objects, IAM
resources, or SQS queues.

## Scope

- AWS account for controller and GPU jobs: `107014413969` (`aws03`)
- EKS context: `leap-world-aws03-usw2`
- EKS cluster: `leap-world-aws03-usw2`
- Region: `us-west-2`
- Namespace: `default`
- Controller deployment: `sglang-video-controller`
- Main SQS request queue: `https://sqs.us-west-2.amazonaws.com/829115578968/lwdp-sglang-video-request`
- SQS owner account: `829115578968`

## Pre-Scale-Down State

### Controller

- Deployment: `default/sglang-video-controller`
- Replicas before scale-down: `1`
- Current pod before scale-down:
  - `sglang-video-controller-7656bf9785-wxqc5`
  - node: `ip-172-31-34-158.us-west-2.compute.internal`
- Container image: `python:3.11-slim`
- Runtime repo: `https://github.com/lkejun237-ops/sglang.git`
- Runtime branch fetch: `origin codex/lingbot-action-override`
- Controller `CODE_GIT_REF`: `3805081754416e76529225dbf0e087e3f247dd71`
- Runner source bundle:
  - `s3://leap-world-us-east-2/world-model/sglang-video/runner-sources/sglang-3805081754416e76529225dbf0e087e3f247dd71.tar.gz`
- Runner ConfigMap: `sglang-video-runner-50bade92d31d`
- Runner ConfigMap keys:
  - `benchmark_evalset.py`
  - `prepare_capacity_smoke_720p.py`
  - `run_capacity_smoke_720p.sh`
  - `run_t2i_video_batch.py`
  - `t2i_video_batch.py`
  - `thirdperson_actions.py`

### Job Runtime Settings

- Job image: `lmsysorg/sglang:dev@sha256:8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7`
- Service account for jobs: `sglang-video-job`
- FSx PVC: `xacct-fsx-pvc`
- FSx mount path: `/fsx`
- Work dir prefix: `/fsx/sglang-video`
- Model: `robbyant/lingbot-world-v2-14b-causal-fast-diffusers`
- Model revision: `59cccf49f2d2dd27418ae7a04b82b10868d455c2`
- `HF_HOME`: `/fsx/hf-lb2`
- `PIP_CACHE_DIR`: `/fsx/world-model/cache/pip/sglang-e21b0e`
- Job GPU per pod: `8`
- Max active jobs: `5`
- B300 max active jobs: `5`
- B300 max active GPUs: `40`
- Fallback/B200 max active jobs: `0`
- Fallback/B200 max active GPUs: `0`
- B200 backend max nodes: `0`

### Backend Placement

Primary backend:

- name: `b300-capacity-block`
- capacity type: `CAPACITY_BLOCK`
- nodegroup: `wan22-cb-p6b300-0715-20c`
- instance type: `p6-b300.48xlarge`
- scheduler: `default-scheduler`

Fallback backend:

- name: `b200-spot`
- capacity type: `SPOT`
- nodegroup: `minwm-spot-p6-b200-0703`
- instance type: `p6-b200.48xlarge`
- max nodes: `0`

### FSx Binding

- PVC: `default/xacct-fsx-pvc`
- PV: `xacct-fsx-pv`
- FSx file system id: `fs-05b2d065253562dd3`
- CSI driver: `fsx.csi.aws.com`
- DNS name in PV: `10.20.41.42`
- Mount name: `vmpzfb4v`
- Reclaim policy: `Retain`

### SQS Health Before Scale-Down

Queue checked with the main account profile:

```bash
AWS_PROFILE=default aws sqs get-queue-attributes \
  --region us-west-2 \
  --queue-url https://sqs.us-west-2.amazonaws.com/829115578968/lwdp-sglang-video-request \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible ApproximateNumberOfMessagesDelayed
```

Observed before scale-down:

- `ApproximateNumberOfMessages`: `0`
- `ApproximateNumberOfMessagesNotVisible`: `0`
- `ApproximateNumberOfMessagesDelayed`: `0`

Controller logs also reported repeated idle status:

```json
{"status":"idle","started":0,"pending":0,"completed":0,"failed":0,"callback_pending":0}
```

### Related CronJobs

These are not the SGLang controller itself, but they run on the same CPU system
nodegroup and may be confused with SGLang/SQS support work:

- `loomvideo-spot-idle-monitor`
  - schedule: `*/5 * * * *`
  - `suspend`: `false` before scale-down
  - nodegroup selector: `wan22-cpu-system-0603`
- `loomvideo-checkpoint-sweeper-xacct`
  - schedule: `*/15 * * * *`
  - `suspend`: `false` before scale-down
  - scans `/fsx/world-model/loomvideo/checkpoints/instantstart`
  - archives to `s3://leap-world-us-east-2/world-model/loomvideo/checkpoints/instantstart-archive`
  - current active job at inspection time: `loomvideo-checkpoint-sweeper-xacct-29753835`

Do not delete or kill active sweeper jobs unless the owner explicitly asks for
checkpoint/archive operations to stop immediately.

## Scale Down

Recommended minimal scale-down:

```bash
kubectl --context leap-world-aws03-usw2 -n default scale deployment/sglang-video-controller --replicas=0
```

Optional noise reduction after confirming the checkpoint sweeper policy is not
needed:

```bash
kubectl --context leap-world-aws03-usw2 -n default patch cronjob/loomvideo-spot-idle-monitor \
  --type merge -p '{"spec":{"suspend":true}}'
```

The checkpoint sweeper can touch FSx and S3 archive state. Suspend it only with
explicit owner confirmation:

```bash
kubectl --context leap-world-aws03-usw2 -n default patch cronjob/loomvideo-checkpoint-sweeper-xacct \
  --type merge -p '{"spec":{"suspend":true}}'
```

Avoid scaling the `wan22-cpu-system-0603` nodegroup to zero as part of SGLang
scale-down. The nodegroup is shared by cluster system components such as
CloudWatch, CSI, CoreDNS, KubeRay, and Volcano; terminating it is a cluster-level
operation, not an SGLang-only scale-down.

## Restore

Fast restore of the SGLang video controller:

```bash
kubectl --context leap-world-aws03-usw2 -n default scale deployment/sglang-video-controller --replicas=1
kubectl --context leap-world-aws03-usw2 -n default rollout status deployment/sglang-video-controller --timeout=180s
kubectl --context leap-world-aws03-usw2 -n default logs deploy/sglang-video-controller --tail=80
```

If the spot monitor was suspended and should run again:

```bash
kubectl --context leap-world-aws03-usw2 -n default patch cronjob/loomvideo-spot-idle-monitor \
  --type merge -p '{"spec":{"suspend":false}}'
```

If the checkpoint sweeper was explicitly suspended and should run again:

```bash
kubectl --context leap-world-aws03-usw2 -n default patch cronjob/loomvideo-checkpoint-sweeper-xacct \
  --type merge -p '{"spec":{"suspend":false}}'
```

## Verification

After scale-down:

```bash
kubectl --context leap-world-aws03-usw2 -n default get deploy sglang-video-controller
kubectl --context leap-world-aws03-usw2 -n default get pods | grep -E 'sglang-video|NAME'
AWS_PROFILE=default aws sqs get-queue-attributes \
  --region us-west-2 \
  --queue-url https://sqs.us-west-2.amazonaws.com/829115578968/lwdp-sglang-video-request \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible ApproximateNumberOfMessagesDelayed
```

Expected after minimal scale-down:

- `sglang-video-controller` shows `0/0` replicas.
- No `sglang-video-controller-*` pod is Running.
- Existing completed or failed historical `sglang-video-*` Jobs may remain.
- SQS messages are not consumed while the controller is scaled to zero.

Observed after the 2026-07-28 scale-down:

- `sglang-video-controller`: `READY 0/0`, `UP-TO-DATE 0`, `AVAILABLE 0`
- deployment status JSON: `replicas=0`, `ready=0`, `available=0`, `updated=0`
- SQS queue attributes:
  - `ApproximateNumberOfMessages`: `0`
  - `ApproximateNumberOfMessagesNotVisible`: `0`
  - `ApproximateNumberOfMessagesDelayed`: `0`
- No `sglang-video-controller-*` pod remained Running.
- Listed `sglang-video-*` pods were historical `Completed` or `Error` pods from 11 days earlier.
- `loomvideo-spot-idle-monitor` and `loomvideo-checkpoint-sweeper-xacct` were not changed.

## Cost Note

The controller itself ran on a shared `c7i.4xlarge` on-demand node:

- EC2: approximately `$17.136/day`
- root EBS: approximately `$0.53/day`

Scaling the controller deployment to zero stops SGLang queue consumption and
prevents new SGLang GPU Jobs, but it does not automatically remove the shared CPU
system node cost. Reducing that node cost requires a separate cluster-system
consolidation plan.

## FSx Job-Artifact Cleanup Plan

Inspection time: 2026-07-28 17:33 CST
Primary cleanup applied: 2026-07-28 17:43 CST

This section covers only SGLang job-created files on the cross-account FSx mount.
It excludes model/cache paths needed for faster restart.

Target FSx:

- PVC: `default/xacct-fsx-pvc`
- PV: `xacct-fsx-pv`
- FSx file system id: `fs-05b2d065253562dd3`
- Mount path in pods: `/fsx`

Current FSx capacity at inspection:

- Size: `4.5T`
- Used: `3.2T`
- Available: `1.3T`
- Use: `72%`

Cleanup candidates:

1. Primary SGLang video job work directory:
   - path: `/fsx/sglang-video`
   - size: `6.7G`
   - top-level job directories: `81`
   - files: `38839`
   - observed file types include `json`, `jsonl`, `log`, `mp4`, `csv`, and Triton
     compile artifacts such as `ttir`, `ttgir`, `ptx`, `cubin`, `llir`.
2. Optional SGLang reference/eval job output:
   - path: `/fsx/world-model/eval/lingbot2/sglang-reference/20260721-b300-d9a7e0e`
   - size: `515M`

Explicit exclusions:

- `/fsx/hf-lb2`
  - HuggingFace/model cache.
  - Size at inspection: `81G`.
- `/fsx/world-model/cache/pip`
  - Python/pip/runtime dependency cache.
  - Size at inspection: `476G`.
  - Includes current controller restart cache:
    `/fsx/world-model/cache/pip/sglang-e21b0e`.
- Kubernetes Jobs, Pods, PVCs, PVs, SQS queues, S3 objects, and FSx file system
  resources.

The primary cleanup leaves the parent `/fsx/sglang-video` directory in place and
removes only its `sglang-video-*` child directories.

Primary cleanup command:

```bash
kubectl --context leap-world-aws03-usw2 -n default exec \
  sfa54k15k-d8e-d8e1b277cdd97f77eb5aa462679974675e1aff08-0-w82kq \
  -c minwm-training -- bash -lc '
set -euo pipefail
test -d /fsx/sglang-video
find /fsx/sglang-video -mindepth 1 -maxdepth 1 -type d -name "sglang-video-*" -print
find /fsx/sglang-video -mindepth 1 -maxdepth 1 -type d -name "sglang-video-*" -exec rm -rf -- {} +
find /fsx/sglang-video -mindepth 1 -maxdepth 1 -type d -name "sglang-video-*" -print
du -sh /fsx/sglang-video
'
```

Optional reference cleanup command:

```bash
kubectl --context leap-world-aws03-usw2 -n default exec \
  sfa54k15k-d8e-d8e1b277cdd97f77eb5aa462679974675e1aff08-0-w82kq \
  -c minwm-training -- bash -lc '
set -euo pipefail
test -d /fsx/world-model/eval/lingbot2/sglang-reference/20260721-b300-d9a7e0e
rm -rf -- /fsx/world-model/eval/lingbot2/sglang-reference/20260721-b300-d9a7e0e
du -sh /fsx/world-model/eval/lingbot2/sglang-reference || true
'
```

Rollback note:

- There is no direct filesystem rollback for `rm -rf`.
- This plan backs up the operational metadata in git, not the file contents.
- The SGLang controller is already scaled to zero and SQS was empty at inspection,
  so no active SGLang job should be writing these paths.
- If data-level recovery is required, first archive the target paths to S3 and
  then delete from FSx after validating the archive.

Top-level `/fsx/sglang-video` directories observed before cleanup:

```text
sglang-video-debuginvalid720-20260716-2017
sglang-video-gen-020df978b16a4997
sglang-video-gen-093f0d9bde05496f
sglang-video-gen-0f00da3822cb460b
sglang-video-gen-0f80e5df94ab45b8
sglang-video-gen-14bfaf073c6a4f40
sglang-video-gen-18d7dde8287e4bc1
sglang-video-gen-204428888e6d4339
sglang-video-gen-26b4dcfde2e243fb
sglang-video-gen-2704c2506fff42ca
sglang-video-gen-27759768648545e8
sglang-video-gen-2796347f449b4eea
sglang-video-gen-2d077903620c44be
sglang-video-gen-3417f37614cd4083
sglang-video-gen-3b07d67a87074987-r2
sglang-video-gen-3b07d67a87074987-r3
sglang-video-gen-3b07d67a87074987-r4
sglang-video-gen-3f8ecd5a2548411e
sglang-video-gen-400e4b3fd7614756
sglang-video-gen-44480e5a39b643ab
sglang-video-gen-44480e5a39b643ab-r2
sglang-video-gen-458c49a600b2470e
sglang-video-gen-46cd2389bab34b68
sglang-video-gen-517854fb53764640-r2
sglang-video-gen-517854fb53764640-r3
sglang-video-gen-517854fb53764640-r4
sglang-video-gen-603fa85773644a2f-r2
sglang-video-gen-603fa85773644a2f-r3
sglang-video-gen-603fa85773644a2f-r4
sglang-video-gen-642cebe7828b47f9
sglang-video-gen-6c12f543d72148c3
sglang-video-gen-703258d0787b4cdd
sglang-video-gen-73420e55c1954386
sglang-video-gen-73420e55c1954386-r2
sglang-video-gen-7530ea85f2fe4706
sglang-video-gen-7623f48b86574d20
sglang-video-gen-7f119390f40843cf
sglang-video-gen-81231e7b326e499e
sglang-video-gen-81231e7b326e499e-r2
sglang-video-gen-81231e7b326e499e-r3
sglang-video-gen-840f814563a74786
sglang-video-gen-8a28f21ee71c4968
sglang-video-gen-8ce8db5ab3fb4a95
sglang-video-gen-8d5bbb09814c4cfe
sglang-video-gen-95a841e28d164a8f
sglang-video-gen-96d8081f532a4e8f
sglang-video-gen-9a2dc6617bc84e64
sglang-video-gen-9e4fda6543194cbe
sglang-video-gen-a45cda4cc185428d
sglang-video-gen-a8fa69b2fe7640dd
sglang-video-gen-a98c61f65a414e8e-r2
sglang-video-gen-a98c61f65a414e8e-r3
sglang-video-gen-aa5463173c7a480d
sglang-video-gen-b39ad5f3ed4c4c48
sglang-video-gen-b8f3c3785d0c4853
sglang-video-gen-bb9d299b14d74527
sglang-video-gen-c0e7abb2c1774fc9
sglang-video-gen-c0e7abb2c1774fc9-r2
sglang-video-gen-c0e7abb2c1774fc9-r3
sglang-video-gen-c0fa8e9aa9c74687
sglang-video-gen-c60d50b3c0a74ec9
sglang-video-gen-c73b012ed77748a3
sglang-video-gen-cad0942e4a8d4e5d
sglang-video-gen-cb4812f099e04d79
sglang-video-gen-cbf0758993224ecf
sglang-video-gen-ce2f06dee507428a
sglang-video-gen-ce662c83c6a944ff
sglang-video-gen-ceac7b3e32f048bb
sglang-video-gen-d02183ede6ac4052
sglang-video-gen-d0bca8bbc49c418f
sglang-video-gen-d0bca8bbc49c418f-r2
sglang-video-gen-da53fd9e2c804bb1
sglang-video-gen-dabef41160c34426
sglang-video-gen-e19a306da2e44294
sglang-video-gen-e57617d59c1641ed
sglang-video-gen-e75d9f2181bd40c4
sglang-video-gen-ef0ec3b9250d4adb
sglang-video-gen-f483ad92aa8d43e5
sglang-video-gen-f6a8fd1d74c94ecc
sglang-video-gen-ffc449e5590d4069
sglang-video-threecases-newsched-20260716-211810
```

Primary cleanup execution result:

- command target: `/fsx/sglang-video`
- deleted directories: `81`
- size before cleanup: `6.7G`
- remaining `sglang-video-*` directory count: `0`
- size after cleanup: `25K`
- FSx `df -h /fsx` after cleanup: `Size 4.5T`, `Used 3.1T`,
  `Avail 1.5T`, `Use% 68%`
- optional reference path was not removed:
  `/fsx/world-model/eval/lingbot2/sglang-reference/20260721-b300-d9a7e0e`
- model/cache exclusions were not removed:
  `/fsx/hf-lb2`, `/fsx/world-model/cache/pip`,
  `/fsx/world-model/cache/pip/sglang-e21b0e`

## Main Account FSx SGLang Delivery Cleanup Assessment

Inspection time: 2026-07-28 17:50 CST

This section covers the second FSx discussed during SGLang scale-down. It is in
the main AWS account and is mounted by the main Ray/LWDP pipeline cluster. It is
not safe to clean broadly while the pipeline is active.

Target FSx:

- AWS account: `829115578968`
- EKS context: `leap-world-us-east-2`
- Namespace used for inspection: `ray`
- Ray head used for read-only inspection: `ray-cluster-head-frcdr`
- Region: `us-east-2`
- FSx file system id: `fs-03e9f4e8533b98444`
- FSx tag `Name`: `ray-pipeline-fsx-50t-20260617-v215`
- FSx lifecycle: `FAILED`
- AWS failure detail: `Please delete your file system and create a new one.`
- Deployment type: `SCRATCH_2`
- Storage type: `SSD`
- Provisioned capacity: `52800 GiB`
- Mount path in Ray pods: `/fsx`
- Current `df -h /fsx`: `Size 46T`, `Used 13T`, `Avail 34T`, `Use% 27%`

Current cluster activity observed at inspection:

- `ray-cluster-head-frcdr` was Running.
- Many `ray-cluster-*worker-*` pods were Running and recently created.
- `lwdp-be`, `lwdp-fe`, and `lwdp-pipeline-api` pods were Running.
- `lwdp-generation-job-reconciler` and `lwdp-ray-lease-sweeper` jobs had recent
  activity.

Conclusion: this FSx is still a shared live pipeline filesystem. Do not delete
the whole file system, do not delete `/fsx/pipeline`, and do not delete whole
`/fsx/pipeline/lwdp_generation/gen_*` directories without a separate migration
and service cutover plan.

### SGLang-Specific Cleanup Candidate

The safest SGLang-specific candidate on this FSx is the delivery subdirectory
for generation directories that contain a SGLang manifest:

```text
/fsx/pipeline/lwdp_generation/*/delivery/sglang_video_manifest.jsonl
```

Candidate scope from read-only inspection:

- candidate `delivery` directories: `298`
- total manifest rows, treated as video records: `220396`
- exact candidate `delivery` footprint: `604775944 KiB` / `576.759 GiB`
- oldest manifest mtime: `2026-07-17 12:20:56 CST`
- newest manifest mtime: `2026-07-28 12:00:18 CST`
- manifest read errors: `0`

Full candidate manifest:

- `benchmark/lingbot2_offline_batch/main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt`

The file uses CSV format but carries a `.txt` suffix because the repository
ignores `*.csv`.

Candidate distribution by manifest mtime:

```text
date,candidate_dirs,manifest_rows
2026-07-17,4,22
2026-07-22,3,1000
2026-07-23,27,23814
2026-07-24,60,41585
2026-07-25,52,38887
2026-07-26,73,52475
2026-07-27,52,40766
2026-07-28,27,21847
```

### Explicit Exclusions

Do not include these in the SGLang delivery cleanup:

- `/fsx/pipeline/lwdp_generation` as a whole.
- Any `gen_*` directory as a whole.
- `/fsx/pipeline/code_packages`.
- `/fsx/pipeline/review_sites`.
- Model directories.
- Shared cache directories.
- `tasks/*/inputs/lingbot_video.mp4`, which are LingBot input videos rather than
  SGLang output delivery artifacts.

### Proposed Delete Command, Not Yet Executed

The command below deletes only the candidate `delivery` directories listed in
the CSV manifest. Run it only after confirming downstream consumers no longer
need these delivery artifacts or after archiving them to S3.

```bash
manifest=/path/to/main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt
kubectl --context leap-world-us-east-2 -n ray cp \
  "$manifest" ray-cluster-head-frcdr:/tmp/main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt \
  -c ray-head
kubectl --context leap-world-us-east-2 -n ray exec ray-cluster-head-frcdr \
  -c ray-head -- bash -lc '
set -euo pipefail
manifest=/tmp/main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt
python3 - <<'"'"'PY'"'"'
import csv, os, shutil
manifest = "/tmp/main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt"
expected_prefix = "/fsx/pipeline/lwdp_generation/"
deleted = 0
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        path = row["delivery_dir"]
        if not path.startswith(expected_prefix) or not path.endswith("/delivery"):
            raise SystemExit(f"refusing unexpected path: {path}")
        marker = os.path.join(path, "sglang_video_manifest.jsonl")
        if not os.path.isfile(marker):
            print(f"skip_missing_marker {path}")
            continue
        shutil.rmtree(path)
        deleted += 1
        print(f"deleted {path}")
print(f"deleted_delivery_dirs={deleted}")
PY
'
```

Rollback note:

- `rm -rf` / `shutil.rmtree` has no direct filesystem rollback.
- The CSV manifest backs up path metadata only, not file contents.
- If recovery of the actual generated outputs matters, archive these `delivery`
  directories to S3 before deleting them from FSx.
- Deleting these directories frees about `576.759 GiB` on FSx but does not reduce
  FSx provisioned-capacity billing. Reducing FSx billing requires a separate
  delete/recreate or migration plan for `fs-03e9f4e8533b98444`, which is a
  higher-risk shared-storage operation.

### Confirmation Gate

Before executing the delete command, obtain explicit human confirmation for this
exact operation:

- target account: `829115578968`
- target cluster/context: `leap-world-us-east-2`
- target FSx: `fs-03e9f4e8533b98444`
- target paths: the `298` `delivery` directories listed in
  `main_fsx_sglang_delivery_cleanup_manifest_20260728.csv.txt`
- expected deletion size: about `576.759 GiB`
- expected affected manifest rows: `220396`
- irreversible unless the files are archived first
