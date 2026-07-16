from aws03_video_controller import (
    active_gpu_requests,
    can_start_job,
    process_one_message,
    render_job_manifest,
)


def _request() -> dict:
    return {
        "schema_version": "sglang-video-batch.v1",
        "generation_job_id": "gen_t2i_001",
        "idempotency_key": "gen_t2i_001:sglang-video:v1",
        "input": {
            "video_manifest_uri": "s3://bucket/t2i/sglang_video_manifest.jsonl",
        },
        "output": {"video_s3_prefix": "s3://bucket/t2i/videos"},
        "video": {"videos_per_image": 5, "action_seed": 20260715},
        "callback": {
            "url": "https://pipeline.example.com/api/v1/generation/jobs/gen_t2i_001/progress",
            "auth_secret_name": "lwdp-generation-callback-token",
        },
        "limits": {
            "max_active_gpus": 32,
            "gpu_per_pod": 8,
            "job_parallelism": 4,
            "timeout_seconds": 21600,
        },
    }


def test_active_gpu_requests_sums_only_non_terminal_pods():
    pods = [
        {
            "status": {"phase": "Running"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]},
        },
        {
            "status": {"phase": "Pending"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": 8}}}]},
        },
        {
            "status": {"phase": "Succeeded"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]},
        },
    ]

    assert active_gpu_requests(pods) == 16


def test_can_start_job_respects_max_active_gpu_cap():
    assert can_start_job(active_gpus=0, max_active_gpus=32, gpu_per_pod=8, parallelism=4)
    assert can_start_job(active_gpus=16, max_active_gpus=32, gpu_per_pod=8, parallelism=2)
    assert not can_start_job(active_gpus=24, max_active_gpus=32, gpu_per_pod=8, parallelism=2)


def test_render_job_manifest_uses_controller_infra_defaults():
    manifest = render_job_manifest(
        _request(),
        {
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "sglang-video-job",
            "fsx_claim_name": "fsx-claim",
            "fsx_mount_path": "/fsx",
            "work_dir_prefix": "/fsx/sglang-video",
        },
    )

    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "default"
    assert manifest["spec"]["parallelism"] == 4
    assert manifest["spec"]["completions"] == 4
    pod = manifest["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "sglang-video-job"
    container = pod["containers"][0]
    assert container["image"] == "lmsysorg/sglang:dev@sha256:test"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 8
    assert container["env"][0]["name"] == "SGLANG_VIDEO_BATCH_REQUEST_JSON"
    assert container["volumeMounts"][0]["mountPath"] == "/fsx"


def test_render_job_manifest_can_match_existing_b300_batch_runtime_shape():
    manifest = render_job_manifest(
        _request(),
        {
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "default",
            "fsx_claim_name": "xacct-fsx-pvc",
            "fsx_mount_path": "/fsx",
            "work_dir_prefix": "/fsx/sglang-video",
            "command": ["/bin/bash", "-lc"],
            "args": ["python3 /workspace/sglang/benchmark/lingbot2_offline_batch/run_t2i_video_batch.py"],
            "request_cpu": "160",
            "request_memory": "1200Gi",
            "limit_cpu": "180",
            "limit_memory": "1600Gi",
            "shm_size": "512Gi",
            "priority_class_name": "wan22-debug-low",
            "scheduler_name": "volcano",
            "node_selector": {
                "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
                "eks.amazonaws.com/nodegroup": "wan22-cb-p6b300-0715-20c",
                "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
            },
            "tolerations": [{"operator": "Exists"}],
            "security_context": {"capabilities": {"add": ["SYS_ADMIN"]}},
            "extra_env": [
                {"name": "HF_HOME", "value": "/fsx/hf-lb2"},
                {
                    "name": "GITHUB_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "github-token", "key": "token"}},
                },
            ],
        },
    )

    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["command"] == ["/bin/bash", "-lc"]
    assert container["args"][0].startswith("python3 /workspace/sglang/")
    assert container["resources"]["requests"]["cpu"] == "160"
    assert container["resources"]["limits"]["memory"] == "1600Gi"
    assert container["securityContext"]["capabilities"]["add"] == ["SYS_ADMIN"]
    assert container["env"][-2]["name"] == "HF_HOME"
    assert container["env"][-1]["valueFrom"]["secretKeyRef"]["name"] == "github-token"
    assert pod["nodeSelector"]["eks.amazonaws.com/nodegroup"] == "wan22-cb-p6b300-0715-20c"
    assert pod["priorityClassName"] == "wan22-debug-low"
    assert pod["schedulerName"] == "volcano"
    assert pod["tolerations"] == [{"operator": "Exists"}]
    assert any(volume["name"] == "shm" for volume in pod["volumes"])
    assert any(mount["mountPath"] == "/dev/shm" for mount in container["volumeMounts"])


class FakeSQS:
    def __init__(self, request: dict):
        self.request = request
        self.deleted = []
        self.visibility_changes = []

    def receive_message(self, **kwargs):
        return {
            "Messages": [
                {
                    "Body": __import__("json").dumps(self.request),
                    "ReceiptHandle": "receipt-1",
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    def change_message_visibility(self, **kwargs):
        self.visibility_changes.append(kwargs)


class FakeCoreV1:
    def __init__(self, pods):
        self.pods = pods
        self.calls = []

    def list_namespaced_pod(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": self.pods}


class FakeBatchV1:
    def __init__(self):
        self.created = []

    def create_namespaced_job(self, namespace, body):
        self.created.append({"namespace": namespace, "body": body})


def test_process_one_message_creates_job_and_deletes_sqs_message_when_capacity_available():
    sqs = FakeSQS(_request())
    core = FakeCoreV1([])
    batch = FakeBatchV1()

    result = process_one_message(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        config={
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "sglang-video-job",
            "fsx_claim_name": "fsx-claim",
        },
    )

    assert result["status"] == "started"
    assert len(batch.created) == 1
    assert batch.created[0]["body"]["metadata"]["name"].startswith("sglang-video-")
    assert sqs.deleted[0]["ReceiptHandle"] == "receipt-1"


def test_process_one_message_defers_without_deleting_when_gpu_cap_is_exceeded():
    running_pods = [
        {
            "status": {"phase": "Running"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "32"}}}]},
        }
    ]
    sqs = FakeSQS(_request())
    core = FakeCoreV1(running_pods)
    batch = FakeBatchV1()

    result = process_one_message(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        config={
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "sglang-video-job",
            "fsx_claim_name": "fsx-claim",
            "defer_visibility_timeout": 120,
        },
    )

    assert result["status"] == "deferred"
    assert batch.created == []
    assert sqs.deleted == []
    assert sqs.visibility_changes[0]["VisibilityTimeout"] == 120


def test_process_one_message_allows_counting_all_namespace_gpu_pods_for_capacity_guard():
    sqs = FakeSQS(_request())
    core = FakeCoreV1([])
    batch = FakeBatchV1()

    process_one_message(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        config={
            "namespace": "default",
            "pod_label_selector": "",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "sglang-video-job",
            "fsx_claim_name": "fsx-claim",
        },
    )

    assert core.calls[0]["label_selector"] == ""
