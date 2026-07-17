import json

from aws03_video_controller import (
    _env_config,
    _scale_fallback_nodegroups,
    active_gpu_requests,
    backend_free_gpus,
    can_start_job,
    choose_backend,
    controller_tick,
    process_one_message,
    render_job_manifest,
    reconcile_inflight_jobs,
    renew_inflight_messages,
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
    assert manifest["spec"]["parallelism"] == 1
    assert manifest["spec"]["completions"] == 1
    pod = manifest["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "sglang-video-job"
    container = pod["containers"][0]
    assert container["image"] == "lmsysorg/sglang:dev@sha256:test"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 8
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert container["env"][0]["name"] == "SGLANG_VIDEO_BATCH_REQUEST_JSON"
    assert container["volumeMounts"][0]["mountPath"] == "/fsx"


def test_render_job_manifest_ignores_request_gpu_limits_and_uses_controller_config():
    manifest = render_job_manifest(
        _request(),
        {
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "gpu_per_pod": 2,
            "job_parallelism": 3,
        },
    )

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert manifest["spec"]["parallelism"] == 3
    assert manifest["spec"]["completions"] == 3
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 2
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 2


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


def test_render_job_manifest_can_target_b300_and_h100_spot_demand_with_preferred_affinity():
    manifest = render_job_manifest(
        _request(),
        {
            "namespace": "default",
            "job_image": "lmsysorg/sglang:dev@sha256:test",
            "service_account": "sglang-video-job",
            "fsx_claim_name": "fsx-claim",
            "placement_profiles": [
                {
                    "name": "b300-capacity-block",
                    "weight": 80,
                    "node_selector": {
                        "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
                        "eks.amazonaws.com/nodegroup": "wan22-cb-p6b300-0715-20c",
                        "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
                    },
                },
                {
                    "name": "h100-spot",
                    "weight": 100,
                    "node_selector": {
                        "eks.amazonaws.com/capacityType": "SPOT",
                        "node.kubernetes.io/instance-type": "p5.48xlarge",
                        "seedleap.ai/workload": "wan22-ti2v",
                    },
                },
                {
                    "name": "h100-demand",
                    "weight": 40,
                    "node_selector": {
                        "eks.amazonaws.com/capacityType": "ON_DEMAND",
                        "node.kubernetes.io/instance-type": "p5.48xlarge",
                        "seedleap.ai/workload": "wan22-ti2v",
                    },
                },
            ],
            "ttl_seconds_after_finished": "3600",
        },
    )

    assert manifest["spec"]["ttlSecondsAfterFinished"] == 3600
    pod = manifest["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod
    affinity = pod["affinity"]["nodeAffinity"]
    required_terms = affinity["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ]
    assert len(required_terms) == 3
    required_keys = {
        tuple((item["key"], tuple(item["values"])) for item in term["matchExpressions"])
        for term in required_terms
    }
    assert any(
        ("node.kubernetes.io/instance-type", ("p6-b300.48xlarge",)) in keys
        for keys in required_keys
    )
    assert any(
        ("node.kubernetes.io/instance-type", ("p5.48xlarge",)) in keys
        and ("eks.amazonaws.com/capacityType", ("SPOT",)) in keys
        for keys in required_keys
    )
    preferred = affinity["preferredDuringSchedulingIgnoredDuringExecution"]
    weights_by_capacity = {
        expression["values"][0]: item["weight"]
        for item in preferred
        for expression in item["preference"]["matchExpressions"]
        if expression["key"] == "eks.amazonaws.com/capacityType"
    }
    assert weights_by_capacity["SPOT"] > weights_by_capacity["ON_DEMAND"]


def test_env_config_reads_placement_profiles_and_job_ttl(monkeypatch):
    placement_profiles = [
        {
            "name": "h100-spot",
            "weight": 100,
            "node_selector": {
                "eks.amazonaws.com/capacityType": "SPOT",
                "node.kubernetes.io/instance-type": "p5.48xlarge",
            },
        }
    ]

    monkeypatch.setenv("SGLANG_VIDEO_JOB_IMAGE", "lmsysorg/sglang:dev@sha256:test")
    monkeypatch.setenv(
        "SGLANG_VIDEO_JOB_PLACEMENT_PROFILES_JSON",
        json.dumps(placement_profiles),
    )
    monkeypatch.setenv("SGLANG_VIDEO_JOB_TTL_SECONDS_AFTER_FINISHED", "900")

    config = _env_config()

    assert config["max_active_gpus"] == "8"
    assert config["gpu_per_pod"] == "8"
    assert config["job_parallelism"] == "1"
    assert config["placement_profiles"] == placement_profiles
    assert config["ttl_seconds_after_finished"] == "900"


class FakeSQS:
    def __init__(self, request: dict | list[dict]):
        self.requests = request if isinstance(request, list) else [request]
        self.deleted = []
        self.visibility_changes = []

    def receive_message(self, **kwargs):
        max_messages = int(kwargs.get("MaxNumberOfMessages") or 1)
        messages = [
            {
                "Body": json.dumps(request),
                "ReceiptHandle": f"receipt-{index + 1}",
                "MessageId": f"message-{index + 1}",
            }
            for index, request in enumerate(self.requests[:max_messages])
        ]
        return {
            "Messages": messages
        }

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    def change_message_visibility(self, **kwargs):
        self.visibility_changes.append(kwargs)


class FakeCoreV1:
    def __init__(self, pods, nodes=None):
        self.pods = pods
        self.nodes = nodes or []
        self.calls = []

    def list_namespaced_pod(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": self.pods}

    def list_node(self, **kwargs):
        return {"items": self.nodes}


class FakeBatchV1:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.jobs = {}

    def create_namespaced_job(self, namespace, body):
        self.created.append({"namespace": namespace, "body": body})

    def read_namespaced_job_status(self, name, namespace):
        return self.jobs.get(name, {"status": {}})

    def delete_namespaced_job(self, name, namespace, **kwargs):
        self.deleted.append({"namespace": namespace, "name": name, **kwargs})


class FakeEks:
    def __init__(self):
        self.updates = []

    def update_nodegroup_config(self, **kwargs):
        self.updates.append(kwargs)
        return {"update": {"status": "InProgress"}}


def _node(name: str, selector: dict, gpus: int = 8, ready: bool = True) -> dict:
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    return {
        "metadata": {"name": name, "labels": selector},
        "status": {"allocatable": {"nvidia.com/gpu": str(gpus)}, "conditions": conditions},
    }


def _gpu_pod(name: str, node_name: str, gpus: int = 8, phase: str = "Running") -> dict:
    return {
        "metadata": {"name": name},
        "status": {"phase": phase},
        "spec": {
            "nodeName": node_name,
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": str(gpus)}}}],
        },
    }


def _backend_config() -> dict:
    return {
        "namespace": "default",
        "job_image": "lmsysorg/sglang:dev@sha256:test",
        "service_account": "sglang-video-job",
        "fsx_claim_name": "fsx-claim",
        "gpu_per_pod": 8,
        "job_parallelism": 1,
        "max_active_gpus": 40,
        "h100_max_active_gpus": 32,
        "h100_node_gpus": 8,
        "h100_cluster_name": "leap-world-aws03-usw2",
        "h100_nodegroup_name": "sglang-h100-spot",
        "backends": [
            {
                "name": "b300-capacity-block",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
                    "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
                },
            },
            {
                "name": "h100-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-h100-spot",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                    "seedleap.ai/workload": "sglang-video",
                },
                "scale_nodegroup": True,
            },
        ],
    }


def test_process_one_message_creates_job_without_deleting_sqs_message_when_capacity_available():
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
    assert result["requested_gpus"] == 8
    assert result["max_active_gpus"] == 8
    assert len(batch.created) == 1
    assert batch.created[0]["body"]["metadata"]["name"].startswith("sglang-video-")
    assert sqs.deleted == []


def test_backend_free_gpus_subtracts_gpu_pods_on_matching_nodes():
    b300_selector = {
        "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
        "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
    }
    nodes = [_node("b300-a", b300_selector), _node("b300-b", b300_selector)]
    pods = [_gpu_pod("training", "b300-a", 8)]

    assert backend_free_gpus(nodes, pods, b300_selector) == 8


def test_choose_backend_prefers_b300_when_free_and_uses_h100_when_b300_is_full():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])

    assert choose_backend(config, [b300], [], requested_gpus=8)["name"] == "b300-capacity-block"

    selected = choose_backend(
        config,
        [b300, h100],
        [_gpu_pod("other-job", "b300-a", 8)],
        requested_gpus=8,
    )

    assert selected["name"] == "h100-spot"


def test_choose_backend_can_use_b200_as_fallback_when_b300_is_full():
    config = {
        **_backend_config(),
        "backends": [
            _backend_config()["backends"][0],
            {
                "name": "b200-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-b200-spot",
                    "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
                    "seedleap.ai/workload": "sglang-video",
                },
                "scale_nodegroup": True,
            },
        ],
    }
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    b200 = _node("b200-a", config["backends"][1]["node_selector"])

    selected = choose_backend(
        config,
        [b300, b200],
        [_gpu_pod("training", "b300-a", 8)],
        requested_gpus=8,
    )

    assert selected["name"] == "b200-spot"


def test_choose_backend_does_not_double_count_h100_pods_and_inflight_state():
    config = _backend_config()
    h100_selector = config["backends"][1]["node_selector"]
    h100_pods = [
        {
            "metadata": {"labels": {"sglang.seedleap.io/backend": "h100-spot"}},
            "status": {"phase": "Running"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "16"}}}]},
        }
    ]
    state = {
        "inflight": [
            {"backend": "h100-spot", "requested_gpus": 8},
            {"backend": "h100-spot", "requested_gpus": 8},
        ]
    }

    selected = choose_backend(
        config,
        [_node("h100-a", h100_selector)],
        h100_pods,
        requested_gpus=8,
        state=state,
    )

    assert selected["name"] == "h100-spot"


def test_choose_backend_does_not_use_h100_demand_by_default():
    config = {
        **_backend_config(),
        "h100_max_active_gpus": 8,
        "backends": [
            {
                "name": "h100-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                },
            },
            {
                "name": "h100-demand",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "ON_DEMAND",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                },
            },
        ],
    }
    state = {"inflight": [{"backend": "h100-spot", "requested_gpus": 8}]}

    assert choose_backend(config, [], [], requested_gpus=8, state=state) is None


def test_choose_backend_applies_one_gpu_cap_across_b200_and_h100_fallbacks():
    config = {
        **_backend_config(),
        "h100_max_active_gpus": 8,
        "backends": [
            {
                "name": "b200-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-b200-spot",
                    "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
                },
            },
            {
                "name": "h100-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-h100-spot",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                },
            },
        ],
    }
    state = {"inflight": [{"backend": "b200-spot", "requested_gpus": 8}]}

    assert choose_backend(config, [], [], requested_gpus=8, state=state) is None


def test_scale_fallback_nodegroups_scales_each_backend_nodegroup_independently():
    config = {
        **_backend_config(),
        "backends": [
            {
                "name": "b200-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-b200-spot",
                    "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
                },
                "scale_nodegroup": True,
                "max_nodes": 2,
            },
            {
                "name": "h100-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "eks.amazonaws.com/nodegroup": "sglang-h100-spot",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                },
                "scale_nodegroup": True,
                "max_nodes": 4,
            },
        ],
    }
    eks = FakeEks()
    state = {
        "inflight": [
            {"backend": "b200-spot", "requested_gpus": 8},
            {"backend": "h100-spot", "requested_gpus": 16},
        ]
    }

    _scale_fallback_nodegroups(eks, config, state)

    updates = {
        item["nodegroupName"]: item["scalingConfig"]["desiredSize"]
        for item in eks.updates
    }
    assert updates == {"sglang-b200-spot": 1, "sglang-h100-spot": 2}


def test_controller_tick_starts_multiple_messages_up_to_h100_gpu_cap_and_scales_nodes():
    requests = [{**_request(), "generation_job_id": f"gen_t2i_{index}"} for index in range(5)]
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300])
    batch = FakeBatchV1()
    sqs = FakeSQS(requests)
    eks = FakeEks()

    state = {"inflight": []}
    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config={**config, "sqs_max_messages": 10},
        state=state,
        now=1000.0,
    )

    assert result["started"] == 4
    assert result["deferred"] == 1
    assert len(batch.created) == 4
    assert all(
        job["body"]["metadata"]["labels"]["sglang.seedleap.io/backend"] == "h100-spot"
        for job in batch.created
    )
    assert sqs.deleted == []
    assert sqs.visibility_changes[-1]["ReceiptHandle"] == "receipt-5"
    assert sqs.visibility_changes[-1]["VisibilityTimeout"] > 0
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 4
    assert len(state["inflight"]) == 4


def test_renew_inflight_messages_extends_visibility_without_deleting():
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "receipt_handle": "receipt-1",
                "job_name": "sglang-video-gen",
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
            }
        ]
    }

    renew_inflight_messages(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        config={
            "message_visibility_seconds": 900,
            "message_renew_interval_seconds": 60,
            "message_max_lease_seconds": 28800,
        },
        state=state,
        now=1061.0,
    )

    assert sqs.visibility_changes == [
        {
            "QueueUrl": "https://sqs.us-west-2.amazonaws.com/123/video",
            "ReceiptHandle": "receipt-1",
            "VisibilityTimeout": 900,
        }
    ]
    assert sqs.deleted == []
    assert state["inflight"][0]["last_renewed_at"] == 1061.0


def test_reconcile_recreates_failed_h100_job_under_same_message_receipt():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300, h100])
    batch = FakeBatchV1()
    batch.jobs["sglang-video-old"] = {"status": {"failed": 1}}
    sqs = FakeSQS(_request())
    eks = FakeEks()
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-old",
                "backend": "h100-spot",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["restarted"] == 1
    assert sqs.deleted == []
    assert batch.deleted[0]["name"] == "sglang-video-old"
    assert len(batch.created) == 1
    assert state["inflight"][0]["attempts"] == 2
    assert state["inflight"][0]["job_name"].startswith("sglang-video-")


def test_reconcile_deletes_message_only_after_job_succeeds():
    config = _backend_config()
    batch = FakeBatchV1()
    batch.jobs["sglang-video-done"] = {"status": {"succeeded": 1}}
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-done",
                "backend": "b300-capacity-block",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], []),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["completed"] == 1
    assert sqs.deleted[0]["ReceiptHandle"] == "receipt-1"
    assert state["inflight"] == []


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
    assert result["requested_gpus"] == 8
    assert result["max_active_gpus"] == 8
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
