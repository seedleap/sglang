import json
import io

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
    monkeypatch.setenv("SGLANG_VIDEO_B300_MAX_ACTIVE_GPUS", "32")
    monkeypatch.setenv("SGLANG_VIDEO_FALLBACK_MAX_ACTIVE_GPUS", "160")

    config = _env_config()

    assert config["max_active_gpus"] == "8"
    assert config["max_active_jobs"] == "7"
    assert config["b300_max_active_jobs"] == "5"
    assert config["fallback_max_active_jobs"] == "2"
    assert config["b300_max_active_gpus"] == "32"
    assert config["fallback_max_active_gpus"] == "160"
    assert config["fallback_prewarm_ttl_seconds"] == "600"
    assert config["fallback_inflight_without_pod_grace_seconds"] == "300"
    assert config["fallback_nodegroup_not_active_cooldown_seconds"] == "120"
    assert config["fallback_capacity_cooldown_seconds"] == "600"
    assert config["fallback_hard_failure_cooldown_seconds"] == "3600"
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

    def list_namespaced_job(self, namespace, label_selector=None):
        if not label_selector:
            return {"items": list(self.jobs.values())}
        key, _, expected = label_selector.partition("=")
        items = []
        for job in self.jobs.values():
            labels = job.get("metadata", {}).get("labels", {})
            if labels.get(key) == expected:
                items.append(job)
        return {"items": items}

    def delete_namespaced_job(self, name, namespace, **kwargs):
        self.deleted.append({"namespace": namespace, "name": name, **kwargs})


class FakeNotFoundBatchV1(FakeBatchV1):
    def read_namespaced_job_status(self, name, namespace):
        error = Exception("NotFound")
        error.status = 404
        raise error


class FakeEks:
    def __init__(self):
        self.updates = []

    def update_nodegroup_config(self, **kwargs):
        self.updates.append(kwargs)
        return {"update": {"status": "InProgress"}}


class FakeEksWithDesired(FakeEks):
    def __init__(self, desired_size: int):
        super().__init__()
        self.desired_size = desired_size

    def describe_nodegroup(self, **kwargs):
        return {"nodegroup": {"scalingConfig": {"desiredSize": self.desired_size}}}


class FakeEksWithHealth(FakeEksWithDesired):
    def __init__(self, desired_size: int, issues: list[dict]):
        super().__init__(desired_size)
        self.issues = issues

    def describe_nodegroup(self, **kwargs):
        return {
            "nodegroup": {
                "scalingConfig": {"desiredSize": self.desired_size},
                "health": {"issues": self.issues},
            }
        }


class FakeEksWithStatus(FakeEksWithDesired):
    def __init__(self, desired_size: int, status: str):
        super().__init__(desired_size)
        self.status = status

    def describe_nodegroup(self, **kwargs):
        return {
            "nodegroup": {
                "status": self.status,
                "scalingConfig": {"desiredSize": self.desired_size},
            }
        }


class FakeEksResourceInUse(FakeEksWithDesired):
    def update_nodegroup_config(self, **kwargs):
        error = Exception("Nodegroup cannot be updated as it is currently not in Active State")
        error.response = {"Error": {"Code": "ResourceInUseException"}}
        raise error


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], dict]):
        self.objects = objects
        self.reads = []

    def get_object(self, *, Bucket, Key):
        self.reads.append({"Bucket": Bucket, "Key": Key})
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(json.dumps(payload).encode("utf-8"))}


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


def _backend_gpu_pod(backend_name: str, gpus: int = 8, phase: str = "Running") -> dict:
    return {
        "metadata": {"labels": {"sglang.seedleap.io/backend": backend_name}},
        "status": {"phase": phase},
        "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": str(gpus)}}}]},
    }


def _job_pending_pod(job_name: str, backend_name: str, gpus: int = 8) -> dict:
    return {
        "metadata": {
            "name": f"{job_name}-pod",
            "labels": {
                "job-name": job_name,
                "batch.kubernetes.io/job-name": job_name,
                "sglang.seedleap.io/backend": backend_name,
            },
        },
        "status": {"phase": "Pending"},
        "spec": {
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": str(gpus)}}}]
        },
    }


def _job_scheduled_pending_pod(
    job_name: str,
    backend_name: str,
    node_name: str = "node-a",
    gpus: int = 8,
) -> dict:
    pod = _job_pending_pod(job_name, backend_name, gpus=gpus)
    pod["spec"]["nodeName"] = node_name
    pod["status"]["conditions"] = [{"type": "PodScheduled", "status": "True"}]
    return pod


def _backend_config() -> dict:
    return {
        "namespace": "default",
        "job_image": "lmsysorg/sglang:dev@sha256:test",
        "service_account": "sglang-video-job",
        "fsx_claim_name": "fsx-claim",
        "gpu_per_pod": 8,
        "job_parallelism": 1,
        "max_active_jobs": 7,
        "b300_max_active_jobs": 5,
        "fallback_max_active_jobs": 2,
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


def test_choose_backend_respects_b300_cap_before_fallback_pool_cap():
    config = {
        **_backend_config(),
        "b300_max_active_gpus": 32,
        "fallback_max_active_gpus": 160,
    }
    b300 = _node("b300-free", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    b300_pods_at_cap = [
        _backend_gpu_pod("b300-capacity-block", 8),
        _backend_gpu_pod("b300-capacity-block", 8),
        _backend_gpu_pod("b300-capacity-block", 8),
        _backend_gpu_pod("b300-capacity-block", 8),
    ]

    selected = choose_backend(
        config,
        [b300, h100],
        b300_pods_at_cap,
        requested_gpus=8,
    )

    assert selected["name"] == "h100-spot"


def test_choose_backend_allows_fallback_pool_to_160_gpus():
    config = {
        **_backend_config(),
        "fallback_max_active_gpus": 160,
    }
    h100_nodes = [
        _node(f"h100-{index}", config["backends"][1]["node_selector"])
        for index in range(20)
    ]
    state = {"inflight": [{"backend": "h100-spot", "requested_gpus": 152}]}

    selected = choose_backend(config, h100_nodes, [], requested_gpus=8, state=state)
    blocked = choose_backend(
        config,
        h100_nodes,
        [],
        requested_gpus=8,
        state={"inflight": [{"backend": "h100-spot", "requested_gpus": 160}]},
    )

    assert selected["name"] == "h100-spot"
    assert blocked is None


def test_choose_backend_treats_p5e_and_p5en_as_fallback_gpu_backends():
    for instance_type in ("p5e.48xlarge", "p5en.48xlarge"):
        config = {
            **_backend_config(),
            "fallback_max_active_gpus": 160,
            "backends": [
                {
                    "name": f"h200-{instance_type.split('.')[0]}-spot",
                    "node_selector": {
                        "eks.amazonaws.com/capacityType": "SPOT",
                        "node.kubernetes.io/instance-type": instance_type,
                    },
                    "scale_nodegroup": True,
                },
            ],
        }

        node = _node("fallback-a", config["backends"][0]["node_selector"])
        selected = choose_backend(config, [node], [], requested_gpus=8)

        assert selected["node_selector"]["node.kubernetes.io/instance-type"] == instance_type


def test_choose_backend_spreads_fallback_messages_to_least_busy_backend():
    config = {
        **_backend_config(),
        "fallback_max_active_gpus": 160,
        "fallback_max_active_jobs": 10,
        "backends": [
            {
                "name": "b200-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
                },
            },
            {
                "name": "h100-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "node.kubernetes.io/instance-type": "p5.48xlarge",
                },
            },
            {
                "name": "h200-p5e-spot",
                "node_selector": {
                    "eks.amazonaws.com/capacityType": "SPOT",
                    "node.kubernetes.io/instance-type": "p5e.48xlarge",
                },
            },
        ],
    }
    state = {
        "inflight": [
            {"backend": "b200-spot", "requested_gpus": 16},
            {"backend": "h100-spot", "requested_gpus": 8},
        ]
    }
    nodes = [
        _node("b200-a", config["backends"][0]["node_selector"]),
        _node("h100-a", config["backends"][1]["node_selector"]),
        _node("h200-a", config["backends"][2]["node_selector"]),
    ]

    selected = choose_backend(config, nodes, [], requested_gpus=8, state=state)

    assert selected["name"] == "h200-p5e-spot"


def test_choose_backend_prefers_b300_until_five_jobs_then_uses_spot():
    config = {
        **_backend_config(),
        "max_active_jobs": 7,
        "b300_max_active_jobs": 5,
        "fallback_max_active_jobs": 2,
        "b300_max_active_gpus": 40,
        "fallback_max_active_gpus": 160,
    }
    b300_nodes = [
        _node(f"b300-{index}", config["backends"][0]["node_selector"])
        for index in range(5)
    ]
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    four_b300_jobs = {
        "inflight": [
            {
                "backend": "b300-capacity-block",
                "job_name": f"sglang-video-b300-{index}",
                "requested_gpus": 8,
            }
            for index in range(4)
        ]
    }
    five_b300_jobs = {
        "inflight": [
            {
                "backend": "b300-capacity-block",
                "job_name": f"sglang-video-b300-{index}",
                "requested_gpus": 8,
            }
            for index in range(5)
        ]
    }

    selected_b300 = choose_backend(
        config,
        [*b300_nodes, h100],
        [],
        requested_gpus=8,
        state=four_b300_jobs,
    )
    selected_spot = choose_backend(
        config,
        [*b300_nodes, h100],
        [],
        requested_gpus=8,
        state=five_b300_jobs,
    )

    assert selected_b300["name"] == "b300-capacity-block"
    assert selected_spot["name"] == "h100-spot"


def test_choose_backend_does_not_count_callback_pending_jobs_against_b300_capacity():
    config = {
        **_backend_config(),
        "max_active_jobs": 7,
        "b300_max_active_jobs": 5,
        "fallback_max_active_jobs": 2,
        "b300_max_active_gpus": 40,
        "fallback_max_active_gpus": 160,
    }
    b300_nodes = [
        _node(f"b300-{index}", config["backends"][0]["node_selector"])
        for index in range(5)
    ]
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    state = {
        "inflight": [
            {
                "backend": "b300-capacity-block",
                "job_name": f"sglang-video-b300-{index}",
                "requested_gpus": 8,
            }
            for index in range(4)
        ]
        + [
            {
                "backend": "b300-capacity-block",
                "job_name": "sglang-video-callback-pending",
                "requested_gpus": 8,
                "callback_pending": True,
                "gpu_released_at": 1100.0,
            }
        ]
    }

    selected = choose_backend(
        config,
        [*b300_nodes, h100],
        [],
        requested_gpus=8,
        state=state,
    )

    assert selected["name"] == "b300-capacity-block"


def test_choose_backend_limits_fallback_to_two_jobs_even_when_gpu_cap_is_higher():
    config = {
        **_backend_config(),
        "max_active_jobs": 7,
        "b300_max_active_jobs": 5,
        "fallback_max_active_jobs": 2,
        "b300_max_active_gpus": 40,
        "fallback_max_active_gpus": 160,
    }
    h100_nodes = [
        _node(f"h100-{index}", config["backends"][1]["node_selector"])
        for index in range(2)
    ]
    one_spot_job = {
        "inflight": [
            {
                "backend": "b300-capacity-block",
                "job_name": f"sglang-video-b300-{index}",
                "requested_gpus": 8,
            }
            for index in range(5)
        ]
        + [
            {
                "backend": "h100-spot",
                "job_name": "sglang-video-h100-0",
                "requested_gpus": 8,
            }
        ]
    }
    two_spot_jobs = {
        "inflight": one_spot_job["inflight"]
        + [
            {
                "backend": "h100-spot",
                "job_name": "sglang-video-h100-1",
                "requested_gpus": 8,
            }
        ]
    }

    selected = choose_backend(config, h100_nodes, [], requested_gpus=8, state=one_spot_job)
    blocked = choose_backend(config, h100_nodes, [], requested_gpus=8, state=two_spot_jobs)

    assert selected["name"] == "h100-spot"
    assert blocked is None


def test_choose_backend_enforces_total_job_cap_across_b300_and_spot():
    config = {
        **_backend_config(),
        "max_active_jobs": 7,
        "b300_max_active_jobs": 5,
        "fallback_max_active_jobs": 2,
        "b300_max_active_gpus": 40,
        "fallback_max_active_gpus": 160,
    }
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    state = {
        "inflight": [
            {
                "backend": "b300-capacity-block",
                "job_name": f"sglang-video-b300-{index}",
                "requested_gpus": 8,
            }
            for index in range(5)
        ]
        + [
            {
                "backend": "h100-spot",
                "job_name": f"sglang-video-h100-{index}",
                "requested_gpus": 8,
            }
            for index in range(2)
        ]
    }

    assert choose_backend(config, [b300, h100], [], requested_gpus=8, state=state) is None


def test_choose_backend_does_not_double_count_h100_pods_and_inflight_state():
    config = {**_backend_config(), "fallback_max_active_jobs": 10}
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


def test_scale_fallback_nodegroups_counts_existing_backend_pods_after_restart():
    config = _backend_config()
    eks = FakeEks()
    state = {"inflight": []}
    pods = [
        {
            "metadata": {"labels": {"sglang.seedleap.io/backend": "h100-spot"}},
            "status": {"phase": "Pending"},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]},
        }
    ]

    _scale_fallback_nodegroups(eks, config, state, pods=pods)

    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 1


def test_scale_fallback_nodegroups_holds_recent_inflight_before_pod_appears():
    config = {
        **_backend_config(),
        "fallback_inflight_without_pod_grace_seconds": 300,
    }
    eks = FakeEksWithDesired(desired_size=0)
    state = {
        "inflight": [
            {
                "backend": "h100-spot",
                "job_name": "sglang-video-starting",
                "requested_gpus": 8,
                "started_at": 1000.0,
            }
        ]
    }

    _scale_fallback_nodegroups(eks, config, state, pods=[], now=1060.0)

    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 1


def test_scale_fallback_nodegroups_releases_stale_inflight_without_active_pod():
    config = {
        **_backend_config(),
        "fallback_scale_down_grace_seconds": 0,
        "fallback_inflight_without_pod_grace_seconds": 300,
    }
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [
            {
                "backend": "h100-spot",
                "job_name": "sglang-video-gone",
                "requested_gpus": 8,
                "started_at": 1000.0,
            }
        ],
        "fallback_scale_down": {"leap-world-aws03-usw2/sglang-h100-spot": 1500.0},
    }

    _scale_fallback_nodegroups(eks, config, state, pods=[], now=1600.0)

    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_scale_fallback_nodegroups_counts_prewarmed_backend_capacity():
    config = _backend_config()
    eks = FakeEksWithDesired(desired_size=0)
    state = {
        "inflight": [],
        "fallback_prewarm": {
            "h100-spot": {
                "requested_gpus": 8,
                "started_at": 1000.0,
                "updated_at": 1000.0,
            }
        },
    }

    _scale_fallback_nodegroups(eks, config, state, pods=[], now=1000.0)

    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 1


def test_scale_fallback_nodegroups_skips_noop_desired_size_update():
    config = _backend_config()
    eks = FakeEksWithDesired(desired_size=0)

    _scale_fallback_nodegroups(eks, config, {"inflight": []}, pods=[])

    assert eks.updates == []


def test_scale_fallback_nodegroups_holds_positive_nodegroup_before_scale_down():
    config = {**_backend_config(), "fallback_scale_down_grace_seconds": 900}
    eks = FakeEksWithDesired(desired_size=1)
    state = {"inflight": []}

    _scale_fallback_nodegroups(eks, config, state, pods=[], now=1000.0)

    assert eks.updates == []
    assert state["fallback_scale_down"]["leap-world-aws03-usw2/sglang-h100-spot"] == 1000.0


def test_scale_fallback_nodegroups_scales_down_after_grace_period():
    config = {**_backend_config(), "fallback_scale_down_grace_seconds": 900}
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [],
        "fallback_scale_down": {"leap-world-aws03-usw2/sglang-h100-spot": 1000.0},
    }

    _scale_fallback_nodegroups(eks, config, state, pods=[], now=2000.0)

    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_controller_tick_idle_still_scales_down_empty_fallback_nodegroup():
    config = {**_backend_config(), "fallback_scale_down_grace_seconds": 60}
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [],
        "fallback_scale_down": {"leap-world-aws03-usw2/sglang-h100-spot": 1000.0},
    }

    result = controller_tick(
        sqs_client=FakeSQS([]),
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], []),
        batch_client=FakeBatchV1(),
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["status"] == "idle"
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_controller_tick_idle_releases_unused_fallback_prewarm():
    config = {**_backend_config(), "fallback_scale_down_grace_seconds": 60}
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [],
        "fallback_prewarm": {
            "h100-spot": {
                "requested_gpus": 8,
                "requested_jobs": 1,
                "started_at": 1000.0,
                "updated_at": 1000.0,
            }
        },
    }

    result = controller_tick(
        sqs_client=FakeSQS([]),
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], []),
        batch_client=FakeBatchV1(),
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["status"] == "idle"
    assert result["released_prewarm"] == 1
    assert state["fallback_prewarm"] == {}
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_controller_tick_releases_unused_fallback_prewarm_even_with_messages():
    config = {
        **_backend_config(),
        "fallback_scale_down_grace_seconds": 0,
    }
    batch = FakeBatchV1()
    eks = FakeEksWithDesired(desired_size=1)
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    state = {
        "inflight": [],
        "fallback_prewarm": {
            "h100-spot": {
                "requested_gpus": 8,
                "requested_jobs": 1,
                "started_at": 1000.0,
                "updated_at": 1000.0,
            }
        },
    }

    result = controller_tick(
        sqs_client=FakeSQS(_request()),
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], [b300]),
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["status"] == "started"
    assert result["released_prewarm"] == 1
    assert len(batch.created) == 1
    assert batch.created[0]["body"]["metadata"]["labels"]["sglang.seedleap.io/backend"] == "b300-capacity-block"
    assert state["fallback_prewarm"] == {}
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_reconcile_releases_stale_prewarm_when_pending_fallback_job_is_released():
    config = {
        **_backend_config(),
        "pending_job_grace_seconds": 180,
        "fallback_scale_down_grace_seconds": 0,
    }
    batch = FakeBatchV1()
    batch.jobs["sglang-video-stuck"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-stuck",
                "backend": "h100-spot",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
                "pending_job_seen_at": 1000.0,
            }
        ],
        "fallback_prewarm": {
            "h100-spot": {
                "requested_gpus": 8,
                "requested_jobs": 1,
                "started_at": 1000.0,
                "updated_at": 1000.0,
            }
        },
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([_job_pending_pod("sglang-video-stuck", "h100-spot")], []),
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1300.0,
    )

    assert result["released"] == 1
    assert state["fallback_prewarm"] == {}
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_scale_fallback_nodegroups_ignores_resource_in_use_update_conflict():
    config = _backend_config()
    eks = FakeEksResourceInUse(desired_size=0)
    state = {"inflight": [{"backend": "h100-spot", "requested_gpus": 8}]}

    _scale_fallback_nodegroups(eks, config, state, pods=[])


def test_controller_tick_adopts_existing_job_after_restart_and_scales_nodes():
    config = _backend_config()
    request = _request()
    existing_job = render_job_manifest(
        request,
        {
            **config,
            "placement_profiles": [config["backends"][1]],
            "selected_backend": "h100-spot",
        },
    )
    batch = FakeBatchV1()
    batch.jobs[existing_job["metadata"]["name"]] = existing_job
    sqs = FakeSQS(request)
    eks = FakeEks()
    core = FakeCoreV1(
        [
            {
                "metadata": {"labels": {"sglang.seedleap.io/backend": "h100-spot"}},
                "status": {"phase": "Pending"},
                "spec": {
                    "containers": [
                        {"resources": {"requests": {"nvidia.com/gpu": "8"}}}
                    ]
                },
            }
        ],
        [],
    )
    state = {"inflight": []}

    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1000.0,
    )

    assert result["adopted"] == 1
    assert result["started"] == 0
    assert batch.created == []
    assert state["inflight"][0]["job_name"] == existing_job["metadata"]["name"]
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 1


def test_controller_tick_starts_multiple_messages_up_to_fallback_job_cap_and_scales_nodes():
    requests = [{**_request(), "generation_job_id": f"gen_t2i_{index}"} for index in range(5)]
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100_a = _node("h100-a", config["backends"][1]["node_selector"])
    h100_b = _node("h100-b", config["backends"][1]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300, h100_a, h100_b])
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

    assert result["started"] == 2
    assert result["deferred"] == 3
    assert len(batch.created) == 2
    assert all(
        job["body"]["metadata"]["labels"]["sglang.seedleap.io/backend"] == "h100-spot"
        for job in batch.created
    )
    assert sqs.deleted == []
    assert sqs.visibility_changes[-1]["ReceiptHandle"] == "receipt-5"
    assert sqs.visibility_changes[-1]["VisibilityTimeout"] > 0
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 2
    assert len(state["inflight"]) == 2


def test_controller_tick_b300_start_consumes_fallback_prewarm():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    batch = FakeBatchV1()
    sqs = FakeSQS(_request())
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [],
        "fallback_prewarm": {
            "h100-spot": {
                "requested_gpus": 8,
                "requested_jobs": 1,
                "started_at": 1000.0,
                "updated_at": 1000.0,
            }
        },
    }

    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], [b300]),
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["started"] == 1
    assert batch.created[0]["body"]["metadata"]["labels"]["sglang.seedleap.io/backend"] == (
        "b300-capacity-block"
    )
    assert state["fallback_prewarm"] == {}
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_controller_tick_prewarms_fallback_without_creating_pending_job_when_no_spot_node_ready():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300])
    batch = FakeBatchV1()
    sqs = FakeSQS(_request())
    eks = FakeEksWithDesired(desired_size=0)
    state = {"inflight": []}

    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1000.0,
    )

    assert result["started"] == 0
    assert result["deferred"] == 1
    assert result["prewarmed"] == 1
    assert batch.created == []
    assert state["fallback_prewarm"]["h100-spot"]["requested_gpus"] == 8
    assert sqs.visibility_changes[-1]["ReceiptHandle"] == "receipt-1"
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 1


def test_controller_tick_does_not_overcommit_single_ready_fallback_node():
    requests = [{**_request(), "generation_job_id": f"gen_t2i_{index}"} for index in range(3)]
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300, h100])
    batch = FakeBatchV1()
    sqs = FakeSQS(requests)
    eks = FakeEksWithDesired(desired_size=1)
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

    assert result["started"] == 1
    assert result["deferred"] == 2
    assert result["prewarmed"] == 1
    assert len(batch.created) == 1
    assert len(state["inflight"]) == 1
    assert state["fallback_prewarm"]["h100-spot"]["requested_gpus"] == 8
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 2


def test_controller_tick_skips_unhealthy_fallback_prewarm_and_does_not_create_job():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300])
    batch = FakeBatchV1()
    sqs = FakeSQS(_request())
    eks = FakeEksWithHealth(
        desired_size=0,
        issues=[
            {
                "code": "InvalidFleetConfiguration",
                "message": "p5e.48xlarge is not supported in your requested Availability Zone",
            }
        ],
    )
    state = {"inflight": []}

    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1000.0,
    )

    assert result["started"] == 0
    assert result["deferred"] == 1
    assert result["prewarm_skipped"] == 1
    assert batch.created == []
    assert eks.updates == []
    assert state["fallback_backend_cooldown"]["h100-spot"] > 1000.0


def test_controller_tick_skips_non_active_fallback_prewarm_while_nodegroup_updates():
    config = _backend_config()
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    core = FakeCoreV1([_gpu_pod("busy", "b300-a", 8)], [b300])
    batch = FakeBatchV1()
    sqs = FakeSQS(_request())
    eks = FakeEksWithStatus(desired_size=0, status="UPDATING")
    state = {"inflight": []}

    result = controller_tick(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=core,
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1000.0,
    )

    assert result["started"] == 0
    assert result["deferred"] == 1
    assert result["prewarm_skipped"] == 1
    assert batch.created == []
    assert eks.updates == []
    assert state["fallback_prewarm"] == {}
    assert state["fallback_backend_cooldown"]["h100-spot"] > 1000.0


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
    batch.jobs["sglang-video-old"] = {
        "status": {"conditions": [{"type": "Failed", "status": "True"}]}
    }
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


def test_reconcile_releases_failed_fallback_job_when_no_backend_is_available():
    config = {**_backend_config(), "fallback_scale_down_grace_seconds": 60}
    batch = FakeBatchV1()
    batch.jobs["sglang-video-old"] = {
        "status": {"conditions": [{"type": "Failed", "status": "True"}]}
    }
    sqs = FakeSQS(_request())
    eks = FakeEksWithDesired(desired_size=1)
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
                "requested_gpus": 8,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], []),
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["released"] == 1
    assert result["restarted"] == 0
    assert batch.deleted[0]["name"] == "sglang-video-old"
    assert sqs.visibility_changes == [
        {
            "QueueUrl": "https://sqs.us-west-2.amazonaws.com/123/video",
            "ReceiptHandle": "receipt-1",
            "VisibilityTimeout": 60,
        }
    ]
    assert state["inflight"] == []
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_reconcile_keeps_inflight_job_when_failed_count_is_not_terminal():
    config = _backend_config()
    batch = FakeBatchV1()
    batch.jobs["sglang-video-pending"] = {"status": {"active": 1, "failed": 1}}
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-pending",
                "backend": "b200-spot",
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

    assert result["restarted"] == 0
    assert result["failed"] == 0
    assert batch.deleted == []
    assert sqs.deleted == []
    assert state["inflight"][0]["job_name"] == "sglang-video-pending"


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


def test_reconcile_repairs_final_progress_from_s3_report_before_deleting_message():
    config = _backend_config()
    batch = FakeBatchV1()
    batch.jobs["sglang-video-done"] = {"status": {"succeeded": 1}}
    request = _request()
    request["output"]["report_s3_uri"] = "s3://bucket/t2i/reports/sglang_video_report.json"
    report = {
        "summary": {
            "video_status": "succeeded",
            "video_expected_count": 1,
            "video_succeeded_count": 1,
            "video_failed_count": 0,
            "video_running_count": 0,
            "video_output_prefix": "s3://bucket/t2i/videos",
        },
        "counters": {"total": 1, "succeeded": 1, "failed": 0, "running": 0},
        "results": [
            {
                "case_id": "img001-action-00-traj001",
                "status": "succeeded",
                "video_uri": "s3://bucket/t2i/videos/img001/00_traj001.mp4",
                "traj_id": "traj001",
                "movement_key": "w",
                "ending_movement_key": "a",
                "movement_pair": "w+a",
                "camera_key": "",
                "action_seed": 20260715,
                "action_pattern": "api:combo:w-a",
            }
        ],
    }
    s3 = FakeS3({("bucket", "t2i/reports/sglang_video_report.json"): report})
    sent = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(http_request, timeout):
        sent["url"] = http_request.full_url
        sent["body"] = json.loads(http_request.data.decode("utf-8"))
        sent["timeout"] = timeout
        return FakeResponse()

    sqs = FakeSQS(request)
    state = {
        "inflight": [
            {
                "request": request,
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
        s3_client=s3,
        callback_urlopen=fake_urlopen,
    )

    assert result["completed"] == 1
    assert result["callback_repaired"] == 1
    assert sqs.deleted[0]["ReceiptHandle"] == "receipt-1"
    assert s3.reads == [{"Bucket": "bucket", "Key": "t2i/reports/sglang_video_report.json"}]
    assert sent["url"].endswith("/api/v1/generation/jobs/gen_t2i_001/progress")
    assert sent["body"]["status"] == "succeeded"
    assert sent["body"]["summary"]["video_succeeded_count"] == 1
    assert sent["body"]["items"][0]["item_id"] == "img001"
    assert sent["body"]["items"][0]["metadata"]["videos"][0]["traj_id"] == "traj001"
    assert state["inflight"] == []


def test_reconcile_keeps_sqs_message_when_final_progress_repair_fails():
    config = _backend_config()
    batch = FakeBatchV1()
    batch.jobs["sglang-video-done"] = {"status": {"succeeded": 1}}
    request = _request()
    request["output"]["report_s3_uri"] = "s3://bucket/t2i/reports/sglang_video_report.json"
    report = {
        "summary": {
            "video_status": "succeeded",
            "video_expected_count": 1,
            "video_succeeded_count": 1,
            "video_failed_count": 0,
            "video_running_count": 0,
        },
        "counters": {"total": 1, "succeeded": 1, "failed": 0, "running": 0},
        "results": [
            {
                "case_id": "img001-action-00-traj001",
                "status": "succeeded",
                "video_uri": "s3://bucket/t2i/videos/img001/00_traj001.mp4",
            }
        ],
    }
    s3 = FakeS3({("bucket", "t2i/reports/sglang_video_report.json"): report})

    def failing_urlopen(_http_request, timeout):
        raise OSError("callback unavailable")

    sqs = FakeSQS(request)
    state = {
        "inflight": [
            {
                "request": request,
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
        s3_client=s3,
        callback_urlopen=failing_urlopen,
    )

    assert result["completed"] == 0
    assert result["callback_pending"] == 1
    assert sqs.deleted == []
    assert state["inflight"][0]["job_name"] == "sglang-video-done"
    assert state["inflight"][0]["callback_pending"] is True
    assert state["inflight"][0]["gpu_released_at"] == 1100.0
    assert "callback unavailable" in state["inflight"][0]["callback_error"]


def test_reconcile_repairs_callback_pending_without_restarting_deleted_completed_job():
    config = _backend_config()
    request = _request()
    request["output"]["report_s3_uri"] = "s3://bucket/t2i/reports/sglang_video_report.json"
    report = {
        "summary": {
            "video_status": "succeeded",
            "video_expected_count": 1,
            "video_succeeded_count": 1,
            "video_failed_count": 0,
            "video_running_count": 0,
        },
        "counters": {"total": 1, "succeeded": 1, "failed": 0, "running": 0},
        "results": [
            {
                "case_id": "img001-action-00-traj001",
                "status": "succeeded",
                "video_uri": "s3://bucket/t2i/videos/img001/00_traj001.mp4",
            }
        ],
    }
    s3 = FakeS3({("bucket", "t2i/reports/sglang_video_report.json"): report})

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(_http_request, timeout):
        return FakeResponse()

    sqs = FakeSQS(request)
    batch = FakeNotFoundBatchV1()
    state = {
        "inflight": [
            {
                "request": request,
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-done",
                "backend": "b300-capacity-block",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "callback_pending": True,
                "gpu_released_at": 1050.0,
                "callback_error": "OSError: callback unavailable",
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], [_node("b300-a", config["backends"][0]["node_selector"])]),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1100.0,
        s3_client=s3,
        callback_urlopen=fake_urlopen,
    )

    assert result["completed"] == 1
    assert result["callback_repaired"] == 1
    assert batch.created == []
    assert sqs.deleted[0]["ReceiptHandle"] == "receipt-1"
    assert state["inflight"] == []


def test_reconcile_keeps_inflight_job_when_status_read_temporarily_404s():
    config = _backend_config()
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-pending",
                "backend": "b200-spot",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
            }
        ]
    }
    batch = FakeNotFoundBatchV1()

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

    assert result["failed"] == 0
    assert result["restarted"] == 0
    assert result["missing"] == 1
    assert batch.deleted == []
    assert sqs.deleted == []
    assert state["inflight"][0]["job_name"] == "sglang-video-pending"
    assert state["inflight"][0]["missing_job_seen_at"] == 1100.0


def test_reconcile_marks_pending_job_before_recreating_on_fallback():
    config = {**_backend_config(), "pending_job_grace_seconds": 180}
    batch = FakeBatchV1()
    batch.jobs["sglang-video-stuck"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-stuck",
                "backend": "b300-capacity-block",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1(
            [_job_pending_pod("sglang-video-stuck", "b300-capacity-block")],
            [],
        ),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["pending"] == 1
    assert result["restarted"] == 0
    assert batch.deleted == []
    assert batch.created == []
    assert sqs.deleted == []
    assert state["inflight"][0]["job_name"] == "sglang-video-stuck"
    assert state["inflight"][0]["pending_job_seen_at"] == 1100.0


def test_reconcile_recreates_unschedulable_b300_job_on_fallback_after_grace_period():
    config = {**_backend_config(), "pending_job_grace_seconds": 180}
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    h100 = _node("h100-a", config["backends"][1]["node_selector"])
    batch = FakeBatchV1()
    batch.jobs["sglang-video-stuck"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-stuck",
                "backend": "b300-capacity-block",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
                "pending_job_seen_at": 1000.0,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1(
            [_job_pending_pod("sglang-video-stuck", "b300-capacity-block")],
            [b300, h100],
        ),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1300.0,
    )

    assert result["pending"] == 1
    assert result["restarted"] == 1
    assert result["failed"] == 0
    assert sqs.deleted == []
    assert batch.deleted[0]["name"] == "sglang-video-stuck"
    assert len(batch.created) == 1
    created = batch.created[0]["body"]
    assert created["metadata"]["labels"]["sglang.seedleap.io/backend"] == "h100-spot"
    assert state["inflight"][0]["backend"] == "h100-spot"
    assert state["inflight"][0]["attempts"] == 2
    assert state["inflight"][0]["started_at"] == 1300.0
    assert state["inflight"][0]["job_name"].startswith("sglang-video-")
    assert "pending_job_seen_at" not in state["inflight"][0]


def test_reconcile_keeps_scheduled_pending_job_while_container_is_starting():
    config = {**_backend_config(), "pending_job_grace_seconds": 180}
    batch = FakeBatchV1()
    batch.jobs["sglang-video-starting"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-starting",
                "backend": "h100-spot",
                "attempts": 2,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
                "pending_job_seen_at": 1000.0,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1(
            [_job_scheduled_pending_pod("sglang-video-starting", "h100-spot")],
            [],
        ),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1300.0,
    )

    assert result["pending"] == 0
    assert result["restarted"] == 0
    assert batch.deleted == []
    assert batch.created == []
    assert state["inflight"][0]["job_name"] == "sglang-video-starting"


def test_reconcile_releases_pending_fallback_job_when_active_job_caps_are_exceeded():
    config = {**_backend_config(), "pending_job_grace_seconds": 180}
    batch = FakeBatchV1()
    batch.jobs["sglang-video-overflow"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    pods = [_job_pending_pod("sglang-video-overflow", "h100-spot")]
    pods.extend(
        _job_pending_pod(f"sglang-video-b300-{index}", "b300-capacity-block")
        for index in range(5)
    )
    pods.extend(
        _job_pending_pod(f"sglang-video-h100-{index}", "h100-spot")
        for index in range(2)
    )
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-overflow",
                "backend": "h100-spot",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1(pods, []),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1100.0,
    )

    assert result["released"] == 1
    assert result["restarted"] == 0
    assert batch.deleted[0]["name"] == "sglang-video-overflow"
    assert sqs.visibility_changes == [
        {
            "QueueUrl": "https://sqs.us-west-2.amazonaws.com/123/video",
            "ReceiptHandle": "receipt-1",
            "VisibilityTimeout": 60,
        }
    ]
    assert state["inflight"] == []


def test_reconcile_retargets_pending_fallback_job_to_b300_and_cancels_fallback_scale_up():
    config = {
        **_backend_config(),
        "pending_job_grace_seconds": 180,
        "fallback_scale_down_grace_seconds": 900,
    }
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    batch = FakeBatchV1()
    batch.jobs["sglang-video-stuck"] = {"status": {"active": 1}}
    sqs = FakeSQS(_request())
    eks = FakeEksWithDesired(desired_size=1)
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-stuck",
                "backend": "h100-spot",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
                "pending_job_seen_at": 1000.0,
            }
        ]
    }

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1(
            [_job_pending_pod("sglang-video-stuck", "h100-spot")],
            [b300],
        ),
        batch_client=batch,
        eks_client=eks,
        config=config,
        state=state,
        now=1300.0,
    )

    assert result["pending"] == 1
    assert result["restarted"] == 1
    assert result["failed"] == 0
    assert sqs.deleted == []
    assert batch.deleted[0]["name"] == "sglang-video-stuck"
    assert len(batch.created) == 1
    created = batch.created[0]["body"]
    assert created["metadata"]["labels"]["sglang.seedleap.io/backend"] == "b300-capacity-block"
    assert state["inflight"][0]["backend"] == "b300-capacity-block"
    assert state["inflight"][0]["attempts"] == 2
    assert state["inflight"][0]["started_at"] == 1300.0
    assert state["inflight"][0]["job_name"].startswith("sglang-video-")
    assert "pending_job_seen_at" not in state["inflight"][0]
    assert eks.updates[-1]["nodegroupName"] == "sglang-h100-spot"
    assert eks.updates[-1]["scalingConfig"]["desiredSize"] == 0


def test_reconcile_recreates_missing_inflight_job_after_grace_period():
    config = {**_backend_config(), "missing_job_grace_seconds": 180}
    b300 = _node("b300-a", config["backends"][0]["node_selector"])
    sqs = FakeSQS(_request())
    state = {
        "inflight": [
            {
                "request": _request(),
                "receipt_handle": "receipt-1",
                "message_id": "message-1",
                "job_name": "sglang-video-missing",
                "backend": "b300-capacity-block",
                "attempts": 1,
                "started_at": 1000.0,
                "last_renewed_at": 1000.0,
                "requested_gpus": 8,
            }
        ]
    }
    batch = FakeNotFoundBatchV1()

    result = reconcile_inflight_jobs(
        sqs_client=sqs,
        queue_url="https://sqs.us-west-2.amazonaws.com/123/video",
        core_client=FakeCoreV1([], [b300]),
        batch_client=batch,
        eks_client=FakeEks(),
        config=config,
        state=state,
        now=1300.0,
    )

    assert result["missing"] == 1
    assert result["restarted"] == 1
    assert result["failed"] == 0
    assert sqs.deleted == []
    assert batch.deleted == []
    assert len(batch.created) == 1
    assert state["inflight"][0]["attempts"] == 2
    assert state["inflight"][0]["job_name"].startswith("sglang-video-")
    assert "missing_job_seen_at" not in state["inflight"][0]


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
