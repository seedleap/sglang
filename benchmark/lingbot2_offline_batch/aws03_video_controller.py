#!/usr/bin/env python3
"""Controller helpers for launching t2i video batch jobs on aws03."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
DEFAULT_CONTAINER_COMMAND = [
    "python3",
    "/opt/bench/run_t2i_video_batch.py",
]


def _int_or_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _gpu_request(container: dict[str, Any]) -> int:
    resources = container.get("resources") if isinstance(container.get("resources"), dict) else {}
    requests = resources.get("requests") if isinstance(resources.get("requests"), dict) else {}
    return _int_or_default(requests.get("nvidia.com/gpu"), 0)


def active_gpu_requests(pods: list[dict[str, Any]]) -> int:
    """Return GPUs requested by non-terminal SGLang batch pods."""
    total = 0
    for pod in pods:
        status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        if status.get("phase") in TERMINAL_POD_PHASES:
            continue
        spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
        containers = spec.get("containers") if isinstance(spec.get("containers"), list) else []
        total += sum(_gpu_request(container) for container in containers)
    return total


def can_start_job(
    *,
    active_gpus: int,
    max_active_gpus: int,
    gpu_per_pod: int,
    parallelism: int,
) -> bool:
    """Return whether starting a new batch job stays within the GPU cap."""
    requested_gpus = gpu_per_pod * parallelism
    return active_gpus + requested_gpus <= max_active_gpus


def _safe_name(value: str, fallback: str) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return (safe or fallback)[:48].rstrip("-") or fallback


def _metadata_name(request: dict[str, Any]) -> str:
    raw = str(
        request.get("generation_job_id")
        or request.get("request_id")
        or request.get("idempotency_key")
        or "t2i-video"
    )
    return f"sglang-video-{_safe_name(raw, 't2i-video')}"


def _configured_resources(config: dict[str, Any], gpu_per_pod: int) -> dict[str, dict[str, Any]]:
    requests: dict[str, Any] = {"nvidia.com/gpu": gpu_per_pod}
    limits: dict[str, Any] = {"nvidia.com/gpu": gpu_per_pod}
    if config.get("request_cpu"):
        requests["cpu"] = str(config["request_cpu"])
    if config.get("request_memory"):
        requests["memory"] = str(config["request_memory"])
    if config.get("limit_cpu"):
        limits["cpu"] = str(config["limit_cpu"])
    if config.get("limit_memory"):
        limits["memory"] = str(config["limit_memory"])
    return {"limits": limits, "requests": requests}


def _node_selector_term(selector: dict[str, Any]) -> dict[str, Any]:
    return {
        "matchExpressions": [
            {
                "key": str(key),
                "operator": "In",
                "values": [str(item) for item in value]
                if isinstance(value, list)
                else [str(value)],
            }
            for key, value in sorted(selector.items())
            if value not in (None, "")
        ],
    }


def _placement_profile_term(profile: dict[str, Any]) -> dict[str, Any] | None:
    raw_term = profile.get("node_selector_term")
    if isinstance(raw_term, dict):
        return raw_term
    selector = profile.get("node_selector")
    if isinstance(selector, dict) and selector:
        return _node_selector_term(selector)
    return None


def _placement_profiles_affinity(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    required_terms: list[dict[str, Any]] = []
    preferred_terms: list[dict[str, Any]] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        term = _placement_profile_term(profile)
        if not term:
            continue
        required_terms.append(term)
        weight = _int_or_default(profile.get("weight"), 0)
        if weight > 0:
            preferred_terms.append(
                {
                    "weight": min(weight, 100),
                    "preference": term,
                }
            )

    if not required_terms:
        return {}

    node_affinity: dict[str, Any] = {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": required_terms,
        },
    }
    if preferred_terms:
        node_affinity["preferredDuringSchedulingIgnoredDuringExecution"] = preferred_terms
    return {"nodeAffinity": node_affinity}


def _configured_affinity(config: dict[str, Any]) -> dict[str, Any]:
    affinity = config.get("affinity") if isinstance(config.get("affinity"), dict) else {}
    profiles = (
        config.get("placement_profiles")
        if isinstance(config.get("placement_profiles"), list)
        else []
    )
    profile_affinity = _placement_profiles_affinity(profiles)
    if not profile_affinity:
        return dict(affinity)
    merged = dict(affinity)
    merged["nodeAffinity"] = profile_affinity["nodeAffinity"]
    return merged


def render_job_manifest(request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Render a Kubernetes Job manifest for one SGLang t2i video batch."""
    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), 8)
    parallelism = _int_or_default(config.get("job_parallelism"), 1)
    timeout_seconds = _int_or_default(
        config.get("timeout_seconds"),
        21600,
    )
    fsx_mount_path = str(config.get("fsx_mount_path") or "/fsx")
    fsx_claim_name = str(config.get("fsx_claim_name") or "fsx-claim")
    job_name = _metadata_name(request)
    volume_mounts = [
        {
            "name": "fsx",
            "mountPath": fsx_mount_path,
        }
    ]
    volumes = [
        {
            "name": "fsx",
            "persistentVolumeClaim": {"claimName": fsx_claim_name},
        }
    ]
    shm_size = str(config.get("shm_size") or "").strip()
    if shm_size:
        volume_mounts.append({"name": "shm", "mountPath": "/dev/shm"})
        volumes.append({"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": shm_size}})

    container = {
        "name": "sglang-video-batch",
        "image": config["job_image"],
        "imagePullPolicy": config.get("image_pull_policy", "IfNotPresent"),
        "command": config.get("command", DEFAULT_CONTAINER_COMMAND),
        "env": [
            {
                "name": "SGLANG_VIDEO_BATCH_REQUEST_JSON",
                "value": json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            {
                "name": "SGLANG_VIDEO_BATCH_WORK_DIR",
                "value": f"{config.get('work_dir_prefix', fsx_mount_path).rstrip('/')}/{job_name}",
            },
            *config.get("extra_env", []),
        ],
        "resources": _configured_resources(config, gpu_per_pod),
        "volumeMounts": volume_mounts,
    }
    if config.get("args"):
        container["args"] = config["args"]
    if config.get("security_context"):
        container["securityContext"] = config["security_context"]

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccountName": config.get("service_account", "sglang-video-job"),
        "containers": [container],
        "volumes": volumes,
    }
    if config.get("node_selector"):
        pod_spec["nodeSelector"] = config["node_selector"]
    affinity = _configured_affinity(config)
    if affinity:
        pod_spec["affinity"] = affinity
    if config.get("priority_class_name"):
        pod_spec["priorityClassName"] = config["priority_class_name"]
    if config.get("scheduler_name"):
        pod_spec["schedulerName"] = config["scheduler_name"]
    if config.get("tolerations"):
        pod_spec["tolerations"] = config["tolerations"]

    job_spec: dict[str, Any] = {
        "parallelism": parallelism,
        "completions": parallelism,
        "backoffLimit": _int_or_default(config.get("backoff_limit"), 0),
        "activeDeadlineSeconds": timeout_seconds,
        "template": {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/name": "sglang-video-batch",
                    "app.kubernetes.io/component": "t2i-video-job",
                },
            },
            "spec": {
                **pod_spec,
            },
        },
    }
    ttl_seconds = config.get("ttl_seconds_after_finished")
    if ttl_seconds not in (None, ""):
        job_spec["ttlSecondsAfterFinished"] = _int_or_default(ttl_seconds, 0)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": config.get("namespace", "default"),
            "labels": {
                "app.kubernetes.io/name": "sglang-video-batch",
                "app.kubernetes.io/component": "t2i-video-job",
                "sglang.seedleap.io/generation-job-id": str(
                    request.get("generation_job_id") or ""
                ),
            },
        },
        "spec": job_spec,
    }


def _object_to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _list_controller_pods(core_client: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    namespace = config.get("namespace", "default")
    label_selector = config.get(
        "pod_label_selector",
        "app.kubernetes.io/name=sglang-video-batch",
    )
    response = core_client.list_namespaced_pod(
        namespace=namespace,
        label_selector=label_selector,
    )
    response = _object_to_dict(response)
    items = response.get("items", []) if isinstance(response, dict) else response.items
    return [_object_to_dict(item) for item in items]


def _decode_request(message: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(message["Body"])
    if isinstance(payload, dict) and isinstance(payload.get("Message"), str):
        nested = json.loads(payload["Message"])
        if isinstance(nested, dict):
            return nested
    if not isinstance(payload, dict):
        raise ValueError("SQS message body must be a JSON object")
    return payload


def _already_exists(error: Exception) -> bool:
    return getattr(error, "status", None) == 409 or "AlreadyExists" in str(error)


def _json_env(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    if not value:
        return default
    return json.loads(value)


def process_one_message(
    *,
    sqs_client: Any,
    queue_url: str,
    core_client: Any,
    batch_client: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Receive at most one SQS request and create a Kubernetes Job if capacity allows."""
    response = sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=_int_or_default(config.get("sqs_wait_time_seconds"), 20),
        VisibilityTimeout=_int_or_default(config.get("sqs_visibility_timeout"), 300),
    )
    messages = response.get("Messages", [])
    if not messages:
        return {"status": "idle"}

    message = messages[0]
    request = _decode_request(message)
    pods = _list_controller_pods(core_client, config)
    active_gpus = active_gpu_requests(pods)
    max_active_gpus = _int_or_default(config.get("max_active_gpus"), 8)
    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), 8)
    parallelism = _int_or_default(config.get("job_parallelism"), 1)
    if not can_start_job(
        active_gpus=active_gpus,
        max_active_gpus=max_active_gpus,
        gpu_per_pod=gpu_per_pod,
        parallelism=parallelism,
    ):
        sqs_client.change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=message["ReceiptHandle"],
            VisibilityTimeout=_int_or_default(config.get("defer_visibility_timeout"), 60),
        )
        return {
            "status": "deferred",
            "active_gpus": active_gpus,
            "requested_gpus": gpu_per_pod * parallelism,
            "max_active_gpus": max_active_gpus,
        }

    manifest = render_job_manifest(
        request,
        {**config, **{"gpu_per_pod": gpu_per_pod, "job_parallelism": parallelism}},
    )
    namespace = manifest["metadata"]["namespace"]
    try:
        batch_client.create_namespaced_job(namespace=namespace, body=manifest)
    except Exception as error:
        if not _already_exists(error):
            raise
    sqs_client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=message["ReceiptHandle"],
    )
    return {
        "status": "started",
        "job_name": manifest["metadata"]["name"],
        "active_gpus": active_gpus,
        "requested_gpus": gpu_per_pod * parallelism,
        "max_active_gpus": max_active_gpus,
    }


def _env_config() -> dict[str, Any]:
    return {
        "namespace": os.environ.get("SGLANG_VIDEO_NAMESPACE", "default"),
        "job_image": os.environ["SGLANG_VIDEO_JOB_IMAGE"],
        "service_account": os.environ.get("SGLANG_VIDEO_SERVICE_ACCOUNT", "sglang-video-job"),
        "fsx_claim_name": os.environ.get("SGLANG_VIDEO_FSX_CLAIM", "fsx-claim"),
        "fsx_mount_path": os.environ.get("SGLANG_VIDEO_FSX_MOUNT", "/fsx"),
        "work_dir_prefix": os.environ.get("SGLANG_VIDEO_WORK_DIR_PREFIX", "/fsx/sglang-video"),
        "max_active_gpus": os.environ.get("SGLANG_VIDEO_MAX_ACTIVE_GPUS", "8"),
        "gpu_per_pod": os.environ.get("SGLANG_VIDEO_GPU_PER_POD", "8"),
        "job_parallelism": os.environ.get("SGLANG_VIDEO_JOB_PARALLELISM", "1"),
        "timeout_seconds": os.environ.get("SGLANG_VIDEO_TIMEOUT_SECONDS", "21600"),
        "defer_visibility_timeout": os.environ.get(
            "SGLANG_VIDEO_DEFER_VISIBILITY_TIMEOUT",
            "60",
        ),
        "pod_label_selector": os.environ.get(
            "SGLANG_VIDEO_POD_LABEL_SELECTOR",
            "app.kubernetes.io/name=sglang-video-batch",
        ),
        "command": _json_env("SGLANG_VIDEO_JOB_COMMAND_JSON", DEFAULT_CONTAINER_COMMAND),
        "args": _json_env("SGLANG_VIDEO_JOB_ARGS_JSON", []),
        "extra_env": _json_env("SGLANG_VIDEO_JOB_EXTRA_ENV_JSON", []),
        "node_selector": _json_env("SGLANG_VIDEO_JOB_NODE_SELECTOR_JSON", {}),
        "affinity": _json_env("SGLANG_VIDEO_JOB_AFFINITY_JSON", {}),
        "placement_profiles": _json_env("SGLANG_VIDEO_JOB_PLACEMENT_PROFILES_JSON", []),
        "tolerations": _json_env("SGLANG_VIDEO_JOB_TOLERATIONS_JSON", []),
        "security_context": _json_env("SGLANG_VIDEO_JOB_SECURITY_CONTEXT_JSON", {}),
        "request_cpu": os.environ.get("SGLANG_VIDEO_JOB_REQUEST_CPU", ""),
        "request_memory": os.environ.get("SGLANG_VIDEO_JOB_REQUEST_MEMORY", ""),
        "limit_cpu": os.environ.get("SGLANG_VIDEO_JOB_LIMIT_CPU", ""),
        "limit_memory": os.environ.get("SGLANG_VIDEO_JOB_LIMIT_MEMORY", ""),
        "shm_size": os.environ.get("SGLANG_VIDEO_JOB_SHM_SIZE", ""),
        "priority_class_name": os.environ.get("SGLANG_VIDEO_JOB_PRIORITY_CLASS_NAME", ""),
        "scheduler_name": os.environ.get("SGLANG_VIDEO_JOB_SCHEDULER_NAME", ""),
        "ttl_seconds_after_finished": os.environ.get(
            "SGLANG_VIDEO_JOB_TTL_SECONDS_AFTER_FINISHED",
            "",
        ),
    }


def main() -> None:
    import boto3
    from kubernetes import client, config as kube_config

    queue_url = os.environ["SGLANG_VIDEO_QUEUE_URL"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    sqs_client = boto3.client("sqs", region_name=region)
    kube_config.load_incluster_config()
    core_client = client.CoreV1Api()
    batch_client = client.BatchV1Api()
    config = _env_config()
    while True:
        result = process_one_message(
            sqs_client=sqs_client,
            queue_url=queue_url,
            core_client=core_client,
            batch_client=batch_client,
            config=config,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] == "idle":
            time.sleep(1)


if __name__ == "__main__":
    main()
