#!/usr/bin/env python3
"""Controller helpers for launching t2i video batch jobs on aws03."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
FALLBACK_GPU_BACKEND_NAMES = {"h100", "h100-spot", "h200", "h200-spot", "b200", "b200-spot"}
FALLBACK_GPU_INSTANCE_TYPES = {
    "p5.48xlarge",
    "p5e.48xlarge",
    "p5en.48xlarge",
    "p6-b200.48xlarge",
}
B300_BACKEND_NAMES = {"b300", "b300-capacity-block"}
DEFAULT_CONTAINER_COMMAND = [
    "python3",
    "/opt/bench/run_t2i_video_batch.py",
]


def _int_or_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _condition_true(status: dict[str, Any], condition_type: str) -> bool:
    conditions = status.get("conditions") if isinstance(status.get("conditions"), list) else []
    return any(
        condition.get("type") == condition_type and condition.get("status") == "True"
        for condition in conditions
        if isinstance(condition, dict)
    )


def _job_succeeded(status: dict[str, Any]) -> bool:
    return _int_or_default(status.get("succeeded"), 0) > 0 or _condition_true(status, "Complete")


def _job_failed(status: dict[str, Any]) -> bool:
    return _condition_true(status, "Failed")


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


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    return value.get("metadata") if isinstance(value.get("metadata"), dict) else {}


def _labels(value: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(value)
    return metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}


def _node_name(node: dict[str, Any]) -> str:
    return str(_metadata(node).get("name") or "")


def _pod_node_name(pod: dict[str, Any]) -> str:
    spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
    return str(spec.get("nodeName") or "")


def _node_ready(node: dict[str, Any]) -> bool:
    status = node.get("status") if isinstance(node.get("status"), dict) else {}
    conditions = status.get("conditions") if isinstance(status.get("conditions"), list) else []
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
        if isinstance(condition, dict)
    )


def _node_allocatable_gpus(node: dict[str, Any]) -> int:
    status = node.get("status") if isinstance(node.get("status"), dict) else {}
    allocatable = status.get("allocatable") if isinstance(status.get("allocatable"), dict) else {}
    return _int_or_default(allocatable.get("nvidia.com/gpu"), 0)


def _matches_selector(labels: dict[str, Any], selector: dict[str, Any]) -> bool:
    for key, expected in selector.items():
        if expected in (None, ""):
            continue
        values = [str(item) for item in expected] if isinstance(expected, list) else [str(expected)]
        if str(labels.get(str(key))) not in values:
            return False
    return True


def _pod_gpu_requests_on_node(pods: list[dict[str, Any]], node_name: str) -> int:
    total = 0
    for pod in pods:
        status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        if status.get("phase") in TERMINAL_POD_PHASES:
            continue
        if _pod_node_name(pod) != node_name:
            continue
        spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
        containers = spec.get("containers") if isinstance(spec.get("containers"), list) else []
        total += sum(_gpu_request(container) for container in containers)
    return total


def backend_free_gpus(
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    selector: dict[str, Any],
) -> int:
    """Return currently free GPUs on Ready nodes matching a backend selector."""
    total = 0
    for node in nodes:
        if not _node_ready(node) or not _matches_selector(_labels(node), selector):
            continue
        total += max(
            0,
            _node_allocatable_gpus(node) - _pod_gpu_requests_on_node(pods, _node_name(node)),
        )
    return total


def _backend_name(value: dict[str, Any]) -> str:
    return str(value.get("name") or "").strip()


def _is_h100_backend(backend: dict[str, Any]) -> bool:
    name = _backend_name(backend).lower()
    selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
    return (
        name in FALLBACK_GPU_BACKEND_NAMES
        or "h100" in name
        or "h200" in name
        or "b200" in name
        or selector.get("node.kubernetes.io/instance-type") in FALLBACK_GPU_INSTANCE_TYPES
    )


def _is_b300_backend(backend: dict[str, Any]) -> bool:
    name = _backend_name(backend).lower()
    selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
    return (
        name in B300_BACKEND_NAMES
        or "b300" in name
        or selector.get("node.kubernetes.io/instance-type") == "p6-b300.48xlarge"
    )


def _is_demand_backend(backend: dict[str, Any]) -> bool:
    name = _backend_name(backend).lower()
    selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
    return "demand" in name or selector.get("eks.amazonaws.com/capacityType") == "ON_DEMAND"


def _backend_gpu_requests(pods: list[dict[str, Any]], backend_name: str) -> int:
    total = 0
    for pod in pods:
        if _labels(pod).get("sglang.seedleap.io/backend") != backend_name:
            continue
        metadata = _metadata(pod)
        if metadata.get("deletionTimestamp"):
            continue
        status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        if status.get("phase") in TERMINAL_POD_PHASES:
            continue
        spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
        containers = spec.get("containers") if isinstance(spec.get("containers"), list) else []
        total += sum(_gpu_request(container) for container in containers)
    return total


def _fallback_backend_names(config: dict[str, Any]) -> set[str]:
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    return {
        _backend_name(backend)
        for backend in backends
        if isinstance(backend, dict) and _is_h100_backend(backend) and _backend_name(backend)
    }


def _b300_backend_names(config: dict[str, Any]) -> set[str]:
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    return {
        _backend_name(backend)
        for backend in backends
        if isinstance(backend, dict) and _is_b300_backend(backend) and _backend_name(backend)
    }


def _fallback_gpu_requests(
    pods: list[dict[str, Any]],
    backend_names: set[str],
) -> int:
    return sum(_backend_gpu_requests(pods, name) for name in backend_names)


def _inflight_backend_gpus(state: dict[str, Any], backend_name: str, gpu_per_pod: int) -> int:
    return sum(
        _int_or_default(item.get("requested_gpus"), gpu_per_pod)
        for item in state.get("inflight", [])
        if item.get("backend") == backend_name
    )


def _fallback_inflight_gpus(
    state: dict[str, Any],
    backend_names: set[str],
    gpu_per_pod: int,
) -> int:
    return sum(
        _int_or_default(item.get("requested_gpus"), gpu_per_pod)
        for item in state.get("inflight", [])
        if str(item.get("backend") or "") in backend_names
    )


def choose_backend(
    config: dict[str, Any],
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    *,
    requested_gpus: int,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Pick B300 within its cap; otherwise pick fallback Spot GPUs within their shared cap."""
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    state = state or {"inflight": []}
    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), requested_gpus)
    b300_max_gpus = _int_or_default(
        config.get("b300_max_active_gpus"),
        _int_or_default(config.get("max_active_gpus"), 32),
    )
    b300_names = _b300_backend_names(config)
    b300_active = max(
        _fallback_gpu_requests(pods, b300_names),
        _fallback_inflight_gpus(state, b300_names, gpu_per_pod),
    )
    for backend in backends:
        if not isinstance(backend, dict) or not _is_b300_backend(backend):
            continue
        if b300_active + requested_gpus > b300_max_gpus:
            continue
        selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
        if backend_free_gpus(nodes, pods, selector) >= requested_gpus:
            return backend

    fallback_max_gpus = _int_or_default(
        config.get("fallback_max_active_gpus"),
        _int_or_default(config.get("h100_max_active_gpus"), 32),
    )
    allow_h100_demand = str(config.get("allow_h100_demand") or "").lower() in {"1", "true", "yes"}
    fallback_names = _fallback_backend_names(config)
    fallback_active = max(
        _fallback_gpu_requests(pods, fallback_names),
        _fallback_inflight_gpus(state, fallback_names, gpu_per_pod),
    )
    for backend in backends:
        if not isinstance(backend, dict) or not _is_h100_backend(backend):
            continue
        if _is_demand_backend(backend) and not allow_h100_demand:
            continue
        if fallback_active + requested_gpus <= fallback_max_gpus:
            return backend
    return None


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
    suffix = _safe_name(str(config.get("job_name_suffix") or ""), "")
    if suffix:
        job_name = f"{job_name}-{suffix}"[:63].rstrip("-")
    selected_backend = str(config.get("selected_backend") or "")
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
                    "sglang.seedleap.io/backend": selected_backend,
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
                "sglang.seedleap.io/backend": selected_backend,
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


def _list_capacity_pods(core_client: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    namespace = config.get("namespace", "default")
    response = core_client.list_namespaced_pod(namespace=namespace)
    response = _object_to_dict(response)
    items = response.get("items", []) if isinstance(response, dict) else response.items
    return [_object_to_dict(item) for item in items]


def _list_nodes(core_client: Any) -> list[dict[str, Any]]:
    response = core_client.list_node()
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


def _not_found(error: Exception) -> bool:
    return getattr(error, "status", None) == 404 or "NotFound" in str(error)


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
    return {
        "status": "started",
        "job_name": manifest["metadata"]["name"],
        "active_gpus": active_gpus,
        "requested_gpus": gpu_per_pod * parallelism,
        "max_active_gpus": max_active_gpus,
    }


def _message_visibility_seconds(config: dict[str, Any]) -> int:
    return _int_or_default(config.get("message_visibility_seconds"), 900)


def renew_inflight_messages(
    *,
    sqs_client: Any,
    queue_url: str,
    config: dict[str, Any],
    state: dict[str, Any],
    now: float,
) -> dict[str, int]:
    renewed = expired = 0
    renew_interval = _int_or_default(config.get("message_renew_interval_seconds"), 60)
    max_lease = _int_or_default(config.get("message_max_lease_seconds"), 28800)
    visibility = _message_visibility_seconds(config)
    remaining = []
    all_inflight = list(state.get("inflight", []))
    for item in all_inflight:
        if now - float(item.get("started_at", now)) > max_lease:
            sqs_client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=item["receipt_handle"],
                VisibilityTimeout=0,
            )
            expired += 1
            continue
        if now - float(item.get("last_renewed_at", 0)) >= renew_interval:
            sqs_client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=item["receipt_handle"],
                VisibilityTimeout=visibility,
            )
            item["last_renewed_at"] = now
            renewed += 1
        remaining.append(item)
    state["inflight"] = remaining
    return {"renewed": renewed, "expired": expired}


def _job_status(job: Any) -> dict[str, Any]:
    job = _object_to_dict(job)
    if isinstance(job, dict):
        return job.get("status") if isinstance(job.get("status"), dict) else {}
    return _object_to_dict(getattr(job, "status", {})) or {}


def _read_job_status(batch_client: Any, namespace: str, job_name: str) -> dict[str, Any]:
    try:
        return _job_status(batch_client.read_namespaced_job_status(name=job_name, namespace=namespace))
    except Exception as error:
        if _not_found(error):
            return {}
        return {"failed": 1}


def _list_jobs_for_generation(
    batch_client: Any,
    namespace: str,
    generation_job_id: str,
) -> list[dict[str, Any]]:
    if not generation_job_id:
        return []
    response = batch_client.list_namespaced_job(
        namespace=namespace,
        label_selector=f"sglang.seedleap.io/generation-job-id={generation_job_id}",
    )
    response = _object_to_dict(response)
    items = response.get("items", []) if isinstance(response, dict) else response.items
    return [_object_to_dict(item) for item in items]


def _existing_job_for_request(
    batch_client: Any,
    namespace: str,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    generation_job_id = str(request.get("generation_job_id") or "").strip()
    jobs = _list_jobs_for_generation(batch_client, namespace, generation_job_id)
    if not jobs:
        return None
    jobs.sort(key=lambda job: str(_metadata(job).get("creationTimestamp") or ""), reverse=True)
    for job in jobs:
        if not _job_failed(_job_status(job)):
            return job
    return jobs[0]


def _render_for_backend(
    request: dict[str, Any],
    config: dict[str, Any],
    backend: dict[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    return render_job_manifest(
        request,
        {
            **config,
            "placement_profiles": [backend],
            "selected_backend": _backend_name(backend),
            "job_name_suffix": f"r{attempt}" if attempt > 1 else "",
        },
    )


def _backend_scale_target(
    backend: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_h100_backend(backend):
        return None
    selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
    should_scale = bool(
        backend.get("scale_nodegroup")
        or backend.get("scale_nodegroup_name")
        or backend.get("nodegroup_name")
        or selector.get("eks.amazonaws.com/nodegroup")
    )
    if not should_scale:
        return None
    cluster_name = str(
        backend.get("scale_cluster_name")
        or backend.get("cluster_name")
        or config.get("h100_cluster_name")
        or ""
    ).strip()
    nodegroup_name = str(
        backend.get("scale_nodegroup_name")
        or backend.get("nodegroup_name")
        or selector.get("eks.amazonaws.com/nodegroup")
        or config.get("h100_nodegroup_name")
        or ""
    ).strip()
    if not cluster_name or not nodegroup_name:
        return None
    return {
        "cluster_name": cluster_name,
        "nodegroup_name": nodegroup_name,
        "node_gpus": max(
            1,
            _int_or_default(backend.get("node_gpus") or config.get("h100_node_gpus"), 8),
        ),
        "max_nodes": _int_or_default(backend.get("max_nodes") or config.get("h100_max_nodes"), 4),
    }


def _scale_nodegroup(
    eks_client: Any,
    *,
    cluster_name: str,
    nodegroup_name: str,
    node_gpus: int,
    max_nodes: int,
    desired_gpus: int,
) -> None:
    if eks_client is None:
        return
    desired_nodes = min(max_nodes, (max(0, desired_gpus) + node_gpus - 1) // node_gpus)
    try:
        response = eks_client.describe_nodegroup(
            clusterName=cluster_name,
            nodegroupName=nodegroup_name,
        )
        nodegroup = response.get("nodegroup") if isinstance(response, dict) else {}
        scaling = (
            nodegroup.get("scalingConfig")
            if isinstance(nodegroup.get("scalingConfig"), dict)
            else {}
        )
        if _int_or_default(scaling.get("desiredSize"), -1) == desired_nodes:
            return
    except Exception:
        pass
    eks_client.update_nodegroup_config(
        clusterName=cluster_name,
        nodegroupName=nodegroup_name,
        scalingConfig={"desiredSize": desired_nodes},
    )


def _scale_fallback_nodegroups(
    eks_client: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    pods: list[dict[str, Any]] | None = None,
) -> None:
    if eks_client is None:
        return
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        target = _backend_scale_target(backend, config)
        if target is None:
            continue
        key = (target["cluster_name"], target["nodegroup_name"])
        targets.setdefault(
            key,
            {
                **target,
                "backend_names": set(),
            },
        )
        targets[key]["backend_names"].add(_backend_name(backend))
        targets[key]["max_nodes"] = max(targets[key]["max_nodes"], target["max_nodes"])

    for target in targets.values():
        inflight_gpus = sum(
            _int_or_default(item.get("requested_gpus"), target["node_gpus"])
            for item in state.get("inflight", [])
            if str(item.get("backend") or "") in target["backend_names"]
        )
        pod_gpus = sum(
            _backend_gpu_requests(pods or [], backend_name)
            for backend_name in target["backend_names"]
        )
        desired_gpus = max(inflight_gpus, pod_gpus)
        _scale_nodegroup(
            eks_client,
            cluster_name=target["cluster_name"],
            nodegroup_name=target["nodegroup_name"],
            node_gpus=target["node_gpus"],
            max_nodes=target["max_nodes"],
            desired_gpus=desired_gpus,
        )


def reconcile_inflight_jobs(
    *,
    sqs_client: Any,
    queue_url: str,
    core_client: Any,
    batch_client: Any,
    eks_client: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    now: float,
) -> dict[str, int]:
    namespace = config.get("namespace", "default")
    max_attempts = _int_or_default(config.get("max_job_attempts"), 5)
    nodes = _list_nodes(core_client)
    pods = _list_capacity_pods(core_client, config)
    completed = restarted = released = failed = 0
    remaining = []
    all_inflight = list(state.get("inflight", []))

    for item in all_inflight:
        status = _read_job_status(batch_client, namespace, item["job_name"])
        if _job_succeeded(status):
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=item["receipt_handle"],
            )
            completed += 1
            continue

        if _job_failed(status):
            attempts = _int_or_default(item.get("attempts"), 1)
            if attempts >= max_attempts:
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=item["receipt_handle"],
                    VisibilityTimeout=0,
                )
                failed += 1
                continue

            try:
                batch_client.delete_namespaced_job(
                    name=item["job_name"],
                    namespace=namespace,
                    propagation_policy="Background",
                )
            except Exception as error:
                if not _not_found(error):
                    raise
            requested_gpus = _int_or_default(item.get("requested_gpus"), 8)
            backend = choose_backend(
                config,
                nodes,
                pods,
                requested_gpus=requested_gpus,
                state={"inflight": [candidate for candidate in all_inflight if candidate is not item]},
            )
            if backend is None:
                remaining.append(item)
                released += 1
                continue
            next_attempt = attempts + 1
            manifest = _render_for_backend(item["request"], config, backend, attempt=next_attempt)
            batch_client.create_namespaced_job(namespace=namespace, body=manifest)
            item = {
                **item,
                "job_name": manifest["metadata"]["name"],
                "backend": _backend_name(backend),
                "attempts": next_attempt,
                "last_renewed_at": now,
                "requested_gpus": requested_gpus,
            }
            restarted += 1

        remaining.append(item)

    state["inflight"] = remaining
    _scale_fallback_nodegroups(eks_client, config, state, pods=pods)
    return {
        "completed": completed,
        "restarted": restarted,
        "released": released,
        "failed": failed,
    }


def controller_tick(
    *,
    sqs_client: Any,
    queue_url: str,
    core_client: Any,
    batch_client: Any,
    eks_client: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    state.setdefault("inflight", [])
    renew_result = renew_inflight_messages(
        sqs_client=sqs_client,
        queue_url=queue_url,
        config=config,
        state=state,
        now=now,
    )
    reconcile_result = reconcile_inflight_jobs(
        sqs_client=sqs_client,
        queue_url=queue_url,
        core_client=core_client,
        batch_client=batch_client,
        eks_client=eks_client,
        config=config,
        state=state,
        now=now,
    )

    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), 8)
    parallelism = _int_or_default(config.get("job_parallelism"), 1)
    requested_gpus = gpu_per_pod * parallelism
    response = sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=min(10, _int_or_default(config.get("sqs_max_messages"), 1)),
        WaitTimeSeconds=_int_or_default(config.get("sqs_wait_time_seconds"), 20),
        VisibilityTimeout=_message_visibility_seconds(config),
    )
    messages = response.get("Messages", [])
    if not messages:
        return {"status": "idle", **renew_result, **reconcile_result, "started": 0, "deferred": 0}

    nodes = _list_nodes(core_client)
    pods = _list_capacity_pods(core_client, config)
    started = deferred = adopted = completed_existing = 0
    for message in messages:
        request = _decode_request(message)
        namespace = config.get("namespace", "default")
        existing_job = _existing_job_for_request(batch_client, namespace, request)
        if existing_job is not None:
            status = _job_status(existing_job)
            if _job_succeeded(status):
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )
                completed_existing += 1
                continue
            if not _job_failed(status):
                labels = _labels(existing_job)
                metadata = _metadata(existing_job)
                backend_name = str(labels.get("sglang.seedleap.io/backend") or "")
                job_name = str(metadata.get("name") or "")
                if job_name and not any(
                    item.get("job_name") == job_name for item in state.get("inflight", [])
                ):
                    state["inflight"].append(
                        {
                            "request": request,
                            "receipt_handle": message["ReceiptHandle"],
                            "message_id": message.get("MessageId", ""),
                            "job_name": job_name,
                            "backend": backend_name,
                            "attempts": 1,
                            "started_at": now,
                            "last_renewed_at": now,
                            "requested_gpus": requested_gpus,
                        }
                    )
                _scale_fallback_nodegroups(eks_client, config, state, pods=pods)
                adopted += 1
                continue

        backend = choose_backend(
            config,
            nodes,
            pods,
            requested_gpus=requested_gpus,
            state=state,
        )
        if backend is None:
            sqs_client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=message["ReceiptHandle"],
                VisibilityTimeout=_int_or_default(config.get("defer_visibility_timeout"), 60),
            )
            deferred += 1
            continue

        manifest = _render_for_backend(request, config, backend, attempt=1)
        namespace = manifest["metadata"]["namespace"]
        try:
            batch_client.create_namespaced_job(namespace=namespace, body=manifest)
        except Exception as error:
            if not _already_exists(error):
                raise
        state["inflight"].append(
            {
                "request": request,
                "receipt_handle": message["ReceiptHandle"],
                "message_id": message.get("MessageId", ""),
                "job_name": manifest["metadata"]["name"],
                "backend": _backend_name(backend),
                "attempts": 1,
                "started_at": now,
                "last_renewed_at": now,
                "requested_gpus": requested_gpus,
            }
        )
        _scale_fallback_nodegroups(eks_client, config, state, pods=pods)
        started += 1

    return {
        "status": "started" if started else "adopted" if adopted else "deferred",
        **renew_result,
        **reconcile_result,
        "started": started,
        "deferred": deferred,
        "adopted": adopted,
        "completed_existing": completed_existing,
        "inflight": len(state.get("inflight", [])),
    }


def _backend_profiles_from_env(placement_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configured = _json_env("SGLANG_VIDEO_BACKENDS_JSON", [])
    if isinstance(configured, list) and configured:
        return configured
    backends: list[dict[str, Any]] = []
    for profile in placement_profiles:
        if not isinstance(profile, dict):
            continue
        backend = dict(profile)
        if _is_h100_backend(backend):
            backend.setdefault("scale_nodegroup", True)
        backends.append(backend)
    return backends


def _env_config() -> dict[str, Any]:
    placement_profiles = _json_env("SGLANG_VIDEO_JOB_PLACEMENT_PROFILES_JSON", [])
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
        "sqs_max_messages": os.environ.get("SGLANG_VIDEO_SQS_MAX_MESSAGES", "1"),
        "sqs_wait_time_seconds": os.environ.get("SGLANG_VIDEO_SQS_WAIT_TIME_SECONDS", "20"),
        "sqs_visibility_timeout": os.environ.get("SGLANG_VIDEO_SQS_VISIBILITY_TIMEOUT", "300"),
        "message_visibility_seconds": os.environ.get(
            "SGLANG_VIDEO_MESSAGE_VISIBILITY_SECONDS",
            "900",
        ),
        "message_renew_interval_seconds": os.environ.get(
            "SGLANG_VIDEO_MESSAGE_RENEW_INTERVAL_SECONDS",
            "60",
        ),
        "message_max_lease_seconds": os.environ.get(
            "SGLANG_VIDEO_MESSAGE_MAX_LEASE_SECONDS",
            "28800",
        ),
        "max_job_attempts": os.environ.get("SGLANG_VIDEO_MAX_JOB_ATTEMPTS", "5"),
        "b300_max_active_gpus": os.environ.get(
            "SGLANG_VIDEO_B300_MAX_ACTIVE_GPUS",
            os.environ.get("SGLANG_VIDEO_MAX_ACTIVE_GPUS", "32"),
        ),
        "fallback_max_active_gpus": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_MAX_ACTIVE_GPUS",
            os.environ.get("SGLANG_VIDEO_H100_MAX_ACTIVE_GPUS", "32"),
        ),
        "h100_max_active_gpus": os.environ.get("SGLANG_VIDEO_H100_MAX_ACTIVE_GPUS", "32"),
        "h100_node_gpus": os.environ.get("SGLANG_VIDEO_H100_NODE_GPUS", "8"),
        "h100_max_nodes": os.environ.get("SGLANG_VIDEO_H100_MAX_NODES", "4"),
        "h100_cluster_name": os.environ.get("SGLANG_VIDEO_H100_EKS_CLUSTER_NAME", ""),
        "h100_nodegroup_name": os.environ.get("SGLANG_VIDEO_H100_NODEGROUP_NAME", ""),
        "allow_h100_demand": os.environ.get("SGLANG_VIDEO_ALLOW_H100_DEMAND", "false"),
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
        "placement_profiles": placement_profiles,
        "backends": _backend_profiles_from_env(placement_profiles),
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
    eks_client = boto3.client("eks", region_name=region)
    kube_config.load_incluster_config()
    core_client = client.CoreV1Api()
    batch_client = client.BatchV1Api()
    config = _env_config()
    state: dict[str, Any] = {"inflight": []}
    while True:
        result = controller_tick(
            sqs_client=sqs_client,
            queue_url=queue_url,
            core_client=core_client,
            batch_client=batch_client,
            eks_client=eks_client,
            config=config,
            state=state,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] == "idle":
            time.sleep(1)


if __name__ == "__main__":
    main()
