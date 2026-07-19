#!/usr/bin/env python3
"""Controller helpers for launching t2i video batch jobs on aws03."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any
from urllib.parse import urlparse


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
HARD_FALLBACK_NODEGROUP_ISSUE_TOKENS = {
    "invalidfleetconfiguration",
    "invalidlaunchtemplate",
    "ec2launchtemplatenotfound",
    "unsupported",
    "not supported",
}
CAPACITY_FALLBACK_NODEGROUP_ISSUE_TOKENS = {
    "unfulfillablecapacity",
    "insufficientinstancecapacity",
    "capacity is not available",
    "spot capacity",
}


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


def _job_missing(status: dict[str, Any]) -> bool:
    return bool(status.get("not_found"))


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


def _pod_phase(pod: dict[str, Any]) -> str:
    status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
    return str(status.get("phase") or "")


def _pod_job_name(pod: dict[str, Any]) -> str:
    labels = _labels(pod)
    job_name = labels.get("batch.kubernetes.io/job-name") or labels.get("job-name")
    if job_name:
        return str(job_name)
    owner_references = _metadata(pod).get("ownerReferences")
    if isinstance(owner_references, list):
        for owner in owner_references:
            if isinstance(owner, dict) and owner.get("kind") == "Job" and owner.get("name"):
                return str(owner["name"])
    return ""


def _pending_pods_for_job(pods: list[dict[str, Any]], job_name: str) -> list[dict[str, Any]]:
    result = []
    for pod in pods:
        metadata = _metadata(pod)
        if metadata.get("deletionTimestamp"):
            continue
        if _pod_job_name(pod) != job_name:
            continue
        if _pod_phase(pod) == "Pending" and not _pod_node_name(pod):
            result.append(pod)
    return result


def _pods_except_job(pods: list[dict[str, Any]], job_name: str) -> list[dict[str, Any]]:
    return [pod for pod in pods if _pod_job_name(pod) != job_name]


def _pods_except_jobs(pods: list[dict[str, Any]], job_names: set[str]) -> list[dict[str, Any]]:
    if not job_names:
        return pods
    return [pod for pod in pods if _pod_job_name(pod) not in job_names]


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


def _backend_effective_free_gpus(
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    selector: dict[str, Any],
    state: dict[str, Any],
    backend_name: str,
    gpu_per_pod: int,
) -> int:
    free_gpus = backend_free_gpus(nodes, pods, selector)
    pod_gpus = _backend_gpu_requests(pods, backend_name)
    inflight_gpus = _inflight_backend_gpus(state, backend_name, gpu_per_pod)
    return max(0, free_gpus - max(0, inflight_gpus - pod_gpus))


def _backend_name(value: dict[str, Any]) -> str:
    return str(value.get("name") or "").strip()


def _inflight_counts_for_capacity(item: dict[str, Any]) -> bool:
    return not (
        bool(item.get("callback_pending"))
        or item.get("gpu_released_at") not in (None, "")
    )


def _mark_callback_pending(item: dict[str, Any], error: Exception, *, now: float) -> dict[str, Any]:
    item["callback_error"] = f"{type(error).__name__}: {error}"
    item["callback_pending"] = True
    item.setdefault("gpu_released_at", now)
    item.pop("missing_job_seen_at", None)
    item.pop("pending_job_seen_at", None)
    return item


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


def _backend_job_keys(
    pods: list[dict[str, Any]],
    state: dict[str, Any],
    backend_names: set[str],
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for index, pod in enumerate(pods):
        backend_name = str(_labels(pod).get("sglang.seedleap.io/backend") or "")
        if backend_name not in backend_names:
            continue
        metadata = _metadata(pod)
        if metadata.get("deletionTimestamp"):
            continue
        status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        if status.get("phase") in TERMINAL_POD_PHASES:
            continue
        job_key = _pod_job_name(pod) or str(metadata.get("name") or f"pod-{index}")
        keys.add((backend_name, job_key))

    inflight = state.get("inflight") if isinstance(state.get("inflight"), list) else []
    for index, item in enumerate(inflight):
        if not isinstance(item, dict):
            continue
        if not _inflight_counts_for_capacity(item):
            continue
        backend_name = str(item.get("backend") or "")
        if backend_name not in backend_names:
            continue
        job_key = str(
            item.get("job_name")
            or item.get("message_id")
            or item.get("receipt_handle")
            or f"inflight-{index}"
        )
        keys.add((backend_name, job_key))
    return keys


def _backend_active_jobs(
    pods: list[dict[str, Any]],
    state: dict[str, Any],
    backend_name: str,
) -> int:
    return len(_backend_job_keys(pods, state, {backend_name}))


def _backend_group_active_jobs(
    pods: list[dict[str, Any]],
    state: dict[str, Any],
    backend_names: set[str],
) -> int:
    return len(_backend_job_keys(pods, state, backend_names))


def _active_job_caps_exceeded(
    config: dict[str, Any],
    pods: list[dict[str, Any]],
    state: dict[str, Any],
    backend_name: str,
) -> bool:
    b300_names = _b300_backend_names(config)
    fallback_names = _fallback_backend_names(config)
    max_active_jobs = _int_or_default(config.get("max_active_jobs"), 7)
    if max_active_jobs > 0:
        active_jobs = _backend_group_active_jobs(pods, state, b300_names | fallback_names)
        if active_jobs > max_active_jobs:
            return True

    if backend_name in b300_names:
        b300_max_jobs = _int_or_default(config.get("b300_max_active_jobs"), 5)
        return (
            b300_max_jobs > 0
            and _backend_group_active_jobs(pods, state, b300_names) > b300_max_jobs
        )
    if backend_name in fallback_names:
        fallback_max_jobs = _int_or_default(config.get("fallback_max_active_jobs"), 2)
        return (
            fallback_max_jobs > 0
            and _backend_group_active_jobs(pods, state, fallback_names) > fallback_max_jobs
        )
    return False


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
        if item.get("backend") == backend_name and _inflight_counts_for_capacity(item)
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
        and _inflight_counts_for_capacity(item)
    )


def _active_backend_job_names(
    pods: list[dict[str, Any]],
    backend_names: set[str],
) -> set[tuple[str, str]]:
    active: set[tuple[str, str]] = set()
    for pod in pods:
        metadata = _metadata(pod)
        if metadata.get("deletionTimestamp"):
            continue
        if _pod_phase(pod) in TERMINAL_POD_PHASES:
            continue
        backend_name = str(_labels(pod).get("sglang.seedleap.io/backend") or "")
        if backend_name not in backend_names:
            continue
        job_name = _pod_job_name(pod)
        if job_name:
            active.add((backend_name, job_name))
    return active


def _fallback_inflight_gpus_for_nodegroup_scaling(
    state: dict[str, Any],
    pods: list[dict[str, Any]],
    backend_names: set[str],
    gpu_per_pod: int,
    *,
    now: float,
    orphan_grace_seconds: int,
) -> int:
    active_job_names = _active_backend_job_names(pods, backend_names)
    total = 0
    for item in state.get("inflight", []):
        if not _inflight_counts_for_capacity(item):
            continue
        backend_name = str(item.get("backend") or "")
        if backend_name not in backend_names:
            continue
        job_name = str(item.get("job_name") or "")
        if job_name and (backend_name, job_name) in active_job_names:
            total += _int_or_default(item.get("requested_gpus"), gpu_per_pod)
            continue
        try:
            started_at = float(item.get("started_at", now))
        except (TypeError, ValueError):
            started_at = now
        if now - started_at <= orphan_grace_seconds:
            total += _int_or_default(item.get("requested_gpus"), gpu_per_pod)
    return total


def _fallback_prewarm_state(state: dict[str, Any]) -> dict[str, Any]:
    prewarm = state.setdefault("fallback_prewarm", {})
    if not isinstance(prewarm, dict):
        prewarm = {}
        state["fallback_prewarm"] = prewarm
    return prewarm


def _fallback_backend_cooldown_state(state: dict[str, Any]) -> dict[str, Any]:
    cooldown = state.setdefault("fallback_backend_cooldown", {})
    if not isinstance(cooldown, dict):
        cooldown = {}
        state["fallback_backend_cooldown"] = cooldown
    return cooldown


def _cleanup_fallback_prewarm(
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    now: float,
) -> None:
    ttl_seconds = _int_or_default(config.get("fallback_prewarm_ttl_seconds"), 600)
    prewarm = _fallback_prewarm_state(state)
    for backend_name, item in list(prewarm.items()):
        if not isinstance(item, dict):
            prewarm.pop(backend_name, None)
            continue
        updated_at = float(item.get("updated_at") or item.get("started_at") or now)
        if ttl_seconds >= 0 and now - updated_at > ttl_seconds:
            prewarm.pop(backend_name, None)


def _cleanup_fallback_backend_cooldowns(state: dict[str, Any], *, now: float) -> None:
    cooldown = _fallback_backend_cooldown_state(state)
    for backend_name, until in list(cooldown.items()):
        try:
            if now >= float(until):
                cooldown.pop(backend_name, None)
        except (TypeError, ValueError):
            cooldown.pop(backend_name, None)


def _fallback_backend_in_cooldown(
    state: dict[str, Any],
    backend_name: str,
    *,
    now: float,
) -> bool:
    _cleanup_fallback_backend_cooldowns(state, now=now)
    cooldown = _fallback_backend_cooldown_state(state)
    try:
        return now < float(cooldown.get(backend_name) or 0)
    except (TypeError, ValueError):
        cooldown.pop(backend_name, None)
        return False


def _fallback_prewarm_gpus(
    state: dict[str, Any],
    backend_names: set[str],
    gpu_per_pod: int,
) -> int:
    total = 0
    for backend_name, item in _fallback_prewarm_state(state).items():
        if backend_name not in backend_names or not isinstance(item, dict):
            continue
        total += _int_or_default(item.get("requested_gpus"), gpu_per_pod)
    return total


def _backend_prewarm_gpus(
    state: dict[str, Any],
    backend_name: str,
    gpu_per_pod: int,
) -> int:
    return _fallback_prewarm_gpus(state, {backend_name}, gpu_per_pod)


def _fallback_prewarm_jobs(state: dict[str, Any], backend_names: set[str]) -> int:
    total = 0
    for backend_name, item in _fallback_prewarm_state(state).items():
        if backend_name not in backend_names or not isinstance(item, dict):
            continue
        requested_jobs = item.get("requested_jobs")
        if requested_jobs in (None, ""):
            total += 1 if _int_or_default(item.get("requested_gpus"), 0) > 0 else 0
        else:
            total += _int_or_default(requested_jobs, 0)
    return total


def _backend_prewarm_jobs(state: dict[str, Any], backend_name: str) -> int:
    return _fallback_prewarm_jobs(state, {backend_name})


def _add_fallback_prewarm(
    state: dict[str, Any],
    backend_name: str,
    *,
    requested_gpus: int,
    now: float,
) -> None:
    prewarm = _fallback_prewarm_state(state)
    item = prewarm.get(backend_name)
    if not isinstance(item, dict):
        item = {"requested_gpus": 0, "requested_jobs": 0, "started_at": now}
    item["requested_gpus"] = _int_or_default(item.get("requested_gpus"), 0) + requested_gpus
    item["requested_jobs"] = _int_or_default(item.get("requested_jobs"), 0) + 1
    item["updated_at"] = now
    prewarm[backend_name] = item


def _consume_fallback_prewarm(
    state: dict[str, Any],
    backend_name: str,
    *,
    requested_gpus: int,
) -> None:
    prewarm = _fallback_prewarm_state(state)
    item = prewarm.get(backend_name)
    if not isinstance(item, dict):
        return
    remaining_gpus = max(0, _int_or_default(item.get("requested_gpus"), 0) - requested_gpus)
    remaining_jobs = max(0, _int_or_default(item.get("requested_jobs"), 1) - 1)
    if remaining_gpus == 0 or remaining_jobs == 0:
        prewarm.pop(backend_name, None)
        return
    item["requested_gpus"] = remaining_gpus
    item["requested_jobs"] = remaining_jobs


def _consume_any_fallback_prewarm(
    state: dict[str, Any],
    backend_names: set[str],
    *,
    requested_gpus: int,
) -> set[str]:
    prewarm = _fallback_prewarm_state(state)
    for backend_name in sorted(backend_names):
        item = prewarm.get(backend_name)
        if not isinstance(item, dict):
            continue
        _consume_fallback_prewarm(
            state,
            backend_name,
            requested_gpus=requested_gpus,
        )
        return {backend_name}
    return set()


def _release_fallback_prewarm_for_backends(
    state: dict[str, Any],
    backend_names: set[str],
) -> set[str]:
    prewarm = _fallback_prewarm_state(state)
    released: set[str] = set()
    for backend_name in list(backend_names):
        if backend_name in prewarm:
            prewarm.pop(backend_name, None)
            released.add(backend_name)
    return released


def _release_unused_fallback_prewarm(
    config: dict[str, Any],
    state: dict[str, Any],
    pods: list[dict[str, Any]],
) -> set[str]:
    fallback_names = _fallback_backend_names(config)
    if not fallback_names:
        return set()
    active_backend_names = {
        str(item.get("backend") or "")
        for item in state.get("inflight", [])
        if str(item.get("backend") or "") in fallback_names
        and _inflight_counts_for_capacity(item)
    }
    for pod in pods:
        metadata = _metadata(pod)
        if metadata.get("deletionTimestamp"):
            continue
        if _pod_phase(pod) in {"Succeeded", "Failed"}:
            continue
        backend_name = str(_labels(pod).get("sglang.seedleap.io/backend") or "")
        if backend_name in fallback_names:
            active_backend_names.add(backend_name)

    prewarm = _fallback_prewarm_state(state)
    released: set[str] = set()
    for backend_name in list(prewarm):
        if backend_name in fallback_names and backend_name not in active_backend_names:
            prewarm.pop(backend_name, None)
            released.add(backend_name)
    return released


def _backend_active_gpus(
    pods: list[dict[str, Any]],
    state: dict[str, Any],
    backend_name: str,
    gpu_per_pod: int,
) -> int:
    return max(
        _backend_gpu_requests(pods, backend_name),
        _inflight_backend_gpus(state, backend_name, gpu_per_pod),
    )


def choose_backend(
    config: dict[str, Any],
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    *,
    requested_gpus: int,
    state: dict[str, Any] | None = None,
    excluded_backend_names: set[str] | None = None,
) -> dict[str, Any] | None:
    """Pick B300 within its cap; otherwise pick fallback Spot GPUs within their shared cap."""
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    state = state or {"inflight": []}
    excluded_backend_names = excluded_backend_names or set()
    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), requested_gpus)
    b300_names = _b300_backend_names(config)
    fallback_names = _fallback_backend_names(config)
    all_backend_names = b300_names | fallback_names
    max_active_jobs = _int_or_default(config.get("max_active_jobs"), 7)
    if max_active_jobs > 0:
        active_jobs = _backend_group_active_jobs(pods, state, all_backend_names)
        if active_jobs + 1 > max_active_jobs:
            return None

    b300_max_gpus = _int_or_default(
        config.get("b300_max_active_gpus"),
        _int_or_default(config.get("max_active_gpus"), 32),
    )
    b300_max_jobs = _int_or_default(config.get("b300_max_active_jobs"), 5)
    b300_active = max(
        _fallback_gpu_requests(pods, b300_names),
        _fallback_inflight_gpus(state, b300_names, gpu_per_pod),
    )
    b300_active_jobs = _backend_group_active_jobs(pods, state, b300_names)
    for backend in backends:
        if not isinstance(backend, dict) or not _is_b300_backend(backend):
            continue
        if _backend_name(backend) in excluded_backend_names:
            continue
        if b300_max_jobs > 0 and b300_active_jobs + 1 > b300_max_jobs:
            continue
        if b300_active + requested_gpus > b300_max_gpus:
            continue
        selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
        backend_name = _backend_name(backend)
        if (
            _backend_effective_free_gpus(
                nodes,
                pods,
                selector,
                state,
                backend_name,
                gpu_per_pod,
            )
            >= requested_gpus
        ):
            return backend

    fallback_max_gpus = _int_or_default(
        config.get("fallback_max_active_gpus"),
        _int_or_default(config.get("h100_max_active_gpus"), 32),
    )
    fallback_max_jobs = _int_or_default(config.get("fallback_max_active_jobs"), 2)
    allow_h100_demand = str(config.get("allow_h100_demand") or "").lower() in {"1", "true", "yes"}
    fallback_active = max(
        _fallback_gpu_requests(pods, fallback_names),
        _fallback_inflight_gpus(state, fallback_names, gpu_per_pod),
    )
    fallback_active_jobs = _backend_group_active_jobs(pods, state, fallback_names)
    if fallback_max_jobs > 0 and fallback_active_jobs + 1 > fallback_max_jobs:
        return None
    if fallback_active + requested_gpus > fallback_max_gpus:
        return None

    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, backend in enumerate(backends):
        if not isinstance(backend, dict) or not _is_h100_backend(backend):
            continue
        if _is_demand_backend(backend) and not allow_h100_demand:
            continue
        backend_name = _backend_name(backend)
        if not backend_name:
            continue
        if backend_name in excluded_backend_names:
            continue
        selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
        if (
            _backend_effective_free_gpus(
                nodes,
                pods,
                selector,
                state,
                backend_name,
                gpu_per_pod,
            )
            < requested_gpus
        ):
            continue
        candidates.append(
            (
                _backend_active_jobs(pods, state, backend_name),
                _backend_active_gpus(pods, state, backend_name, gpu_per_pod),
                index,
                backend,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def choose_fallback_prewarm_backend(
    config: dict[str, Any],
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    *,
    requested_gpus: int,
    state: dict[str, Any] | None = None,
    excluded_backend_names: set[str] | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Pick a fallback backend to scale before creating a Kubernetes Job."""
    now = time.time() if now is None else now
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    state = state or {"inflight": []}
    excluded_backend_names = excluded_backend_names or set()
    _cleanup_fallback_prewarm(config, state, now=now)
    _cleanup_fallback_backend_cooldowns(state, now=now)

    gpu_per_pod = _int_or_default(config.get("gpu_per_pod"), requested_gpus)
    b300_names = _b300_backend_names(config)
    fallback_names = _fallback_backend_names(config)
    all_backend_names = b300_names | fallback_names
    max_active_jobs = _int_or_default(config.get("max_active_jobs"), 7)
    if max_active_jobs > 0:
        active_jobs = _backend_group_active_jobs(pods, state, all_backend_names)
        prewarm_jobs = _fallback_prewarm_jobs(state, fallback_names)
        if active_jobs + prewarm_jobs + 1 > max_active_jobs:
            return None

    fallback_max_jobs = _int_or_default(config.get("fallback_max_active_jobs"), 2)
    fallback_active_jobs = _backend_group_active_jobs(pods, state, fallback_names)
    fallback_prewarm_jobs = _fallback_prewarm_jobs(state, fallback_names)
    if fallback_max_jobs > 0 and fallback_active_jobs + fallback_prewarm_jobs + 1 > fallback_max_jobs:
        return None

    fallback_max_gpus = _int_or_default(
        config.get("fallback_max_active_gpus"),
        _int_or_default(config.get("h100_max_active_gpus"), 32),
    )
    fallback_active = max(
        _fallback_gpu_requests(pods, fallback_names),
        _fallback_inflight_gpus(state, fallback_names, gpu_per_pod),
    )
    fallback_prewarm_gpus = _fallback_prewarm_gpus(state, fallback_names, gpu_per_pod)
    if fallback_active + fallback_prewarm_gpus + requested_gpus > fallback_max_gpus:
        return None

    allow_h100_demand = str(config.get("allow_h100_demand") or "").lower() in {"1", "true", "yes"}
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, backend in enumerate(backends):
        if not isinstance(backend, dict) or not _is_h100_backend(backend):
            continue
        if _is_demand_backend(backend) and not allow_h100_demand:
            continue
        backend_name = _backend_name(backend)
        if not backend_name or backend_name in excluded_backend_names:
            continue
        if _fallback_backend_in_cooldown(state, backend_name, now=now):
            continue
        target = _backend_scale_target(backend, config)
        if target is None:
            continue
        selector = backend.get("node_selector") if isinstance(backend.get("node_selector"), dict) else {}
        if (
            _backend_effective_free_gpus(
                nodes,
                pods,
                selector,
                state,
                backend_name,
                gpu_per_pod,
            )
            >= requested_gpus
        ):
            continue
        candidates.append(
            (
                _backend_active_jobs(pods, state, backend_name)
                + _backend_prewarm_jobs(state, backend_name),
                _backend_active_gpus(pods, state, backend_name, gpu_per_pod)
                + _backend_prewarm_gpus(state, backend_name, gpu_per_pod),
                index,
                backend,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


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


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _read_json_s3_uri(uri: str, s3_client: Any) -> dict[str, Any]:
    bucket, key = _parse_s3_uri(uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


def _item_id_from_video_result(result: dict[str, Any]) -> str:
    video_uri = str(result.get("video_uri") or "")
    parsed = urlparse(video_uri)
    parts = [part for part in parsed.path.split("/") if part]
    if "videos" in parts:
        index = parts.index("videos")
        if index + 1 < len(parts):
            return parts[index + 1]
    case_id = str(result.get("case_id") or "")
    marker = "-action-"
    if marker in case_id:
        return case_id.split(marker, 1)[0]
    return case_id


def _video_status_from_counts(total: int, succeeded: int, failed: int, running: int) -> str:
    if total and running:
        return "running"
    if failed and succeeded:
        return "completed_with_failures"
    if failed:
        return "failed"
    if total and succeeded == total:
        return "succeeded"
    return "running"


def _progress_payload_from_report(report: dict[str, Any]) -> dict[str, Any]:
    payload = report.get("callback_payload")
    if isinstance(payload, dict):
        return payload

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    counters = report.get("counters") if isinstance(report.get("counters"), dict) else {}
    results = report.get("results") if isinstance(report.get("results"), list) else []
    total = _int_or_default(summary.get("video_expected_count"), _int_or_default(counters.get("total"), 0))
    succeeded = _int_or_default(
        summary.get("video_succeeded_count"),
        _int_or_default(counters.get("succeeded"), 0),
    )
    failed = _int_or_default(
        summary.get("video_failed_count"),
        _int_or_default(counters.get("failed"), 0),
    )
    running = _int_or_default(
        summary.get("video_running_count"),
        max(0, total - succeeded - failed),
    )
    video_status = str(summary.get("video_status") or "").strip() or _video_status_from_counts(
        total,
        succeeded,
        failed,
        running,
    )

    items_by_id: dict[str, dict[str, Any]] = {}
    video_fields = (
        "case_id",
        "video_uri",
        "movement_key",
        "ending_movement_key",
        "movement_pair",
        "camera_key",
        "traj_id",
        "traj_type",
        "action_source",
        "action_index",
        "action_seed",
        "action_pattern",
        "status",
        "error",
    )
    for result in results:
        if not isinstance(result, dict):
            continue
        item_id = _item_id_from_video_result(result)
        if not item_id:
            continue
        status = str(result.get("status") or "running")
        video = {field: result.get(field) for field in video_fields if field in result}
        video["status"] = status
        video["error"] = result.get("error") or ""
        item = items_by_id.setdefault(
            item_id,
            {
                "item_id": item_id,
                "status": status,
                "stage": "sglang_video_generation",
                "output_uri": str(result.get("video_uri") or ""),
                "error": "",
                "metadata": {"video_status": status, "videos": []},
            },
        )
        if status != "succeeded" and item["status"] == "succeeded":
            item["status"] = status
        if not item.get("output_uri") and result.get("video_uri"):
            item["output_uri"] = str(result["video_uri"])
        item["metadata"]["videos"].append(video)

    for item in items_by_id.values():
        videos = item["metadata"]["videos"]
        item_total = len(videos)
        item_succeeded = sum(video.get("status") == "succeeded" for video in videos)
        item_failed = sum(video.get("status") in {"failed", "rejected"} for video in videos)
        item_running = max(0, item_total - item_succeeded - item_failed)
        item_video_status = _video_status_from_counts(
            item_total,
            item_succeeded,
            item_failed,
            item_running,
        )
        item["metadata"]["video_status"] = item_video_status
        if item_video_status == "completed_with_failures":
            item["status"] = "completed"
        else:
            item["status"] = item_video_status

    if video_status == "succeeded":
        status = "succeeded"
    elif video_status in {"completed_with_failures", "failed"}:
        status = "completed" if video_status == "completed_with_failures" else "failed"
    else:
        status = "running"
    return {
        "status": status,
        "stage": "sglang_video_generation",
        "summary": summary,
        "counters": counters,
        "items": list(items_by_id.values()),
    }


def _callback_token() -> str:
    return (
        os.environ.get("SGLANG_VIDEO_CALLBACK_TOKEN")
        or os.environ.get("LWDP_GENERATION_API_TOKEN")
        or ""
    ).strip()


def _post_final_progress_callback(
    request: dict[str, Any],
    payload: dict[str, Any],
    *,
    callback_urlopen: Any | None = None,
) -> None:
    callback = request.get("callback") if isinstance(request.get("callback"), dict) else {}
    url = callback.get("url")
    if not url:
        return
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = _callback_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-LWDP-Token"] = token
    http_request = urllib.request.Request(
        str(url),
        data=body,
        headers=headers,
        method="PUT",
    )
    opener = callback_urlopen or urllib.request.urlopen
    with opener(http_request, timeout=60) as response:
        if response.status >= 300:
            raise RuntimeError(f"callback failed with HTTP {response.status}")


def repair_final_progress_from_report(
    request: dict[str, Any],
    *,
    s3_client: Any | None = None,
    callback_urlopen: Any | None = None,
) -> bool:
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    report_s3_uri = str(output.get("report_s3_uri") or "")
    if not report_s3_uri:
        return True
    if s3_client is None:
        import boto3

        region = os.environ.get("SGLANG_VIDEO_S3_REGION") or os.environ.get("AWS_REGION")
        s3_client = boto3.client("s3", region_name=region)
    report = _read_json_s3_uri(report_s3_uri, s3_client)
    payload = _progress_payload_from_report(report)
    _post_final_progress_callback(
        request,
        payload,
        callback_urlopen=callback_urlopen,
    )
    return True


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
            return {"not_found": True}
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


def _retry_excluded_backends(config: dict[str, Any], backend_name: str) -> set[str]:
    if backend_name in _b300_backend_names(config):
        return _b300_backend_names(config)
    return {backend_name} if backend_name else set()


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


def _scale_target_key(*, cluster_name: str, nodegroup_name: str) -> str:
    return f"{cluster_name}/{nodegroup_name}"


def _aws_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        error_info = response.get("Error")
        if isinstance(error_info, dict):
            return str(error_info.get("Code") or "")
    return ""


def _nodegroup_health_issues(nodegroup: dict[str, Any]) -> list[dict[str, Any]]:
    health = nodegroup.get("health") if isinstance(nodegroup.get("health"), dict) else {}
    issues = health.get("issues") if isinstance(health.get("issues"), list) else []
    return [issue for issue in issues if isinstance(issue, dict)]


def _fallback_nodegroup_issue_kind(issues: list[dict[str, Any]]) -> str:
    issue_text = " ".join(
        f"{issue.get('code') or ''} {issue.get('message') or ''}".lower()
        for issue in issues
    )
    if any(token in issue_text for token in HARD_FALLBACK_NODEGROUP_ISSUE_TOKENS):
        return "hard"
    if any(token in issue_text for token in CAPACITY_FALLBACK_NODEGROUP_ISSUE_TOKENS):
        return "capacity"
    return ""


def _set_fallback_backend_cooldown(
    config: dict[str, Any],
    state: dict[str, Any],
    backend_name: str,
    *,
    kind: str,
    now: float,
) -> float:
    if kind == "hard":
        cooldown_seconds = _int_or_default(
            config.get("fallback_hard_failure_cooldown_seconds"),
            3600,
        )
    else:
        cooldown_seconds = _int_or_default(
            config.get("fallback_capacity_cooldown_seconds"),
            600,
        )
    until = now + max(0, cooldown_seconds)
    _fallback_backend_cooldown_state(state)[backend_name] = until
    return until


def _describe_nodegroup(
    eks_client: Any,
    *,
    cluster_name: str,
    nodegroup_name: str,
) -> dict[str, Any] | None:
    try:
        response = eks_client.describe_nodegroup(
            clusterName=cluster_name,
            nodegroupName=nodegroup_name,
        )
        nodegroup = response.get("nodegroup") if isinstance(response, dict) else {}
        return nodegroup if isinstance(nodegroup, dict) else {}
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "nodegroup_describe_failed",
                    "nodegroup": nodegroup_name,
                    "error": str(error),
                    "error_code": _aws_error_code(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return None


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
    nodegroup = _describe_nodegroup(
        eks_client,
        cluster_name=cluster_name,
        nodegroup_name=nodegroup_name,
    )
    if nodegroup is not None:
        scaling = (
            nodegroup.get("scalingConfig")
            if isinstance(nodegroup.get("scalingConfig"), dict)
            else {}
        )
        if _int_or_default(scaling.get("desiredSize"), -1) == desired_nodes:
            return
    try:
        eks_client.update_nodegroup_config(
            clusterName=cluster_name,
            nodegroupName=nodegroup_name,
            scalingConfig={"desiredSize": desired_nodes},
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "nodegroup_scale_skipped",
                    "nodegroup": nodegroup_name,
                    "desired_nodes": desired_nodes,
                    "error": str(error),
                    "error_code": _aws_error_code(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _scale_fallback_nodegroups(
    eks_client: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    pods: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> None:
    if eks_client is None:
        return
    now = time.time() if now is None else now
    _cleanup_fallback_prewarm(config, state, now=now)
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

    scale_down_state = state.setdefault("fallback_scale_down", {})
    if not isinstance(scale_down_state, dict):
        scale_down_state = {}
        state["fallback_scale_down"] = scale_down_state
    scale_down_grace = _int_or_default(config.get("fallback_scale_down_grace_seconds"), 900)
    orphan_grace = _int_or_default(
        config.get("fallback_inflight_without_pod_grace_seconds"),
        300,
    )
    for target in targets.values():
        inflight_gpus = _fallback_inflight_gpus_for_nodegroup_scaling(
            state,
            pods or [],
            target["backend_names"],
            target["node_gpus"],
            now=now,
            orphan_grace_seconds=orphan_grace,
        )
        pod_gpus = sum(
            _backend_gpu_requests(pods or [], backend_name)
            for backend_name in target["backend_names"]
        )
        prewarm_gpus = _fallback_prewarm_gpus(
            state,
            target["backend_names"],
            target["node_gpus"],
        )
        desired_gpus = max(inflight_gpus, pod_gpus) + prewarm_gpus
        target_key = _scale_target_key(
            cluster_name=target["cluster_name"],
            nodegroup_name=target["nodegroup_name"],
        )
        if desired_gpus > 0:
            scale_down_state.pop(target_key, None)
        else:
            first_empty_at = scale_down_state.get(target_key)
            if first_empty_at is None:
                scale_down_state[target_key] = now
                continue
            if now - float(first_empty_at) < scale_down_grace:
                continue
        _scale_nodegroup(
            eks_client,
            cluster_name=target["cluster_name"],
            nodegroup_name=target["nodegroup_name"],
            node_gpus=target["node_gpus"],
            max_nodes=target["max_nodes"],
            desired_gpus=desired_gpus,
        )


def _prewarm_fallback_backend(
    eks_client: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    backend: dict[str, Any],
    *,
    requested_gpus: int,
    now: float,
    pods: list[dict[str, Any]] | None = None,
) -> bool:
    if eks_client is None:
        return False
    backend_name = _backend_name(backend)
    if not backend_name:
        return False
    if _fallback_backend_in_cooldown(state, backend_name, now=now):
        return False
    target = _backend_scale_target(backend, config)
    if target is None:
        return False
    nodegroup = _describe_nodegroup(
        eks_client,
        cluster_name=target["cluster_name"],
        nodegroup_name=target["nodegroup_name"],
    )
    if nodegroup is None:
        until = _set_fallback_backend_cooldown(
            config,
            state,
            backend_name,
            kind="capacity",
            now=now,
        )
        print(
            json.dumps(
                {
                    "status": "fallback_prewarm_skipped",
                    "backend": backend_name,
                    "nodegroup": target["nodegroup_name"],
                    "reason": "nodegroup_describe_failed",
                    "cooldown_until": until,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return False
    nodegroup_status = str(nodegroup.get("status") or "ACTIVE")
    if nodegroup_status != "ACTIVE":
        cooldown_seconds = _int_or_default(
            config.get("fallback_nodegroup_not_active_cooldown_seconds"),
            120,
        )
        until = now + max(0, cooldown_seconds)
        _fallback_backend_cooldown_state(state)[backend_name] = until
        print(
            json.dumps(
                {
                    "status": "fallback_prewarm_skipped",
                    "backend": backend_name,
                    "nodegroup": target["nodegroup_name"],
                    "nodegroup_status": nodegroup_status,
                    "reason": "nodegroup_not_active",
                    "cooldown_until": until,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return False
    issues = _nodegroup_health_issues(nodegroup)
    issue_kind = _fallback_nodegroup_issue_kind(issues)
    if issue_kind:
        until = _set_fallback_backend_cooldown(
            config,
            state,
            backend_name,
            kind=issue_kind,
            now=now,
        )
        print(
            json.dumps(
                {
                    "status": "fallback_prewarm_skipped",
                    "backend": backend_name,
                    "nodegroup": target["nodegroup_name"],
                    "reason": f"nodegroup_{issue_kind}_issue",
                    "issues": issues,
                    "cooldown_until": until,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return False

    _add_fallback_prewarm(
        state,
        backend_name,
        requested_gpus=requested_gpus,
        now=now,
    )
    _scale_fallback_nodegroups(eks_client, config, state, pods=pods or [], now=now)
    return True


def _prime_fallback_scale_down(
    config: dict[str, Any],
    state: dict[str, Any],
    backend_names: set[str],
    *,
    now: float,
) -> None:
    fallback_names = _fallback_backend_names(config)
    target_backend_names = backend_names & fallback_names
    if not target_backend_names:
        return

    scale_down_state = state.setdefault("fallback_scale_down", {})
    if not isinstance(scale_down_state, dict):
        scale_down_state = {}
        state["fallback_scale_down"] = scale_down_state
    scale_down_grace = _int_or_default(config.get("fallback_scale_down_grace_seconds"), 900)
    backends = config.get("backends") if isinstance(config.get("backends"), list) else []
    for backend in backends:
        if not isinstance(backend, dict) or _backend_name(backend) not in target_backend_names:
            continue
        target = _backend_scale_target(backend, config)
        if target is None:
            continue
        target_key = _scale_target_key(
            cluster_name=target["cluster_name"],
            nodegroup_name=target["nodegroup_name"],
        )
        scale_down_state[target_key] = now - scale_down_grace


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
    s3_client: Any | None = None,
    callback_urlopen: Any | None = None,
) -> dict[str, int]:
    namespace = config.get("namespace", "default")
    max_attempts = _int_or_default(config.get("max_job_attempts"), 5)
    nodes = _list_nodes(core_client)
    pods = _list_capacity_pods(core_client, config)
    completed = restarted = released = failed = missing = pending = 0
    released_prewarm = 0
    callback_repaired = callback_pending = 0
    remaining = []
    all_inflight = list(state.get("inflight", []))
    deleted_job_names: set[str] = set()
    fallback_backends_retargeted_to_b300: set[str] = set()
    fallback_backends_released: set[str] = set()

    for item in all_inflight:
        old_job_name = str(item.get("job_name") or "")
        old_backend_name = str(item.get("backend") or "")
        if not _inflight_counts_for_capacity(item):
            try:
                if repair_final_progress_from_report(
                    item["request"],
                    s3_client=s3_client,
                    callback_urlopen=callback_urlopen,
                ):
                    callback_repaired += 1
            except Exception as error:
                item = _mark_callback_pending(item, error, now=now)
                remaining.append(item)
                callback_pending += 1
                continue
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=item["receipt_handle"],
            )
            completed += 1
            continue

        status = _read_job_status(batch_client, namespace, item["job_name"])
        if _job_succeeded(status):
            try:
                if repair_final_progress_from_report(
                    item["request"],
                    s3_client=s3_client,
                    callback_urlopen=callback_urlopen,
                ):
                    callback_repaired += 1
            except Exception as error:
                item = _mark_callback_pending(item, error, now=now)
                remaining.append(item)
                callback_pending += 1
                continue
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=item["receipt_handle"],
            )
            completed += 1
            continue

        if _job_missing(status):
            missing += 1
            missing_grace = _int_or_default(config.get("missing_job_grace_seconds"), 180)
            started_at = float(item.get("started_at", now))
            first_missing_at = float(item.get("missing_job_seen_at", now))
            if "missing_job_seen_at" not in item:
                item["missing_job_seen_at"] = now
                first_missing_at = now
            if now - started_at < missing_grace and now - first_missing_at < missing_grace:
                remaining.append(item)
                continue

            attempts = _int_or_default(item.get("attempts"), 1)
            if attempts >= max_attempts:
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=item["receipt_handle"],
                    VisibilityTimeout=0,
                )
                failed += 1
                continue

            requested_gpus = _int_or_default(item.get("requested_gpus"), 8)
            backend = choose_backend(
                config,
                nodes,
                pods,
                requested_gpus=requested_gpus,
                state={"inflight": [candidate for candidate in all_inflight if candidate is not item]},
            )
            if backend is None:
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=item["receipt_handle"],
                    VisibilityTimeout=_int_or_default(
                        config.get("defer_visibility_timeout"),
                        60,
                    ),
                )
                if old_backend_name in _fallback_backend_names(config):
                    fallback_backends_released.add(old_backend_name)
                released += 1
                continue
            next_attempt = attempts + 1
            next_backend_name = _backend_name(backend)
            manifest = _render_for_backend(item["request"], config, backend, attempt=next_attempt)
            batch_client.create_namespaced_job(namespace=namespace, body=manifest)
            item = {
                **item,
                "job_name": manifest["metadata"]["name"],
                "backend": next_backend_name,
                "attempts": next_attempt,
                "started_at": now,
                "last_renewed_at": now,
                "requested_gpus": requested_gpus,
            }
            if (
                old_backend_name in _fallback_backend_names(config)
                and next_backend_name in _b300_backend_names(config)
            ):
                fallback_backends_retargeted_to_b300.add(old_backend_name)
            item.pop("missing_job_seen_at", None)
            item.pop("pending_job_seen_at", None)
            restarted += 1

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
                    name=old_job_name,
                    namespace=namespace,
                    propagation_policy="Background",
                )
                deleted_job_names.add(old_job_name)
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
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=item["receipt_handle"],
                    VisibilityTimeout=_int_or_default(
                        config.get("defer_visibility_timeout"),
                        60,
                    ),
                )
                if old_backend_name in _fallback_backend_names(config):
                    fallback_backends_released.add(old_backend_name)
                released += 1
                continue
            next_attempt = attempts + 1
            next_backend_name = _backend_name(backend)
            manifest = _render_for_backend(item["request"], config, backend, attempt=next_attempt)
            batch_client.create_namespaced_job(namespace=namespace, body=manifest)
            item = {
                **item,
                "job_name": manifest["metadata"]["name"],
                "backend": next_backend_name,
                "attempts": next_attempt,
                "started_at": now,
                "last_renewed_at": now,
                "requested_gpus": requested_gpus,
            }
            if (
                old_backend_name in _fallback_backend_names(config)
                and next_backend_name in _b300_backend_names(config)
            ):
                fallback_backends_retargeted_to_b300.add(old_backend_name)
            item.pop("missing_job_seen_at", None)
            item.pop("pending_job_seen_at", None)
            restarted += 1

        if not _job_missing(status) and not _job_failed(status):
            pending_pods = _pending_pods_for_job(pods, str(item.get("job_name") or ""))
            if pending_pods:
                pending += 1
                if _active_job_caps_exceeded(
                    config,
                    pods,
                    {"inflight": all_inflight},
                    str(item.get("backend") or ""),
                ):
                    try:
                        batch_client.delete_namespaced_job(
                            name=old_job_name,
                            namespace=namespace,
                            propagation_policy="Background",
                        )
                        deleted_job_names.add(old_job_name)
                    except Exception as error:
                        if not _not_found(error):
                            raise
                    sqs_client.change_message_visibility(
                        QueueUrl=queue_url,
                        ReceiptHandle=item["receipt_handle"],
                        VisibilityTimeout=_int_or_default(
                            config.get("defer_visibility_timeout"),
                            60,
                        ),
                    )
                    released += 1
                    continue

                pending_grace = _int_or_default(config.get("pending_job_grace_seconds"), 900)
                started_at = float(item.get("started_at", now))
                first_pending_at = float(item.get("pending_job_seen_at", now))
                if "pending_job_seen_at" not in item:
                    item["pending_job_seen_at"] = now
                    first_pending_at = now
                if now - started_at < pending_grace and now - first_pending_at < pending_grace:
                    remaining.append(item)
                    continue

                attempts = _int_or_default(item.get("attempts"), 1)
                if attempts >= max_attempts:
                    sqs_client.change_message_visibility(
                        QueueUrl=queue_url,
                        ReceiptHandle=item["receipt_handle"],
                        VisibilityTimeout=0,
                    )
                    failed += 1
                    continue

                requested_gpus = _int_or_default(item.get("requested_gpus"), 8)
                backend = choose_backend(
                    config,
                    nodes,
                    _pods_except_job(pods, str(item.get("job_name") or "")),
                    requested_gpus=requested_gpus,
                    state={
                        "inflight": [
                            candidate for candidate in all_inflight if candidate is not item
                        ]
                    },
                    excluded_backend_names=_retry_excluded_backends(
                        config,
                        str(item.get("backend") or ""),
                    ),
                )
                if backend is None:
                    if old_backend_name in _fallback_backend_names(config):
                        try:
                            batch_client.delete_namespaced_job(
                                name=old_job_name,
                                namespace=namespace,
                                propagation_policy="Background",
                            )
                            deleted_job_names.add(old_job_name)
                        except Exception as error:
                            if not _not_found(error):
                                raise
                        sqs_client.change_message_visibility(
                            QueueUrl=queue_url,
                            ReceiptHandle=item["receipt_handle"],
                            VisibilityTimeout=_int_or_default(
                                config.get("defer_visibility_timeout"),
                                60,
                            ),
                        )
                        fallback_backends_released.add(old_backend_name)
                        released += 1
                        continue
                    remaining.append(item)
                    released += 1
                    continue

                try:
                    batch_client.delete_namespaced_job(
                        name=old_job_name,
                        namespace=namespace,
                        propagation_policy="Background",
                    )
                    deleted_job_names.add(old_job_name)
                except Exception as error:
                    if not _not_found(error):
                        raise
                next_attempt = attempts + 1
                next_backend_name = _backend_name(backend)
                manifest = _render_for_backend(
                    item["request"],
                    config,
                    backend,
                    attempt=next_attempt,
                )
                batch_client.create_namespaced_job(namespace=namespace, body=manifest)
                item = {
                    **item,
                    "job_name": manifest["metadata"]["name"],
                    "backend": next_backend_name,
                    "attempts": next_attempt,
                    "started_at": now,
                    "last_renewed_at": now,
                    "requested_gpus": requested_gpus,
                }
                if (
                    old_backend_name in _fallback_backend_names(config)
                    and next_backend_name in _b300_backend_names(config)
                ):
                    fallback_backends_retargeted_to_b300.add(old_backend_name)
                item.pop("missing_job_seen_at", None)
                item.pop("pending_job_seen_at", None)
                restarted += 1

        remaining.append(item)

    state["inflight"] = remaining
    scale_down_backends = fallback_backends_retargeted_to_b300 | fallback_backends_released
    released_prewarm_backends = _release_fallback_prewarm_for_backends(
        state,
        scale_down_backends,
    )
    released_prewarm = len(released_prewarm_backends)
    _prime_fallback_scale_down(
        config,
        state,
        scale_down_backends | released_prewarm_backends,
        now=now,
    )
    _scale_fallback_nodegroups(
        eks_client,
        config,
        state,
        pods=_pods_except_jobs(pods, deleted_job_names),
        now=now,
    )
    return {
        "completed": completed,
        "restarted": restarted,
        "released": released,
        "failed": failed,
        "missing": missing,
        "pending": pending,
        "callback_repaired": callback_repaired,
        "callback_pending": callback_pending,
        "released_prewarm": released_prewarm,
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
    s3_client: Any | None = None,
    callback_urlopen: Any | None = None,
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
        s3_client=s3_client,
        callback_urlopen=callback_urlopen,
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
        pods = _list_capacity_pods(core_client, config)
        released_prewarm = _release_unused_fallback_prewarm(config, state, pods)
        if released_prewarm:
            _prime_fallback_scale_down(
                config,
                state,
                released_prewarm,
                now=now,
            )
            _scale_fallback_nodegroups(eks_client, config, state, pods=pods, now=now)
        return {
            "status": "idle",
            **renew_result,
            **reconcile_result,
            "started": 0,
            "deferred": 0,
            "released_prewarm": len(released_prewarm),
        }

    nodes = _list_nodes(core_client)
    pods = _list_capacity_pods(core_client, config)
    started = deferred = adopted = completed_existing = 0
    prewarmed = prewarm_skipped = 0
    released_prewarm_count = _int_or_default(reconcile_result.get("released_prewarm"), 0)
    callback_repaired_existing = callback_pending_existing = 0
    for message in messages:
        request = _decode_request(message)
        namespace = config.get("namespace", "default")
        existing_job = _existing_job_for_request(batch_client, namespace, request)
        if existing_job is not None:
            status = _job_status(existing_job)
            if _job_succeeded(status):
                try:
                    if repair_final_progress_from_report(
                        request,
                        s3_client=s3_client,
                        callback_urlopen=callback_urlopen,
                    ):
                        callback_repaired_existing += 1
                except Exception:
                    sqs_client.change_message_visibility(
                        QueueUrl=queue_url,
                        ReceiptHandle=message["ReceiptHandle"],
                        VisibilityTimeout=_int_or_default(
                            config.get("defer_visibility_timeout"),
                            60,
                        ),
                    )
                    callback_pending_existing += 1
                    continue
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
                _scale_fallback_nodegroups(eks_client, config, state, pods=pods, now=now)
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
            prewarm_backend = choose_fallback_prewarm_backend(
                config,
                nodes,
                pods,
                requested_gpus=requested_gpus,
                state=state,
                now=now,
            )
            if prewarm_backend is not None:
                if _prewarm_fallback_backend(
                    eks_client,
                    config,
                    state,
                    prewarm_backend,
                    requested_gpus=requested_gpus,
                    now=now,
                    pods=pods,
                ):
                    prewarmed += 1
                else:
                    prewarm_skipped += 1
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
        backend_name = _backend_name(backend)
        if backend_name in _fallback_backend_names(config):
            _consume_fallback_prewarm(
                state,
                backend_name,
                requested_gpus=requested_gpus,
            )
        elif backend_name in _b300_backend_names(config):
            released_prewarm_backends = _consume_any_fallback_prewarm(
                state,
                _fallback_backend_names(config),
                requested_gpus=requested_gpus,
            )
            if released_prewarm_backends:
                released_prewarm_count += len(released_prewarm_backends)
                _prime_fallback_scale_down(
                    config,
                    state,
                    released_prewarm_backends,
                    now=now,
                )
        _scale_fallback_nodegroups(eks_client, config, state, pods=pods, now=now)
        started += 1

    return {
        "status": "started" if started else "adopted" if adopted else "deferred",
        **renew_result,
        **reconcile_result,
        "started": started,
        "deferred": deferred,
        "adopted": adopted,
        "prewarmed": prewarmed,
        "prewarm_skipped": prewarm_skipped,
        "released_prewarm": released_prewarm_count,
        "completed_existing": completed_existing,
        "callback_repaired_existing": callback_repaired_existing,
        "callback_pending_existing": callback_pending_existing,
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
        "missing_job_grace_seconds": os.environ.get(
            "SGLANG_VIDEO_MISSING_JOB_GRACE_SECONDS",
            "180",
        ),
        "pending_job_grace_seconds": os.environ.get(
            "SGLANG_VIDEO_PENDING_JOB_GRACE_SECONDS",
            "900",
        ),
        "max_active_jobs": os.environ.get("SGLANG_VIDEO_MAX_ACTIVE_JOBS", "7"),
        "b300_max_active_jobs": os.environ.get("SGLANG_VIDEO_B300_MAX_ACTIVE_JOBS", "5"),
        "fallback_max_active_jobs": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_MAX_ACTIVE_JOBS",
            "2",
        ),
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
        "fallback_scale_down_grace_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_SCALE_DOWN_GRACE_SECONDS",
            "60",
        ),
        "fallback_inflight_without_pod_grace_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_INFLIGHT_WITHOUT_POD_GRACE_SECONDS",
            "300",
        ),
        "fallback_nodegroup_not_active_cooldown_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_NODEGROUP_NOT_ACTIVE_COOLDOWN_SECONDS",
            "120",
        ),
        "fallback_prewarm_ttl_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_PREWARM_TTL_SECONDS",
            "600",
        ),
        "fallback_capacity_cooldown_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_CAPACITY_COOLDOWN_SECONDS",
            "600",
        ),
        "fallback_hard_failure_cooldown_seconds": os.environ.get(
            "SGLANG_VIDEO_FALLBACK_HARD_FAILURE_COOLDOWN_SECONDS",
            "3600",
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
    s3_region = os.environ.get("SGLANG_VIDEO_S3_REGION") or os.environ.get("AWS_REGION")
    s3_client = boto3.client("s3", region_name=s3_region)
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
            s3_client=s3_client,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] == "idle":
            time.sleep(1)


if __name__ == "__main__":
    main()
