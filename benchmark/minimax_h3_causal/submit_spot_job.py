# SPDX-License-Identifier: Apache-2.0
"""Render or submit a pinned MiniMax H3 B200/B300 Spot benchmark Job."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from typing import Any


_DEFAULT_IMAGE = (
    "lmsysorg/sglang:dev@sha256:"
    "8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7"
)
_DEFAULT_MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
_CLUSTER_PROFILES = {
    "use1-atl2": {
        "context": "codex-seed-leap-use1",
        "nodepool": "minwm-test-atl2-p6-spot",
        "capacity_pool": "minwm-test-atl2-karpenter",
        "zone": "us-east-1-atl-2a",
    },
    "usw2d-sp12": {
        "context": "codex-minwm-test-phx2",
        "nodepool": "minwm-sp12-usw2d-p6-spot",
        "capacity_pool": "minwm-sp12-usw2d-karpenter",
        "zone": "us-west-2d",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("attention-probe", "e2e"), required=True)
    parser.add_argument("--hardware", choices=("b200", "b300"), required=True)
    parser.add_argument(
        "--cluster-profile",
        choices=tuple(_CLUSTER_PROFILES),
        default="use1-atl2",
    )
    parser.add_argument("--git-ref", required=True, help="Pushed immutable commit SHA")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--causal-mode", choices=("off", "flex"), default="flex")
    parser.add_argument(
        "--mask-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse structural FlexAttention block masks across requests.",
    )
    parser.add_argument("--nfe", nargs="+", type=int, default=[3])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--model-revision", default=_DEFAULT_MODEL_REVISION)
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument("--job-name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Submit the Job. Without this flag, print the manifest only.",
    )
    return parser.parse_args()


def _env(name: str, value: object) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def _job(args) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", args.git_ref) is None:
        raise ValueError("--git-ref must be a 40-character commit SHA")
    if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
        raise ValueError("--model-revision must be a 40-character commit SHA")
    if any(nfe <= 0 for nfe in args.nfe):
        raise ValueError("--nfe values must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.seconds <= 0:
        raise ValueError("warmup/repeats/seconds are out of range")

    gpu_count = 1 if args.phase == "attention-probe" else 8
    if args.phase == "attention-probe":
        if args.tp_size != 1 or args.ulysses_degree != 1:
            raise ValueError("attention-probe requires TP=1 and Ulysses=1")
    elif args.tp_size * args.ulysses_degree != gpu_count:
        raise ValueError("e2e requires TP * Ulysses = 8")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = (
        "probe"
        if args.phase == "attention-probe"
        else f"tp{args.tp_size}-u{args.ulysses_degree}"
    )
    default_name = f"minimax-h3-{args.hardware}-{suffix}-{timestamp}"
    job_name = args.job_name or default_name
    if re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", job_name) is None:
        raise ValueError("--job-name must be a valid lowercase Kubernetes name")
    if len(job_name) > 63:
        raise ValueError("--job-name must contain at most 63 characters")

    instance_type = f"p6-{args.hardware}.48xlarge"
    cluster_profile = _CLUSTER_PROFILES[args.cluster_profile]
    run_script = r"""
set -euo pipefail
repo=/workspace/sglang
repo_url=https://github.com/seedleap/sglang.git
auth_header="Authorization: Basic $(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')"
git init -q "${repo}"
git -C "${repo}" remote add origin "${repo_url}"
git -C "${repo}" -c http.extraHeader="${auth_header}" fetch --depth=1 origin "${SGLANG_GIT_REF}"
unset auth_header GITHUB_TOKEN
git -C "${repo}" checkout -q --detach "${SGLANG_GIT_REF}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
cd "${repo}"
exec bash benchmark/minimax_h3_causal/aws_entrypoint.sh
""".strip()

    env = [
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "github-token", "key": "token"}},
        },
        _env("SGLANG_GIT_REF", args.git_ref),
        _env("H3_PHASE", args.phase),
        _env("H3_RUN_ID", job_name),
        _env("H3_HARDWARE", args.hardware),
        _env("H3_GPU_COUNT", gpu_count),
        _env("H3_TP_SIZE", args.tp_size),
        _env("H3_ULYSSES_DEGREE", args.ulysses_degree),
        _env("H3_CAUSAL_MODE", args.causal_mode),
        _env("H3_MASK_CACHE", str(args.mask_cache).lower()),
        _env("H3_NFE", " ".join(str(value) for value in args.nfe)),
        _env("H3_WARMUP", args.warmup),
        _env("H3_REPEATS", args.repeats),
        _env("H3_SECONDS", args.seconds),
        _env("H3_MODEL_ID", args.model_id),
        _env("H3_MODEL_REVISION", args.model_revision),
        {
            "name": "POD_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
        {
            "name": "NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
    ]
    if gpu_count == 1:
        requests = {"cpu": "32", "memory": "200Gi", "nvidia.com/gpu": "1"}
        limits = {"cpu": "64", "memory": "400Gi", "nvidia.com/gpu": "1"}
        shm_size = "64Gi"
    else:
        requests = {
            "cpu": "64",
            "memory": "512Gi",
            "ephemeral-storage": "180Gi",
            "nvidia.com/gpu": "8",
        }
        limits = {
            "cpu": "192",
            "memory": "1Ti",
            "ephemeral-storage": "350Gi",
            "nvidia.com/gpu": "8",
        }
        shm_size = "128Gi"

    labels = {
        "app.kubernetes.io/name": job_name,
        "seedleap.ai/owner": "chenshengdong",
        "seedleap.ai/task": "minimax-h3-causal-realtime",
        "seedleap.ai/run-id": job_name,
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": labels,
            "annotations": {
                "seedleap.ai/repo": "seedleap/sglang",
                "seedleap.ai/sglang-git-ref": args.git_ref,
                "seedleap.ai/runtime-image": args.image,
                "seedleap.ai/model-revision": args.model_revision,
                "seedleap.ai/capacity-type": "spot",
                "seedleap.ai/cluster-profile": args.cluster_profile,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 7200 if gpu_count == 1 else 21600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 60,
                    "securityContext": {"seLinuxOptions": {"type": "spc_t"}},
                    "nodeSelector": {
                        "karpenter.sh/capacity-type": "spot",
                        "karpenter.sh/nodepool": cluster_profile["nodepool"],
                        "seedleap.ai/capacity-pool": cluster_profile["capacity_pool"],
                        "node.kubernetes.io/instance-type": instance_type,
                        "topology.kubernetes.io/zone": cluster_profile["zone"],
                    },
                    "tolerations": [
                        {
                            "key": "seedleap.ai/capacity-pool",
                            "operator": "Equal",
                            "value": cluster_profile["capacity_pool"],
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {"name": "work", "emptyDir": {}},
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": shm_size},
                        },
                        {
                            "name": "cache",
                            "hostPath": {
                                "path": "/var/lib/minimax-h3-benchmark/cache",
                                "type": "DirectoryOrCreate",
                            },
                        },
                    ],
                    "containers": [
                        {
                            "name": "benchmark",
                            "image": args.image,
                            "imagePullPolicy": "IfNotPresent",
                            "workingDir": "/workspace",
                            "command": ["/bin/bash", "-lc", run_script],
                            "env": env,
                            "resources": {"requests": requests, "limits": limits},
                            "volumeMounts": [
                                {"name": "work", "mountPath": "/work"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                                {"name": "cache", "mountPath": "/root/.cache"},
                            ],
                        }
                    ],
                },
            },
        },
    }


def main() -> None:
    args = parse_args()
    manifest = json.dumps(_job(args), indent=2)
    if not args.apply:
        print(manifest)
        return
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            _CLUSTER_PROFILES[args.cluster_profile]["context"],
            "apply",
            "-f",
            "-",
        ],
        input=manifest,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
