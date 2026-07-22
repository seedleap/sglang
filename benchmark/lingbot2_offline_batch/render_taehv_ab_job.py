#!/usr/bin/env python3
"""Render one isolated B300 LingBot VAE A/B Kubernetes Job."""

from __future__ import annotations

import argparse
import json
from typing import Any


def env(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def render_job(
    *,
    name: str,
    image: str,
    variant: str,
    run_id: str,
    output_s3_prefix: str,
) -> dict[str, Any]:
    if variant not in {"baseline", "taehv"}:
        raise ValueError("variant must be baseline or taehv")
    results_root = f"/fsx/world-model/eval/lingbot2/taehv_ab/{run_id}/{variant}"
    output_uri = f"{output_s3_prefix.rstrip('/')}/{variant}"
    variables = [
        env("RESULTS_ROOT", results_root),
        env("TAEHV_AB_VARIANT", variant),
        env("TAEHV_AB_RUN_ID", run_id),
        env("TAEHV_AB_OUTPUT_S3_URI", output_uri),
        env(
            "TAEHV_AB_SOURCE_S3_URI",
            "s3://leap-world-us-east-2/world-model/eval/platform/eval_sets/minWM/testset100_v2/messages.jsonl",
        ),
        env("SGLANG_VIDEO_CASE_LIMIT", "100"),
        env("SGLANG_VIDEO_GPU_TOTAL", "8"),
        env("SGLANG_VIDEO_GPUS_PER_SERVER", "1"),
        env("SGLANG_VIDEO_TOPOLOGY", "8x1"),
        env("SGLANG_VIDEO_WIDTH", "832"),
        env("SGLANG_VIDEO_HEIGHT", "480"),
        env("SGLANG_VIDEO_FPS", "16"),
        env("STREAM_UPLOAD", "false"),
        env("RESUME", "false"),
        env("MODEL_ID", "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"),
        env("MODEL_REVISION", "59cccf49f2d2dd27418ae7a04b82b10868d455c2"),
        env("HF_HOME", "/fsx/hf-lb2"),
        env("HF_HUB_ENABLE_HF_TRANSFER", "1"),
    ]
    if variant == "taehv":
        variables.append(env("TAEHV_CHECKPOINT_PATH", "/opt/taehv/taew2_1.pth"))
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "default",
            "labels": {
                "app.kubernetes.io/name": "lingbot-taehv-ab",
                "seedleap.ai/task": "lingbot-taehv-ab-testset100",
                "seedleap.ai/variant": variant,
            },
            "annotations": {
                "seedleap.ai/test-only": "true",
                "seedleap.ai/production-switch": "false",
                "seedleap.ai/benchmark-contract": "testset100-v2-first100-832x480-16fps-4steps-8x1",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 14400,
            "ttlSecondsAfterFinished": 172800,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "lingbot-taehv-ab",
                        "seedleap.ai/task": "lingbot-taehv-ab-testset100",
                        "seedleap.ai/variant": variant,
                    }
                },
                "spec": {
                    "serviceAccountName": "sglang-video-job",
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 60,
                    "schedulerName": "default-scheduler",
                    "nodeSelector": {
                        "eks.amazonaws.com/capacityType": "CAPACITY_BLOCK",
                        "eks.amazonaws.com/nodegroup": "wan22-cb-p6b300-0715-20c",
                        "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
                    },
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "benchmark",
                            "image": image,
                            "imagePullPolicy": "Always",
                            "command": ["bash", "/opt/bench/run_taehv_ab_test.sh"],
                            "env": variables,
                            "resources": {
                                "requests": {
                                    "cpu": "160",
                                    "memory": "1200Gi",
                                    "nvidia.com/gpu": "8",
                                },
                                "limits": {
                                    "cpu": "180",
                                    "memory": "1600Gi",
                                    "nvidia.com/gpu": "8",
                                },
                            },
                            "securityContext": {"capabilities": {"add": ["SYS_ADMIN"]}},
                            "volumeMounts": [
                                {"name": "fsx", "mountPath": "/fsx"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "fsx",
                            "persistentVolumeClaim": {"claimName": "xacct-fsx-pvc"},
                        },
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "512Gi"},
                        },
                    ],
                },
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--variant", choices=("baseline", "taehv"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-s3-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            render_job(
                name=args.name,
                image=args.image,
                variant=args.variant,
                run_id=args.run_id,
                output_s3_prefix=args.output_s3_prefix,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
