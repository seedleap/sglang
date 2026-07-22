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
    source_s3_uri: str,
    source_revision: str,
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
        variables.extend(
            [
                env("TAEHV_CHECKPOINT_PATH", "/opt/taehv/taew2_1.pth"),
                env("PYTHONPATH", "/opt/taehv/python:/opt/sglang/python"),
            ]
        )
    else:
        variables.append(env("PYTHONPATH", "/opt/sglang/python"))

    runtime_volume_mounts = [
        {"name": "fsx", "mountPath": "/fsx"},
        {"name": "shm", "mountPath": "/dev/shm"},
        {"name": "sglang-source", "mountPath": "/opt/sglang"},
        {
            "name": "sglang-source",
            "mountPath": "/opt/bench",
            "subPath": "benchmark/lingbot2_offline_batch",
        },
    ]
    init_containers = [
        {
            "name": "prepare-sglang-source",
            "image": image,
            "imagePullPolicy": "Always",
            "command": ["bash", "-ceu"],
            "args": [
                """
set -x
python3 -m pip install --no-cache-dir --target /bootstrap/python boto3
echo "source-init: boto3 installation completed"
PYTHONPATH=/bootstrap/python python3 - <<'PY'
import os
import traceback
from pathlib import Path
import boto3

try:
    uri = os.environ["TAEHV_AB_SOURCE_BUNDLE_S3_URI"]
    print(f"source-init: requested bundle={uri}", flush=True)
    if not uri.startswith("s3://"):
        raise RuntimeError(f"expected s3 uri, got {uri!r}")
    bucket, key = uri[5:].split("/", 1)
    target = Path("/bootstrap/source.tar.gz")
    identity = boto3.client("sts").get_caller_identity()["Arn"]
    print(f"source-init: caller={identity}", flush=True)
    boto3.client("s3").download_file(bucket, key, str(target))
    print(f"source-init: downloaded_bytes={target.stat().st_size}", flush=True)
except Exception:
    traceback.print_exc()
    raise
PY
tar -xzf /bootstrap/source.tar.gz -C /opt/sglang
test -f /opt/sglang/python/sglang/multimodal_gen/vae/vae_decoder.py
printf '%s\\n' "$TAEHV_AB_SOURCE_REVISION" > /opt/sglang/TAEHV_AB_SOURCE_REVISION
""",
            ],
            "env": [
                env("TAEHV_AB_SOURCE_BUNDLE_S3_URI", source_s3_uri),
                env("TAEHV_AB_SOURCE_REVISION", source_revision),
            ],
            "resources": {
                "requests": {"cpu": "8", "memory": "24Gi"},
                "limits": {"cpu": "16", "memory": "48Gi"},
            },
            "volumeMounts": [
                {"name": "sglang-source", "mountPath": "/opt/sglang"},
                {"name": "bootstrap", "mountPath": "/bootstrap"},
            ],
        }
    ]
    if variant == "taehv":
        init_containers.append(
            {
                "name": "install-taehv",
                "image": image,
                "imagePullPolicy": "Always",
                "command": ["bash", "-ceu"],
                "args": [
                    """
python3 -m pip install --no-cache-dir --no-deps --target /opt/taehv/python \\
  'taehv @ git+https://github.com/madebyollin/taehv.git@093b918971d59001a0bad6dfd6e0409b5e1752cf'
python3 - <<'PY'
import hashlib
import urllib.request
from pathlib import Path

url = "https://raw.githubusercontent.com/madebyollin/taehv/093b918971d59001a0bad6dfd6e0409b5e1752cf/taew2_1.pth"
target = Path("/opt/taehv/taew2_1.pth")
urllib.request.urlretrieve(url, target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
expected = "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e"
if digest != expected:
    raise SystemExit(f"TAEHV checkpoint sha256 mismatch: {digest}")
PY
""",
                ],
                "resources": {
                    "requests": {"cpu": "8", "memory": "24Gi"},
                    "limits": {"cpu": "16", "memory": "48Gi"},
                },
                "volumeMounts": [{"name": "taehv-runtime", "mountPath": "/opt/taehv"}],
            }
        )
        runtime_volume_mounts.append(
            {"name": "taehv-runtime", "mountPath": "/opt/taehv"}
        )
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
                    "initContainers": init_containers,
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
                            "volumeMounts": runtime_volume_mounts,
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
                        {"name": "sglang-source", "emptyDir": {}},
                        {"name": "bootstrap", "emptyDir": {}},
                        {"name": "taehv-runtime", "emptyDir": {}},
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
    parser.add_argument("--source-s3-uri", required=True)
    parser.add_argument("--source-revision", required=True)
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
                source_s3_uri=args.source_s3_uri,
                source_revision=args.source_revision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
