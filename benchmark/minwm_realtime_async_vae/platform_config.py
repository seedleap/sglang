"""Loopit platform application drafts for the World Model product group.

These are API payloads/goldens, not Kubernetes or Argo CD manifests. Image
placeholders are accepted only in review mode; a deployment render must resolve
every placeholder to a registry reference pinned with ``@sha256``.
"""

from __future__ import annotations

import copy
import re
from typing import Any


BUSINESS_LINE = "world-model"
NAMESPACE = "world-model"
APP_GROUP = "world-studio"
REGISTRY = "829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime"
DENOISER_IMAGE = (
    f"{REGISTRY}@sha256:"
    "77b975f6758e642462c984dec3e1e51ef806622eb9bf3b9304330f6e072c3209"
)
WEBUI_IMAGE = (
    f"{REGISTRY}@sha256:"
    "38bfc1802736805d98f43aed08c43bf239da010d64d7605b5a96a7f0cb2335cc"
)
IMAGE_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
IMAGE_PLACEHOLDERS = {
    "${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}",
    "${WORLD_REALTIME_COORDINATOR_IMAGE_DIGEST}",
    "${WORLD_REALTIME_GATEWAY_IMAGE_DIGEST}",
    "${WORLD_REALTIME_VAE_IMAGE_DIGEST}",
}

MINWM_RELEASE = (
    "models/minwm/wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2/"
    "gs3200-ema-student-v1/releases/20260810T042157Z-c302d572/model"
)
LINGBOT2_REVISION = "59cccf49f2d2dd27418ae7a04b82b10868d455c2"
LINGBOT2_RELEASE = (
    "models/lingbot2/robbyant-lingbot-world-v2-14b-causal-fast-diffusers/"
    f"{LINGBOT2_REVISION}/releases/${{LINGBOT2_RELEASE_ID}}/model"
)


def _env(name: str, value: str) -> dict[str, Any]:
    return {"name": name, "value": value}


def _field_env(name: str, field_path: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": field_path}}}


def _secret_env(name: str, key: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {
            "secretKeyRef": {"name": "world-studio-runtime", "key": key}
        },
    }


def _http_probe(
    path: str,
    port: int,
    *,
    period: int,
    timeout: int = 5,
    failures: int = 6,
    initial_delay: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "httpGet",
        "path": path,
        "port": port,
        "periodSeconds": period,
        "timeoutSeconds": timeout,
        "failureThreshold": failures,
    }
    if initial_delay is not None:
        result["initialDelaySeconds"] = initial_delay
    return result


def _spread(name: str) -> list[dict[str, Any]]:
    return [
        {
            "maxSkew": 1,
            "topologyKey": "topology.kubernetes.io/zone",
            "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {
                "matchLabels": {"loopit.me/service-id": name}
            },
        }
    ]


def _network_policy(
    *, ingress_from: list[str], egress_to: list[str], external: list[str] | None = None
) -> dict[str, Any]:
    return {
        "enabled": True,
        "defaultDeny": True,
        "ingressFromServiceIds": ingress_from,
        "egressToServiceIds": egress_to,
        "externalEgress": external or [],
    }


def _base_application(
    *,
    name: str,
    app_name: str,
    deploy_type: str,
    image: str,
    replicas: int,
    service_account: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "application",
        "name": name,
        "appName": app_name,
        "businessLineId": BUSINESS_LINE,
        "clusterName": "world-model",
        "namespace": NAMESPACE,
        "appGroup": APP_GROUP,
        "lane": "default",
        "deployType": deploy_type,
        "replicas": replicas,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "serviceAccountName": service_account,
        "labels": {
            "loopit.me/business-line": BUSINESS_LINE,
            "loopit.me/managed-by": "platform",
            "loopit.me/service-id": name,
            "loopit.me/lane": "default",
            "app.kubernetes.io/part-of": APP_GROUP,
        },
        "annotations": {
            "logs.loopit.me/enabled": "true",
            "logs.loopit.me/bucket": BUSINESS_LINE,
        },
        "command": [],
        "args": [],
        "env": [],
        "resources": {"requests": {}, "limits": {}},
        "initContainers": [],
        "volumes": [],
        "volumeMounts": [],
        "nodeSelector": {},
        "tolerations": [],
        "affinity": {},
        "topologySpreadConstraints": [],
        "terminationGracePeriodSeconds": 30,
    }


def _coordinator() -> dict[str, Any]:
    name = "world-realtime-coordinator"
    app = _base_application(
        name=name,
        app_name="World Realtime Coordinator",
        deploy_type="deployment",
        image="${WORLD_REALTIME_COORDINATOR_IMAGE_DIGEST}",
        replicas=2,
        service_account="wm-coordinator",
    )
    app.update(
        {
            "command": ["/bin/sh", "-ec"],
            "args": [
                "exec python -m sglang.multimodal_gen.runtime.entrypoints."
                "realtime_coordinator_server --host=0.0.0.0 --port=18081 "
                "--backend=dynamodb --table-name=${COORDINATOR_TABLE} "
                "--region=${AWS_REGION} --ttl-s=30 --worker-ttl-s=15 "
                "--wait-timeout-s=10 --candidate-limit=64 "
                "--denoiser-capacity-limit=4 --vae-capacity-limit=16"
            ],
            "env": [
                _env("COORDINATOR_TABLE", "world-model-realtime"),
                _env("AWS_REGION", "us-east-2"),
                _env("OTEL_SERVICE_NAME", name),
                _env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://adot-collector.monitoring:4317"),
                _env(
                    "OTEL_RESOURCE_ATTRIBUTES",
                    "service.namespace=world-model,deployment.environment=production",
                ),
            ],
            "resources": {
                "requests": {"cpu": "250m", "memory": "256Mi"},
                "limits": {"cpu": "1", "memory": "1Gi"},
            },
            "serviceSpec": {
                "headless": False,
                "ports": [{"name": "http", "port": 18081, "targetPort": 18081}],
            },
            "startupProbe": _http_probe("/healthz", 18081, period=2, failures=30),
            "readinessProbe": _http_probe("/healthz", 18081, period=5),
            "livenessProbe": _http_probe("/healthz", 18081, period=15),
            "topologySpreadConstraints": _spread(name),
            "podDisruptionBudget": {"minAvailable": 1},
            "networkPolicy": _network_policy(
                ingress_from=[
                    "world-realtime-gateway",
                    "minwm-denoiser",
                    "lingbot2-denoiser",
                    "minwm-vae",
                    "lingbot2-vae",
                ],
                egress_to=[],
                external=["dynamodb.us-east-2.amazonaws.com", "adot-collector.monitoring"],
            ),
        }
    )
    return app


def _vae(*, lingbot2: bool) -> dict[str, Any]:
    name = "lingbot2-vae" if lingbot2 else "minwm-vae"
    fingerprint = "taew2_1-d26151e7" if lingbot2 else "taew2_2-d053e216"
    checkpoint = "/opt/taehv/taew2_1.pth" if lingbot2 else "/opt/taehv/taew2_2.pth"
    app = _base_application(
        name=name,
        app_name="LingBot2 VAE" if lingbot2 else "minWM VAE",
        deploy_type="deployment",
        image="${WORLD_REALTIME_VAE_IMAGE_DIGEST}",
        replicas=1,
        service_account="wm-worker-discovery",
    )
    app["labels"]["loopit.me/gpu-role"] = "vae"
    app.update(
        {
            "strategy": {"type": "Recreate"},
            "args": [
                f"--checkpoint-path={checkpoint}",
                "--device=cuda",
                "--dtype=bfloat16",
                "--max-sessions=16",
                "--queue-depth-per-session=1",
                "--encoded-frames-per-batch=1",
                "--encode-workers=4",
                "--max-message-mb=64",
                "--host=0.0.0.0",
                "--port=18081",
            ],
            "env": [
                _env("WORKER_EPOCH_FILE", "/var/run/minwm-worker/epoch"),
                _env("OTEL_SERVICE_NAME", name),
                _env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://adot-collector.monitoring:4317"),
                _env(
                    "OTEL_RESOURCE_ATTRIBUTES",
                    f"service.namespace=world-model,worker.role=vae,model.name={'lingbot2' if lingbot2 else 'minwm'}",
                ),
            ],
            "resources": {
                "requests": {
                    "cpu": "6",
                    "memory": "24Gi",
                    "ephemeral-storage": "20Gi",
                    "nvidia.com/gpu": "1",
                },
                "limits": {
                    "cpu": "7",
                    "memory": "28Gi",
                    "ephemeral-storage": "40Gi",
                    "nvidia.com/gpu": "1",
                },
            },
            "initContainers": [
                {
                    "name": "vae-heartbeat",
                    "image": "${WORLD_REALTIME_VAE_IMAGE_DIGEST}",
                    "imagePullPolicy": "IfNotPresent",
                    "restartPolicy": "Always",
                    "command": ["/bin/sh", "-ec"],
                    "args": [
                        "exec python3 -m sglang.multimodal_gen.runtime.entrypoints."
                        "realtime_worker_heartbeat "
                        "--coordinator-url=http://world-realtime-coordinator:18081 "
                        "--health-url=http://127.0.0.1:18081/health "
                        "--state-url=http://127.0.0.1:18081/v1/realtime_worker/state "
                        "--worker-id=${POD_UID} --worker-epoch-file=/var/run/minwm-worker/epoch "
                        "--role=vae --endpoint=ws://${POD_IP}:18081/v1/realtime_vae/decode "
                        "--reservation-endpoint=http://${POD_IP}:18081/v1/realtime_worker "
                        f"--node-name=${{NODE_NAME}} --capacity=16 --model-revision=all --vae-fingerprint={fingerprint} --interval-s=5"
                    ],
                    "env": [
                        _env("SGLANG_LIGHTWEIGHT_RUNTIME", "1"),
                        _field_env("POD_UID", "metadata.uid"),
                        _field_env("POD_IP", "status.podIP"),
                        _field_env("NODE_NAME", "spec.nodeName"),
                    ],
                    "resources": {
                        "requests": {"cpu": "25m", "memory": "64Mi"},
                        "limits": {"cpu": "200m", "memory": "256Mi"},
                    },
                    "volumeMounts": [
                        {"name": "worker-epoch", "mountPath": "/var/run/minwm-worker"}
                    ],
                }
            ],
            "volumes": [{"name": "worker-epoch", "type": "emptyDir", "emptyDir": {}}],
            "volumeMounts": [
                {"name": "worker-epoch", "mountPath": "/var/run/minwm-worker"}
            ],
            "nodeSelector": {"loopit.me/gpu-pool": "l4"},
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                }
            ],
            "affinity": {
                "podAntiAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "topologyKey": "kubernetes.io/hostname",
                            "labelSelector": {
                                "matchLabels": {"loopit.me/gpu-role": "vae"}
                            },
                        }
                    ]
                }
            },
            "terminationGracePeriodSeconds": 60,
            "startupProbe": _http_probe("/health", 18081, period=10, failures=180),
            "readinessProbe": _http_probe("/health", 18081, period=5),
            "livenessProbe": _http_probe(
                "/health", 18081, period=30, initial_delay=30
            ),
            "podDisruptionBudget": {"maxUnavailable": 1},
            "networkPolicy": _network_policy(
                ingress_from=[
                    "minwm-denoiser" if not lingbot2 else "lingbot2-denoiser"
                ],
                egress_to=["world-realtime-coordinator"],
                external=["adot-collector.monitoring"],
            ),
        }
    )
    return app


def _model_stager(*, prefix: str, revision: str) -> dict[str, Any]:
    return {
        "name": "model-stager",
        "image": DENOISER_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": "/opt/sglang",
        "command": ["/bin/bash", "-lc"],
        "args": [
            "exec python3 benchmark/minwm_realtime_async_vae/download_model_artifact.py "
            "--bucket ${MODEL_BUCKET} --prefix ${MODEL_PREFIX} "
            "--destination /model-cache/model --lock-path /model-cache/.download.lock "
            "--region ${AWS_REGION} --expected-revision ${MODEL_REVISION} "
            "--concurrency 128 --part-size-mib 16"
        ],
        "env": [
            _env("AWS_REGION", "us-east-2"),
            _env("AWS_DEFAULT_REGION", "us-east-2"),
            _env("AWS_EC2_METADATA_DISABLED", "true"),
            _env("MODEL_BUCKET", "leap-world-model-serving-829115578968-us-east-2"),
            _env("MODEL_PREFIX", prefix),
            _env("MODEL_REVISION", revision),
        ],
        "resources": {
            "requests": {"cpu": "8", "memory": "32Gi", "ephemeral-storage": "32Gi"},
            "limits": {"cpu": "16", "memory": "64Gi", "ephemeral-storage": "64Gi"},
        },
        "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache"}],
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": False,
        },
    }


def _denoiser_heartbeat(*, lingbot2: bool) -> dict[str, Any]:
    name = "lingbot2" if lingbot2 else "minwm"
    capacity = 4 if lingbot2 else 1
    fingerprint = "taew2_1-d26151e7" if lingbot2 else "taew2_2-d053e216"
    revision = (
        "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
        if lingbot2
        else "wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2"
    )
    return {
        "name": "denoiser-heartbeat",
        "image": DENOISER_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "restartPolicy": "Always",
        "command": ["/bin/sh", "-ec"],
        "args": [
            "exec python3 -m sglang.multimodal_gen.runtime.entrypoints."
            "realtime_worker_heartbeat "
            "--coordinator-url=http://world-realtime-coordinator:18081 "
            "--health-url=http://127.0.0.1:30000/health "
            "--state-url=http://127.0.0.1:30000/v1/realtime_worker/state "
            "--worker-id=${POD_UID} --worker-epoch-file=/var/run/minwm-worker/epoch "
            "--role=denoiser --endpoint=ws://${POD_IP}:30000/v1/realtime_video/generate "
            "--reservation-endpoint=http://${POD_IP}:30000/v1/realtime_worker "
            f"--node-name=${{NODE_NAME}} --capacity={capacity} --model-revision={revision} "
            f"--vae-fingerprint={fingerprint} --interval-s=5"
        ],
        "env": [
            _env("SGLANG_LIGHTWEIGHT_RUNTIME", "1"),
            _env("MODEL_NAME", name),
            _field_env("POD_UID", "metadata.uid"),
            _field_env("POD_IP", "status.podIP"),
            _field_env("NODE_NAME", "spec.nodeName"),
        ],
        "resources": {
            "requests": {"cpu": "25m", "memory": "64Mi"},
            "limits": {"cpu": "200m", "memory": "256Mi"},
        },
        "volumeMounts": [
            {"name": "worker-epoch", "mountPath": "/var/run/minwm-worker"}
        ],
    }


def _denoiser(*, lingbot2: bool) -> dict[str, Any]:
    name = "lingbot2-denoiser" if lingbot2 else "minwm-denoiser"
    release = LINGBOT2_RELEASE if lingbot2 else MINWM_RELEASE
    expected_revision = LINGBOT2_REVISION if lingbot2 else "gs3200-ema-student-v1"
    app = _base_application(
        name=name,
        app_name="LingBot2 Denoiser" if lingbot2 else "minWM Denoiser",
        deploy_type="statefulset",
        image=DENOISER_IMAGE,
        replicas=1,
        service_account="wm-model-fetcher",
    )
    app["annotations"].update(
        {
            "cloudwatch.aws.amazon.com/auto-annotate-python": "false",
            "instrumentation.opentelemetry.io/inject-python": "false",
        }
    )
    app["labels"]["loopit.me/gpu-role"] = "denoiser"
    if lingbot2:
        launch = (
            "python3 -m sglang.multimodal_gen.runtime.launch_server "
            "--model-path ${MODEL_PATH} --model-id robbyant/lingbot-world-v2-14b-causal-fast-diffusers "
            "--pipeline-class-name LingBotWorldCausalDMDPipeline --attention-backend fa "
            "--attention-backend-config lingbot_causal_fa_num_splits=2 --performance-mode speed "
            "--num-gpus 4 --tp-size 1 --sp-degree 4 --ulysses-degree 4 --ring-degree 1 "
            "--enable-cfg-parallel false --dit-cpu-offload false --text-encoder-cpu-offload true "
            "--pin-cpu-memory true --enable-torch-compile false --warmup-mode server "
            "--batching-max-size 4 --batching-delay-ms 2 --realtime-max-sessions 4 "
            "--realtime-max-sessions-per-worker 4 --realtime-remote-vae-enabled "
            "--realtime-session-idle-timeout-s 60 --realtime-session-max-lifetime-s 90 "
            "--realtime-admission-wait-s 10 --enable-trace "
            "--otlp-traces-endpoint ${OTEL_EXPORTER_OTLP_ENDPOINT} --host 0.0.0.0 --port 30000"
        )
        resources = {
            "requests": {
                "cpu": "40",
                "memory": "600Gi",
                "ephemeral-storage": "128Gi",
                "nvidia.com/gpu": "4",
            },
            "limits": {
                "cpu": "88",
                "memory": "1200Gi",
                "ephemeral-storage": "256Gi",
                "nvidia.com/gpu": "4",
            },
        }
        cache_size = "256Gi"
        shm_size = "128Gi"
    else:
        launch = (
            "python3 -m sglang.multimodal_gen.runtime.launch_server "
            "--model-path ${MODEL_PATH} --pipeline-class-name MinWMCausalDMDPipeline "
            "--attention-backend fa --performance-mode speed --num-gpus 2 --tp-size 1 "
            "--sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-cuda-graph "
            "--enable-cfg-parallel false --enable-torch-compile false --warmup-mode off "
            "--batching-max-size 1 --batching-delay-ms 2 --realtime-max-sessions 1 "
            "--realtime-max-sessions-per-worker 1 --realtime-remote-vae-enabled "
            "--realtime-session-idle-timeout-s 60 --realtime-session-max-lifetime-s 90 "
            "--realtime-admission-wait-s 10 --realtime-causal-sink-size 8 "
            "--realtime-causal-kv-cache-num-frames 32 --enable-trace "
            "--otlp-traces-endpoint ${OTEL_EXPORTER_OTLP_ENDPOINT} --host 0.0.0.0 --port 30000"
        )
        resources = {
            "requests": {
                "cpu": "24",
                "memory": "300Gi",
                "ephemeral-storage": "32Gi",
                "nvidia.com/gpu": "2",
            },
            "limits": {
                "cpu": "40",
                "memory": "400Gi",
                "ephemeral-storage": "64Gi",
                "nvidia.com/gpu": "2",
            },
        }
        cache_size = "80Gi"
        shm_size = "64Gi"
    command = (
        "set -euo pipefail; test -f ${MODEL_PATH}/_READY; "
        "test -f ${MODEL_PATH}/model_index.json; test -d ${MODEL_PATH}/transformer; "
        f"{launch} & child=$!; "
        "terminate() { kill -TERM ${child} 2>/dev/null || true; wait ${child} || true; exit 143; }; "
        "trap terminate TERM INT; wait ${child}"
    )
    app.update(
        {
            "statefulSet": {
                "serviceName": f"{name}-headless",
                "podManagementPolicy": "Parallel",
                "updateStrategy": {"type": "OnDelete"},
            },
            "serviceSpec": {
                "headless": True,
                "publishNotReadyAddresses": True,
                "ports": [{"name": "api", "port": 30000, "targetPort": 30000}],
            },
            "command": ["/bin/bash", "-lc"],
            "args": [command],
            "env": [
                _env("PYTHONUNBUFFERED", "1"),
                _env("MODEL_PATH", "/model-cache/model"),
                _env("MODEL_REVISION", expected_revision),
                _env("WORKER_EPOCH_FILE", "/var/run/minwm-worker/epoch"),
                _env("OTEL_SERVICE_NAME", name),
                _env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://adot-collector.monitoring:4317"),
            ],
            "resources": resources,
            "initContainers": [
                _model_stager(prefix=release, revision=expected_revision),
                _denoiser_heartbeat(lingbot2=lingbot2),
            ],
            "volumes": [
                {
                    "name": "model-cache",
                    "type": "emptyDir",
                    "emptyDir": {"sizeLimit": cache_size},
                },
                {
                    "name": "shm",
                    "type": "emptyDir",
                    "emptyDir": {"medium": "Memory", "sizeLimit": shm_size},
                },
                {"name": "worker-epoch", "type": "emptyDir", "emptyDir": {}},
            ],
            "volumeMounts": [
                {"name": "model-cache", "mountPath": "/model-cache", "readOnly": True},
                {"name": "shm", "mountPath": "/dev/shm"},
                {"name": "worker-epoch", "mountPath": "/var/run/minwm-worker"},
            ],
            "nodeSelector": {"loopit.me/gpu-pool": "h100"},
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                }
            ],
            "terminationGracePeriodSeconds": 90,
            "startupProbe": _http_probe("/health", 30000, period=20, failures=270),
            "readinessProbe": _http_probe("/health", 30000, period=10),
            "livenessProbe": _http_probe(
                "/health", 30000, period=30, initial_delay=60
            ),
            "podDisruptionBudget": {"maxUnavailable": 1},
            "networkPolicy": _network_policy(
                ingress_from=["world-realtime-gateway"],
                egress_to=[
                    "world-realtime-coordinator",
                    "lingbot2-vae" if lingbot2 else "minwm-vae",
                ],
                external=[
                    "s3.us-east-2.amazonaws.com",
                    "adot-collector.monitoring",
                ],
            ),
        }
    )
    return app


def _gateway() -> dict[str, Any]:
    name = "world-realtime-gateway"
    app = _base_application(
        name=name,
        app_name="World Realtime Gateway",
        deploy_type="deployment",
        image="${WORLD_REALTIME_GATEWAY_IMAGE_DIGEST}",
        replicas=2,
        service_account="wm-gateway",
    )
    app.update(
        {
            "command": ["/bin/sh", "-ec"],
            "args": [
                "exec python -m sglang.multimodal_gen.runtime.entrypoints."
                "realtime_gateway_server --host=0.0.0.0 --port=18080 "
                "--coordinator-url=http://world-realtime-coordinator:18081 "
                "--model-revision=${MODEL_REVISION} "
                "--lingbot2-model-revision=${LINGBOT2_MODEL_REVISION} "
                "--vae-fingerprint=${VAE_FINGERPRINT} "
                "--lingbot2-vae-fingerprint=${LINGBOT2_VAE_FINGERPRINT} "
                "--internal-output-url=ws://${POD_IP}:18080/v1/internal/realtime_output "
                "--output-queue-depth=64 --output-enqueue-timeout-s=0 "
                "--output-drain-timeout-s=90 --lease-renew-interval-s=10 "
                "--release-grace-s=0.5 --trace-log-group=${TRACE_LOG_GROUP}"
            ],
            "env": [
                _field_env("POD_IP", "status.podIP"),
                _env("MODEL_REVISION", "wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2"),
                _env("VAE_FINGERPRINT", "taew2_2-d053e216"),
                _env(
                    "LINGBOT2_MODEL_REVISION",
                    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                ),
                _env("LINGBOT2_VAE_FINGERPRINT", "taew2_1-d26151e7"),
                _env("TRACE_LOG_GROUP", "/aws/eks/world-model/realtime-traces"),
                _env("AWS_REGION", "us-east-2"),
                _env("AWS_DEFAULT_REGION", "us-east-2"),
                _env("OTEL_SERVICE_NAME", name),
                _env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://adot-collector.monitoring:4317"),
            ],
            "resources": {
                "requests": {"cpu": "500m", "memory": "512Mi"},
                "limits": {"cpu": "2", "memory": "2Gi"},
            },
            "serviceSpec": {
                "headless": False,
                "ports": [{"name": "http", "port": 18080, "targetPort": 18080}],
            },
            "startupProbe": _http_probe("/healthz", 18080, period=2, failures=30),
            "readinessProbe": _http_probe("/readyz", 18080, period=5),
            "livenessProbe": _http_probe("/healthz", 18080, period=15),
            "topologySpreadConstraints": _spread(name),
            "podDisruptionBudget": {"minAvailable": 1},
            "networkPolicy": _network_policy(
                ingress_from=["world-studio-webui", "istio-ingressgateway"],
                egress_to=[
                    "world-realtime-coordinator",
                    "minwm-denoiser",
                    "lingbot2-denoiser",
                ],
                external=[
                    "logs.us-east-2.amazonaws.com",
                    "adot-collector.monitoring",
                ],
            ),
        }
    )
    return app


def _webui() -> dict[str, Any]:
    name = "world-studio-webui"
    app = _base_application(
        name=name,
        app_name="World Studio WebUI",
        deploy_type="deployment",
        image=WEBUI_IMAGE,
        replicas=2,
        service_account="wm-webui",
    )
    app.update(
        {
            "command": ["python", "server.py"],
            "env": [
                _secret_env("HAPPYOYSTER_API_KEY", "happyoyster-api-key"),
                _env(
                    "HAPPYOYSTER_API_BASE_URL",
                    "https://llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com/api/v2/apps/happyoyster-1.0/",
                ),
                _env(
                    "HAPPYOYSTER_TOKEN_BASE_URL",
                    "https://llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com",
                ),
                _env("HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL", "https://seedleap-world.loopit.me"),
                _env("REALTIME_UPSTREAM_HTTP", "http://world-realtime-gateway:18080"),
                _env("REALTIME_UPSTREAM_WS", "ws://world-realtime-gateway:18080"),
                _env(
                    "MINWM_UPSTREAM_HTTP",
                    "http://world-realtime-gateway:18080/backends/minwm",
                ),
                _env(
                    "MINWM_UPSTREAM_WS",
                    "ws://world-realtime-gateway:18080/backends/minwm",
                ),
                _env(
                    "LINGBOT2_UPSTREAM_HTTP",
                    "http://world-realtime-gateway:18080/backends/lingbot2",
                ),
                _env(
                    "LINGBOT2_UPSTREAM_WS",
                    "ws://world-realtime-gateway:18080/backends/lingbot2",
                ),
                _env(
                    "REALTIME_UI_CONFIG_JSON",
                    '{"generationModes":["i2v","t2v"],"defaultGenerationMode":"i2v",'
                    '"t2vFrameStep":4,"t2vDefaultNumFrames":121,"sessionMaxLifetimeSeconds":90,'
                    '"singleExperience":true,"singleExperienceUserIds":{"minwm":"showcase:world-model:zing",'
                    '"lingbot2":"showcase:world-model:lingbot2"},"smoothCatchupRateMax":1.1,'
                    '"dualModels":{"minwm":{"label":"Zing"},"lingbot2":{"label":"LingBot2",'
                    '"targetFps":16,"sinkSize":9,"windowFrames":18}}}',
                ),
                _env(
                    "VIDEO_PROMPT_REWRITE_CREDENTIALS",
                    "/run/secrets/realtime-webui/prompt-rewriter-vertex.json",
                ),
                _env(
                    "CREATE_WORLD_IMAGE_CONFIG",
                    "/run/secrets/realtime-webui/world-image-model-config.json",
                ),
            ],
            "resources": {
                "requests": {"cpu": "250m", "memory": "384Mi"},
                "limits": {"cpu": "1", "memory": "1Gi"},
            },
            "serviceSpec": {
                "headless": False,
                "ports": [{"name": "http", "port": 18080, "targetPort": 18080}],
            },
            "volumes": [
                {
                    "name": "runtime-secret",
                    "type": "secret",
                    "secret": {
                        "secretName": "world-studio-runtime",
                        "items": [
                            {
                                "key": "prompt-rewriter-vertex.json",
                                "path": "prompt-rewriter-vertex.json",
                            },
                            {
                                "key": "world-image-model-config.json",
                                "path": "world-image-model-config.json",
                            },
                        ],
                    },
                },
                {
                    "name": "generated-images",
                    "type": "emptyDir",
                    "emptyDir": {"sizeLimit": "1Gi"},
                },
                {
                    "name": "tmp",
                    "type": "emptyDir",
                    "emptyDir": {"sizeLimit": "256Mi"},
                },
            ],
            "volumeMounts": [
                {
                    "name": "runtime-secret",
                    "mountPath": "/run/secrets/realtime-webui",
                    "readOnly": True,
                },
                {
                    "name": "generated-images",
                    "mountPath": "/opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui_generated",
                },
                {"name": "tmp", "mountPath": "/tmp"},
            ],
            "startupProbe": _http_probe("/runtime-config.js", 18080, period=2, failures=30),
            "readinessProbe": _http_probe("/runtime-config.js", 18080, period=5),
            "livenessProbe": _http_probe("/", 18080, period=15),
            "topologySpreadConstraints": _spread(name),
            "podDisruptionBudget": {"minAvailable": 1},
            "networkPolicy": _network_policy(
                ingress_from=["istio-ingressgateway"],
                egress_to=["world-realtime-gateway"],
                external=[
                    "llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com",
                    "aiplatform.googleapis.com",
                ],
            ),
        }
    )
    return app


def artifact_publisher_task() -> dict[str, Any]:
    name = "world-model-artifact-publisher"
    return {
        "schemaVersion": 1,
        "kind": "task",
        "name": name,
        "appName": "World Model Artifact Publisher",
        "businessLineId": BUSINESS_LINE,
        "clusterName": "world-model",
        "namespace": NAMESPACE,
        "appGroup": APP_GROUP,
        "lane": "default",
        "deployType": "job",
        "image": "${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}",
        "imagePullPolicy": "IfNotPresent",
        "serviceAccountName": "wm-artifact-publisher",
        "labels": {
            "loopit.me/business-line": BUSINESS_LINE,
            "loopit.me/managed-by": "platform",
            "loopit.me/service-id": name,
            "loopit.me/lane": "default",
            "app.kubernetes.io/part-of": APP_GROUP,
        },
        "annotations": {"logs.loopit.me/enabled": "true"},
        "command": [
            "python3",
            "/opt/sglang/benchmark/minwm_realtime_async_vae/copy_model_release.py",
        ],
        "args": [
            "--release",
            "/opt/sglang/benchmark/minwm_realtime_async_vae/model_releases/lingbot2/"
            f"{LINGBOT2_REVISION}/release-spec.template.json",
            "--offline-plan",
        ],
        "requiredExecutionInputs": [
            "reviewed release spec generated by build_model_release_spec.py",
            "--execute",
            "--confirm-release-id=<reviewed release_id>",
        ],
        "resources": {
            "requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "16Gi"},
            "limits": {"cpu": "8", "memory": "16Gi", "ephemeral-storage": "32Gi"},
        },
        "nodeSelector": {},
        "tolerations": [],
        "volumes": [],
        "volumeMounts": [],
        "activeDeadlineSeconds": 10800,
        "backoffLimit": 1,
        "networkPolicy": _network_policy(
            ingress_from=[],
            egress_to=[],
            external=["s3.us-west-2.amazonaws.com", "s3.us-east-2.amazonaws.com"],
        ),
    }


def applications() -> list[dict[str, Any]]:
    """Return applications in the frozen dependency order D-03."""
    return [
        _coordinator(),
        _vae(lingbot2=False),
        _vae(lingbot2=True),
        _denoiser(lingbot2=False),
        _denoiser(lingbot2=True),
        _gateway(),
        _webui(),
    ]


def all_platform_configs() -> list[dict[str, Any]]:
    return applications() + [artifact_publisher_task()]


def resolve_image_inputs(
    configs: list[dict[str, Any]], image_inputs: dict[str, str]
) -> list[dict[str, Any]]:
    resolved = copy.deepcopy(configs)

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str) and value in IMAGE_PLACEHOLDERS:
            if value not in image_inputs:
                raise ValueError(f"missing image input for {value}")
            image = image_inputs[value]
            if not IMAGE_RE.fullmatch(image):
                raise ValueError(f"image input for {value} is not digest-pinned")
            return image
        return value

    return replace(resolved)


def validate_configs(
    configs: list[dict[str, Any]], *, require_resolved_images: bool
) -> None:
    names = [config["name"] for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("platform service names must be unique")
    if names != [
        "world-realtime-coordinator",
        "minwm-vae",
        "lingbot2-vae",
        "minwm-denoiser",
        "lingbot2-denoiser",
        "world-realtime-gateway",
        "world-studio-webui",
        "world-model-artifact-publisher",
    ]:
        raise ValueError("platform configs are not in the frozen release order")
    for config in configs:
        if config["businessLineId"] != BUSINESS_LINE or config["namespace"] != NAMESPACE:
            raise ValueError("platform config escaped the world-model ownership boundary")
        if "karpenter.sh/capacity-type" in config.get("nodeSelector", {}):
            raise ValueError(f"{config['name']} pins a Kubernetes capacity type")
        images = [config["image"]] + [
            container["image"] for container in config.get("initContainers", [])
        ]
        for image in images:
            if image in IMAGE_PLACEHOLDERS and not require_resolved_images:
                continue
            if not IMAGE_RE.fullmatch(image):
                raise ValueError(f"{config['name']} has an image not pinned by digest")
