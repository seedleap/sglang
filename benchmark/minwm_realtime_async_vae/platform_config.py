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
PRODUCTION_BASE_GIT_SHA = "d8019542103c83047997cf6dc2e7014cba8565e3"
PRODUCTION_BASE_BRANCH = "codex/minwm-lingbot2-dual-webui-opt-20260812"
FROZEN_GIT_SHA = "5ded4b5de2702d063cb9421d5c7049c0570c013b"
FROZEN_BRANCH = "codex/wm09-sglang-platform-config-20260814"
GIT_REPO = "git@github.com:seedleap/sglang.git"
IMAGE_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
CALLBACK_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CREATE_SERVICE_FIELDS = {
    "name",
    "appName",
    "appGroup",
    "description",
    "language",
    "deployType",
    "gitRepo",
    "jenkinsJob",
    "owners",
    "developers",
    "owner",
    "pipeline",
    "command",
    "args",
    "envVars",
    "env",
    "initContainers",
    "labels",
    "annotations",
    "networkPolicy",
    "configMaps",
    "affinityRules",
    "tolerations",
    "resources",
    "startupProbe",
    "readinessProbe",
    "livenessProbe",
    "nodeSelector",
    "affinity",
    "topologySpreadConstraints",
    "terminationGracePeriodSeconds",
    "podDisruptionBudget",
    "statefulSet",
    "clusterId",
    "namespace",
    "image",
    "serviceAccountName",
    "envFromSecrets",
    "runtimeJobNamespaces",
    "taskExecutionPolicy",
    "cronJob",
    "containerPort",
    "servicePort",
    "extraPorts",
    "volumes",
    "volumeMounts",
    "serviceSpec",
    "replicas",
    "businessLineId",
}
PLATFORM_OWNERSHIP_LABELS = {
    "business_line_id",
    "cluster_id",
    "namespace",
    "service_id",
}
IMAGE_INPUT_SPECS = {
    "world-realtime-coordinator": {
        "placeholder": "${WORLD_REALTIME_COORDINATOR_IMAGE_DIGEST}",
        "tagPrefix": "world-realtime-coordinator",
        "jenkinsJob": "loopit/world-model/world-realtime-coordinator",
    },
    "minwm-vae": {
        "placeholder": "${MINWM_VAE_IMAGE_DIGEST}",
        "tagPrefix": "minwm-vae",
        "jenkinsJob": "loopit/world-model/minwm-vae",
    },
    "lingbot2-vae": {
        "placeholder": "${LINGBOT2_VAE_IMAGE_DIGEST}",
        "tagPrefix": "lingbot2-vae",
        "jenkinsJob": "loopit/world-model/lingbot2-vae",
    },
    "minwm-denoiser": {
        "placeholder": "${MINWM_DENOISER_IMAGE_DIGEST}",
        "tagPrefix": "minwm-denoiser",
        "jenkinsJob": "loopit/world-model/minwm-denoiser",
    },
    "lingbot2-denoiser": {
        "placeholder": "${LINGBOT2_DENOISER_IMAGE_DIGEST}",
        "tagPrefix": "lingbot2-denoiser",
        "jenkinsJob": "loopit/world-model/lingbot2-denoiser",
    },
    "world-realtime-gateway": {
        "placeholder": "${WORLD_REALTIME_GATEWAY_IMAGE_DIGEST}",
        "tagPrefix": "world-realtime-gateway",
        "jenkinsJob": "loopit/world-model/world-realtime-gateway",
    },
    "world-studio-webui": {
        "placeholder": "${WORLD_STUDIO_WEBUI_IMAGE_DIGEST}",
        "tagPrefix": "world-studio-webui",
        "jenkinsJob": "loopit/world-model/world-studio-webui",
    },
    "world-model-artifact-publisher": {
        "placeholder": "${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}",
        "tagPrefix": "model-artifact-publish",
        "jenkinsJob": "loopit/world-model/model-artifact-publish",
    },
}
IMAGE_PLACEHOLDERS = {spec["placeholder"] for spec in IMAGE_INPUT_SPECS.values()}
NETWORK_REQUIRED_INPUTS = {
    "world-realtime-coordinator": [
        ("egress", "dynamodb-us-east-2", "TCP", 443),
        ("egress", "adot-collector-monitoring", "TCP", 4317),
    ],
    "minwm-vae": [("egress", "adot-collector-monitoring", "TCP", 4317)],
    "lingbot2-vae": [("egress", "adot-collector-monitoring", "TCP", 4317)],
    "minwm-denoiser": [
        ("egress", "s3-us-east-2", "TCP", 443),
        ("egress", "adot-collector-monitoring", "TCP", 4317),
    ],
    "lingbot2-denoiser": [
        ("egress", "s3-us-east-2", "TCP", 443),
        ("egress", "adot-collector-monitoring", "TCP", 4317),
    ],
    "world-realtime-gateway": [
        ("ingress", "istio-ingressgateway", "TCP", 18080),
        ("egress", "cloudwatch-logs-us-east-2", "TCP", 443),
        ("egress", "adot-collector-monitoring", "TCP", 4317),
    ],
    "world-studio-webui": [
        ("ingress", "istio-ingressgateway", "TCP", 18080),
        ("egress", "happyoyster-api", "TCP", 443),
        ("egress", "google-vertex-ai", "TCP", 443),
    ],
    "world-model-artifact-publisher": [
        ("egress", "s3-us-west-2-source", "TCP", 443),
        ("egress", "s3-us-east-2-destination", "TCP", 443),
    ],
}

MODEL_BUCKET = "leap-world-model-serving-829115578968-us-east-2"
MINWM_MODEL_NAME = "minwm-async-denoiser-0"
MINWM_MODEL_VERSION = "wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2-gs3200-ema-student-v1"
MINWM_SOURCE_REVISION = "gs3200-ema-student-v1"
MINWM_RELEASE_ID = "20260810T042157Z-c302d572"
LINGBOT2_REVISION = "59cccf49f2d2dd27418ae7a04b82b10868d455c2"
LINGBOT2_MODEL_NAME = "lingbot2-denoiser"
LINGBOT2_MODEL_VERSION = "robbyant-lingbot-world-v2-14b-causal-fast-diffusers"
LINGBOT2_RELEASE_ID = "20260814T054118Z-e0650875"
S3_EXACT_BUNDLE_SHA256 = (
    "944d828d3eb4c3db52f761847046c2910b8243a23579553fe6bee2defa8b29c7"
)


def _image_placeholder(service_name: str) -> str:
    return IMAGE_INPUT_SPECS[service_name]["placeholder"]


def required_inputs_document() -> dict[str, Any]:
    image_callbacks = []
    for service_name, spec in IMAGE_INPUT_SPECS.items():
        image_callbacks.append(
            {
                "serviceName": service_name,
                "state": "missing",
                "requiredStatus": "success",
                "requiredBranch": FROZEN_BRANCH,
                "requiredAuditTag": (
                    f"{REGISTRY}:{spec['tagPrefix']}-{FROZEN_GIT_SHA}"
                ),
                "requiredImageDigestPattern": "sha256:<64-lowercase-hex>",
                "requiredJenkinsJob": spec["jenkinsJob"],
                "requiredOperator": "jenkins",
                "requiredBuildEvidence": ["buildId", "buildUrl"],
                "deploymentImageSource": "callback.imageDigest",
            }
        )
    network_peers = []
    for service_name, requirements in NETWORK_REQUIRED_INPUTS.items():
        for direction, destination, protocol, port in requirements:
            network_peers.append(
                {
                    "serviceName": service_name,
                    "direction": direction,
                    "destinationId": destination,
                    "protocol": protocol,
                    "port": port,
                    "state": "missing",
                    "requiredTypedPeer": (
                        "podSelector+namespaceSelector or approved ipBlock"
                    ),
                }
            )
    return {
        "schemaVersion": 1,
        "executionReady": False,
        "hardGate": "all required inputs must be resolved before deployment render",
        "frozenSource": {
            "branch": FROZEN_BRANCH,
            "gitSha": FROZEN_GIT_SHA,
        },
        "productionBaseline": {
            "branch": PRODUCTION_BASE_BRANCH,
            "gitSha": PRODUCTION_BASE_GIT_SHA,
        },
        "requiredInputs": {
            "wm08BuildContract": {
                "state": "missing",
                "currentFrozenBranch": PRODUCTION_BASE_BRANCH,
                "currentFrozenGitSha": PRODUCTION_BASE_GIT_SHA,
                "requiredBranch": FROZEN_BRANCH,
                "requiredGitSha": FROZEN_GIT_SHA,
                "requiredTagFormat": "<service-tag-prefix>-<full-40-character-git-sha>",
                "requiredCallbackDigestField": "imageDigest",
                "runtimeImageFormat": "<ecr-repository>@<callback.imageDigest>",
                "reason": (
                    "the production baseline does not contain the WM-09 release "
                    "spec, CRT downloader compatibility, or publisher verifier"
                ),
            },
            "publisherExecutionBundle": {
                "state": "missing",
                "requiredApprovalFields": [
                    "releaseSpecSha256",
                    "publisherBundleSha256",
                    "executionBundleSha256",
                ],
                "requiredExecuteArgs": [
                    "--confirm-release-id",
                    "--confirm-release-spec-sha256",
                    "--confirm-execution-bundle-sha256",
                ],
                "reason": (
                    "the checked-in 26-object LingBot2 release spec predates the "
                    "23-object runtime allowlist and complete publisher bundle contract"
                ),
            },
            "clusterRegistration": {
                "state": "missing",
                "clusterName": "world-model",
                "requiredField": "clusterId",
            },
            "imageBuildCallbacks": image_callbacks,
            "networkPolicyPeers": network_peers,
        },
        "resolvedInputs": {
            "lingbot2Release": {
                "releaseId": "20260814T054118Z-e0650875",
                "manifestSha256": (
                    "6a790fd04daecfa66bede8cc71f18ed96dda617bc74cecda51e5ce72c4cf19af"
                ),
                "objectCount": 23,
                "bytes": 86068529220,
                "controlRevisionField": "revision",
            },
            "s3MigrationBundle": {
                "sha256": S3_EXACT_BUNDLE_SHA256,
                "payloadObjectCount": 41,
                "controlObjectCount": 6,
                "totalObjectCount": 47,
                "payloadBytes": 110277729372,
                "controlBytes": 8656,
                "totalBytes": 110277738028,
            }
        },
    }


def _env(name: str, value: str) -> dict[str, Any]:
    return {"name": name, "value": value}


def _field_env(name: str, field_path: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": field_path}}}


def _secret_env(name: str, key: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"secretKeyRef": {"name": "world-studio-runtime", "key": key}},
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
        "enabled": True,
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
            "labelSelector": {"matchLabels": {"app.kubernetes.io/name": name}},
        }
    ]


SERVICE_PORTS = {
    "world-realtime-coordinator": 18081,
    "minwm-vae": 18081,
    "lingbot2-vae": 18081,
    "minwm-denoiser": 30000,
    "lingbot2-denoiser": 30000,
    "world-realtime-gateway": 18080,
    "world-studio-webui": 18080,
}


def _service_peer(name: str) -> dict[str, Any]:
    return {"podSelector": {"matchLabels": {"app.kubernetes.io/name": name}}}


def _dns_rules() -> list[dict[str, Any]]:
    peer = {
        "namespaceSelector": {
            "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
        }
    }
    return [
        {"to": [peer], "ports": [{"protocol": protocol, "port": 53}]}
        for protocol in ("UDP", "TCP")
    ]


def _network_policy(
    *,
    service_name: str,
    ingress_from: list[str],
    egress_to: list[str],
    external: list[str] | None = None,
) -> dict[str, Any]:
    del external
    ingress_peers = [
        _service_peer(name) for name in ingress_from if name in SERVICE_PORTS
    ]
    ingress = []
    if ingress_peers:
        ingress.append(
            {
                "from": ingress_peers,
                "ports": [{"protocol": "TCP", "port": SERVICE_PORTS[service_name]}],
            }
        )
    egress = _dns_rules()
    for name in egress_to:
        if name in SERVICE_PORTS:
            egress.append(
                {
                    "to": [_service_peer(name)],
                    "ports": [{"protocol": "TCP", "port": SERVICE_PORTS[name]}],
                }
            )
    return {
        "enabled": True,
        "defaultDeny": True,
        "ingress": ingress,
        "egress": egress,
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
        "name": name,
        "appName": app_name,
        "businessLineId": BUSINESS_LINE,
        "clusterId": "${WORLD_MODEL_CLUSTER_ID}",
        "namespace": NAMESPACE,
        "appGroup": APP_GROUP,
        "deployType": deploy_type,
        "gitRepo": GIT_REPO,
        "jenkinsJob": IMAGE_INPUT_SPECS[name]["jenkinsJob"],
        "replicas": replicas,
        "image": image,
        "serviceAccountName": service_account,
        "labels": {
            "app.kubernetes.io/name": name,
            "world-model.loopit.me/application-group": APP_GROUP,
            "world-model.loopit.me/service": name,
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
        image=_image_placeholder(name),
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
                _env(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://adot-collector.monitoring:4317",
                ),
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
                service_name=name,
                ingress_from=[
                    "world-realtime-gateway",
                    "minwm-denoiser",
                    "lingbot2-denoiser",
                    "minwm-vae",
                    "lingbot2-vae",
                ],
                egress_to=[],
                external=[
                    "dynamodb.us-east-2.amazonaws.com",
                    "adot-collector.monitoring",
                ],
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
        image=_image_placeholder(name),
        replicas=1,
        service_account="wm-worker-discovery",
    )
    app["labels"]["loopit.me/gpu-role"] = "vae"
    app.update(
        {
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
                _env(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://adot-collector.monitoring:4317",
                ),
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
                    "image": _image_placeholder(name),
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
            "livenessProbe": _http_probe("/health", 18081, period=30, initial_delay=30),
            "podDisruptionBudget": {"maxUnavailable": 1},
            "networkPolicy": _network_policy(
                service_name=name,
                ingress_from=[
                    "minwm-denoiser" if not lingbot2 else "lingbot2-denoiser"
                ],
                egress_to=["world-realtime-coordinator"],
                external=["adot-collector.monitoring"],
            ),
        }
    )
    return app


def _model_stager(
    *,
    model_name: str,
    model_version: str,
    release_id: str,
    source_revision: str,
    image: str,
) -> dict[str, Any]:
    return {
        "name": "model-stager",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": "/opt/sglang",
        "command": ["/bin/bash", "-lc"],
        "args": [
            "exec python3 benchmark/minwm_realtime_async_vae/download_model_artifact.py "
            '--bucket "${MODEL_BUCKET}" --model-s3-uri "${MODEL_S3_URI:-}" '
            '--model-name "${MODEL_NAME}" --model-version "${MODEL_VERSION}" '
            '--model-release-id "${MODEL_RELEASE_ID}" '
            "--destination /model-cache/model --lock-path /model-cache/.download.lock "
            '--region "${AWS_REGION}" --expected-revision "${MODEL_SOURCE_REVISION}" '
            "--concurrency 128 --part-size-mib 16"
        ],
        "env": [
            _env("AWS_REGION", "us-east-2"),
            _env("AWS_DEFAULT_REGION", "us-east-2"),
            _env("AWS_EC2_METADATA_DISABLED", "true"),
            _env("MODEL_BUCKET", MODEL_BUCKET),
            _env("MODEL_NAME", model_name),
            _env("MODEL_VERSION", model_version),
            _env("MODEL_RELEASE_ID", release_id),
            _env("MODEL_SOURCE_REVISION", source_revision),
            # Non-empty values override the derived serving path for an explicit
            # immutable rollback while retaining metadata identity validation.
            _env("MODEL_S3_URI", ""),
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
    service_name = "lingbot2-denoiser" if lingbot2 else "minwm-denoiser"
    capacity = 4 if lingbot2 else 1
    fingerprint = "taew2_1-d26151e7" if lingbot2 else "taew2_2-d053e216"
    revision = (
        "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
        if lingbot2
        else "wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2"
    )
    return {
        "name": "denoiser-heartbeat",
        "image": _image_placeholder(service_name),
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
    image = _image_placeholder(name)
    model_name = LINGBOT2_MODEL_NAME if lingbot2 else MINWM_MODEL_NAME
    model_version = LINGBOT2_MODEL_VERSION if lingbot2 else MINWM_MODEL_VERSION
    release_id = LINGBOT2_RELEASE_ID if lingbot2 else MINWM_RELEASE_ID
    expected_revision = LINGBOT2_REVISION if lingbot2 else MINWM_SOURCE_REVISION
    app = _base_application(
        name=name,
        app_name="LingBot2 Denoiser" if lingbot2 else "minWM Denoiser",
        deploy_type="statefulset",
        image=image,
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
                _env(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://adot-collector.monitoring:4317",
                ),
            ],
            "resources": resources,
            "initContainers": [
                _model_stager(
                    model_name=model_name,
                    model_version=model_version,
                    release_id=release_id,
                    source_revision=expected_revision,
                    image=image,
                ),
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
            "livenessProbe": _http_probe("/health", 30000, period=30, initial_delay=60),
            "podDisruptionBudget": {"maxUnavailable": 1},
            "networkPolicy": _network_policy(
                service_name=name,
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
        image=_image_placeholder(name),
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
                _env(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "http://adot-collector.monitoring:4317",
                ),
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
                service_name=name,
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
        image=_image_placeholder(name),
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
                _env(
                    "HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL",
                    "https://seedleap-world.loopit.me",
                ),
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
            "startupProbe": _http_probe(
                "/runtime-config.js", 18080, period=2, failures=30
            ),
            "readinessProbe": _http_probe("/runtime-config.js", 18080, period=5),
            "livenessProbe": _http_probe("/", 18080, period=15),
            "topologySpreadConstraints": _spread(name),
            "podDisruptionBudget": {"minAvailable": 1},
            "networkPolicy": _network_policy(
                service_name=name,
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
    command = [
        "python3",
        "/opt/sglang/benchmark/minwm_realtime_async_vae/copy_model_release.py",
    ]
    release_path = (
        "/opt/sglang/benchmark/minwm_realtime_async_vae/model_releases/lingbot2/"
        f"{LINGBOT2_REVISION}/release-spec.json"
    )
    return {
        "name": name,
        "appName": "World Model Artifact Publisher",
        "businessLineId": BUSINESS_LINE,
        "clusterId": "${WORLD_MODEL_CLUSTER_ID}",
        "namespace": NAMESPACE,
        "appGroup": APP_GROUP,
        "deployType": "job",
        "gitRepo": GIT_REPO,
        "jenkinsJob": IMAGE_INPUT_SPECS[name]["jenkinsJob"],
        "image": _image_placeholder(name),
        "serviceAccountName": "wm-artifact-publisher",
        "labels": {
            "app.kubernetes.io/name": name,
            "world-model.loopit.me/application-group": APP_GROUP,
            "world-model.loopit.me/service": name,
        },
        "annotations": {
            "logs.loopit.me/enabled": "true",
            "logs.loopit.me/bucket": BUSINESS_LINE,
        },
        "command": command,
        "args": [
            "--release",
            release_path,
            "--offline-plan",
        ],
        "taskExecutionPolicy": {
            "requireDigestImage": True,
            "commandRules": [
                {
                    "command": command,
                    "argsExact": ["--release", release_path, "--offline-plan"],
                }
            ],
        },
        "resources": {
            "requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "16Gi"},
            "limits": {"cpu": "8", "memory": "16Gi", "ephemeral-storage": "32Gi"},
        },
        "nodeSelector": {},
        "tolerations": [],
        "volumes": [],
        "volumeMounts": [],
        "networkPolicy": _network_policy(
            service_name=name,
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
    configs: list[dict[str, Any]], image_inputs: dict[str, Any]
) -> list[dict[str, Any]]:
    if image_inputs.get("schemaVersion") != 1:
        raise ValueError("image inputs must use schemaVersion 1")
    if image_inputs.get("frozenGitSha") != FROZEN_GIT_SHA:
        raise ValueError("image inputs do not match the frozen full Git SHA")
    callbacks = image_inputs.get("callbacks")
    if not isinstance(callbacks, dict) or set(callbacks) != set(IMAGE_INPUT_SPECS):
        raise ValueError("image inputs must contain exactly one callback per service")

    resolved_images: dict[str, str] = {}
    for service_name, spec in IMAGE_INPUT_SPECS.items():
        callback = callbacks[service_name]
        if not isinstance(callback, dict):
            raise ValueError(f"image callback for {service_name} must be an object")
        expected_tag = f"{REGISTRY}:{spec['tagPrefix']}-{FROZEN_GIT_SHA}"
        if callback.get("serviceName") != service_name:
            raise ValueError(f"image callback serviceName mismatch for {service_name}")
        if str(callback.get("status", "")).lower() != "success":
            raise ValueError(f"image callback did not succeed for {service_name}")
        if callback.get("branch") != FROZEN_BRANCH:
            raise ValueError(f"image callback branch mismatch for {service_name}")
        if callback.get("jenkinsJob") != spec["jenkinsJob"]:
            raise ValueError(f"image callback Jenkins Job mismatch for {service_name}")
        if callback.get("operator") != "jenkins":
            raise ValueError(f"image callback operator mismatch for {service_name}")
        if not isinstance(callback.get("buildId"), str) or not callback["buildId"]:
            raise ValueError(f"image callback buildId is missing for {service_name}")
        if not isinstance(callback.get("buildUrl"), str) or not callback["buildUrl"]:
            raise ValueError(f"image callback buildUrl is missing for {service_name}")
        if callback.get("image") != expected_tag:
            raise ValueError(
                f"image callback tag for {service_name} is not the full-SHA audit tag"
            )
        digest = callback.get("imageDigest")
        if not isinstance(digest, str) or not CALLBACK_DIGEST_RE.fullmatch(digest):
            raise ValueError(f"image callback digest is invalid for {service_name}")
        repository = expected_tag.rsplit(":", 1)[0]
        resolved_images[spec["placeholder"]] = f"{repository}@{digest}"

    resolved = copy.deepcopy(configs)

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str) and value in IMAGE_PLACEHOLDERS:
            if value not in resolved_images:
                raise ValueError(f"missing image input for {value}")
            image = resolved_images[value]
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
        unknown_fields = set(config) - CREATE_SERVICE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"{config['name']} contains non-typed service fields: "
                f"{sorted(unknown_fields)}"
            )
        if (
            config["businessLineId"] != BUSINESS_LINE
            or config["namespace"] != NAMESPACE
        ):
            raise ValueError(
                "platform config escaped the world-model ownership boundary"
            )
        if PLATFORM_OWNERSHIP_LABELS.intersection(config.get("labels", {})):
            raise ValueError(f"{config['name']} overrides platform ownership labels")
        policy = config.get("networkPolicy")
        if not isinstance(policy, dict) or set(policy) != {
            "enabled",
            "defaultDeny",
            "ingress",
            "egress",
        }:
            raise ValueError(f"{config['name']} has a non-typed network policy")
        if not policy["enabled"] or not policy["defaultDeny"]:
            raise ValueError(f"{config['name']} must use typed default-deny networking")
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
