#!/usr/bin/env python3
"""Policy checks for the disposable MinWM async-VAE benchmark topology."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_MANIFESTS = (
    "namespace.yaml",
    "west-s3-volume.yaml",
    "east2-model-serving-s3-volume.yaml",
    "observability.yaml",
    "8gpu-nodeclass.yaml",
    "coordinator.yaml",
    "gateway.yaml",
    "worker-discovery.yaml",
    "autoscaling.yaml",
    "h100-denoiser.yaml",
    "h200-denoiser-capacity.yaml",
    "lingbot2-h100-denoiser.yaml",
    "l4-vae.yaml",
    "network-policy.yaml",
    "gpu-replica-safety.yaml",
    "gateway-service.yaml",
)


def load_documents(paths: tuple[str, ...] = BASE_MANIFESTS) -> list[dict]:
    documents: list[dict] = []
    for relative_path in paths:
        path = ROOT / relative_path
        with path.open() as stream:
            documents.extend(
                document
                for document in yaml.safe_load_all(stream)
                if isinstance(document, dict)
            )
    return documents


def find(documents: list[dict], kind: str, name: str) -> dict:
    for document in documents:
        if (
            document.get("kind") == kind
            and document.get("metadata", {}).get("name") == name
        ):
            return document
    raise AssertionError(f"missing {kind}/{name}")


def requirement_values(nodepool: dict, key: str) -> list[str]:
    requirements = nodepool["spec"]["template"]["spec"]["requirements"]
    for requirement in requirements:
        if requirement.get("key") == key:
            return list(requirement.get("values") or [])
    raise AssertionError(f"missing NodePool requirement {key}")


def validate(documents: list[dict]) -> None:
    denoiser = find(documents, "NodePool", "minwm-async-denoiser-h100")
    denoiser_8x = find(documents, "NodePool", "minwm-async-denoiser-h100-8x")
    denoiser_h200_8x = find(documents, "NodePool", "minwm-async-denoiser-h200-8x")
    vae = find(documents, "NodePool", "minwm-async-vae-l4")
    vae_spot = find(documents, "NodePool", "minwm-async-vae-l4-spot")
    assert requirement_values(denoiser, "karpenter.sh/capacity-type") == ["spot"]
    assert requirement_values(denoiser_8x, "karpenter.sh/capacity-type") == ["spot"]
    assert requirement_values(denoiser_h200_8x, "karpenter.sh/capacity-type") == [
        "spot"
    ]
    assert requirement_values(vae, "karpenter.sh/capacity-type") == ["on-demand"]
    assert requirement_values(vae_spot, "karpenter.sh/capacity-type") == ["spot"]
    assert vae["spec"]["weight"] > vae_spot["spec"]["weight"]
    assert requirement_values(denoiser, "node.kubernetes.io/instance-type") == [
        "p5.48xlarge",
    ]
    assert requirement_values(denoiser_8x, "node.kubernetes.io/instance-type") == [
        "p5.48xlarge",
    ]
    assert requirement_values(denoiser_h200_8x, "node.kubernetes.io/instance-type") == [
        "p5e.48xlarge",
        "p6-b200.48xlarge",
    ]
    denoiser_nodeclass = find(
        documents, "EC2NodeClass", "minwm-async-denoiser-8gpu-nvme-ec2"
    )
    assert denoiser_8x["spec"]["template"]["spec"]["nodeClassRef"]["name"] == (
        "minwm-async-denoiser-8gpu-nvme-ec2"
    )
    assert denoiser["spec"]["template"]["spec"]["nodeClassRef"]["name"] == (
        "minwm-async-denoiser-8gpu-nvme-ec2"
    )
    assert denoiser_nodeclass["spec"]["instanceStorePolicy"] == "RAID0"
    assert (
        denoiser_nodeclass["spec"]["blockDeviceMappings"][0]["ebs"]["volumeSize"]
        == "100Gi"
    )
    assert (
        denoiser_h200_8x["spec"]["template"]["spec"]["nodeClassRef"]["name"]
        == "minwm-async-denoiser-8gpu-nvme-ec2"
    )
    for pool in (denoiser, denoiser_8x, denoiser_h200_8x):
        assert (
            pool["spec"]["template"]["metadata"]["labels"][
                "seedleap.ai/model-cache-storage"
            ]
            == "local-nvme"
        )
    assert [
        interface["networkCardIndex"]
        for interface in denoiser_nodeclass["spec"]["networkInterfaces"]
    ] == list(range(8))
    assert requirement_values(denoiser_8x, "topology.kubernetes.io/zone") == [
        "us-east-2a",
        "us-east-2b",
        "us-east-2c",
    ]
    assert all(
        value.startswith("g6.")
        for value in requirement_values(vae, "node.kubernetes.io/instance-type")
    )
    assert 1 <= int(denoiser["spec"]["limits"]["nvidia.com/gpu"]) <= 8
    assert int(denoiser_8x["spec"]["limits"]["nvidia.com/gpu"]) == 8
    assert int(denoiser_h200_8x["spec"]["limits"]["nvidia.com/gpu"]) == 8
    assert 1 <= int(vae["spec"]["limits"]["nvidia.com/gpu"]) <= 8

    workloads = (
        (find(documents, "StatefulSet", "minwm-async-denoiser"), "2"),
        (find(documents, "StatefulSet", "lingbot2-async-denoiser"), 4),
        (find(documents, "Deployment", "minwm-async-vae"), "1"),
        (find(documents, "Deployment", "lingbot2-async-vae"), "1"),
    )
    for workload, expected_gpus in workloads:
        labels = workload["metadata"]["labels"]
        assert labels["seedleap.ai/test-run"] == "minwm-async-vae-benchmark"
        assert labels["seedleap.ai/ttl-after-test"] == "required"
        container = workload["spec"]["template"]["spec"]["containers"][0]
        resources = container["resources"]
        assert resources.get("requests")
        assert resources.get("limits")
        assert resources["requests"]["nvidia.com/gpu"] == expected_gpus
        assert resources["limits"]["nvidia.com/gpu"] == expected_gpus

    denoiser = find(documents, "StatefulSet", "minwm-async-denoiser")
    lingbot = find(documents, "StatefulSet", "lingbot2-async-denoiser")
    assert denoiser["spec"]["replicas"] == "REPLACE_WITH_DENOISER_BASE_REPLICAS"
    assert lingbot["spec"]["replicas"] == "REPLACE_WITH_LINGBOT_DENOISER_REPLICAS"
    assert denoiser["spec"]["ordinals"]["start"] == 2
    assert lingbot["spec"]["ordinals"]["start"] == 1
    assert denoiser["spec"]["updateStrategy"]["type"] == "OnDelete"
    assert lingbot["spec"]["updateStrategy"]["type"] == "OnDelete"
    for workload in (denoiser, lingbot):
        selector = workload["spec"]["template"]["spec"]["nodeSelector"]
        assert selector["seedleap.ai/denoiser-worker"] == "true"
        assert "karpenter.sh/nodepool" not in selector
        model_cache = next(
            volume
            for volume in workload["spec"]["template"]["spec"]["volumes"]
            if volume["name"] == "model-cache"
        )
        assert model_cache["hostPath"] == {
            "path": "/mnt/k8s-disks/0/minwm-model-cache",
            "type": "DirectoryOrCreate",
        }
        assert (
            workload["spec"]["template"]["spec"]["nodeSelector"][
                "seedleap.ai/model-cache-storage"
            ]
            == "local-nvme"
        )
    containers = denoiser["spec"]["template"]["spec"]["containers"]
    assert {container["name"] for container in containers} == {"denoiser"}
    command = " ".join(containers[0]["args"])
    assert "--realtime-vae-backend taehv_remote" in command
    assert "--realtime-vae-worker-url" not in command
    init_containers = denoiser["spec"]["template"]["spec"]["initContainers"]
    heartbeat = next(
        container
        for container in init_containers
        if container["name"] == "denoiser-heartbeat"
    )
    assert heartbeat["restartPolicy"] == "Always"
    assert "realtime_worker_heartbeat" in " ".join(heartbeat["args"])

    gateway_service = find(documents, "Service", "zing-lingbot-public")
    assert gateway_service["spec"]["selector"] == {
        "app.kubernetes.io/name": "minwm-realtime-gateway"
    }
    assert find(documents, "Namespace", "minwm-realtime")


def main() -> None:
    validate(load_documents())
    print(
        "MinWM async-VAE manifests satisfy capacity, availability, and safety policies."
    )


if __name__ == "__main__":
    main()
