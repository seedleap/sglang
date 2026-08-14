from __future__ import annotations

import json

import pytest

from platform_config import (
    DENOISER_IMAGE,
    IMAGE_PLACEHOLDERS,
    all_platform_configs,
    applications,
    resolve_image_inputs,
    validate_configs,
)
from render_platform_config import DEFAULT_OUTPUT, render_documents


def _by_name(name: str):
    return next(config for config in all_platform_configs() if config["name"] == name)


def test_seven_applications_and_task_follow_frozen_release_order():
    configs = all_platform_configs()

    assert len(applications()) == 7
    assert [config["name"] for config in configs] == [
        "world-realtime-coordinator",
        "minwm-vae",
        "lingbot2-vae",
        "minwm-denoiser",
        "lingbot2-denoiser",
        "world-realtime-gateway",
        "world-studio-webui",
        "world-model-artifact-publisher",
    ]
    assert configs[-1]["kind"] == "task"
    validate_configs(configs, require_resolved_images=False)


@pytest.mark.parametrize(
    ("name", "gpu_count", "release_fragment"),
    [
        ("minwm-denoiser", "2", "/releases/20260810T042157Z-c302d572/model"),
        ("lingbot2-denoiser", "4", "/releases/${LINGBOT2_RELEASE_ID}/model"),
    ],
)
def test_denoisers_use_crt_nvme_immutable_release_and_digest_images(
    name, gpu_count, release_fragment
):
    config = _by_name(name)
    stager, heartbeat = config["initContainers"]
    command = " ".join(stager["args"])
    env = {item["name"]: item.get("value") for item in stager["env"]}
    cache = next(volume for volume in config["volumes"] if volume["name"] == "model-cache")

    assert config["deployType"] == "statefulset"
    assert config["statefulSet"]["updateStrategy"] == {"type": "OnDelete"}
    assert config["serviceSpec"]["headless"] is True
    assert config["image"] == DENOISER_IMAGE
    assert stager["image"] == DENOISER_IMAGE
    assert "download_model_artifact.py" in command
    assert "--concurrency 128" in command
    assert "--part-size-mib 16" in command
    assert release_fragment in env["MODEL_PREFIX"]
    assert cache["type"] == "emptyDir"
    assert config["nodeSelector"] == {"loopit.me/gpu-pool": "h100"}
    assert "karpenter.sh/capacity-type" not in config["nodeSelector"]
    assert config["resources"]["requests"]["nvidia.com/gpu"] == gpu_count
    assert config["resources"]["limits"]["nvidia.com/gpu"] == gpu_count
    assert heartbeat["restartPolicy"] == "Always"
    assert config["startupProbe"]["failureThreshold"] == 270


def test_vae_workers_request_one_l4_each_and_are_anti_affined():
    for name in ("minwm-vae", "lingbot2-vae"):
        config = _by_name(name)
        assert config["replicas"] == 1
        assert config["strategy"] == {"type": "Recreate"}
        assert config["nodeSelector"] == {"loopit.me/gpu-pool": "l4"}
        assert config["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert config["initContainers"][0]["restartPolicy"] == "Always"
        required = config["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        assert required[0]["topologyKey"] == "kubernetes.io/hostname"


def test_cpu_control_plane_is_ha_and_spread_across_zones():
    for name in (
        "world-realtime-coordinator",
        "world-realtime-gateway",
        "world-studio-webui",
    ):
        config = _by_name(name)
        assert config["replicas"] == 2
        assert config["podDisruptionBudget"] == {"minAvailable": 1}
        assert config["topologySpreadConstraints"][0]["topologyKey"] == (
            "topology.kubernetes.io/zone"
        )


def test_webui_keeps_external_secret_references_and_ephemeral_volumes():
    webui = _by_name("world-studio-webui")
    env = {item["name"]: item for item in webui["env"]}
    volumes = {item["name"]: item for item in webui["volumes"]}

    assert env["HAPPYOYSTER_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "world-studio-runtime",
        "key": "happyoyster-api-key",
    }
    assert volumes["runtime-secret"]["type"] == "secret"
    assert volumes["generated-images"]["type"] == "emptyDir"
    assert volumes["tmp"]["type"] == "emptyDir"


def test_every_config_has_ownership_logging_and_default_deny_network_policy():
    for config in all_platform_configs():
        assert config["businessLineId"] == "world-model"
        assert config["namespace"] == "world-model"
        assert config["labels"]["loopit.me/business-line"] == "world-model"
        assert config["labels"]["loopit.me/managed-by"] == "platform"
        assert config["annotations"]["logs.loopit.me/enabled"] == "true"
        assert config["networkPolicy"]["defaultDeny"] is True


def test_publisher_defaults_to_offline_plan_and_requires_explicit_execute_inputs():
    task = _by_name("world-model-artifact-publisher")

    assert task["deployType"] == "job"
    assert task["args"][-1] == "--offline-plan"
    assert "--execute" in task["requiredExecutionInputs"]
    assert task["image"] == "${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}"
    assert task["nodeSelector"] == {}


def test_image_inputs_must_resolve_to_digests_before_deploy_render():
    inputs = {
        placeholder: "example.invalid/world-model@sha256:" + str(index) * 64
        for index, placeholder in enumerate(sorted(IMAGE_PLACEHOLDERS), start=1)
    }
    resolved = resolve_image_inputs(all_platform_configs(), inputs)
    validate_configs(resolved, require_resolved_images=True)

    bad = dict(inputs)
    bad[next(iter(bad))] = "example.invalid/world-model:latest"
    with pytest.raises(ValueError, match="not digest-pinned"):
        resolve_image_inputs(all_platform_configs(), bad)


def test_renderer_matches_checked_in_goldens():
    documents = render_documents()
    assert len(documents) == 8
    for name, content in documents.items():
        assert json.loads(content)["name"] in name
        assert (DEFAULT_OUTPUT / name).read_text(encoding="utf-8") == content
