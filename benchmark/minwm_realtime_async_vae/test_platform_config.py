from __future__ import annotations

import json
import subprocess
import sys

import pytest

from platform_config import (
    FROZEN_BRANCH,
    FROZEN_GIT_SHA,
    IMAGE_INPUT_SPECS,
    IMAGE_PLACEHOLDERS,
    PRODUCTION_BASE_BRANCH,
    PRODUCTION_BASE_GIT_SHA,
    REGISTRY,
    all_platform_configs,
    applications,
    resolve_image_inputs,
    required_inputs_document,
    validate_configs,
)
from render_platform_config import DEFAULT_OUTPUT, render_documents


def _by_name(name: str):
    return next(config for config in all_platform_configs() if config["name"] == name)


def _image_callbacks() -> dict:
    return {
        "schemaVersion": 1,
        "frozenGitSha": FROZEN_GIT_SHA,
        "callbacks": {
            service_name: {
                "serviceName": service_name,
                "status": "success",
                "branch": FROZEN_BRANCH,
                "jenkinsJob": spec["jenkinsJob"],
                "operator": "jenkins",
                "buildId": str(index),
                "buildUrl": f"https://jenkins.example/job/{index}/",
                "image": (
                    f"{REGISTRY}:{spec['tagPrefix']}-{FROZEN_GIT_SHA}"
                ),
                "imageDigest": "sha256:" + str(index) * 64,
            }
            for index, (service_name, spec) in enumerate(
                IMAGE_INPUT_SPECS.items(), start=1
            )
        },
    }


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
    assert configs[-1]["deployType"] == "job"
    validate_configs(configs, require_resolved_images=False)

    encoded = json.dumps(configs, sort_keys=True)
    assert len(IMAGE_PLACEHOLDERS) == 8
    assert all(placeholder in encoded for placeholder in IMAGE_PLACEHOLDERS)
    assert ":latest" not in encoded


@pytest.mark.parametrize(
    ("name", "gpu_count", "release_fragment"),
    [
        ("minwm-denoiser", "2", "/releases/20260810T042157Z-c302d572/model"),
        ("lingbot2-denoiser", "4", "/releases/20260814T054118Z-e0650875/model"),
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
    assert config["image"] == IMAGE_INPUT_SPECS[name]["placeholder"]
    assert stager["image"] == IMAGE_INPUT_SPECS[name]["placeholder"]
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
        assert config["clusterId"] == "${WORLD_MODEL_CLUSTER_ID}"
        assert config["labels"]["app.kubernetes.io/name"] == config["name"]
        assert not {
            "business_line_id",
            "cluster_id",
            "namespace",
            "service_id",
        }.intersection(config["labels"])
        assert config["annotations"]["logs.loopit.me/enabled"] == "true"
        assert config["networkPolicy"]["defaultDeny"] is True
        assert set(config["networkPolicy"]) == {
            "enabled",
            "defaultDeny",
            "ingress",
            "egress",
        }


def test_network_policy_uses_only_typed_selectors_ports_and_ip_blocks():
    for config in all_platform_configs():
        policy = config["networkPolicy"]
        encoded = json.dumps(policy, sort_keys=True)
        assert "ingressFromServiceIds" not in encoded
        assert "egressToServiceIds" not in encoded
        assert "externalEgress" not in encoded
        for direction, peer_field in (("ingress", "from"), ("egress", "to")):
            for rule in policy[direction]:
                assert rule[peer_field]
                assert rule["ports"]
                assert all(set(peer) <= {
                    "podSelector",
                    "namespaceSelector",
                    "ipBlock",
                } for peer in rule[peer_field])
                assert all(port["protocol"] in {"TCP", "UDP", "SCTP"} for port in rule["ports"])


def test_required_inputs_keep_unresolved_deployment_non_executable():
    document = required_inputs_document()

    assert document["executionReady"] is False
    assert document["productionBaseline"] == {
        "branch": PRODUCTION_BASE_BRANCH,
        "gitSha": PRODUCTION_BASE_GIT_SHA,
    }
    assert document["frozenSource"] == {
        "branch": FROZEN_BRANCH,
        "gitSha": FROZEN_GIT_SHA,
    }
    assert document["requiredInputs"]["wm08BuildContract"] == {
        "state": "missing",
        "currentFrozenBranch": PRODUCTION_BASE_BRANCH,
        "currentFrozenGitSha": PRODUCTION_BASE_GIT_SHA,
        "requiredBranch": FROZEN_BRANCH,
        "requiredGitSha": FROZEN_GIT_SHA,
        "requiredTagFormat": "<service-tag-prefix>-<full-40-character-git-sha>",
        "requiredCallbackDigestField": "imageDigest",
        "runtimeImageFormat": "<ecr-repository>@<callback.imageDigest>",
        "reason": (
            "the production baseline does not contain the WM-09 release spec, "
            "CRT downloader compatibility, or publisher verifier"
        ),
    }
    assert document["requiredInputs"]["clusterRegistration"]["state"] == "missing"
    assert len(document["requiredInputs"]["imageBuildCallbacks"]) == 8
    assert all(
        item["state"] == "missing"
        for item in document["requiredInputs"]["networkPolicyPeers"]
    )
    assert document["resolvedInputs"]["lingbot2Release"] == {
        "releaseId": "20260814T054118Z-e0650875",
        "manifestSha256": (
            "e065087570bde5ae45cac0f678239d6da5dafb7c1af3a2a1be0ddd6ea8929fdd"
        ),
        "objectCount": 26,
        "bytes": 86071995490,
        "releaseSpec": (
            "model_releases/lingbot2/"
            "59cccf49f2d2dd27418ae7a04b82b10868d455c2/release-spec.json"
        ),
    }


def test_publisher_defaults_to_offline_plan_and_requires_explicit_execute_inputs():
    task = _by_name("world-model-artifact-publisher")

    assert task["deployType"] == "job"
    assert task["args"][-1] == "--offline-plan"
    assert task["args"][-2].endswith("/release-spec.json")
    assert task["taskExecutionPolicy"]["requireDigestImage"] is True
    assert task["taskExecutionPolicy"]["commandRules"][1]["argsPrefix"][-2:] == [
        "--execute",
        "--confirm-release-id",
    ]
    assert task["image"] == "${WORLD_MODEL_ARTIFACT_PUBLISHER_IMAGE_DIGEST}"
    assert task["nodeSelector"] == {}


def test_image_inputs_must_resolve_to_digests_before_deploy_render():
    inputs = _image_callbacks()
    resolved = resolve_image_inputs(all_platform_configs(), inputs)
    validate_configs(resolved, require_resolved_images=True)
    assert all(
        "@sha256:" in config["image"] and ":latest" not in config["image"]
        for config in resolved
    )

    bad = _image_callbacks()
    bad["callbacks"]["world-studio-webui"]["image"] = f"{REGISTRY}:latest"
    with pytest.raises(ValueError, match="full-SHA audit tag"):
        resolve_image_inputs(all_platform_configs(), bad)


def test_image_inputs_reject_wrong_branch_or_unverified_digest():
    wrong_branch = _image_callbacks()
    wrong_branch["callbacks"]["minwm-denoiser"]["branch"] = "main"
    with pytest.raises(ValueError, match="branch mismatch"):
        resolve_image_inputs(all_platform_configs(), wrong_branch)

    missing_digest = _image_callbacks()
    missing_digest["callbacks"]["lingbot2-denoiser"]["imageDigest"] = ""
    with pytest.raises(ValueError, match="digest is invalid"):
        resolve_image_inputs(all_platform_configs(), missing_digest)

    missing_build = _image_callbacks()
    missing_build["callbacks"]["world-realtime-gateway"]["buildId"] = ""
    with pytest.raises(ValueError, match="buildId is missing"):
        resolve_image_inputs(all_platform_configs(), missing_build)


def test_cli_validates_callbacks_but_blocks_resolved_payload_write(tmp_path):
    image_inputs = tmp_path / "callbacks.json"
    image_inputs.write_text(json.dumps(_image_callbacks()), encoding="utf-8")
    script = DEFAULT_OUTPUT.parents[2] / "render_platform_config.py"

    checked = subprocess.run(
        [
            sys.executable,
            str(script),
            "--image-inputs",
            str(image_inputs),
            "--check-image-inputs",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0
    assert json.loads(checked.stdout) == {
        "imageInputsValid": True,
        "executionReady": False,
    }

    blocked = subprocess.run(
        [sys.executable, str(script), "--image-inputs", str(image_inputs)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "deployment render blocked" in blocked.stderr


def test_renderer_matches_checked_in_goldens():
    documents = render_documents()
    assert len(documents) == 8
    for name, content in documents.items():
        assert json.loads(content)["name"] in name
        assert (DEFAULT_OUTPUT / name).read_text(encoding="utf-8") == content
