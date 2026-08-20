import hashlib
import json
import sys
from pathlib import Path

import pytest

from sglang.multimodal_gen.tools import minwm_profile_launcher as launcher
from sglang.multimodal_gen.tools.minwm_profile_launcher import (
    GPUInfo,
    PROFILE_RESIDENT_SPEED,
    PROFILE_SM120_32G_SPEED,
    PROFILE_SM120_HIGHMEM_SPEED,
    build_launch_contract,
    resolve_profile,
)


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (GPUInfo("GeForce RTX 5090", (12, 0), 32768), PROFILE_SM120_32G_SPEED),
        (
            GPUInfo("RTX PRO 6000 Blackwell Server Edition", (12, 0), 97887),
            PROFILE_SM120_HIGHMEM_SPEED,
        ),
        (GPUInfo("NVIDIA H200", (9, 0), 143771), PROFILE_RESIDENT_SPEED),
        (GPUInfo("NVIDIA B200", (10, 0), 183359), PROFILE_RESIDENT_SPEED),
    ],
)
def test_auto_profile_selection(gpu, expected):
    assert resolve_profile("auto", gpu) == expected


def test_auto_profile_rejects_unvalidated_sm120_memory_size():
    with pytest.raises(ValueError, match="does not match a validated"):
        resolve_profile("auto", GPUInfo("Future SM120", (12, 0), 49152))


def test_explicit_sm120_profile_rejects_wrong_architecture():
    with pytest.raises(ValueError, match="requires SM120"):
        resolve_profile(
            PROFILE_SM120_32G_SPEED,
            GPUInfo("RTX 6000 Ada", (8, 9), 49140),
        )


def test_explicit_32g_profile_can_be_qualified_on_high_memory_sm120():
    assert (
        resolve_profile(
            PROFILE_SM120_32G_SPEED,
            GPUInfo("RTX PRO 6000 Blackwell Server Edition", (12, 0), 97887),
        )
        == PROFILE_SM120_32G_SPEED
    )


def test_auto_profile_rejects_ada():
    with pytest.raises(ValueError, match="unsupported MinWM inference GPU"):
        resolve_profile("auto", GPUInfo("RTX 6000 Ada", (8, 9), 49140))


def test_32g_contract_offloads_only_text_encoder():
    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
        taehv_path=Path("/models/taehv/taew2_2.pth"),
        server_args=["--model-path", "/models/minwm", "--port", "30000"],
        validate_artifacts=False,
    )

    command = contract["command"]
    assert contract["resolved_profile"] == PROFILE_SM120_32G_SPEED
    assert command[:3] == [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
    ]
    assert command[command.index("--text-encoder-cpu-offload") + 1] == "true"
    assert command[command.index("--vae-cpu-offload") + 1] == "false"
    assert command[command.index("--dit-cpu-offload") + 1] == "false"
    assert command[command.index("--dit-layerwise-offload") + 1] == "false"
    assert contract["environment"]["MINWM_ATTENTION_IMPL"] == "packed"
    assert contract["environment"]["MINWM_PACKED_ATTENTION_DETERMINISTIC"] == "false"
    assert contract["environment"]["SGLANG_MINWM_REQUIRE_SM120_FA4"] == "1"


def test_profile_rejects_managed_override():
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_launch_contract(
            requested_profile="auto",
            gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
            taehv_path=Path("/models/taehv/taew2_2.pth"),
            server_args=[
                "--model-path",
                "/models/minwm",
                "--dit-layerwise-offload",
                "true",
            ],
            validate_artifacts=False,
        )


def test_contract_validates_taehv_and_tianpeng_model(tmp_path, monkeypatch):
    taehv_path = tmp_path / "taew2_2.pth"
    taehv_path.write_bytes(b"test-taehv")
    monkeypatch.setattr(
        launcher,
        "TAEHV_SHA256",
        hashlib.sha256(taehv_path.read_bytes()).hexdigest(),
    )
    model_path = tmp_path / "model"
    config_path = model_path / "transformer" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(launcher.EXPECTED_MODEL_CONFIG))

    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
        taehv_path=taehv_path,
        server_args=["--model-path", str(model_path)],
    )

    assert contract["resolved_profile"] == PROFILE_SM120_32G_SPEED
