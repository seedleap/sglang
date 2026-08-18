from types import SimpleNamespace
import json

import pytest

from sglang.multimodal_gen.tools import minwm_image_runtime_probe as probe
from sglang.multimodal_gen.tools.minwm_image_runtime_probe import (
    expected_attention_for_capability,
)


@pytest.mark.parametrize(
    ("capability", "family", "backend"),
    [
        ((9, 0), "hopper", "fa3"),
        ((10, 0), "blackwell", "fa4"),
        ((10, 3), "blackwell", "fa4"),
    ],
)
def test_expected_attention_for_supported_gpu(capability, family, backend):
    assert expected_attention_for_capability(capability) == (family, backend)


@pytest.mark.parametrize("capability", [(8, 9), (9, 1), (12, 0)])
def test_expected_attention_rejects_unsupported_gpu(capability):
    with pytest.raises(ValueError, match="unsupported compute capability"):
        expected_attention_for_capability(capability)


class _FakeLockfile:
    def __init__(self, value):
        self.value = value

    def is_file(self):
        return True

    def read_text(self, encoding):
        assert encoding == "utf-8"
        return json.dumps(
            [
                {
                    "repo_id": probe.FA3_KERNEL_REPOSITORY,
                    "sha": probe.EXPECTED_FA3_KERNEL_SHA,
                    "variants": {
                        probe.EXPECTED_FA3_VARIANT: {
                            "hash": probe.EXPECTED_FA3_VARIANT_HASH
                        }
                    },
                }
            ]
        )

    def __str__(self):
        return self.value


def _patch_valid_software(monkeypatch):
    versions = {
        "flash-attn-4": "4.0.0b15",
        "kernels": "0.14.1",
        "nvidia-cutlass-dsl": "4.5.2",
        "sglang-kernel": "0.4.4",
        "torch": "2.11.0+cu130",
        "flash-attn": None,
    }
    monkeypatch.setattr(probe, "package_version", versions.get)
    monkeypatch.setattr(probe, "Path", _FakeLockfile)
    monkeypatch.setattr(
        probe,
        "system_cuda_version",
        lambda: {"release": "13.0", "compiler": "13.0.88"},
    )
    monkeypatch.setattr(
        probe,
        "module_spec_info",
        lambda name: {"available": name != "flash_attn_interface", "origin": name},
    )
    monkeypatch.setenv("SGLANG_BUILD_COMMIT", "a" * 40)
    monkeypatch.setenv("SGLANG_USE_SGL_FA3_KERNEL", "0")
    return SimpleNamespace(
        __version__="2.11.0+cu130", version=SimpleNamespace(cuda="13.0")
    )


def test_software_validation_does_not_import_native_modules(monkeypatch):
    torch_module = _patch_valid_software(monkeypatch)
    monkeypatch.setattr(
        probe.importlib,
        "import_module",
        lambda name: pytest.fail(f"software-only imported {name}"),
    )

    software, errors = probe._validate_software(torch_module, "a" * 40)

    assert errors == []
    assert software["fa3_kernel_lock_contains_repository"] is True


def test_software_validation_rejects_missing_required_module(monkeypatch):
    torch_module = _patch_valid_software(monkeypatch)
    monkeypatch.setattr(
        probe,
        "module_spec_info",
        lambda name: {
            "available": name not in {"flash_attn.cute", "flash_attn_interface"},
            "origin": name,
        },
    )

    _, errors = probe._validate_software(torch_module, "a" * 40)

    assert "required module spec 'flash_attn.cute' is missing" in errors


def test_software_validation_rejects_wrong_source_commit(monkeypatch):
    torch_module = _patch_valid_software(monkeypatch)

    _, errors = probe._validate_software(torch_module, "b" * 40)

    assert any("image source commit" in error for error in errors)


def test_software_validation_rejects_wrong_fa3_provider_selection(monkeypatch):
    torch_module = _patch_valid_software(monkeypatch)
    monkeypatch.setenv("SGLANG_USE_SGL_FA3_KERNEL", "1")

    _, errors = probe._validate_software(torch_module, "a" * 40)

    assert any("SGLANG_USE_SGL_FA3_KERNEL" in error for error in errors)


def test_fa3_compat_omits_unsupported_default_keywords():
    from sglang.jit_kernel.flash_attention_v3 import _call_fa3_kernel

    calls = []

    def legacy_kernel(*, value):
        calls.append(value)
        return value

    assert _call_fa3_kernel(legacy_kernel, value=7, only_qv=False, out=None) == 7
    assert calls == [7]


def test_fa3_compat_forwards_supported_nondefault_only_qv():
    from sglang.jit_kernel.flash_attention_v3 import _call_fa3_kernel

    def modern_kernel(*, only_qv=False):
        return only_qv

    assert _call_fa3_kernel(modern_kernel, only_qv=True) is True


def test_fa3_compat_rejects_unsupported_nondefault_only_qv():
    from sglang.jit_kernel.flash_attention_v3 import _call_fa3_kernel

    def legacy_kernel():
        return None

    with pytest.raises(NotImplementedError, match="does not support only_qv=True"):
        _call_fa3_kernel(legacy_kernel, only_qv=True)
