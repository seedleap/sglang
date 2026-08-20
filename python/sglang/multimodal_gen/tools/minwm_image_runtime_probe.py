#!/usr/bin/env python3
"""Fail-fast runtime contract for the unified MinWM CUDA inference image."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet

CONTRACT_VERSION = "minwm-image-runtime-contract/v2"
EXPECTED_TAEHV_REVISION = "093b918971d59001a0bad6dfd6e0409b5e1752cf"
EXPECTED_PACKAGE_SPECS = {
    "cryptography": "==50.0.0",
    "distro": "==1.9.0",
    "flash-attn-4": "==4.0.0b21",
    "kernels": "==0.14.1",
    "nvidia-cutlass-dsl": "==4.6.0.dev0",
    "nvidia-cutlass-dsl-libs-base": "==4.6.0.dev0",
    "nvidia-cutlass-dsl-libs-cu13": "==4.6.0.dev0",
    "pillow": ">=12.2.0",
    "pyjwt": "==2.13.0",
    "pyparsing": "==3.3.2",
    "quack-kernels": "==0.5.3",
    "sglang-kernel": "==0.4.4",
    "taehv": "==0.1.0",
    "torch": "==2.11.0+cu130",
}
FA3_KERNEL_REPOSITORY = "kernels-community/sgl-flash-attn3"
EXPECTED_FA3_KERNEL_SHA = "15c17db0bf9ce6599db795fa02a8f27467c92860"
EXPECTED_FA3_VARIANT = "torch211-cxx11-cu130-x86_64-linux"
EXPECTED_FA3_VARIANT_HASH = (
    "sha256-99e603e8511cd0dabd503c7c4ad08e7f1d4cb86ad11fe7ab27618cf91bf82be6"
)
EXPECTED_SGLANG_CACHE_DIR = "/root/.cache/sglang"
EXPECTED_SGLANG_USE_SGL_FA3_KERNEL = "0"
REQUIRED_MODULE_SPECS = (
    "flash_attn",
    "flash_attn.cute",
    "kernels",
    "sgl_kernel.flash_attn",
    "sglang.jit_kernel.flash_attention_v3",
    "sglang.multimodal_gen.runtime.layers.quantization.fp8",
    "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
    "taehv",
)


def expected_attention_for_capability(capability: tuple[int, int]) -> tuple[str, str]:
    if capability == (9, 0):
        return "hopper", "fa3"
    if capability[0] == 10:
        return "blackwell", "fa4"
    if capability[0] == 12:
        return "sm120", "fa4"
    raise ValueError(
        f"unsupported compute capability {capability}; expected SM90, SM10x, or SM12x"
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_vcs_revision(name: str) -> str | None:
    try:
        direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        return None


def system_cuda_version() -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            ["/usr/local/cuda/bin/nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release ([0-9.]+), V([^\s]+)", completed.stdout)
    if match is None:
        return None
    return {"release": match.group(1), "compiler": match.group(2)}


def module_spec_info(name: str) -> dict[str, Any]:
    """Resolve a module without executing package or native-extension code."""

    search_path = None
    qualified_name = ""
    try:
        for component in name.split("."):
            qualified_name = (
                f"{qualified_name}.{component}" if qualified_name else component
            )
            spec = (
                importlib.util.find_spec(qualified_name)
                if search_path is None
                else importlib.machinery.PathFinder.find_spec(
                    qualified_name, search_path
                )
            )
            if spec is None:
                return {"available": False, "origin": None}
            search_path = spec.submodule_search_locations
    except Exception as exc:
        return {"available": False, "origin": None, "error": repr(exc)}
    return {"available": True, "origin": spec.origin}


def module_import_info(name: str) -> dict[str, Any]:
    result = module_spec_info(name)
    if not result["available"]:
        result["imported"] = False
        return result
    try:
        importlib.import_module(name)
        result["imported"] = True
    except Exception as exc:
        result["imported"] = False
        result["error"] = repr(exc)
    return result


def _validate_software(
    torch_module: Any, expected_source_commit: str | None
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    packages = {name: package_version(name) for name in EXPECTED_PACKAGE_SPECS}
    for name, expected_spec in EXPECTED_PACKAGE_SPECS.items():
        actual = packages[name]
        if actual is None:
            errors.append(f"required distribution {name!r} is missing")
        elif actual not in SpecifierSet(expected_spec):
            errors.append(
                f"distribution {name!r} is {actual!r}, expected {expected_spec!r}"
            )

    taehv_revision = package_vcs_revision("taehv")
    if taehv_revision != EXPECTED_TAEHV_REVISION:
        errors.append(
            f"TAEHV source revision is {taehv_revision!r}, expected "
            f"{EXPECTED_TAEHV_REVISION!r}"
        )

    classic_flash_attn = package_version("flash-attn")
    if classic_flash_attn is not None:
        errors.append(
            "classic 'flash-attn' distribution is installed and can overwrite the "
            "flash-attn-4 namespace"
        )

    moviepy = package_version("moviepy")
    if moviepy is not None:
        errors.append(
            "MoviePy is installed even though its released Pillow<12 dependency "
            "conflicts with the MinWM image's security-fixed Pillow runtime"
        )

    nixl_distributions = {
        name: package_version(name) for name in ("nixl", "nixl-cu12", "nixl-cu13")
    }
    if any(version is not None for version in nixl_distributions.values()):
        errors.append(
            "NIXL distributions are installed even though MinWM does not use "
            "disaggregated transfer and the nixl meta package requires both CUDA "
            "12 and CUDA 13 backends"
        )

    torch_version = str(torch_module.__version__)
    torch_cuda = str(torch_module.version.cuda)
    if torch_version != "2.11.0+cu130":
        errors.append(f"torch runtime is {torch_version!r}, expected '2.11.0+cu130'")
    if torch_cuda != "13.0":
        errors.append(f"torch CUDA runtime is {torch_cuda!r}, expected '13.0'")
    cuda_toolkit = system_cuda_version()
    if cuda_toolkit is None or cuda_toolkit["release"] != "13.0":
        errors.append(
            f"system CUDA toolkit is {cuda_toolkit!r}, expected release '13.0'"
        )

    build_commit = os.environ.get("SGLANG_BUILD_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", build_commit):
        errors.append(
            "SGLANG_BUILD_COMMIT must contain the immutable 40-character source SHA"
        )
    elif expected_source_commit is not None and build_commit != expected_source_commit:
        errors.append(
            f"image source commit is {build_commit!r}, expected "
            f"{expected_source_commit!r}"
        )

    use_sgl_fa3_kernel = os.environ.get("SGLANG_USE_SGL_FA3_KERNEL")
    if use_sgl_fa3_kernel != EXPECTED_SGLANG_USE_SGL_FA3_KERNEL:
        errors.append(
            "SGLANG_USE_SGL_FA3_KERNEL must be '0' so Hopper uses the locked "
            "kernels-community FA3 provider; got "
            f"{use_sgl_fa3_kernel!r}"
        )

    cache_dir = os.path.abspath(
        os.path.expanduser(
            os.environ.get("SGLANG_CACHE_DIR", EXPECTED_SGLANG_CACHE_DIR)
        )
    )
    if cache_dir != EXPECTED_SGLANG_CACHE_DIR:
        errors.append(
            f"effective SGLANG_CACHE_DIR is {cache_dir!r}, expected "
            f"{EXPECTED_SGLANG_CACHE_DIR!r}"
        )
    lockfile = Path(os.path.join(cache_dir, "kernels.lock"))
    lock_revision = None
    variant_hash = None
    if not lockfile.is_file():
        errors.append(f"FA3 kernel lockfile is missing: {lockfile}")
        lock_contains_fa3 = False
    else:
        try:
            lock_data = json.loads(lockfile.read_text(encoding="utf-8"))
            matching_entries = [
                entry
                for entry in lock_data
                if entry.get("repo_id") == FA3_KERNEL_REPOSITORY
            ]
            lock_contains_fa3 = len(matching_entries) == 1
            if lock_contains_fa3:
                lock_revision = matching_entries[0].get("sha")
                variant_hash = (
                    matching_entries[0]
                    .get("variants", {})
                    .get(EXPECTED_FA3_VARIANT, {})
                    .get("hash")
                )
        except Exception as exc:
            lock_contains_fa3 = False
            errors.append(f"FA3 kernel lockfile cannot be read: {exc!r}")
        if not lock_contains_fa3:
            errors.append(
                f"FA3 kernel lockfile does not contain {FA3_KERNEL_REPOSITORY!r}"
            )
        if lock_revision != EXPECTED_FA3_KERNEL_SHA:
            errors.append(
                f"FA3 kernel revision is {lock_revision!r}, expected "
                f"{EXPECTED_FA3_KERNEL_SHA!r}"
            )
        if variant_hash != EXPECTED_FA3_VARIANT_HASH:
            errors.append(
                f"FA3 variant {EXPECTED_FA3_VARIANT!r} hash is "
                f"{variant_hash!r}, expected {EXPECTED_FA3_VARIANT_HASH!r}"
            )

    modules = {
        name: module_spec_info(name)
        for name in (*REQUIRED_MODULE_SPECS, "flash_attn_interface")
    }
    for name in REQUIRED_MODULE_SPECS:
        if not modules[name]["available"]:
            errors.append(f"required module spec {name!r} is missing")
    if modules["flash_attn_interface"]["available"]:
        errors.append(
            "unexpected top-level flash_attn_interface; unified image uses the "
            "bundled SGLang FA3 loader"
        )

    return (
        {
            "torch": torch_version,
            "torch_cuda": torch_cuda,
            "system_cuda_toolkit": cuda_toolkit,
            "sglang_commit": build_commit or None,
            "sglang_use_sgl_fa3_kernel": use_sgl_fa3_kernel,
            "image_tag": os.environ.get("SGLANG_IMAGE_TAG") or None,
            "packages": packages,
            "taehv_source_revision": taehv_revision,
            "classic_flash_attn_distribution": classic_flash_attn,
            "moviepy_distribution": moviepy,
            "nixl_distributions": nixl_distributions,
            "fa3_kernel_lockfile": str(lockfile),
            "fa3_kernel_lock_contains_repository": lock_contains_fa3,
            "fa3_kernel_revision": lock_revision,
            "fa3_variant": EXPECTED_FA3_VARIANT,
            "fa3_variant_hash": variant_hash,
            "module_specs": modules,
        },
        errors,
    )


def _unwrap_output(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _reference_attention(query: Any, key: Any, value: Any) -> Any:
    import torch

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())


def _check_attention_output(name: str, output: Any, reference: Any) -> dict[str, Any]:
    import torch

    difference = (output.float() - reference).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    finite = bool(torch.isfinite(output).all().item())
    passed = (
        output.shape == reference.shape
        and output.dtype == torch.bfloat16
        and finite
        and max_abs <= 0.20
        and mean_abs <= 0.02
    )
    return {
        "name": name,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "max_abs_vs_fp32_reference": max_abs,
        "mean_abs_vs_fp32_reference": mean_abs,
        "passed": passed,
    }


def _run_attention_kernel_smoke() -> list[dict[str, Any]]:
    import torch

    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        FlashAttentionImpl,
    )
    from sglang.multimodal_gen.runtime.models.dits.minwm import (
        _minwm_packed_varlen_attention,
    )

    dense_impl = FlashAttentionImpl(
        num_heads=4,
        head_size=128,
        causal=False,
        softmax_scale=1.0 / math.sqrt(128),
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260818)
    results: list[dict[str, Any]] = []
    for label, query_length, key_length in (
        ("self", 128, 128),
        ("cross", 128, 192),
    ):
        shape_q = (1, query_length, 4, 128)
        shape_kv = (1, key_length, 4, 128)
        query = torch.randn(
            shape_q, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        key = torch.randn(
            shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        value = torch.randn(
            shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        reference = _reference_attention(query, key, value)

        packed = _unwrap_output(_minwm_packed_varlen_attention(query, key, value))
        torch.cuda.synchronize()
        results.append(_check_attention_output(f"packed_{label}", packed, reference))

        dense = _unwrap_output(dense_impl.forward(query, key, value))
        torch.cuda.synchronize()
        results.append(_check_attention_output(f"dense_{label}", dense, reference))
    return results


def _check_ffn_output(name: str, output: Any, reference: Any) -> dict[str, Any]:
    import torch

    output_float = output.float()
    reference_float = reference.float()
    difference = output_float - reference_float
    relative_l2 = float(
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference_float)
        )
        .float()
        .item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            output_float.flatten(), reference_float.flatten(), dim=0
        ).item()
    )
    finite = bool(torch.isfinite(output).all().item())
    passed = (
        output.shape == reference.shape
        and output.dtype == torch.bfloat16
        and finite
        and relative_l2 <= 0.08
        and cosine >= 0.995
    )
    return {
        "name": name,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "relative_l2_vs_bf16_reference": relative_l2,
        "cosine_vs_bf16_reference": cosine,
        "passed": passed,
    }


def _run_ffn_kernel_smoke(family: str) -> list[dict[str, Any]]:
    import torch

    from sglang.srt.layers.quantization.fp8_utils import (
        apply_fp8_linear,
        apply_fp8_linear_scaled_mm,
        cutlass_fp8_supported,
        per_token_group_quant_fp8,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260818)
    inputs = torch.randn(
        (64, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight_reference = torch.randn(
        (128, 256), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight_nk, weight_scale_n1 = per_token_group_quant_fp8(
        weight_reference.t().contiguous(), group_size=weight_reference.shape[0]
    )
    weight = weight_nk.t()
    weight_scale = weight_scale_n1.t().contiguous()
    if weight.stride(0) != 1 or weight_scale.numel() != weight.shape[1]:
        raise RuntimeError(
            "channelwise FP8 weight must use the production column-major KxN "
            "layout with one scale per output channel"
        )
    reference = torch.matmul(inputs, weight_reference)

    online = apply_fp8_linear(
        input=inputs,
        weight=weight,
        weight_scale=weight_scale,
        input_scale=None,
        cutlass_fp8_supported=cutlass_fp8_supported(),
    )
    torch.cuda.synchronize()
    results = [_check_ffn_output("online_fp8", online, reference)]

    input_scale = (inputs.float().abs().max() / 448.0).clamp_min(1e-8).reshape(1)
    if family == "hopper":
        static = apply_fp8_linear(
            input=inputs,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cutlass_fp8_supported(),
        )
        static_name = "static_channelwise_fp8"
    else:
        tensor_weight_scale = (weight_reference.float().abs().max() / 448.0).clamp_min(
            1e-8
        )
        tensor_weight_nk = (
            (weight_reference.t().contiguous().float() / tensor_weight_scale)
            .clamp(min=-448, max=448)
            .to(torch.float8_e4m3fn)
        )
        tensor_weight = tensor_weight_nk.t()
        if tensor_weight.stride(0) != 1:
            raise RuntimeError(
                "SM100 FP8 weight must use the production column-major KxN layout"
            )
        static = apply_fp8_linear_scaled_mm(
            input=inputs,
            weight=tensor_weight,
            weight_scale=tensor_weight_scale.reshape(1),
            input_scale=input_scale,
        )
        static_name = "static_sm100_scaled_mm_fp8"
    torch.cuda.synchronize()
    results.append(_check_ffn_output(static_name, static, reference))
    return results


def _fa3_provider_info() -> dict[str, Any]:
    from sgl_kernel.flash_attn import flash_attn_varlen_func as fallback_function

    from sglang.jit_kernel.flash_attention_v3 import _load_fa3_kernels

    function = _load_fa3_kernels()["flash_attn_varlen_func"]
    fallback = function is fallback_function
    return {
        "provider": "sgl-kernel-fallback" if fallback else "kernels-community",
        "module": getattr(function, "__module__", None),
        "type": f"{type(function).__module__}.{type(function).__qualname__}",
    }


def collect_contract(
    *,
    software_only: bool,
    expected_source_commit: str | None,
    expected_family: str | None,
    expected_visible_gpus: int,
    expected_image_digest: str | None,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch

    software, errors = _validate_software(torch, expected_source_commit)
    result: dict[str, Any] = {
        "schema_version": CONTRACT_VERSION,
        "expected_image_digest": expected_image_digest,
        "software": software,
        "hardware": None,
        "attention": None,
        "ffn": None,
        "kernel_smoke": [],
        "errors": errors,
    }
    if software_only:
        result["status"] = "pass" if not errors else "fail"
        return result

    try:
        visible_gpus = torch.cuda.device_count()
    except Exception as exc:
        errors.append(f"hardware discovery raised {exc!r}")
        result["status"] = "fail"
        return result
    if visible_gpus != expected_visible_gpus:
        errors.append(
            f"visible CUDA device count is {visible_gpus}, expected "
            f"{expected_visible_gpus}"
        )
    if visible_gpus < 1:
        result["status"] = "fail"
        return result

    try:
        name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
    except Exception as exc:
        errors.append(f"GPU identity query raised {exc!r}")
        result["hardware"] = {"visible_gpu_count": visible_gpus}
        result["status"] = "fail"
        return result
    try:
        family, expected_backend = expected_attention_for_capability(capability)
    except ValueError as exc:
        errors.append(str(exc))
        result["hardware"] = {
            "visible_gpu_count": visible_gpus,
            "name": name,
            "compute_capability": list(capability),
        }
        result["status"] = "fail"
        return result

    if expected_family is not None and family != expected_family:
        errors.append(f"detected GPU family {family!r}, expected {expected_family!r}")
    if family == "hopper" and not re.search(r"\bH(?:100|200)\b", name):
        errors.append(f"SM90 device name {name!r} is not an H100/H200")
    if family == "blackwell" and not re.search(r"\b(?:B|GB)(?:200|300)\b", name):
        errors.append(f"SM10x device name {name!r} is not a B200/B300/GB200/GB300")
    if family == "sm120" and not re.search(r"\bRTX\b", name):
        errors.append(f"SM12x device name {name!r} is not an RTX GPU")

    required_import = (
        "sglang.jit_kernel.flash_attention_v3"
        if family == "hopper"
        else "flash_attn.cute"
    )
    active_import = module_import_info(required_import)
    if not active_import.get("imported"):
        errors.append(f"active backend module {required_import!r} failed to import")

    result["hardware"] = {
        "profile": family,
        "visible_gpu_count": visible_gpus,
        "name": name,
        "compute_capability": list(capability),
    }
    result["attention"] = {
        "expected": expected_backend,
        "active_module": {"name": required_import, **active_import},
        "dense": None,
        "packed": None,
    }
    ffn_modules = {
        name: module_import_info(name)
        for name in (
            "sglang.multimodal_gen.runtime.layers.quantization.fp8",
            "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
        )
    }
    for name, info in ffn_modules.items():
        if not info.get("imported"):
            errors.append(f"FFN quantization module {name!r} failed to import")
    result["ffn"] = {"quantization_modules": ffn_modules, "kernel_smoke": []}

    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            flash_attn as dense_flash_attn,
        )
        from sglang.multimodal_gen.runtime.models.dits.minwm import (
            _minwm_packed_attention_backend,
        )
        from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatform
        from sglang.multimodal_gen.runtime.platforms.interface import (
            AttentionBackendEnum,
        )

        dense_class = CudaPlatform.get_attn_backend_cls_str(
            selected_backend=AttentionBackendEnum.FA,
            head_size=128,
            dtype=torch.bfloat16,
        )
        dense_backend = f"fa{dense_flash_attn.fa_ver}"
        packed_backend = _minwm_packed_attention_backend(torch.device("cuda:0"))
        result["attention"]["dense"] = {
            "resolved_class": dense_class,
            "version": dense_backend,
        }
        result["attention"]["packed"] = {"version": packed_backend}
        if not dense_class.endswith(".FlashAttentionBackend"):
            errors.append(f"dense FA resolved to unexpected class {dense_class!r}")
        if dense_backend != expected_backend:
            errors.append(
                f"dense backend resolved to {dense_backend}, expected {expected_backend}"
            )
        if packed_backend != expected_backend:
            errors.append(
                f"packed backend resolved to {packed_backend}, expected "
                f"{expected_backend}"
            )
    except Exception as exc:
        errors.append(f"attention backend resolution raised {exc!r}")

    try:
        result["kernel_smoke"] = _run_attention_kernel_smoke()
        failed_kernels = [
            item["name"] for item in result["kernel_smoke"] if not item["passed"]
        ]
        if failed_kernels:
            errors.append(f"attention kernel smoke failed: {failed_kernels}")
        if family == "hopper":
            provider = _fa3_provider_info()
            result["attention"]["fa3_provider"] = provider
            if provider["provider"] != "kernels-community":
                errors.append(
                    "Hopper FA3 loaded the sgl-kernel fallback instead of the "
                    "locked kernels-community artifact"
                )
    except Exception as exc:
        errors.append(f"attention kernel smoke raised {exc!r}")

    if family == "sm120":
        result["ffn"]["profile"] = "bf16"
        result["ffn"]["kernel_smoke"] = []
    else:
        try:
            result["ffn"]["kernel_smoke"] = _run_ffn_kernel_smoke(family)
            failed_ffn_kernels = [
                item["name"]
                for item in result["ffn"]["kernel_smoke"]
                if not item["passed"]
            ]
            if failed_ffn_kernels:
                errors.append(f"FFN FP8 kernel smoke failed: {failed_ffn_kernels}")
        except Exception as exc:
            errors.append(f"FFN FP8 kernel smoke raised {exc!r}")

    result["status"] = "pass" if not errors else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-only", action="store_true")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-family", choices=("hopper", "blackwell", "sm120"))
    parser.add_argument("--expected-visible-gpus", type=int, default=1)
    parser.add_argument("--expected-image-digest")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        contract = collect_contract(
            software_only=args.software_only,
            expected_source_commit=args.expected_source_commit,
            expected_family=args.expected_family,
            expected_visible_gpus=args.expected_visible_gpus,
            expected_image_digest=args.expected_image_digest,
        )
    except Exception as exc:
        contract = {
            "schema_version": CONTRACT_VERSION,
            "expected_image_digest": args.expected_image_digest,
            "software": None,
            "hardware": None,
            "attention": None,
            "ffn": None,
            "kernel_smoke": [],
            "errors": [f"contract initialization raised {exc!r}"],
            "status": "fail",
        }
    rendered = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    sys.stdout.flush()
    sys.stderr.flush()
    # CUTLASS DSL may leave non-daemon compiler workers alive after SM120 JIT.
    # The probe is a one-shot CLI, so terminate after the contract is durable.
    os._exit(0 if contract["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
