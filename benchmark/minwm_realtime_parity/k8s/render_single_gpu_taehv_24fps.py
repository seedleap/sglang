#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
TEMPLATE = HERE / "minwm_hardware_job.template.yaml"
GENERATED = HERE / "generated"
RUNNER = BENCHMARK_ROOT / "run_single_gpu_taehv_24fps.sh"
CLIENT = BENCHMARK_ROOT / "benchmark_realtime_throughput.py"
COMMON = BENCHMARK_ROOT / "common.py"
CASES = BENCHMARK_ROOT / "cases_720p_compile_smoke.json"
QUALITY_CLIENT = BENCHMARK_ROOT / "run_sglang_api.py"
QUALITY_ANALYZER = BENCHMARK_ROOT / "analyze_fa3_quality.py"
QUALITY_ACTION_CASES = BENCHMARK_ROOT / "cases_fa3_quality_actions_720p.json"
QUALITY_LONG_CASES = BENCHMARK_ROOT / "cases_fa3_quality_long_720p.json"
FA2_REFERENCE_PATCH = BENCHMARK_ROOT / "fa2_reference_hopper_validation.patch"

BASE_HARNESS_FILES = {
    "run_single_gpu_taehv_24fps.sh": RUNNER,
    "benchmark_realtime_throughput.py": CLIENT,
    "common.py": COMMON,
    "cases_720p_compile_smoke.json": CASES,
}
QUALITY_HARNESS_FILES = {
    "run_sglang_api.py": QUALITY_CLIENT,
    "analyze_fa3_quality.py": QUALITY_ANALYZER,
    "cases_fa3_quality_actions_720p.json": QUALITY_ACTION_CASES,
    "cases_fa3_quality_long_720p.json": QUALITY_LONG_CASES,
    "fa2_reference_hopper_validation.patch": FA2_REFERENCE_PATCH,
}
HARNESS_FILES = {**BASE_HARNESS_FILES, **QUALITY_HARNESS_FILES}

DEFAULT_SGLANG_GIT_REF = "54bdfea9cd52ac1cd79896e1a7275e18a0257b79"
MINWM_GIT_REF = "4220c8a2dc456b2d9c85ef6c0d9db7fb872d864c"
BASE_IMAGE = (
    "829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-training@"
    "sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a"
)
RESULT_URI_ROOT = (
    "s3://leap-world-us-east-2/world-model/evals/minwm/performance/20260817/"
    "single-gpu-local-taehv-24fps"
)
RESULT_MOUNT_ROOT = (
    "/s3-results/world-model/evals/minwm/performance/20260817/"
    "single-gpu-local-taehv-24fps"
)

COMMON_ENV = {
    "MINWM_BASE_IMAGE": BASE_IMAGE,
    "MINWM_GIT_REF": MINWM_GIT_REF,
    "MINWM_INPUT_ROOT": "/s3-input",
    "MINWM_CHECKPOINT_RELATIVE_PATH": (
        "world-model/evals/minwm/checkpoint-staging/"
        "tianpeng-detailmix-gap12-20260729/global_step_010000/generator/model.pt"
    ),
    "MINWM_PRETRAINED_RELATIVE_PATH": (
        "world-model/checkpoints/minWM/Wan2.2-TI2V-5B-from-diffusers"
    ),
    "MINWM_CHECKPOINT_SOURCE_URI": (
        "s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/"
        "rolling/Wan21/Action2V/bidirectional/"
        "wan22-5B-varlen-multishot-texiao-0725detailed-mix-"
        "dccb050-dmd-0724-5eba381389f-merge/global_step_010000/generator/model.pt"
    ),
    "MINWM_CHECKPOINT_SOURCE_VERSION": "sL6CTylRv4QWY98mVTkuLoe5REbKHlvd",
    "MINWM_CHECKPOINT_SOURCE_ETAG": "b99fdb78f7c4e784fa676964f7054d1f-2386",
    "MINWM_CHECKPOINT_STAGED_VERSION": "yPbAthXMihZVEbfNFK4Ew2AVYs0FGDUq",
    "MINWM_CHECKPOINT_BYTES": "20014120667",
    "MINWM_CHECKPOINT_SHA256": (
        "18a48a2709d74b93ce26f0b808f381d191553853aae81dd72d2438430251d379"
    ),
    "MINWM_CHECKPOINT_CRC64": "WId1a/FowFo=",
    "MINWM_VAE_CPU_OFFLOAD": "false",
    "MINWM_FIRST_FRAME_SOURCE_URI": (
        "s3://leap-world-us-east-2/world-model/eval/platform/"
        "eval_sets/minWM/testset100_v2/img/p02.png"
    ),
    "MINWM_FIRST_FRAME_SOURCE_VERSION": "5q2pfK_Cqr48ufR6Ksl_6gu2qnSwwLVn",
    "MINWM_FIRST_FRAME_SOURCE_ETAG": "6c342c50c60984ad0019bf65dd6e10e5",
    "MINWM_FIRST_FRAME_SOURCE_BYTES": "1878806",
    "MINWM_FIRST_FRAME_SOURCE_SHA256": (
        "d7dc0202e4aaf92c6c82d155e859494d4bd1d0f7d0d43bd43551f1b05d8eb51a"
    ),
    "MINWM_FIRST_FRAME_SOURCE_CRC64": "5rFASr/KHec=",
    "TAEHV_REVISION": "093b918971d59001a0bad6dfd6e0409b5e1752cf",
    "TAEHV_CHECKPOINT_URL": (
        "https://raw.githubusercontent.com/madebyollin/taehv/"
        "093b918971d59001a0bad6dfd6e0409b5e1752cf/taew2_2.pth"
    ),
    "TAEHV_CHECKPOINT_SHA256": (
        "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"
    ),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

HARDWARE = {
    "b200": {
        "context": "minwm-spot",
        "namespace": "ray",
        "instance_type": "p6-b200.48xlarge",
        "nodepool": "minwm-test-b200-spot",
        "zone": None,
        "node_selector": {
            "karpenter.sh/capacity-type": "spot",
            "karpenter.sh/nodepool": "minwm-test-b200-spot",
            "node.kubernetes.io/instance-type": "p6-b200.48xlarge",
            "seedleap.ai/capacity-pool": "minwm-test-b200-karpenter",
        },
        "taint_key": "seedleap.ai/workload",
        "taint_value": "wan22-ti2v",
        "storage": {
            "layout": "shared",
            "input_pvc": "s3-claim",
            "results_pvc": "s3-claim",
            "verified_results_access": "RWX",
        },
        "profile": "experimental-sm100-high-memory",
        "compute_cap": "10.0",
        "min_memory_mib": "180000",
        "max_memory_mib": "",
        "gpu_sku": "B200",
        "protocol_smoke_warmup_chunks": "1",
        "require_full_window_no_offload_smoke": "false",
    },
    "b300": {
        "context": "aws03-usw2",
        "namespace": "default",
        "instance_type": "p6-b300.48xlarge",
        "nodepool": "minwm-spot-p6-b300-0703",
        "zone": "us-west-2a",
        "node_selector": {
            "eks.amazonaws.com/capacityType": "SPOT",
            "eks.amazonaws.com/nodegroup": "minwm-spot-p6-b300-0703",
            "node.kubernetes.io/instance-type": "p6-b300.48xlarge",
            "seedleap.ai/workload": "wan22-ti2v",
            "topology.kubernetes.io/zone": "us-west-2a",
        },
        "taint_key": "seedleap.ai/workload",
        "taint_value": "wan22-ti2v",
        "storage": {
            "layout": "shared",
            "input_pvc": "s3-claim",
            "results_pvc": "s3-claim",
            "verified_results_access": "RWX",
        },
        "profile": "experimental-sm103-high-memory",
        "compute_cap": "10.3",
        "min_memory_mib": "250000",
        "max_memory_mib": "",
        "gpu_sku": "B300",
        "protocol_smoke_warmup_chunks": "1",
        "require_full_window_no_offload_smoke": "false",
    },
    "h100": {
        "context": "aws03-usw2",
        "namespace": "default",
        "instance_type": "p5.48xlarge",
        "nodepool": "minwm-spot-p5-h100-sglang-0718",
        "zone": None,
        "node_selector": {
            "eks.amazonaws.com/capacityType": "SPOT",
            "eks.amazonaws.com/nodegroup": "minwm-spot-p5-h100-sglang-0718",
            "node.kubernetes.io/instance-type": "p5.48xlarge",
            "seedleap.ai/workload": "wan22-ti2v",
        },
        "taint_key": "seedleap.ai/workload",
        "taint_value": "wan22-ti2v",
        "storage": {
            "layout": "shared",
            "input_pvc": "s3-claim",
            "results_pvc": "s3-claim",
            "verified_results_access": "RWX",
        },
        "profile": "experimental-sm90-h100-no-offload",
        "compute_cap": "9.0",
        "min_memory_mib": "80000",
        "max_memory_mib": "90000",
        "gpu_sku": "H100",
        "protocol_smoke_warmup_chunks": "8",
        "require_full_window_no_offload_smoke": "true",
    },
    "h200": {
        "context": "aws03-usw2",
        "namespace": "default",
        "instance_type": "p5en.48xlarge",
        "nodepool": "minwm-spot-p5en-h200-sglang-0718",
        "zone": None,
        "node_selector": {
            "eks.amazonaws.com/capacityType": "SPOT",
            "eks.amazonaws.com/nodegroup": "minwm-spot-p5en-h200-sglang-0718",
            "node.kubernetes.io/instance-type": "p5en.48xlarge",
            "seedleap.ai/workload": "wan22-ti2v",
        },
        "taint_key": "seedleap.ai/workload",
        "taint_value": "wan22-ti2v",
        "storage": {
            "layout": "shared",
            "input_pvc": "s3-claim",
            "results_pvc": "s3-claim",
            "verified_results_access": "RWX",
        },
        "profile": "experimental-sm90-h200-no-offload",
        "compute_cap": "9.0",
        "min_memory_mib": "140000",
        "max_memory_mib": "150000",
        "gpu_sku": "H200",
        "protocol_smoke_warmup_chunks": "8",
        "require_full_window_no_offload_smoke": "true",
    },
}


class LiteralDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


LiteralDumper.add_representer(str, _str_presenter)


def env_entry(name: str, value: str) -> dict:
    return {"name": name, "value": value}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def storage_spec(storage: dict) -> tuple[list[dict], list[dict]]:
    layout = storage.get("layout")
    input_pvc = storage.get("input_pvc")
    results_pvc = storage.get("results_pvc")
    verified_results_access = storage.get("verified_results_access")
    if not input_pvc:
        raise ValueError("storage input_pvc must be explicit")
    if layout == "shared":
        if results_pvc != input_pvc:
            raise ValueError("shared storage requires identical input/results PVCs")
        if verified_results_access != "RWX":
            raise ValueError("shared storage requires a verified RWX results PVC")
        return (
            [
                {
                    "name": "s3-shared",
                    "persistentVolumeClaim": {"claimName": input_pvc},
                }
            ],
            [
                {
                    "name": "s3-shared",
                    "mountPath": "/s3-input",
                    "readOnly": True,
                },
                {"name": "s3-shared", "mountPath": "/s3-results"},
            ],
        )
    if layout == "split":
        if not results_pvc:
            raise ValueError("split storage requires an explicit results_pvc")
        if results_pvc == input_pvc:
            raise ValueError("split storage requires distinct input/results PVCs")
        if verified_results_access != "RWX":
            raise ValueError("split storage requires a verified RWX results PVC")
        return (
            [
                {
                    "name": "s3-input",
                    "persistentVolumeClaim": {"claimName": input_pvc},
                },
                {
                    "name": "s3-results",
                    "persistentVolumeClaim": {"claimName": results_pvc},
                },
            ],
            [
                {"name": "s3-input", "mountPath": "/s3-input", "readOnly": True},
                {"name": "s3-results", "mountPath": "/s3-results"},
            ],
        )
    raise ValueError(f"unsupported storage layout: {layout!r}")


def harness_ref_matches_checkout(harness_git_ref: str) -> bool:
    for path in HARNESS_FILES.values():
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            committed = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "show", f"{harness_git_ref}:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError:
            return False
        if committed != path.read_bytes():
            return False
    return True


def render(
    sku: str,
    mode: str,
    *,
    sglang_git_ref: str,
    harness_git_ref: str,
    harness_ref_verified: bool = True,
    require_24fps: bool = False,
    candidate_evidence: bool = False,
    run_tag: str,
) -> tuple[dict, dict]:
    hardware = HARDWARE[sku]
    if mode == "fa3-quality" and sku not in {"h100", "h200"}:
        raise ValueError("fa3-quality mode requires Hopper H100 or H200")
    storage_volumes, storage_mounts = storage_spec(hardware["storage"])
    candidate_evidence = candidate_evidence or require_24fps or mode == "fa3-quality"
    run_id = f"minwm-taehv24-{sku}-{mode}-{run_tag}"
    configmap_name = f"{run_id}-files"
    template = yaml.safe_load(TEMPLATE.read_text())
    job = copy.deepcopy(template)

    job["metadata"] = {
        "name": run_id,
        "namespace": hardware["namespace"],
        "labels": {
            "app.kubernetes.io/name": run_id,
            "seedleap.ai/owner": "chenshengdong",
            "seedleap.ai/task": "minwm-single-gpu-taehv24",
            "seedleap.ai/hardware-profile": hardware["profile"],
            "seedleap.ai/profile-mode": mode,
        },
        "annotations": {
            "seedleap.ai/repo": "seedleap/sglang",
            "seedleap.ai/sglang-git-ref": sglang_git_ref,
            "seedleap.ai/harness-git-ref": harness_git_ref,
            "seedleap.ai/harness-ref-verified": str(harness_ref_verified).lower(),
            "seedleap.ai/model-git-ref": MINWM_GIT_REF,
            "seedleap.ai/checkpoint-version-id": (
                COMMON_ENV["MINWM_CHECKPOINT_SOURCE_VERSION"]
            ),
            "seedleap.ai/instance-type": hardware["instance_type"],
            "seedleap.ai/capacity-type": "spot",
            "seedleap.ai/cluster-context": hardware["context"],
            "seedleap.ai/vae-cpu-offload": "false",
            "seedleap.ai/expected-compute-cap": hardware["compute_cap"],
            "seedleap.ai/expected-memory-mib": (
                f"{hardware['min_memory_mib']}-"
                f"{hardware['max_memory_mib'] or 'unbounded'}"
            ),
            "seedleap.ai/no-offload-protocol-gate": (
                "full-window"
                if hardware["require_full_window_no_offload_smoke"] == "true"
                else "standard"
            ),
            "seedleap.ai/execution-policy": "serial-quiet-headline",
            "seedleap.ai/storage-layout": hardware["storage"]["layout"],
            "seedleap.ai/input-pvc": hardware["storage"]["input_pvc"],
            "seedleap.ai/results-pvc": hardware["storage"]["results_pvc"],
            "seedleap.ai/results-pvc-access": hardware["storage"][
                "verified_results_access"
            ],
            "seedleap.ai/require-24fps": str(require_24fps).lower(),
            "seedleap.ai/candidate-evidence": str(candidate_evidence).lower(),
            "seedleap.ai/result-uri": (f"{RESULT_URI_ROOT}/{sku}/{mode}/{run_id}"),
        },
    }
    job["spec"] = {
        "backoffLimit": 0,
        "activeDeadlineSeconds": 14400,
        "template": {
            "metadata": {
                "labels": job["metadata"]["labels"],
                "annotations": {
                    "seedleap.ai/sglang-git-ref": sglang_git_ref,
                    "seedleap.ai/gpu-count": "1",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "terminationGracePeriodSeconds": 120,
                "securityContext": {"seLinuxOptions": {"type": "spc_t"}},
                "nodeSelector": copy.deepcopy(hardware["node_selector"]),
                "tolerations": (
                    [
                        {
                            "key": hardware["taint_key"],
                            "operator": "Equal",
                            "value": hardware["taint_value"],
                            "effect": "NoSchedule",
                        }
                    ]
                    if hardware["taint_key"] is not None
                    else []
                ),
                "containers": [],
                "volumes": [
                    {
                        "name": "profile-files",
                        "configMap": {"name": configmap_name, "defaultMode": 0o555},
                    },
                    *storage_volumes,
                    {"name": "work", "emptyDir": {"sizeLimit": "300Gi"}},
                    {
                        "name": "shm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
                    },
                ],
            },
        },
    }
    env = {
        **COMMON_ENV,
        "SGLANG_GIT_REF": sglang_git_ref,
        "MINWM_HARNESS_GIT_REF": harness_git_ref,
        "MINWM_HARNESS_REF_VERIFIED": str(harness_ref_verified).lower(),
        "MINWM_REQUIRE_24FPS": str(require_24fps).lower(),
        "MINWM_REQUIRE_CANDIDATE_EVIDENCE": str(candidate_evidence).lower(),
        "MINWM_RUNNER_SHA256": sha256_file(RUNNER),
        "MINWM_PROFILE_CLIENT_SHA256": sha256_file(CLIENT),
        "MINWM_COMMON_SHA256": sha256_file(COMMON),
        "MINWM_CASES_SHA256": sha256_file(CASES),
        "MINWM_RUN_ID": run_id,
        "MINWM_PROFILE_MODE": mode,
        "MINWM_GPU_SKU": hardware["gpu_sku"],
        "MINWM_HARDWARE_PROFILE": hardware["profile"],
        "MINWM_EXPECTED_COMPUTE_CAP": hardware["compute_cap"],
        "MINWM_EXPECTED_MIN_MEMORY_MIB": hardware["min_memory_mib"],
        "MINWM_EXPECTED_MAX_MEMORY_MIB": hardware["max_memory_mib"],
        "MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS": hardware["protocol_smoke_warmup_chunks"],
        "MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS": (
            "2" if mode in ("baseline", "fa3-quality") else "1"
        ),
        "MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE": hardware[
            "require_full_window_no_offload_smoke"
        ],
        "MINWM_STORAGE_LAYOUT": hardware["storage"]["layout"],
        "MINWM_INPUT_PVC": hardware["storage"]["input_pvc"],
        "MINWM_RESULTS_PVC": hardware["storage"]["results_pvc"],
        "MINWM_RESULTS_PVC_ACCESS": hardware["storage"]["verified_results_access"],
        "MINWM_RESULTS_ROOT": f"{RESULT_MOUNT_ROOT}/{sku}/{mode}",
    }
    harness_files = dict(BASE_HARNESS_FILES)
    if mode == "fa3-quality":
        harness_files.update(QUALITY_HARNESS_FILES)
        env.update(
            {
                "MINWM_QUALITY_CLIENT_SHA256": sha256_file(QUALITY_CLIENT),
                "MINWM_QUALITY_ANALYZER_SHA256": sha256_file(QUALITY_ANALYZER),
                "MINWM_QUALITY_ACTION_CASES_SHA256": sha256_file(QUALITY_ACTION_CASES),
                "MINWM_QUALITY_LONG_CASES_SHA256": sha256_file(QUALITY_LONG_CASES),
                "MINWM_FA2_REFERENCE_PATCH_SHA256": sha256_file(FA2_REFERENCE_PATCH),
            }
        )
    container = {
        "name": "benchmark",
        "image": BASE_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": "/work",
        "env": [
            {
                "name": "GITHUB_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "github-token", "key": "token"}},
            },
            {
                "name": "NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            },
            *[env_entry(name, value) for name, value in env.items()],
        ],
        "command": [
            "/bin/bash",
            "/opt/minwm-profile/run_single_gpu_taehv_24fps.sh",
        ],
        "resources": {
            "requests": {
                "cpu": "64",
                "memory": "320Gi",
                "ephemeral-storage": "120Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {
                "cpu": "64",
                "memory": "320Gi",
                "ephemeral-storage": "300Gi",
                "nvidia.com/gpu": "1",
            },
        },
        "volumeMounts": [
            {
                "name": "profile-files",
                "mountPath": "/opt/minwm-profile",
                "readOnly": True,
            },
            *storage_mounts,
            {"name": "work", "mountPath": "/work"},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    if mode == "nsys":
        container["securityContext"] = {"capabilities": {"add": ["SYS_ADMIN"]}}
    job["spec"]["template"]["spec"]["containers"] = [container]

    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": hardware["namespace"],
            "labels": job["metadata"]["labels"],
        },
        "data": {name: path.read_text() for name, path in harness_files.items()},
    }
    return configmap, job


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one reproducible single-GPU local-TAEHV MinWM job."
    )
    parser.add_argument(
        "--sku", action="append", choices=sorted(HARDWARE), required=True
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("baseline", "nsys", "fa3-quality"),
        required=True,
    )
    parser.add_argument(
        "--sglang-git-ref", default=DEFAULT_SGLANG_GIT_REF, metavar="COMMIT"
    )
    parser.add_argument("--harness-git-ref", required=True, metavar="COMMIT")
    parser.add_argument(
        "--allow-uncommitted-harness-for-dry-run",
        action="store_true",
        help=(
            "Render a manifest that runtime will refuse to execute. This is only "
            "for pre-commit Kubernetes dry-run validation."
        ),
    )
    parser.add_argument(
        "--require-24fps",
        action="store_true",
        help="Fail a baseline Job after recording results when client FPS is below 24.",
    )
    parser.add_argument(
        "--candidate-evidence",
        action="store_true",
        help=(
            "Require candidate-only all-layer, async-output, and action-marker "
            "evidence. Implied by --require-24fps and usable for NSYS Jobs."
        ),
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Unique lowercase suffix included in the Job name and S3 result prefix.",
    )
    parser.add_argument("--output-dir", type=Path, default=GENERATED)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.sglang_git_ref):
        parser.error("--sglang-git-ref must be a full 40-character lowercase commit")
    if not re.fullmatch(r"[0-9a-f]{40}", args.harness_git_ref):
        parser.error("--harness-git-ref must be a full 40-character lowercase commit")
    args.harness_ref_verified = harness_ref_matches_checkout(args.harness_git_ref)
    if not args.harness_ref_verified and not args.allow_uncommitted_harness_for_dry_run:
        parser.error(
            "--harness-git-ref does not contain the exact embedded harness files; "
            "commit them first or use the dry-run-only escape hatch"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", args.run_tag):
        parser.error("--run-tag must match [a-z0-9][a-z0-9.-]+")
    for sku in args.sku:
        for mode in args.mode:
            if len(f"minwm-taehv24-{sku}-{mode}-{args.run_tag}") > 63:
                parser.error(
                    "--run-tag makes the Kubernetes Job name exceed 63 characters"
                )
    if args.require_24fps and any(mode != "baseline" for mode in args.mode):
        parser.error("--require-24fps is only valid for baseline Jobs")
    if "fa3-quality" in args.mode:
        invalid_skus = sorted(set(args.sku) - {"h100", "h200"})
        if invalid_skus:
            parser.error(
                "--mode fa3-quality is only valid for Hopper H100/H200, got "
                + ", ".join(invalid_skus)
            )
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sku in dict.fromkeys(args.sku):
        for mode in dict.fromkeys(args.mode):
            documents = render(
                sku,
                mode,
                sglang_git_ref=args.sglang_git_ref,
                harness_git_ref=args.harness_git_ref,
                harness_ref_verified=args.harness_ref_verified,
                require_24fps=args.require_24fps,
                candidate_evidence=args.candidate_evidence,
                run_tag=args.run_tag,
            )
            output = args.output_dir / (
                f"minwm_single_gpu_taehv24_{sku}_{mode}_{args.run_tag}.yaml"
            )
            output.write_text(
                yaml.dump_all(
                    documents,
                    Dumper=LiteralDumper,
                    explicit_start=True,
                    sort_keys=False,
                    width=1000,
                )
            )
            print(output)


if __name__ == "__main__":
    main()
