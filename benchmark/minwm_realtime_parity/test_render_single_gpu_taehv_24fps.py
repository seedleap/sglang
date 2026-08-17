from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

K8S_DIR = Path(__file__).with_name("k8s")
sys.path.insert(0, str(K8S_DIR))

from render_single_gpu_taehv_24fps import (  # noqa: E402
    BASE_IMAGE,
    HARDWARE,
    MINWM_GIT_REF,
    parse_args,
    render,
)

SGLANG_GIT_REF = "54bdfea9cd52ac1cd79896e1a7275e18a0257b79"
HARNESS_GIT_REF = "ae1aa27b28ac805fd5c33243d7e0ddf2ab563cf0"
RUNNER_PATH = Path(__file__).with_name("run_single_gpu_taehv_24fps.sh")


def _env(container: dict) -> dict[str, str]:
    return {
        entry["name"]: entry["value"] for entry in container["env"] if "value" in entry
    }


def _nsys_sqlite_analyzer() -> str:
    runner = RUNNER_PATH.read_text()
    marker = (
        'python3 - "${sqlite}" "${nsys_dir}/metrics.json" \\\n'
        "    \"${nsys_dir}/sqlite-schema.json\" <<'PY'\n"
    )
    return runner.split(marker, 1)[1].split("\nPY\nfi", 1)[0]


def _write_nsys_sqlite_fixture(
    path: Path,
    *,
    include_action_ranges: bool = False,
    partial_first_range: bool,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE NVTX_EVENTS (
          start INTEGER NOT NULL,
          end INTEGER,
          eventType INTEGER NOT NULL,
          text TEXT,
          textId INTEGER
        );
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE ENUM_NSYS_EVENT_TYPE (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
          start INTEGER NOT NULL,
          end INTEGER NOT NULL,
          deviceId INTEGER NOT NULL
        );
        CREATE TABLE GPU_METRICS (
          timestamp INTEGER NOT NULL,
          typeId INTEGER NOT NULL,
          metricId INTEGER NOT NULL,
          value REAL NOT NULL
        );
        CREATE TABLE TARGET_INFO_GPU_METRICS (
          metricId INTEGER NOT NULL,
          metricName TEXT NOT NULL
        );
        """
    )
    cur.execute(
        "INSERT INTO StringIds VALUES (?,?)",
        (1, "stage_MinWMCausalDMDDenoisingStage"),
    )
    cur.execute(
        "INSERT INTO StringIds VALUES (?,?)",
        (2, "minwm_action_residual_prepare_once_per_chunk"),
    )
    cur.executemany(
        "INSERT INTO ENUM_NSYS_EVENT_TYPE VALUES (?,?)",
        (
            (35, "NvtxPushRange"),
            (36, "NvtxPopRange"),
            (37, "NvtxStartRange"),
            (38, "NvtxEndRange"),
            (59, "NvtxPushPopRange"),
            (60, "NvtxStartEndRange"),
        ),
    )
    cur.executemany(
        "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?)",
        ((3, "SM Active"), (5, "SM Issue"), (18, "Tensor Active"), (19, "DRAM")),
    )
    for index in range(16):
        start = 1_000 + index * 1_000
        event_type = 35 if partial_first_range and index == 0 else 59
        end = None if partial_first_range and index == 0 else start + 800
        cur.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
            (start, end, event_type, None, 1),
        )
        if include_action_ranges:
            cur.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                (start + 200, start + 300, 59, None, 2),
            )
        if end is not None:
            cur.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
                (start + 100, start + 500, 0),
            )
        for metric_id in (3, 5, 18, 19):
            cur.execute(
                "INSERT INTO GPU_METRICS VALUES (?,?,?,?)",
                (start + 200, 0, metric_id, 50.0 + metric_id),
            )
    con.commit()
    con.close()


@pytest.mark.parametrize("sku", ("b200", "b300"))
@pytest.mark.parametrize("mode", ("baseline", "nsys"))
def test_render_preserves_single_gpu_hardware_and_profile_contract(
    sku: str, mode: str
) -> None:
    run_tag = f"20260817-test-{sku}-{mode}"
    configmap, job = render(
        sku,
        mode,
        sglang_git_ref=SGLANG_GIT_REF,
        harness_git_ref=HARNESS_GIT_REF,
        run_tag=run_tag,
    )
    hardware = HARDWARE[sku]
    run_id = f"minwm-taehv24-{sku}-{mode}-{run_tag}"
    pod_spec = job["spec"]["template"]["spec"]
    assert len(pod_spec["containers"]) == 1
    container = pod_spec["containers"][0]
    environment = _env(container)

    assert job["metadata"]["name"] == run_id
    assert configmap["metadata"]["name"] == f"{run_id}-files"
    assert job["metadata"]["annotations"]["seedleap.ai/result-uri"].endswith(
        f"/{sku}/{mode}/{run_id}"
    )
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"

    node_selector = pod_spec["nodeSelector"]
    assert node_selector["karpenter.sh/capacity-type"] == "spot"
    assert node_selector["karpenter.sh/nodepool"] == hardware["nodepool"]
    assert (
        node_selector["node.kubernetes.io/instance-type"] == hardware["instance_type"]
    )
    if hardware["zone"] is None:
        assert "topology.kubernetes.io/zone" not in node_selector
    else:
        assert node_selector["topology.kubernetes.io/zone"] == hardware["zone"]

    expected_toleration = {
        "key": hardware["taint_key"],
        "operator": "Equal",
        "value": hardware["taint_value"],
        "effect": "NoSchedule",
    }
    assert expected_toleration in pod_spec["tolerations"]
    if sku == "b300":
        assert "seedleap.ai/capacity-pool" not in node_selector
    else:
        assert node_selector["seedleap.ai/capacity-pool"] == hardware["capacity_label"]

    if mode == "nsys":
        assert container["securityContext"] == {"capabilities": {"add": ["SYS_ADMIN"]}}
    else:
        assert "securityContext" not in container

    assert (
        job["metadata"]["annotations"]["seedleap.ai/sglang-git-ref"] == SGLANG_GIT_REF
    )
    assert environment["SGLANG_GIT_REF"] == SGLANG_GIT_REF
    assert environment["MINWM_HARNESS_GIT_REF"] == HARNESS_GIT_REF
    assert environment["MINWM_HARNESS_REF_VERIFIED"] == "true"
    assert environment["MINWM_REQUIRE_24FPS"] == "false"
    assert environment["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "false"
    assert environment["MINWM_BASE_IMAGE"] == BASE_IMAGE
    assert environment["MINWM_GIT_REF"] == MINWM_GIT_REF
    assert len(environment["MINWM_GIT_REF"]) == 40
    for name in (
        "MINWM_RUNNER_SHA256",
        "MINWM_PROFILE_CLIENT_SHA256",
        "MINWM_COMMON_SHA256",
        "MINWM_CASES_SHA256",
    ):
        assert len(environment[name]) == 64
    assert environment["MINWM_CHECKPOINT_SOURCE_VERSION"]
    assert len(environment["MINWM_CHECKPOINT_SHA256"]) == 64
    assert environment["MINWM_FIRST_FRAME_SOURCE_URI"].endswith(
        "/world-model/eval/platform/eval_sets/minWM/testset100_v2/img/p02.png"
    )
    assert (
        environment["MINWM_FIRST_FRAME_SOURCE_VERSION"]
        == "5q2pfK_Cqr48ufR6Ksl_6gu2qnSwwLVn"
    )
    assert environment["MINWM_FIRST_FRAME_SOURCE_BYTES"] == "1878806"
    assert len(environment["MINWM_FIRST_FRAME_SOURCE_SHA256"]) == 64
    assert len(environment["TAEHV_REVISION"]) == 40
    assert len(environment["TAEHV_CHECKPOINT_SHA256"]) == 64
    assert environment["MINWM_PROFILE_MODE"] == mode
    assert environment["MINWM_RUN_ID"] == run_id
    runner = configmap["data"]["run_single_gpu_taehv_24fps.sh"]
    assert set(configmap["data"]) == {
        "benchmark_realtime_throughput.py",
        "cases_720p_compile_smoke.json",
        "common.py",
        "run_single_gpu_taehv_24fps.sh",
    }
    assert 'readonly CASES="${PROFILE_DIR}/cases_720p_compile_smoke.json"' in runner
    assert 'export PYTHONPATH="${PROFILE_DIR}:${REPO_ROOT}/python"' in runner
    assert 'export MINWM_S3_MOUNT="${MINWM_INPUT_ROOT}"' in runner
    assert "first-frame-source.json" in runner
    assert 'uri = os.environ["MINWM_FIRST_FRAME_SOURCE_URI"]' in runner
    assert 'Path(os.environ["MINWM_S3_MOUNT"]) / key' in runner
    assert "assert mounted_source.is_file()" in runner
    assert "pip freeze --all" in runner
    assert "PIP_CHECK_STATUS" in runner
    assert (
        '"received_payload_chunks"'
        in configmap["data"]["benchmark_realtime_throughput.py"]
    )
    assert (
        '"received_server_timing_chunks"'
        in configmap["data"]["benchmark_realtime_throughput.py"]
    )
    assert 'assert result["received_payload_chunks"] == expected_total' in runner
    assert 'assert result["received_server_timing_chunks"] == expected_total' in runner
    assert '"protocol_smoke_in_headline": False' in runner
    assert '"protocol_smoke_warmup_chunks": 1' in runner
    assert "--realtime-session-idle-timeout-s 900" in runner
    assert "--realtime-session-max-lifetime-s 900" in runner
    assert "export SGLANG_REALTIME_TRACE_SYNC_CUDA=0" in runner
    assert "export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0" in runner
    assert "PERFORMANCE_PASS" in runner and "PERFORMANCE_FAIL" in runner
    assert "trap 'finish $?' EXIT" in runner
    assert "trap 'finish 130' INT" in runner
    assert "trap 'finish 143' TERM" in runner
    assert "trap - EXIT INT TERM" in runner
    assert 'kill -KILL "${server_pid}"' in runner
    assert runner.count("timeout --kill-after=5s 20s") == 2
    assert 'result["server"]["raw_frame_async_enqueue_ms"]' in runner
    assert 'structured_marker = "MINWM_RUNTIME_ALIGNMENT_JSON "' in runner
    assert '"cache_tokens": "27456"' in runner
    assert '"sink_tokens": "6864"' in runner
    if mode == "baseline":
        assert '--output "${baseline_dir}/protocol-smoke.json"' in runner
        assert (
            'assert_profile_result "${baseline_dir}/protocol-smoke.json" 1 2' in runner
        )
        assert (
            'assert_profile_result "${baseline_dir}/throughput.json" 20 200' in runner
        )
    else:
        assert '--output "${nsys_dir}/protocol-smoke.json"' in runner
        assert 'assert_profile_result "${nsys_dir}/protocol-smoke.json" 1 1' in runner
        assert 'assert_profile_result "${nsys_dir}/throughput.json" 8 8' in runner
        assert '"capture_chunk_count": 16' in runner
        assert 'stage_name = "stage_MinWMCausalDMDDenoisingStage"' in runner
        assert "LEFT JOIN StringIds AS s ON s.id=n.textId" in runner
        assert '"complete": 16' in runner
        assert "stage_kernel_counts" in runner
        assert '"${nsys_dir}/sqlite-schema.json"' in runner
        assert "wait_for_chunk" not in runner
        assert runner.index("nsys start \\") < runner.index(
            '--profile-name "${MINWM_GPU_SKU,,}-local-taehv-main-segment-nsys"'
        )
        assert "run_sglang_api.py" not in runner


def test_render_makes_candidate_names_and_result_prefixes_unique() -> None:
    _, candidate_a = render(
        "b200",
        "baseline",
        sglang_git_ref=SGLANG_GIT_REF,
        harness_git_ref=HARNESS_GIT_REF,
        run_tag="20260817-candidate-a",
    )
    _, candidate_b = render(
        "b200",
        "baseline",
        sglang_git_ref=SGLANG_GIT_REF,
        harness_git_ref=HARNESS_GIT_REF,
        run_tag="20260817-candidate-b",
    )

    assert candidate_a["metadata"]["name"] != candidate_b["metadata"]["name"]
    assert (
        candidate_a["metadata"]["annotations"]["seedleap.ai/result-uri"]
        != candidate_b["metadata"]["annotations"]["seedleap.ai/result-uri"]
    )


@pytest.mark.parametrize(
    "extra_args",
    (
        ["--sglang-git-ref", "not-a-full-commit"],
        ["--harness-git-ref", "not-a-full-commit"],
        ["--run-tag", "x" * 64],
    ),
)
def test_parse_args_rejects_nonimmutable_or_oversized_identity(
    extra_args: list[str],
) -> None:
    args = [
        "--sku",
        "b200",
        "--mode",
        "baseline",
        "--run-tag",
        "20260817-a1",
        "--sglang-git-ref",
        SGLANG_GIT_REF,
        "--harness-git-ref",
        HARNESS_GIT_REF,
        "--allow-uncommitted-harness-for-dry-run",
    ]
    key = extra_args[0]
    index = args.index(key)
    args[index : index + 2] = extra_args

    with pytest.raises(SystemExit):
        parse_args(args)


def test_candidate_request_harness_is_fixed_while_server_commit_changes() -> None:
    candidate_ref = "ae1aa27b28ac805fd5c33243d7e0ddf2ab563cf0"
    main_cm, main_job = render(
        "b200",
        "baseline",
        sglang_git_ref=SGLANG_GIT_REF,
        harness_git_ref=HARNESS_GIT_REF,
        run_tag="20260817-main",
    )
    candidate_cm, candidate_job = render(
        "b200",
        "baseline",
        sglang_git_ref=candidate_ref,
        harness_git_ref=HARNESS_GIT_REF,
        require_24fps=True,
        run_tag="20260817-candidate",
    )

    assert main_cm["data"] == candidate_cm["data"]
    main_env = _env(main_job["spec"]["template"]["spec"]["containers"][0])
    candidate_env = _env(candidate_job["spec"]["template"]["spec"]["containers"][0])
    fixed_names = {
        name
        for name in main_env
        if name
        not in {
            "SGLANG_GIT_REF",
            "MINWM_RUN_ID",
            "MINWM_REQUIRE_24FPS",
            "MINWM_REQUIRE_CANDIDATE_EVIDENCE",
        }
    }
    assert {name: main_env[name] for name in fixed_names} == {
        name: candidate_env[name] for name in fixed_names
    }
    assert main_env["SGLANG_GIT_REF"] == SGLANG_GIT_REF
    assert candidate_env["SGLANG_GIT_REF"] == candidate_ref
    assert main_env["MINWM_REQUIRE_24FPS"] == "false"
    assert candidate_env["MINWM_REQUIRE_24FPS"] == "true"
    assert main_env["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "false"
    assert candidate_env["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true"


def test_nsys_candidate_can_require_candidate_evidence_without_headline_gate() -> None:
    _, job = render(
        "b200",
        "nsys",
        sglang_git_ref="b" * 40,
        harness_git_ref=HARNESS_GIT_REF,
        candidate_evidence=True,
        run_tag="20260817-candidate-nsys",
    )

    environment = _env(job["spec"]["template"]["spec"]["containers"][0])
    assert environment["MINWM_REQUIRE_24FPS"] == "false"
    assert environment["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true"
    assert job["metadata"]["annotations"]["seedleap.ai/candidate-evidence"] == "true"


def test_parse_args_allows_candidate_evidence_for_nsys() -> None:
    args = parse_args(
        [
            "--sku",
            "b200",
            "--mode",
            "nsys",
            "--run-tag",
            "20260817-candidate-nsys",
            "--sglang-git-ref",
            "b" * 40,
            "--harness-git-ref",
            HARNESS_GIT_REF,
            "--candidate-evidence",
            "--allow-uncommitted-harness-for-dry-run",
        ]
    )

    assert args.candidate_evidence is True
    assert args.require_24fps is False


def test_nsys_sqlite_analyzer_resolves_registered_nvtx_stage_names(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    metrics_path = tmp_path / "metrics.json"
    schema_path = tmp_path / "schema.json"
    _write_nsys_sqlite_fixture(sqlite_path, partial_first_range=False)

    result = subprocess.run(
        [sys.executable, "-", str(sqlite_path), str(metrics_path), str(schema_path)],
        input=_nsys_sqlite_analyzer(),
        check=False,
        capture_output=True,
        env={**os.environ, "MINWM_REQUIRE_CANDIDATE_EVIDENCE": "false"},
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(metrics_path.read_text())
    assert metrics["denoise_stage_name"] == "stage_MinWMCausalDMDDenoisingStage"
    assert metrics["denoise_stage_range_status_counts"] == {
        "complete": 16,
        "invalid": 0,
        "partial_left": 0,
        "partial_right": 0,
    }
    assert len(metrics["denoise_stage_ranges"]) == 16
    assert all(
        stage["fully_contained_kernel_count"] == 1
        for stage in metrics["denoise_stage_ranges"]
    )
    assert json.loads(schema_path.read_text())["columns"]["NVTX_EVENTS"] == [
        "start",
        "end",
        "eventType",
        "text",
        "textId",
    ]


def test_nsys_sqlite_analyzer_rejects_partial_target_stage_range(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "partial.sqlite"
    metrics_path = tmp_path / "metrics.json"
    schema_path = tmp_path / "schema.json"
    _write_nsys_sqlite_fixture(sqlite_path, partial_first_range=True)

    result = subprocess.run(
        [sys.executable, "-", str(sqlite_path), str(metrics_path), str(schema_path)],
        input=_nsys_sqlite_analyzer(),
        check=False,
        capture_output=True,
        env={**os.environ, "MINWM_REQUIRE_CANDIDATE_EVIDENCE": "false"},
        text=True,
    )

    assert result.returncode != 0
    assert "partial_right" in result.stderr
    assert schema_path.is_file()


def test_nsys_sqlite_analyzer_requires_nested_action_marker_for_candidate(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "candidate.sqlite"
    metrics_path = tmp_path / "metrics.json"
    schema_path = tmp_path / "schema.json"
    _write_nsys_sqlite_fixture(
        sqlite_path,
        include_action_ranges=True,
        partial_first_range=False,
    )

    result = subprocess.run(
        [sys.executable, "-", str(sqlite_path), str(metrics_path), str(schema_path)],
        input=_nsys_sqlite_analyzer(),
        check=False,
        capture_output=True,
        env={**os.environ, "MINWM_REQUIRE_CANDIDATE_EVIDENCE": "true"},
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(metrics_path.read_text())
    action = metrics["action_prepare_once_per_chunk"]
    assert action["range_status_counts"]["complete"] == 16
    assert len(action["ranges"]) == 16
