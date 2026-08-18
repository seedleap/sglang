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
    storage_spec,
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


def _copy_tree_contents_function() -> str:
    runner = RUNNER_PATH.read_text()
    start = runner.index("copy_tree_contents() {")
    end = runner.index("\n}\n\narchive_results()", start) + len("\n}")
    return runner[start:end]


def _no_offload_protocol_fit_functions() -> str:
    runner = RUNNER_PATH.read_text()
    start = runner.index("record_no_offload_protocol_fit() {")
    end = runner.index("\n\nclone_at()", start)
    return runner[start:end]


def _run_copy_tree_contents(
    tmp_path: Path,
    source: Path,
    destination: Path,
    *,
    command_override: str = "",
) -> subprocess.CompletedProcess[str]:
    local_root = tmp_path / "local-root"
    local_root.mkdir(exist_ok=True)
    cp_compat = ""
    if sys.platform == "darwin":
        cp_compat = """
cp() {
  local args=()
  local arg=""
  for arg in "$@"; do
    if [[ "${arg}" != "--no-preserve=all" ]]; then
      args+=("${arg}")
    fi
  done
  command cp "${args[@]}"
}
"""
    script = f"""
set -u
LOCAL_ROOT="$1"
{_copy_tree_contents_function()}
{cp_compat}
{command_override}
set +e
copy_tree_contents "$2" "$3"
status=$?
exit "${{status}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            script,
            "copy-tree-test",
            str(local_root),
            str(source),
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_copy_tree_contents_copies_nested_and_hidden_regular_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    destination.mkdir()
    (source / "nested" / "payload").write_bytes(b"nested-payload")
    (source / ".hidden").write_bytes(b"hidden-payload")

    result = _run_copy_tree_contents(tmp_path, source, destination)

    assert result.returncode == 0, result.stderr
    assert (destination / "nested" / "payload").read_bytes() == b"nested-payload"
    assert (destination / ".hidden").read_bytes() == b"hidden-payload"


def test_copy_tree_contents_rejects_special_paths_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_bytes(b"payload")
    (source / "link").symlink_to(source / "payload")
    os.mkfifo(source / "fifo")

    result = _run_copy_tree_contents(tmp_path, source, destination)

    assert result.returncode != 0
    assert "unsupported result path type" in result.stderr
    assert list(destination.iterdir()) == []


def test_copy_tree_contents_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (destination / "nested").mkdir(parents=True)
    (source / "nested" / "payload").write_bytes(b"new")
    target = destination / "nested" / "payload"
    target.write_bytes(b"old")

    result = _run_copy_tree_contents(tmp_path, source, destination)

    assert result.returncode != 0
    assert "refusing to overwrite result path" in result.stderr
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize(
    "command_override",
    (
        "find() { return 23; }",
        'mkdir() { if [[ "$*" == *nested* ]]; then return 17; fi; command mkdir "$@"; }',
        "cp() { return 19; }",
    ),
)
def test_copy_tree_contents_propagates_command_failures_with_errexit_disabled(
    tmp_path: Path,
    command_override: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_bytes(b"payload")
    (source / "nested").mkdir()

    result = _run_copy_tree_contents(
        tmp_path,
        source,
        destination,
        command_override=command_override,
    )

    assert result.returncode != 0


@pytest.mark.parametrize(("warmup_chunks", "passes"), ((8, True), (7, False)))
def test_hopper_no_offload_fit_requires_a_full_cache_window(
    tmp_path: Path, warmup_chunks: int, passes: bool
) -> None:
    result_path = tmp_path / "protocol-smoke.json"
    log_path = tmp_path / "server.log"
    gpu_path = tmp_path / "gpu.csv"
    marker_path = tmp_path / "NO_OFFLOAD_PROTOCOL_FIT_PASS.json"
    result_path.write_text(
        json.dumps(
            {
                "comparison_contract": {"latent_frames_per_chunk": 4},
                "warmup_chunks": warmup_chunks,
                "measured_chunks": 2,
                "received_payload_chunks": warmup_chunks + 2,
                "received_server_timing_chunks": warmup_chunks + 2,
            }
        )
    )
    log_path.write_text(
        'server_args: {"vae_cpu_offload": false, '
        '"realtime_vae_backend": "local"}\n'
        "Preloading TAEHV decoder weights\n"
    )
    gpu_path.write_text("NVIDIA H100 80GB HBM3, 81559, 9.0\n")
    script = f"""
set -euo pipefail
server_pid=$$
LOCAL_RESULTS="$1"
{_no_offload_protocol_fit_functions()}
record_no_offload_protocol_fit "$2" "$3" "$4"
assert_no_offload_protocol_fit "$4"
"""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "no-offload-fit-test",
            str(tmp_path),
            str(result_path),
            str(log_path),
            str(marker_path),
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "MINWM_EXPECTED_COMPUTE_CAP": "9.0",
            "MINWM_EXPECTED_MAX_MEMORY_MIB": "90000",
            "MINWM_EXPECTED_MIN_MEMORY_MIB": "80000",
            "MINWM_GPU_SKU": "H100",
            "MINWM_EXPECTED_GPU_COUNT": "1",
            "MINWM_VAE_TOPOLOGY": "local",
            "MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS": "2",
            "MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS": str(warmup_chunks),
            "MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE": "true",
        },
        text=True,
    )

    assert (completed.returncode == 0) is passes, completed.stderr
    assert marker_path.is_file() is passes
    if passes:
        evidence = json.loads(marker_path.read_text())
        assert evidence["warmup_latent_frames"] == 32
        assert evidence["no_offload_protocol_fit_pass"] is True


def _write_nsys_sqlite_fixture(
    path: Path,
    *,
    include_action_ranges: bool = False,
    partial_first_range: bool,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
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
        """)
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


@pytest.mark.parametrize("sku", ("b200", "b300", "h100", "h200"))
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
    assert (
        job["metadata"]["annotations"]["seedleap.ai/cluster-context"]
        == hardware["context"]
    )
    assert job["metadata"]["annotations"]["seedleap.ai/storage-layout"] == "shared"
    assert job["metadata"]["annotations"]["seedleap.ai/input-pvc"] == "s3-claim"
    assert job["metadata"]["annotations"]["seedleap.ai/results-pvc"] == "s3-claim"
    assert job["metadata"]["annotations"]["seedleap.ai/results-pvc-access"] == "RWX"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    s3_volumes = [
        volume for volume in pod_spec["volumes"] if "persistentVolumeClaim" in volume
    ]
    assert s3_volumes == [
        {
            "name": "s3-shared",
            "persistentVolumeClaim": {"claimName": "s3-claim"},
        }
    ]
    s3_mounts = [
        mount for mount in container["volumeMounts"] if mount["name"] == "s3-shared"
    ]
    assert s3_mounts == [
        {"name": "s3-shared", "mountPath": "/s3-input", "readOnly": True},
        {"name": "s3-shared", "mountPath": "/s3-results"},
    ]

    node_selector = pod_spec["nodeSelector"]
    assert node_selector == hardware["node_selector"]

    if hardware["taint_key"] is None:
        assert pod_spec["tolerations"] == []
    else:
        expected_toleration = {
            "key": hardware["taint_key"],
            "operator": "Equal",
            "value": hardware["taint_value"],
            "effect": "NoSchedule",
        }
        assert expected_toleration in pod_spec["tolerations"]
    if sku == "b200":
        assert node_selector["seedleap.ai/capacity-pool"] == "minwm-test-b200-karpenter"
    elif sku in {"b300", "h100", "h200"}:
        assert node_selector["eks.amazonaws.com/capacityType"] == "SPOT"
        assert node_selector["eks.amazonaws.com/nodegroup"] == hardware["nodepool"]
    if sku in {"b200", "b300"}:
        assert hardware["max_memory_mib"] == ""
    else:
        expected = {
            "h100": {
                "profile": "experimental-sm90-h100-no-offload",
                "min_memory_mib": "80000",
                "max_memory_mib": "90000",
                "instance_type": "p5.48xlarge",
                "nodepool": "minwm-spot-p5-h100-sglang-0718",
            },
            "h200": {
                "profile": "experimental-sm90-h200-no-offload",
                "min_memory_mib": "140000",
                "max_memory_mib": "150000",
                "instance_type": "p5en.48xlarge",
                "nodepool": "minwm-spot-p5en-h200-sglang-0718",
            },
        }[sku]
        assert hardware["context"] == "aws03-usw2"
        assert hardware["namespace"] == "default"
        assert hardware["profile"] == expected["profile"]
        assert hardware["compute_cap"] == "9.0"
        assert hardware["min_memory_mib"] == expected["min_memory_mib"]
        assert hardware["max_memory_mib"] == expected["max_memory_mib"]
        assert hardware["protocol_smoke_warmup_chunks"] == "8"
        assert hardware["require_full_window_no_offload_smoke"] == "true"
        assert node_selector == {
            "eks.amazonaws.com/capacityType": "SPOT",
            "eks.amazonaws.com/nodegroup": expected["nodepool"],
            "node.kubernetes.io/instance-type": expected["instance_type"],
            "seedleap.ai/workload": "wan22-ti2v",
        }
        assert (
            job["metadata"]["annotations"]["seedleap.ai/no-offload-protocol-gate"]
            == "full-window"
        )

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
    assert environment["MINWM_EXPECTED_COMPUTE_CAP"] == hardware["compute_cap"]
    assert environment["MINWM_EXPECTED_MIN_MEMORY_MIB"] == hardware["min_memory_mib"]
    assert environment["MINWM_EXPECTED_MAX_MEMORY_MIB"] == hardware["max_memory_mib"]
    assert (
        environment["MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS"]
        == hardware["protocol_smoke_warmup_chunks"]
    )
    assert environment["MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS"] == (
        "2" if mode == "baseline" else "1"
    )
    assert (
        environment["MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE"]
        == hardware["require_full_window_no_offload_smoke"]
    )
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
    assert environment["MINWM_STORAGE_LAYOUT"] == "shared"
    assert environment["MINWM_INPUT_PVC"] == "s3-claim"
    assert environment["MINWM_RESULTS_PVC"] == "s3-claim"
    assert environment["MINWM_RESULTS_PVC_ACCESS"] == "RWX"
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
    assert '"${REMOTE_RESULTS}/STORAGE_WRITE_PROBE"' in runner
    assert '"${LOCAL_ROOT}/STORAGE_WRITE_PROBE"' in runner
    assert '"${LOCAL_RESULTS}/STORAGE_WRITE_PROBE"' not in runner
    assert 'cmp --silent "${LOCAL_ROOT}/STORAGE_WRITE_PROBE"' in runner
    assert runner.index('cp "${LOCAL_ROOT}/STORAGE_WRITE_PROBE"') < runner.index(
        "git clone"
    )
    assert '"${LOCAL_ROOT}/ARCHIVE_COPY_PROBE"' in runner
    assert '"${LOCAL_RESULTS}/ARCHIVE_COPY_PROBE"' not in runner
    assert (
        'copy_tree_contents "${LOCAL_ARCHIVE_COPY_PROBE}" '
        '"${REMOTE_ARCHIVE_COPY_PROBE}"' in runner
    )
    assert runner.index('copy_tree_contents "${LOCAL_ARCHIVE_COPY_PROBE}"') < (
        runner.index("git clone")
    )
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
    assert 'os.environ["MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS"]' in runner
    assert 'os.environ["MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS"]' in runner
    assert "gpu_memory_mib <= MINWM_EXPECTED_MAX_MEMORY_MIB" in runner
    assert "NO_OFFLOAD_PROTOCOL_FIT_PASS.json" in runner
    assert 'assert evidence["no_offload_protocol_fit_pass"] is True' in runner
    assert 'assert evidence["formal_same_process_eligible"] is True' in runner
    assert 'assert evidence["warmup_latent_frames"] >= 32' in runner
    assert runner.index("record_no_offload_protocol_fit \\") < runner.index(
        '--profile-name "${MINWM_GPU_SKU,,}-${MINWM_VAE_TOPOLOGY}-taehv-main-segment"'
    )
    fit_assertion = 'assert_no_offload_protocol_fit "${no_offload_fit_marker}"'
    assert runner.count(fit_assertion) == 2
    if mode == "baseline":
        assert runner.index(
            fit_assertion, runner.index('if [[ "${MINWM_PROFILE_MODE}"')
        ) < runner.index(
            '--profile-name "${MINWM_GPU_SKU,,}-${MINWM_VAE_TOPOLOGY}-taehv-main-segment"'
        )
    else:
        nsys_branch = runner.index("else\n  readonly NSYS_URL=")
        assert runner.index(fit_assertion, nsys_branch) < runner.index(
            '--profile-name "${MINWM_GPU_SKU,,}-local-taehv-main-segment-nsys"'
        )
    assert "--realtime-session-idle-timeout-s 900" in runner
    assert "--realtime-session-max-lifetime-s 900" in runner
    assert "export SGLANG_REALTIME_TRACE_SYNC_CUDA=0" in runner
    assert "export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0" in runner
    assert "PERFORMANCE_PASS" in runner and "PERFORMANCE_FAIL" in runner
    assert '> "${LOCAL_RESULTS}/RUN_COMPLETE"' in runner
    assert "cp -R --no-preserve=all" not in runner
    assert 'find "${source}" -mindepth 1 -print0 > "${entry_list}"' in runner
    assert 'entry_list="$(mktemp "${LOCAL_ROOT}/copy-tree.XXXXXX")"' in runner
    assert "while IFS= read -r -d '' path; do" in runner
    assert 'mkdir -p -- "${target}"' in runner
    assert '( -e "${target}" || -L "${target}" )' in runner
    assert 'cp --no-preserve=all -- "${path}" "${target}"' in runner
    assert 'cp -a "${LOCAL_RESULTS}/." "${REMOTE_RESULTS}/"' not in runner
    assert "archive_attempted=0" in runner
    assert "archive_attempted=1" in runner
    assert "if (( archive_attempted == 0 )); then" in runner
    assert runner.index("archive_attempted=1") < runner.index(
        'copy_tree_contents "${LOCAL_RESULTS}" "${REMOTE_RESULTS}"'
    )
    assert runner.count("archive_results") == 3
    assert '"${LOCAL_RESULTS}/FAILED"' not in runner
    assert 'readonly LOCAL_FAILED="${LOCAL_ROOT}/FAILED"' in runner
    assert 'cp "${LOCAL_FAILED}" "${REMOTE_RESULTS}/FAILED" 2>/dev/null' in runner
    assert 'cp "${LOCAL_RESULTS}/RUN_COMPLETE" "${REMOTE_RESULTS}/SUCCESS"' in runner
    assert 'touch "${REMOTE_RESULTS}/SUCCESS"' not in runner
    completion_log = runner.rindex("MINWM_SINGLE_GPU_TAEHV24_COMPLETE")
    success_publish = runner.rindex(
        'cp "${LOCAL_RESULTS}/RUN_COMPLETE" "${REMOTE_RESULTS}/SUCCESS"'
    )
    disable_traps = runner.rindex("trap - EXIT INT TERM")
    assert completion_log < success_publish < disable_traps
    assert "trap 'finish $?' EXIT" in runner
    assert "trap 'finish 130' INT" in runner
    assert "trap 'finish 143' TERM" in runner
    assert "trap - EXIT INT TERM" in runner
    assert 'kill -KILL "${server_pid}"' in runner
    assert runner.count("timeout --kill-after=5s 20s") == 2
    assert 'result["server"]["raw_frame_async_enqueue_ms"]' in runner
    assert 'vae_decode = result["server"]["model_vae_decode_ms"]' in runner
    assert 'os.environ["MINWM_VAE_TOPOLOGY"] == "local"' in runner
    assert 'vae_decode["sample_count"] == 0' in runner
    assert 'vae_decode["missing_count"] == expected_measured' in runner
    assert '"model_vae_encode_ms",' in runner
    assert 'model_vae_encode_ms"]["sample_count"] == 0' not in runner
    assert 'structured_marker = "MINWM_RUNTIME_ALIGNMENT_JSON "' in runner
    assert '"cache_tokens": "27456"' in runner
    assert '"sink_tokens": "6864"' in runner
    if mode == "baseline":
        assert '--output "${baseline_dir}/protocol-smoke.json"' in runner
        assert (
            '"${baseline_dir}/protocol-smoke.json" \\\n'
            '    "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \\\n'
            '    "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}"' in runner
        )
        assert (
            'assert_profile_result "${baseline_dir}/throughput.json" 20 200' in runner
        )
    else:
        assert '--output "${nsys_dir}/protocol-smoke.json"' in runner
        assert (
            '"${nsys_dir}/protocol-smoke.json" \\\n'
            '    "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \\\n'
            '    "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}"' in runner
        )
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


def test_storage_spec_supports_explicit_split_rw_s3_claims() -> None:
    volumes, mounts = storage_spec(
        {
            "layout": "split",
            "input_pvc": "input-s3",
            "results_pvc": "results-s3",
            "verified_results_access": "RWX",
        }
    )
    assert [volume["persistentVolumeClaim"]["claimName"] for volume in volumes] == [
        "input-s3",
        "results-s3",
    ]
    assert mounts == [
        {"name": "s3-input", "mountPath": "/s3-input", "readOnly": True},
        {"name": "s3-results", "mountPath": "/s3-results"},
    ]


@pytest.mark.parametrize("sku", ("b200", "b300", "h100", "h200"))
def test_render_remote_taehv_uses_two_same_sku_gpus(sku: str) -> None:
    configmap, job = render(
        sku,
        "baseline",
        topology="remote",
        sglang_git_ref=SGLANG_GIT_REF,
        harness_git_ref=HARNESS_GIT_REF,
        run_tag=f"20260818-{sku}-remote",
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    environment = _env(container)
    runner = configmap["data"]["run_single_gpu_taehv_24fps.sh"]

    assert job["metadata"]["labels"]["seedleap.ai/vae-topology"] == "remote"
    assert job["metadata"]["annotations"]["seedleap.ai/result-uri"].endswith(
        f"/{sku}/remote/baseline/{job['metadata']['name']}"
    )
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert environment["MINWM_VAE_TOPOLOGY"] == "remote"
    assert environment["MINWM_EXPECTED_GPU_COUNT"] == "2"
    assert environment["MINWM_RESULTS_ROOT"].endswith(f"/{sku}/remote/baseline")
    assert "CUDA_VISIBLE_DEVICES=1 python3 -m" in runner
    assert "--realtime-vae-backend taehv_remote" in runner
    assert "--realtime-vae-transport shared_memory" in runner
    assert 'CUDA_VISIBLE_DEVICES=0 "${server_command[@]}"' in runner


def test_render_rejects_remote_nsys() -> None:
    with pytest.raises(ValueError, match="baseline mode only"):
        render(
            "b300",
            "nsys",
            topology="remote",
            sglang_git_ref=SGLANG_GIT_REF,
            harness_git_ref=HARNESS_GIT_REF,
            run_tag="20260818-remote-nsys",
        )


@pytest.mark.parametrize(
    "storage,match",
    (
        (
            {
                "layout": "shared",
                "input_pvc": "input-s3",
                "results_pvc": "results-s3",
                "verified_results_access": "RWX",
            },
            "identical",
        ),
        (
            {
                "layout": "split",
                "input_pvc": "input-s3",
                "results_pvc": None,
                "verified_results_access": "RWX",
            },
            "explicit results_pvc",
        ),
        (
            {
                "layout": "split",
                "input_pvc": "input-s3",
                "results_pvc": "results-ebs",
                "verified_results_access": "RWO",
            },
            "verified RWX",
        ),
    ),
)
def test_storage_spec_fails_closed(storage: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        storage_spec(storage)


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
