#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_PROFILE_MODE:?set MINWM_PROFILE_MODE to baseline, nsys, or fa3-quality}"
: "${MINWM_GPU_SKU:?set MINWM_GPU_SKU to B200, B300, H100, or H200}"
: "${MINWM_HARDWARE_PROFILE:?set the experimental hardware profile}"
: "${MINWM_EXPECTED_COMPUTE_CAP:?set expected compute capability}"
: "${MINWM_EXPECTED_MIN_MEMORY_MIB:?set expected minimum visible memory}"
: "${MINWM_EXPECTED_MAX_MEMORY_MIB:=}"
: "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS:?set protocol-smoke warmup chunks}"
: "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS:?set protocol-smoke measured chunks}"
: "${MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE:?set full-window fit gate}"
: "${MINWM_BASE_IMAGE:?set the immutable base image digest}"
: "${SGLANG_GIT_REF:?set the immutable SGLang commit}"
: "${MINWM_GIT_REF:?set the MinWM checkpoint provenance commit}"
: "${MINWM_HARNESS_GIT_REF:?set the immutable profiling harness commit}"
: "${MINWM_HARNESS_REF_VERIFIED:?set whether the harness commit was verified}"
: "${MINWM_REQUIRE_24FPS:?set whether 24 FPS is a completion gate}"
: "${MINWM_REQUIRE_CANDIDATE_EVIDENCE:?set candidate evidence gate}"
: "${MINWM_RUNNER_SHA256:?set the profiling runner SHA-256}"
: "${MINWM_PROFILE_CLIENT_SHA256:?set the profiling client SHA-256}"
: "${MINWM_COMMON_SHA256:?set the fixed common.py SHA-256}"
: "${MINWM_CASES_SHA256:?set the fixed cases manifest SHA-256}"
: "${MINWM_INPUT_ROOT:?set the read-only S3 CSI input mount root}"
: "${MINWM_RESULTS_ROOT:?set the unique writable S3 CSI result prefix}"
: "${MINWM_STORAGE_LAYOUT:?set the verified storage layout}"
: "${MINWM_INPUT_PVC:?set the input PVC name}"
: "${MINWM_RESULTS_PVC:?set the results PVC name}"
: "${MINWM_RESULTS_PVC_ACCESS:?set the verified results PVC access mode}"
: "${MINWM_CHECKPOINT_RELATIVE_PATH:?set checkpoint path below MINWM_INPUT_ROOT}"
: "${MINWM_PRETRAINED_RELATIVE_PATH:?set donor path below MINWM_INPUT_ROOT}"
: "${MINWM_CHECKPOINT_SOURCE_URI:?set immutable source URI}"
: "${MINWM_CHECKPOINT_SOURCE_VERSION:?set immutable source version}"
: "${MINWM_CHECKPOINT_SOURCE_ETAG:?set immutable source ETag}"
: "${MINWM_CHECKPOINT_STAGED_VERSION:?set immutable staged version}"
: "${MINWM_CHECKPOINT_BYTES:?set checkpoint byte size}"
: "${MINWM_CHECKPOINT_SHA256:?set checkpoint SHA-256}"
: "${MINWM_CHECKPOINT_CRC64:?set checkpoint CRC64NVME}"
: "${MINWM_FIRST_FRAME_SOURCE_URI:?set immutable first-frame source URI}"
: "${MINWM_FIRST_FRAME_SOURCE_VERSION:?set immutable first-frame VersionId}"
: "${MINWM_FIRST_FRAME_SOURCE_ETAG:?set immutable first-frame ETag}"
: "${MINWM_FIRST_FRAME_SOURCE_BYTES:?set immutable first-frame byte size}"
: "${MINWM_FIRST_FRAME_SOURCE_SHA256:?set immutable first-frame SHA-256}"
: "${MINWM_FIRST_FRAME_SOURCE_CRC64:?set immutable first-frame CRC64NVME}"
: "${TAEHV_REVISION:?set the immutable TAEHV revision}"
: "${TAEHV_CHECKPOINT_URL:?set the immutable taew2_2 URL}"
: "${TAEHV_CHECKPOINT_SHA256:?set the taew2_2 SHA-256}"

[[ ",baseline,nsys,fa3-quality," == *",${MINWM_PROFILE_MODE},"* ]]
if [[ "${MINWM_PROFILE_MODE}" == "fa3-quality" ]]; then
  [[ "${MINWM_GPU_SKU}" == "H100" || "${MINWM_GPU_SKU}" == "H200" ]]
  : "${MINWM_QUALITY_CLIENT_SHA256:?set the fixed quality client SHA-256}"
  : "${MINWM_QUALITY_ANALYZER_SHA256:?set the fixed quality analyzer SHA-256}"
  : "${MINWM_QUALITY_ACTION_CASES_SHA256:?set the action cases SHA-256}"
  : "${MINWM_QUALITY_LONG_CASES_SHA256:?set the long cases SHA-256}"
  : "${MINWM_FA2_REFERENCE_PATCH_SHA256:?set the FA2 reference patch SHA-256}"
fi
[[ ",B200,B300,H100,H200," == *",${MINWM_GPU_SKU},"* ]]
[[ "${MINWM_RUN_ID}" =~ ^[a-z0-9][a-z0-9.-]+$ ]]
[[ "${MINWM_RESULTS_ROOT}" == /s3-results/world-model/evals/minwm/performance/* ]]
[[ "${MINWM_STORAGE_LAYOUT}" == "shared" || "${MINWM_STORAGE_LAYOUT}" == "split" ]]
[[ -n "${MINWM_INPUT_PVC}" && -n "${MINWM_RESULTS_PVC}" ]]
[[ "${MINWM_RESULTS_PVC_ACCESS}" == "RWX" ]]
[[ "${SGLANG_GIT_REF}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MINWM_GIT_REF}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MINWM_HARNESS_GIT_REF}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MINWM_HARNESS_REF_VERIFIED}" == "true" ]]
[[ "${MINWM_REQUIRE_24FPS}" == "true" || "${MINWM_REQUIRE_24FPS}" == "false" ]]
[[ "${MINWM_REQUIRE_CANDIDATE_EVIDENCE}" == "true" || "${MINWM_REQUIRE_CANDIDATE_EVIDENCE}" == "false" ]]
[[ "${MINWM_EXPECTED_MIN_MEMORY_MIB}" =~ ^[0-9]+$ ]]
if [[ -n "${MINWM_EXPECTED_MAX_MEMORY_MIB}" ]]; then
  [[ "${MINWM_EXPECTED_MAX_MEMORY_MIB}" =~ ^[0-9]+$ ]]
  (( MINWM_EXPECTED_MIN_MEMORY_MIB <= MINWM_EXPECTED_MAX_MEMORY_MIB ))
fi
[[ "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" =~ ^[1-9][0-9]*$ ]]
[[ "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ ]]
[[ "${MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE}" == "true" || "${MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE}" == "false" ]]
[[ "${MINWM_RUNNER_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${MINWM_PROFILE_CLIENT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${MINWM_COMMON_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${MINWM_CASES_SHA256}" =~ ^[0-9a-f]{64}$ ]]
if [[ "${MINWM_PROFILE_MODE}" == "fa3-quality" ]]; then
  [[ "${MINWM_QUALITY_CLIENT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${MINWM_QUALITY_ANALYZER_SHA256}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${MINWM_QUALITY_ACTION_CASES_SHA256}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${MINWM_QUALITY_LONG_CASES_SHA256}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${MINWM_FA2_REFERENCE_PATCH_SHA256}" =~ ^[0-9a-f]{64}$ ]]
fi
[[ "${MINWM_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${MINWM_FIRST_FRAME_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${TAEHV_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]

readonly LOCAL_ROOT="/work/minwm-taehv24/${MINWM_RUN_ID}"
readonly LOCAL_RESULTS="${LOCAL_ROOT}/results"
readonly LOCAL_FAILED="${LOCAL_ROOT}/FAILED"
readonly REMOTE_RESULTS="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
readonly LOCAL_ARCHIVE_COPY_PROBE="${LOCAL_ROOT}/ARCHIVE_COPY_PROBE"
readonly REMOTE_ARCHIVE_COPY_PROBE="${REMOTE_RESULTS}/ARCHIVE_COPY_PROBE"
readonly REPO_ROOT="${LOCAL_ROOT}/sglang"
readonly CHECKPOINT="${LOCAL_ROOT}/input/checkpoint/model.pt"
readonly PRETRAINED="${LOCAL_ROOT}/input/pretrained"
readonly MODEL_DIR="${LOCAL_ROOT}/sglang-model"
readonly TAEHV_DIR="${LOCAL_ROOT}/taehv"
readonly TAEHV_CHECKPOINT="${TAEHV_DIR}/taew2_2.pth"
readonly PROFILE_DIR="/opt/minwm-profile"
readonly CASES="${PROFILE_DIR}/cases_720p_compile_smoke.json"
readonly CASE="00_forward_080_pottery_720p"
readonly PROFILE_CLIENT="${PROFILE_DIR}/benchmark_realtime_throughput.py"
readonly PROFILE_COMMON="${PROFILE_DIR}/common.py"
readonly PROFILE_RUNNER="${PROFILE_DIR}/run_single_gpu_taehv_24fps.sh"
readonly QUALITY_CLIENT="${PROFILE_DIR}/run_sglang_api.py"
readonly QUALITY_ANALYZER="${PROFILE_DIR}/analyze_fa3_quality.py"
readonly QUALITY_ACTION_CASES="${PROFILE_DIR}/cases_fa3_quality_actions_720p.json"
readonly QUALITY_LONG_CASES="${PROFILE_DIR}/cases_fa3_quality_long_720p.json"
readonly FA2_REFERENCE_PATCH="${PROFILE_DIR}/fa2_reference_hopper_validation.patch"

export SGLANG_REALTIME_TRACE_SYNC_CUDA=0
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0

[[ ! -e "${LOCAL_ROOT}" ]]
mkdir -p "${LOCAL_RESULTS}" "${REMOTE_RESULTS}"
if find "${REMOTE_RESULTS}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty result prefix: ${REMOTE_RESULTS}" >&2
  exit 2
fi

server_pid=""
nsys_session=""
capture_active=0
archive_attempted=0

stop_server() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL "${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
}

copy_tree_contents() {
  local source="$1"
  local destination="$2"
  local entry_list=""
  local path=""
  local relative=""
  local target=""
  local rc=0

  [[ ! -L "${source}" && -d "${source}" ]] || return 1
  mkdir -p -- "${destination}" || return $?
  entry_list="$(mktemp "${LOCAL_ROOT}/copy-tree.XXXXXX")" || return $?
  if find "${source}" -mindepth 1 -print0 > "${entry_list}"; then
    :
  else
    rc=$?
    rm -f -- "${entry_list}"
    return "${rc}"
  fi

  while IFS= read -r -d '' path; do
    relative="${path#"${source}/"}"
    target="${destination}/${relative}"
    if [[ -L "${path}" || ( ! -d "${path}" && ! -f "${path}" ) ]]; then
      printf 'unsupported result path type: %s\n' "${path}" >&2
      rc=1
    elif [[ -f "${path}" && ( -e "${target}" || -L "${target}" ) ]]; then
      printf 'refusing to overwrite result path: %s\n' "${target}" >&2
      rc=1
    elif [[ -d "${path}" && ( -L "${target}" || ( -e "${target}" && ! -d "${target}" ) ) ]]; then
      printf 'invalid result directory target: %s\n' "${target}" >&2
      rc=1
    fi
  done < "${entry_list}"
  if (( rc != 0 )); then
    rm -f -- "${entry_list}"
    return "${rc}"
  fi

  while IFS= read -r -d '' path; do
    relative="${path#"${source}/"}"
    target="${destination}/${relative}"
    if [[ ! -L "${path}" && -d "${path}" ]]; then
      mkdir -p -- "${target}" || rc=1
    fi
  done < "${entry_list}"

  while IFS= read -r -d '' path; do
    relative="${path#"${source}/"}"
    target="${destination}/${relative}"
    if [[ ! -L "${path}" && -f "${path}" ]]; then
      if [[ -e "${target}" || -L "${target}" ]]; then
        printf 'refusing to overwrite result path: %s\n' "${target}" >&2
        rc=1
      else
        cp --no-preserve=all -- "${path}" "${target}" || rc=1
      fi
    fi
  done < "${entry_list}"
  rm -f -- "${entry_list}" || rc=1
  return "${rc}"
}

archive_results() {
  archive_attempted=1
  copy_tree_contents "${LOCAL_RESULTS}" "${REMOTE_RESULTS}"
}

finish() {
  local status="${1:-$?}"
  trap - EXIT INT TERM
  set +e
  if (( capture_active == 1 )) && [[ -n "${nsys_session}" ]]; then
    timeout --kill-after=5s 20s \
      nsys stop --session="${nsys_session}" >/dev/null 2>&1
  fi
  stop_server
  if (( status != 0 )); then
    printf 'status=%d\n' "${status}" > "${LOCAL_FAILED}"
    if (( archive_attempted == 0 )); then
      archive_results 2>/dev/null
    fi
    cp "${LOCAL_FAILED}" "${REMOTE_RESULTS}/FAILED" 2>/dev/null
  fi
  exit "${status}"
}
trap 'finish $?' EXIT
trap 'finish 130' INT
trap 'finish 143' TERM

printf 'run_id=%s\nstorage_layout=%s\ninput_pvc=%s\nresults_pvc=%s\n' \
  "${MINWM_RUN_ID}" "${MINWM_STORAGE_LAYOUT}" "${MINWM_INPUT_PVC}" \
  "${MINWM_RESULTS_PVC}" > "${LOCAL_ROOT}/STORAGE_WRITE_PROBE"
cp "${LOCAL_ROOT}/STORAGE_WRITE_PROBE" \
  "${REMOTE_RESULTS}/STORAGE_WRITE_PROBE"
cmp --silent "${LOCAL_ROOT}/STORAGE_WRITE_PROBE" \
  "${REMOTE_RESULTS}/STORAGE_WRITE_PROBE"
mkdir -p "${LOCAL_ARCHIVE_COPY_PROBE}/nested" "${REMOTE_ARCHIVE_COPY_PROBE}"
printf 'run_id=%s\narchive_copy_probe=recursive-no-posix-metadata\n' \
  "${MINWM_RUN_ID}" > "${LOCAL_ARCHIVE_COPY_PROBE}/nested/payload"
copy_tree_contents "${LOCAL_ARCHIVE_COPY_PROBE}" "${REMOTE_ARCHIVE_COPY_PROBE}"
cmp --silent "${LOCAL_ARCHIVE_COPY_PROBE}/nested/payload" \
  "${REMOTE_ARCHIVE_COPY_PROBE}/nested/payload"

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 900); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

assert_runtime_alignment() {
  local log_path="$1"
  python3 - "${log_path}" \
    "${LOCAL_RESULTS}/runtime-alignment-${MINWM_PROFILE_MODE}.json" <<'PY'
import json
import os
import sys

marker = "MINWM_RUNTIME_ALIGNMENT "
lines = [line for line in open(sys.argv[1]) if marker in line]
assert lines, "missing MINWM_RUNTIME_ALIGNMENT"
payload = lines[-1].split(marker, 1)[1].strip()
observed = {}
for item in payload.split():
    key, separator, value = item.partition("=")
    assert separator and key not in observed, item
    observed[key] = value
expected = {
    "allow_growth": "False",
    "cache_tokens": "27456",
    "local_attn_size": "32",
    "prompt_first_frame_pin_enabled": "True",
    "request_sink_size": "8",
    "request_window_size": "32",
    "rope_gap": "12",
    "rope_position_mode": "block_relative",
    "scene_cut_rope_offset": "0",
    "scene_cut_sink_enabled": "False",
    "sink_size": "8",
    "sink_tokens": "6864",
    "window_size": "32",
}
for key, value in expected.items():
    assert observed.get(key) == value, (key, observed.get(key), value, observed)
candidate_evidence_required = (
    os.environ["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true"
)
structured_marker = "MINWM_RUNTIME_ALIGNMENT_JSON "
structured_lines = [
    line for line in open(sys.argv[1]) if structured_marker in line
]
structured = None
if candidate_evidence_required:
    assert structured_lines, "missing candidate all-layer runtime alignment JSON"
    structured = json.loads(
        structured_lines[-1].split(structured_marker, 1)[1].strip()
    )
    assert structured["all_match"] is True, structured
    assert structured["layer_count"] == 30, structured
    assert structured["violations"] == [], structured
    assert structured["expected"]["layer_count"] == 30, structured
    for key, value in {
        "allow_growth": False,
        "cache_tokens": 27456,
        "prompt_first_frame_pin_enabled": True,
        "rope_max_frame_gap": 12,
        "rope_position_mode": "block_relative",
        "scene_cut_rope_offset": 0,
        "scene_cut_sink_enabled": False,
        "sink_tokens": 6864,
    }.items():
        assert structured["expected"][key] == value, (key, structured)
    for key, value in {
        "local_attn_size": 32,
        "request_sink_size": 8,
        "request_window_size": 32,
        "sink_size": 8,
        "window_size": 32,
    }.items():
        assert structured["resolved"][key] == value, (key, structured)
    for key, value in {
        "cache_token_counts": [27456],
        "k_capacity_token_counts": [27456],
        "sink_token_counts": [6864],
        "v_capacity_token_counts": [27456],
    }.items():
        assert structured["observed"][key] == value, (key, structured)
with open(sys.argv[2], "w") as handle:
    json.dump({
        "candidate_all_layer_evidence_required": candidate_evidence_required,
        "evidence_scope": (
            "candidate structured all-layer runtime assertion"
            if candidate_evidence_required
            else "config plus layer0 runtime alignment on reference main"
        ),
        "expected": expected,
        "observed": observed,
        "structured_all_layer_alignment": structured,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(observed, sort_keys=True))
PY
}

assert_server_contract() {
  local log_path="$1"
  python3 - "${log_path}" \
    "${LOCAL_RESULTS}/server-contract-${MINWM_PROFILE_MODE}.json" <<'PY'
import json
import os
import sys

marker = "server_args: "
lines = [line for line in open(sys.argv[1]) if marker in line]
assert lines, "missing structured server_args log"
observed = json.loads(lines[0].split(marker, 1)[1])
expected = {
    "attention_backend": "fa",
    "enable_cfg_parallel": False,
    "enable_cuda_graph": False,
    "enable_layerwise_nvtx_marker": os.environ["MINWM_PROFILE_MODE"] == "nsys",
    "enable_torch_compile": False,
    "num_gpus": 1,
    "performance_mode": "speed",
    "realtime_session_idle_timeout_s": 900.0,
    "realtime_session_max_lifetime_s": 900.0,
    "realtime_vae_backend": "local",
    "sp_degree": 1,
    "vae_cpu_offload": False,
}
for key, value in expected.items():
    assert observed.get(key) == value, (key, observed.get(key), value)
runtime_env = {
    "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": os.environ[
        "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"
    ],
    "SGLANG_REALTIME_TRACE_SYNC_CUDA": os.environ[
        "SGLANG_REALTIME_TRACE_SYNC_CUDA"
    ],
}
assert runtime_env == {
    "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "0",
    "SGLANG_REALTIME_TRACE_SYNC_CUDA": "0",
}
with open(sys.argv[2], "w") as handle:
    json.dump(
        {"expected_server_args": expected, "runtime_env": runtime_env},
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

assert_profile_result() {
  local result_path="$1" expected_warmup="$2" expected_measured="$3"
  python3 - "${result_path}" "${expected_warmup}" "${expected_measured}" <<'PY'
import json
import os
import sys

result = json.load(open(sys.argv[1]))
expected_warmup = int(sys.argv[2])
expected_measured = int(sys.argv[3])
expected_total = expected_warmup + expected_measured
expected_measured_frames = expected_measured * 16
contract = result["comparison_contract"]
assert contract["size"] == "1248x704"
assert contract["steps"] == 4
assert contract["latent_frames_per_chunk"] == 4
assert contract["generated_pixel_frames_per_steady_chunk"] == 16
assert contract["kv_cache_num_frames"] == 32
assert contract["sink_size"] == 8
assert result["warmup_chunks"] == expected_warmup
assert result["measured_chunks"] == expected_measured
assert result["received_payload_chunks"] == expected_total
assert result["received_server_timing_chunks"] == expected_total
assert result["measured_frames"] == expected_measured_frames
assert result["received_frame_contract"] == {
    "bytes_per_frame": 1248 * 704 * 3,
    "channels": 3,
    "content_type": "application/x-raw-rgb",
    "height": 704,
    "width": 1248,
}
request = result["request_evidence"]
assert request["case_id"] == "00_forward_080_pottery_720p"
assert request["first_frame_uri"] == os.environ["MINWM_FIRST_FRAME_SOURCE_URI"]
assert request["first_frame_sha256"] == os.environ["MINWM_FIRST_FRAME_SOURCE_SHA256"]
assert len(request["prompt_sha256"]) == 64
assert len(request["condition_inputs_sha256"]) == 64
assert result["target_fps"] == 24.0
assert result["target_ms_per_chunk"] == 1000 * 16 / 24
for field in (
    "scheduler_forward_ms",
    "chunk_total_ms",
    "model_vae_encode_ms",
    "model_denoise_ms",
    "model_vae_decode_ms",
    "raw_payload_build_ms",
    "ws_write_ms",
):
    assert result["server"][field]["sample_count"] == expected_measured, (
        field,
        result["server"][field],
    )
    assert result["server"][field]["missing_count"] == 0
if os.environ["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true":
    async_enqueue = result["server"]["raw_frame_async_enqueue_ms"]
    assert async_enqueue["sample_count"] == expected_measured, async_enqueue
    assert async_enqueue["missing_count"] == 0, async_enqueue
PY
}

record_performance_gate() {
  local result_path="$1"
  python3 - "${result_path}" "${LOCAL_RESULTS}" "${MINWM_REQUIRE_24FPS}" <<'PY'
import json
import pathlib
import sys

result = json.load(open(sys.argv[1]))
results_dir = pathlib.Path(sys.argv[2])
required = sys.argv[3] == "true"
passed = bool(result["target_24fps_pass"])
summary = {
    "client_fps": result["client"]["steady_received_fps_ratio_of_sums"],
    "required": required,
    "target_24fps_pass": passed,
    "target_chunk_p50_pass": result["target_chunk_p50_pass"],
    "target_chunk_p95_pass": result["target_chunk_p95_pass"],
    "target_fps": result["target_fps"],
    "target_ms_per_chunk": result["target_ms_per_chunk"],
}
status_path = results_dir / ("PERFORMANCE_PASS" if passed else "PERFORMANCE_FAIL")
status_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
if required and not passed:
    raise SystemExit(3)
PY
}

record_no_offload_protocol_fit() {
  local result_path="$1" log_path="$2" marker_path="$3"
  kill -0 "${server_pid}"
  python3 - "${result_path}" "${log_path}" "${LOCAL_RESULTS}/gpu.csv" \
    "${marker_path}" <<'PY'
import csv
import json
import os
import sys

result = json.load(open(sys.argv[1]))
server_log = open(sys.argv[2]).read()
with open(sys.argv[3], newline="") as handle:
    gpu_rows = list(csv.reader(handle))
assert len(gpu_rows) == 1, gpu_rows
gpu_name, gpu_memory_mib, gpu_compute_cap = [item.strip() for item in gpu_rows[0]]

warmup_chunks = int(os.environ["MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS"])
measured_chunks = int(os.environ["MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS"])
latent_frames_per_chunk = int(
    result["comparison_contract"]["latent_frames_per_chunk"]
)
warmup_latent_frames = warmup_chunks * latent_frames_per_chunk
full_window_required = (
    os.environ["MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE"] == "true"
)
if full_window_required:
    assert warmup_latent_frames >= 32, warmup_latent_frames

server_args_marker = "server_args: "
server_args_lines = [
    line for line in server_log.splitlines() if server_args_marker in line
]
assert server_args_lines, "missing structured server_args log"
server_args = json.loads(server_args_lines[0].split(server_args_marker, 1)[1])
assert server_args["vae_cpu_offload"] is False, server_args
assert server_args["realtime_vae_backend"] == "local", server_args
assert "Preloading TAEHV decoder weights" in server_log

assert result["warmup_chunks"] == warmup_chunks, result
assert result["measured_chunks"] == measured_chunks, result
assert result["received_payload_chunks"] == warmup_chunks + measured_chunks, result
assert result["received_server_timing_chunks"] == warmup_chunks + measured_chunks, result
assert gpu_name.find(os.environ["MINWM_GPU_SKU"]) >= 0, gpu_name
assert gpu_compute_cap == os.environ["MINWM_EXPECTED_COMPUTE_CAP"], gpu_compute_cap
memory_mib = int(gpu_memory_mib)
assert int(os.environ["MINWM_EXPECTED_MIN_MEMORY_MIB"]) <= memory_mib
max_memory_mib = os.environ["MINWM_EXPECTED_MAX_MEMORY_MIB"]
if max_memory_mib:
    assert memory_mib <= int(max_memory_mib)

evidence = {
    "formal_same_process_eligible": True,
    "full_window_required": full_window_required,
    "gpu": {
        "compute_cap": gpu_compute_cap,
        "memory_total_mib": memory_mib,
        "name": gpu_name,
    },
    "local_streaming_taehv": True,
    "measured_chunks": measured_chunks,
    "no_offload_protocol_fit_pass": True,
    "processed_latent_frames": (
        (warmup_chunks + measured_chunks) * latent_frames_per_chunk
    ),
    "same_server_process_alive_after_smoke": True,
    "vae_cpu_offload": False,
    "warmup_chunks": warmup_chunks,
    "warmup_latent_frames": warmup_latent_frames,
}
with open(sys.argv[4], "w") as handle:
    json.dump(evidence, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(evidence, sort_keys=True))
PY
}

assert_no_offload_protocol_fit() {
  local marker_path="$1"
  kill -0 "${server_pid}"
  python3 - "${marker_path}" <<'PY'
import json
import sys

evidence = json.load(open(sys.argv[1]))
assert evidence["no_offload_protocol_fit_pass"] is True, evidence
assert evidence["formal_same_process_eligible"] is True, evidence
assert evidence["same_server_process_alive_after_smoke"] is True, evidence
assert evidence["local_streaming_taehv"] is True, evidence
assert evidence["vae_cpu_offload"] is False, evidence
if evidence["full_window_required"]:
    assert evidence["warmup_latent_frames"] >= 32, evidence
PY
}

clone_at() {
  local repository="$1" destination="$2" revision="$3"
  git clone --filter=blob:none --no-checkout \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/seedleap/${repository}.git" \
    "${destination}"
  git -C "${destination}" checkout --detach "${revision}"
  [[ "$(git -C "${destination}" rev-parse HEAD)" == "${revision}" ]]
}

readonly CHECKPOINT_MOUNT_PATH="${MINWM_INPUT_ROOT%/}/${MINWM_CHECKPOINT_RELATIVE_PATH#/}"
readonly PRETRAINED_MOUNT_PATH="${MINWM_INPUT_ROOT%/}/${MINWM_PRETRAINED_RELATIVE_PATH#/}"
[[ -f "${CHECKPOINT_MOUNT_PATH}" ]]
[[ -f "${PRETRAINED_MOUNT_PATH}/model_index.json" ]]
[[ -d "${PRETRAINED_MOUNT_PATH}/transformer" ]]

nvidia-smi --query-gpu=name,memory.total,compute_cap \
  --format=csv,noheader,nounits | tee "${LOCAL_RESULTS}/gpu.csv"
[[ "$(wc -l < "${LOCAL_RESULTS}/gpu.csv")" == "1" ]]
IFS=',' read -r gpu_name gpu_memory_mib gpu_compute_cap \
  < "${LOCAL_RESULTS}/gpu.csv"
gpu_name="$(xargs <<< "${gpu_name}")"
gpu_memory_mib="$(xargs <<< "${gpu_memory_mib}")"
gpu_compute_cap="$(xargs <<< "${gpu_compute_cap}")"
[[ "${gpu_name}" == *"${MINWM_GPU_SKU}"* ]]
[[ "${gpu_compute_cap}" == "${MINWM_EXPECTED_COMPUTE_CAP}" ]]
(( gpu_memory_mib >= MINWM_EXPECTED_MIN_MEMORY_MIB ))
if [[ -n "${MINWM_EXPECTED_MAX_MEMORY_MIB}" ]]; then
  (( gpu_memory_mib <= MINWM_EXPECTED_MAX_MEMORY_MIB ))
fi
[[ "${MINWM_VAE_CPU_OFFLOAD:-false}" == "false" ]]

python3 - <<'PY' | tee "${LOCAL_RESULTS}/runtime-before-install.json"
import json
import os
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print(json.dumps({
    "cuda": torch.version.cuda,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "device_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
}, indent=2, sort_keys=True))
PY

python3 - "${CASES}" "${CASE}" <<'PY' > "${LOCAL_RESULTS}/contract.json"
import hashlib
import json
import os
import sys

cases_bytes = open(sys.argv[1], "rb").read()
cases = json.loads(cases_bytes)
request_contract = cases["contract"]
case = next(item for item in cases["cases"] if item["id"] == sys.argv[2])
assert int(request_contract["width"]) == 1248
assert int(request_contract["height"]) == 704
assert int(request_contract["fps"]) == 24
assert int(request_contract["latent_frames_per_chunk"]) == 4
assert case["first_frame"] == os.environ["MINWM_FIRST_FRAME_SOURCE_URI"]
canonical_action_source = json.dumps(
    {
        "action_label": int(case["action_label"]),
        "action_type": request_contract["action_type"],
    },
    separators=(",", ":"),
    sort_keys=True,
).encode()

contract = {
    "checkpoint": {
        "bytes": int(os.environ["MINWM_CHECKPOINT_BYTES"]),
        "crc64nvme": os.environ["MINWM_CHECKPOINT_CRC64"],
        "sha256": os.environ["MINWM_CHECKPOINT_SHA256"],
        "source_etag": os.environ["MINWM_CHECKPOINT_SOURCE_ETAG"],
        "source_uri": os.environ["MINWM_CHECKPOINT_SOURCE_URI"],
        "source_version": os.environ["MINWM_CHECKPOINT_SOURCE_VERSION"],
        "staged_version": os.environ["MINWM_CHECKPOINT_STAGED_VERSION"],
    },
    "execution": {
        "attention_impl": "packed",
        "cuda_graph": False,
        "gpu_count": 1,
        "local_streaming_taehv": True,
        "measured_chunks": (
            200 if os.environ["MINWM_PROFILE_MODE"] == "baseline" else
            8 if os.environ["MINWM_PROFILE_MODE"] == "nsys" else 0
        ),
        "nsys_capture_chunks": 16 if os.environ["MINWM_PROFILE_MODE"] == "nsys" else 0,
        "nsys_is_headline": False,
        "profile_mode": os.environ["MINWM_PROFILE_MODE"],
        "require_24fps": os.environ["MINWM_REQUIRE_24FPS"] == "true",
        "require_candidate_evidence": (
            os.environ["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true"
        ),
        "protocol_smoke_in_headline": False,
        "protocol_smoke_measured_chunks": int(
            os.environ["MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS"]
        ),
        "protocol_smoke_warmup_chunks": int(
            os.environ["MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS"]
        ),
        "require_full_window_no_offload_smoke": (
            os.environ["MINWM_REQUIRE_FULL_WINDOW_NO_OFFLOAD_SMOKE"] == "true"
        ),
        "realtime_session_idle_timeout_s": 900,
        "realtime_session_max_lifetime_s": 900,
        "segment_compile": True,
        "torch_compile": False,
        "vae_cpu_offload": False,
        "warmup_chunks": (
            20 if os.environ["MINWM_PROFILE_MODE"] == "baseline" else
            8 if os.environ["MINWM_PROFILE_MODE"] == "nsys" else 0
        ),
    },
    "environment": {
        "known_unrelated_preinstalled_extras": [
            "decord",
            "open-clip-torch",
            "wandb",
        ],
        "pip_check_is_inference_gate": False,
        "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": os.environ[
            "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"
        ],
        "SGLANG_REALTIME_TRACE_SYNC_CUDA": os.environ[
            "SGLANG_REALTIME_TRACE_SYNC_CUDA"
        ],
    },
    "first_frame": {
        "bytes": int(os.environ["MINWM_FIRST_FRAME_SOURCE_BYTES"]),
        "crc64nvme": os.environ["MINWM_FIRST_FRAME_SOURCE_CRC64"],
        "etag": os.environ["MINWM_FIRST_FRAME_SOURCE_ETAG"],
        "sha256": os.environ["MINWM_FIRST_FRAME_SOURCE_SHA256"],
        "uri": os.environ["MINWM_FIRST_FRAME_SOURCE_URI"],
        "version_id": os.environ["MINWM_FIRST_FRAME_SOURCE_VERSION"],
    },
    "hardware": {
        "base_image": os.environ["MINWM_BASE_IMAGE"],
        "profile": os.environ["MINWM_HARDWARE_PROFILE"],
        "sku": os.environ["MINWM_GPU_SKU"],
        "expected_compute_cap": os.environ["MINWM_EXPECTED_COMPUTE_CAP"],
        "expected_memory_mib": {
            "min": int(os.environ["MINWM_EXPECTED_MIN_MEMORY_MIB"]),
            "max": (
                int(os.environ["MINWM_EXPECTED_MAX_MEMORY_MIB"])
                if os.environ["MINWM_EXPECTED_MAX_MEMORY_MIB"]
                else None
            ),
        },
    },
    "harness": {
        "cases_sha256": os.environ["MINWM_CASES_SHA256"],
        "common_sha256": os.environ["MINWM_COMMON_SHA256"],
        "git_ref": os.environ["MINWM_HARNESS_GIT_REF"],
        "profile_client_sha256": os.environ["MINWM_PROFILE_CLIENT_SHA256"],
        "ref_content_verified": os.environ["MINWM_HARNESS_REF_VERIFIED"] == "true",
        "runner_sha256": os.environ["MINWM_RUNNER_SHA256"],
        "fa3_quality": (
            {
                "action_cases_sha256": os.environ[
                    "MINWM_QUALITY_ACTION_CASES_SHA256"
                ],
                "analyzer_sha256": os.environ["MINWM_QUALITY_ANALYZER_SHA256"],
                "fa2_reference_patch_sha256": os.environ[
                    "MINWM_FA2_REFERENCE_PATCH_SHA256"
                ],
                "long_cases_sha256": os.environ[
                    "MINWM_QUALITY_LONG_CASES_SHA256"
                ],
                "quality_client_sha256": os.environ["MINWM_QUALITY_CLIENT_SHA256"],
                "reference_scope": "detached validation worktree only",
            }
            if os.environ["MINWM_PROFILE_MODE"] == "fa3-quality"
            else None
        ),
    },
    "request": {
        "action_source_sha256": hashlib.sha256(canonical_action_source).hexdigest(),
        "case_id": case["id"],
        "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
        "first_frame_uri": case["first_frame"],
        "height": 704,
        "kv_cache_num_frames": 32,
        "latent_frames_per_chunk": 4,
        "pixel_frames_per_chunk": 16,
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "sink_size": 8,
        "steps": 4,
        "target_fps": 24,
        "target_ms_per_chunk": 1000 * 16 / 24,
        "width": 1248,
    },
    "sglang_git_ref": os.environ["SGLANG_GIT_REF"],
    "minwm_git_ref": os.environ["MINWM_GIT_REF"],
    "taehv": {
        "checkpoint": "taew2_2.pth",
        "checkpoint_sha256": os.environ["TAEHV_CHECKPOINT_SHA256"],
        "revision": os.environ["TAEHV_REVISION"],
    },
    "tianpeng": {
        "local_attn_size": 32,
        "prompt_first_frame_pin_enabled": True,
        "rope_max_frame_gap": 12,
        "rope_position_mode": "block_relative",
        "sink_size": 8,
        "sliding_window_num_frames": 32,
    },
}
print(json.dumps(contract, indent=2, sort_keys=True))
PY

clone_at sglang "${REPO_ROOT}" "${SGLANG_GIT_REF}"
git -C "${REPO_ROOT}" show -s --format=fuller HEAD \
  | tee "${LOCAL_RESULTS}/sglang-commit.txt"
mkdir -p "${LOCAL_RESULTS}/harness"
for harness_entry in \
  "${MINWM_RUNNER_SHA256}:${PROFILE_RUNNER}" \
  "${MINWM_PROFILE_CLIENT_SHA256}:${PROFILE_CLIENT}" \
  "${MINWM_COMMON_SHA256}:${PROFILE_COMMON}" \
  "${MINWM_CASES_SHA256}:${CASES}"; do
  harness_sha="${harness_entry%%:*}"
  harness_path="${harness_entry#*:}"
  printf '%s  %s\n' "${harness_sha}" "${harness_path}" | sha256sum --check -
  cp "${harness_path}" "${LOCAL_RESULTS}/harness/$(basename "${harness_path}")"
done
if [[ "${MINWM_PROFILE_MODE}" == "fa3-quality" ]]; then
  for harness_entry in \
    "${MINWM_QUALITY_CLIENT_SHA256}:${QUALITY_CLIENT}" \
    "${MINWM_QUALITY_ANALYZER_SHA256}:${QUALITY_ANALYZER}" \
    "${MINWM_QUALITY_ACTION_CASES_SHA256}:${QUALITY_ACTION_CASES}" \
    "${MINWM_QUALITY_LONG_CASES_SHA256}:${QUALITY_LONG_CASES}" \
    "${MINWM_FA2_REFERENCE_PATCH_SHA256}:${FA2_REFERENCE_PATCH}"; do
    harness_sha="${harness_entry%%:*}"
    harness_path="${harness_entry#*:}"
    printf '%s  %s\n' "${harness_sha}" "${harness_path}" | sha256sum --check -
    cp "${harness_path}" "${LOCAL_RESULTS}/harness/$(basename "${harness_path}")"
  done
fi
sha256sum "${PROFILE_DIR}"/* | tee "${LOCAL_RESULTS}/harness-sha256.txt"

if ! command -v cargo >/dev/null; then
  python3 - "${REPO_ROOT}/python/pyproject.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
block = '''[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
'''
if text.count(block) != 1:
    raise RuntimeError("expected exactly one SGLang gRPC Rust extension block")
path.write_text(text.replace(block, ""))
PY
fi
python3 -m pip install -e "${REPO_ROOT}/python[diffusion,tracing]" \
  --root-user-action=ignore
python3 -m pip uninstall -y peft
python3 -m pip install --no-cache-dir --no-deps \
  "taehv @ git+https://github.com/madebyollin/taehv.git@${TAEHV_REVISION}" \
  --root-user-action=ignore
python3 -m pip freeze --all > "${LOCAL_RESULTS}/pip-freeze.txt"
set +e
python3 -m pip check 2>&1 | tee "${LOCAL_RESULTS}/pip-check.txt"
pip_check_status="${PIPESTATUS[0]}"
set -e
printf 'PIP_CHECK_STATUS=%s\n' "${pip_check_status}" \
  | tee "${LOCAL_RESULTS}/pip-check-status.txt"

mkdir -p "$(dirname "${CHECKPOINT}")" "${PRETRAINED}" "${TAEHV_DIR}"
cp "${CHECKPOINT_MOUNT_PATH}" "${CHECKPOINT}"
cp -a "${PRETRAINED_MOUNT_PATH}/." "${PRETRAINED}/"
[[ "$(stat -c '%s' "${CHECKPOINT}")" == "${MINWM_CHECKPOINT_BYTES}" ]]
printf '%s  %s\n' "${MINWM_CHECKPOINT_SHA256}" "${CHECKPOINT}" \
  | sha256sum --check - \
  | tee "${LOCAL_RESULTS}/checkpoint-sha256.txt"
curl --fail --location --retry 3 --output "${TAEHV_CHECKPOINT}" \
  "${TAEHV_CHECKPOINT_URL}"
printf '%s  %s\n' "${TAEHV_CHECKPOINT_SHA256}" "${TAEHV_CHECKPOINT}" \
  | sha256sum --check - \
  | tee "${LOCAL_RESULTS}/taehv-checkpoint-sha256.txt"

python3 "${REPO_ROOT}/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py" \
  --minwm-checkpoint "${CHECKPOINT}" \
  --donor-diffusers-dir "${PRETRAINED}" \
  --output-dir "${MODEL_DIR}" \
  --link-donor \
  --source-uri "${MINWM_CHECKPOINT_SOURCE_URI}" \
  --source-version-id "${MINWM_CHECKPOINT_SOURCE_VERSION}" \
  --source-etag "${MINWM_CHECKPOINT_SOURCE_ETAG}" \
  --action-type auto \
  --local-attn-size 32 \
  --sink-size 8 \
  --sliding-window-num-frames 32 \
  --rope-position-mode block_relative \
  --rope-max-frame-gap 12 \
  --prompt-first-frame-pin-enabled \
  | tee "${LOCAL_RESULTS}/conversion.log"

cp "${MODEL_DIR}/minwm_conversion_manifest.json" \
  "${LOCAL_RESULTS}/minwm_conversion_manifest.json"
cp "${MODEL_DIR}/transformer/config.json" \
  "${LOCAL_RESULTS}/transformer-config.json"
python3 - "${LOCAL_RESULTS}/minwm_conversion_manifest.json" \
  "${LOCAL_RESULTS}/transformer-config.json" <<'PY'
import json
import os
import sys

manifest = json.load(open(sys.argv[1]))
config = json.load(open(sys.argv[2]))
assert manifest["source_checkpoint"] == {
    "etag": os.environ["MINWM_CHECKPOINT_SOURCE_ETAG"],
    "local_size": int(os.environ["MINWM_CHECKPOINT_BYTES"]),
    "selected_state_dict": manifest["source_checkpoint"]["selected_state_dict"],
    "uri": os.environ["MINWM_CHECKPOINT_SOURCE_URI"],
    "version_id": os.environ["MINWM_CHECKPOINT_SOURCE_VERSION"],
}
assert manifest["causal_cache_defaults"] == {
    "local_attn_size": 32,
    "provenance": "converter arguments, not checkpoint tensors",
    "sink_size": 8,
    "sliding_window_num_frames": 32,
}
expected = {
    "local_attn_size": 32,
    "prompt_first_frame_pin_enabled": True,
    "rope_max_frame_gap": 12,
    "rope_position_mode": "block_relative",
    "sink_size": 8,
    "sliding_window_num_frames": 32,
}
for key, value in expected.items():
    assert config[key] == value, (key, config[key], value)
assert config["action_type"] == "primitive_token_residual"
PY

export PYTHONPATH="${PROFILE_DIR}:${REPO_ROOT}/python"
export PYTHONUNBUFFERED=1
export MINWM_S3_MOUNT="${MINWM_INPUT_ROOT}"
export MINWM_ATTENTION_IMPL=packed
export MINWM_PACKED_ATTENTION_DETERMINISTIC=false
export MINWM_NATIVE_COMPONENTS=
export MINWM_SEGMENT_COMPILE=true
export MINWM_PARITY_DETERMINISTIC=0
export MINWM_DETERMINISTIC_ATTENTION=false
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export PYTHONHASHSEED=0

python3 - "${CASES}" "${CASE}" <<'PY' \
  | tee "${LOCAL_RESULTS}/first-frame-source.json"
import hashlib
import json
import os
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
case = next(case for case in manifest["cases"] if case["id"] == sys.argv[2])
uri = os.environ["MINWM_FIRST_FRAME_SOURCE_URI"]
assert case["first_frame"] == uri, (case["first_frame"], uri)
bucket, key = uri.removeprefix("s3://").split("/", 1)
assert bucket == "leap-world-us-east-2", bucket
mounted_source = Path(os.environ["MINWM_S3_MOUNT"]) / key
assert mounted_source.is_file(), mounted_source
digest = hashlib.sha256()
with mounted_source.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(json.dumps({
    "bytes": mounted_source.stat().st_size,
    "case": case["id"],
    "crc64nvme": os.environ["MINWM_FIRST_FRAME_SOURCE_CRC64"],
    "etag": os.environ["MINWM_FIRST_FRAME_SOURCE_ETAG"],
    "mounted_path": str(mounted_source),
    "sha256": digest.hexdigest(),
    "uri": uri,
    "version_id": os.environ["MINWM_FIRST_FRAME_SOURCE_VERSION"],
}, indent=2, sort_keys=True))
assert mounted_source.stat().st_size == int(
    os.environ["MINWM_FIRST_FRAME_SOURCE_BYTES"]
)
assert digest.hexdigest() == os.environ["MINWM_FIRST_FRAME_SOURCE_SHA256"]
PY

server_command=(
  python3 -m sglang.multimodal_gen.runtime.launch_server
  --model-path "${MODEL_DIR}"
  --pipeline-class-name MinWMCausalDMDPipeline
  --attention-backend fa
  --performance-mode speed
  --num-gpus 1
  --sp-degree 1
  --enable-cfg-parallel false
  --enable-torch-compile false
  --enable-cuda-graph false
  --vae-cpu-offload false
  --text-encoder-cpu-offload false
  --dit-cpu-offload false
  --dit-layerwise-offload false
  --realtime-vae-backend local
  --vae-config.taehv-checkpoint-path "${TAEHV_CHECKPOINT}"
  --realtime-causal-sink-size 8
  --realtime-causal-kv-cache-num-frames 32
  --realtime-session-idle-timeout-s 900
  --realtime-session-max-lifetime-s 900
  --warmup-mode off
  --host 127.0.0.1
  --port 30000
)

if [[ "${MINWM_PROFILE_MODE}" == "baseline" ]]; then
  baseline_dir="${LOCAL_RESULTS}/baseline"
  mkdir -p "${baseline_dir}"
  "${server_command[@]}" > "${baseline_dir}/server.log" 2>&1 &
  server_pid=$!
  wait_for_server "${baseline_dir}/server.log"
  assert_server_contract "${baseline_dir}/server.log"
  python3 "${PROFILE_CLIENT}" \
    --cases "${CASES}" \
    --case "${CASE}" \
    --profile-name "${MINWM_GPU_SKU,,}-local-taehv-protocol-smoke" \
    --sink-size 8 \
    --kv-cache-num-frames 32 \
    --warmup-chunks "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \
    --measured-chunks "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}" \
    --timeout 3600 \
    --output "${baseline_dir}/protocol-smoke.json" \
    | tee "${baseline_dir}/protocol-smoke-client.log"
  assert_profile_result \
    "${baseline_dir}/protocol-smoke.json" \
    "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \
    "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}"
  assert_runtime_alignment "${baseline_dir}/server.log"
  grep -F 'Preloading TAEHV decoder weights' "${baseline_dir}/server.log" \
    > "${baseline_dir}/taehv-load-evidence.txt"
  no_offload_fit_marker="${baseline_dir}/NO_OFFLOAD_PROTOCOL_FIT_PASS.json"
  record_no_offload_protocol_fit \
    "${baseline_dir}/protocol-smoke.json" \
    "${baseline_dir}/server.log" \
    "${no_offload_fit_marker}"
  assert_no_offload_protocol_fit "${no_offload_fit_marker}"
  python3 "${PROFILE_CLIENT}" \
    --cases "${CASES}" \
    --case "${CASE}" \
    --profile-name "${MINWM_GPU_SKU,,}-local-taehv-main-segment" \
    --sink-size 8 \
    --kv-cache-num-frames 32 \
    --warmup-chunks 20 \
    --measured-chunks 200 \
    --timeout 3600 \
    --output "${baseline_dir}/throughput.json" \
    | tee "${baseline_dir}/client.log"
  assert_profile_result "${baseline_dir}/throughput.json" 20 200
  record_performance_gate "${baseline_dir}/throughput.json"
  stop_server
elif [[ "${MINWM_PROFILE_MODE}" == "fa3-quality" ]]; then
  quality_dir="${LOCAL_RESULTS}/fa3-quality"
  fa2_repo="${LOCAL_ROOT}/sglang-fa2-reference"
  mkdir -p "${quality_dir}" "${quality_dir}/actions" "${quality_dir}/long"
  git -C "${REPO_ROOT}" worktree add --detach "${fa2_repo}" "${SGLANG_GIT_REF}"
  git -C "${fa2_repo}" apply --check "${FA2_REFERENCE_PATCH}"
  git -C "${fa2_repo}" apply "${FA2_REFERENCE_PATCH}"
  git -C "${fa2_repo}" diff --check
  git -C "${fa2_repo}" diff -- \
    python/sglang/multimodal_gen/runtime/models/dits/minwm.py \
    > "${quality_dir}/fa2-reference-applied.diff"
  cp "${FA2_REFERENCE_PATCH}" "${quality_dir}/fa2-reference-source.patch"
  git -C "${REPO_ROOT}" diff --quiet -- \
    python/sglang/multimodal_gen/runtime/models/dits/minwm.py
  python3 -m pip install --no-cache-dir --no-deps \
    lpips==0.1.4 scikit-image==0.24.0 \
    --root-user-action=ignore

  run_quality_backend() {
    local backend="$1"
    local source_root="$2"
    local backend_dir="${quality_dir}/${backend}"
    mkdir -p "${backend_dir}"
    PYTHONPATH="${PROFILE_DIR}:${source_root}/python" \
    MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
    MINWM_PARITY_DETERMINISTIC=1 \
    MINWM_DETERMINISTIC_ATTENTION=true \
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      "${server_command[@]}" > "${backend_dir}/server.log" 2>&1 &
    server_pid=$!
    wait_for_server "${backend_dir}/server.log"
    assert_server_contract "${backend_dir}/server.log"
    python3 "${PROFILE_CLIENT}" \
      --cases "${CASES}" \
      --case "${CASE}" \
      --profile-name "${MINWM_GPU_SKU,,}-${backend}-quality-protocol-smoke" \
      --sink-size 8 \
      --kv-cache-num-frames 32 \
      --warmup-chunks 8 \
      --measured-chunks 2 \
      --timeout 3600 \
      --output "${backend_dir}/protocol-smoke.json" \
      > "${backend_dir}/protocol-smoke-client.log" 2>&1
    assert_profile_result "${backend_dir}/protocol-smoke.json" 8 2
    assert_runtime_alignment "${backend_dir}/server.log"
    no_offload_fit_marker="${backend_dir}/NO_OFFLOAD_PROTOCOL_FIT_PASS.json"
    record_no_offload_protocol_fit \
      "${backend_dir}/protocol-smoke.json" \
      "${backend_dir}/server.log" \
      "${no_offload_fit_marker}"
    assert_no_offload_protocol_fit "${no_offload_fit_marker}"
    grep -E "MinWM packed-varlen attention backend=${backend} device=cuda(:0)?" \
      "${backend_dir}/server.log" > "${backend_dir}/attention-backend.txt"
    if [[ "${backend}" == "fa3" ]]; then
      ! grep -E 'MinWM packed-varlen attention backend=fa2 device=cuda(:0)?' \
        "${backend_dir}/server.log"
    fi
    for replay in a b; do
      python3 "${QUALITY_CLIENT}" \
        --cases "${QUALITY_ACTION_CASES}" \
        --results "${quality_dir}/actions" \
        --output-prefix "${backend}_${replay}" \
        --engine-name "minwm-${backend}-same-gpu-quality-ab" \
        --sink-size 8 \
        --kv-cache-num-frames 32 \
        --timeout 3600 \
        > "${backend_dir}/actions-${replay}.log" 2>&1
      python3 "${QUALITY_CLIENT}" \
        --cases "${QUALITY_LONG_CASES}" \
        --results "${quality_dir}/long" \
        --output-prefix "${backend}_${replay}" \
        --engine-name "minwm-${backend}-same-gpu-quality-ab" \
        --sink-size 8 \
        --kv-cache-num-frames 32 \
        --timeout 3600 \
        > "${backend_dir}/long-${replay}.log" 2>&1
    done
    stop_server
  }

  run_quality_backend fa2 "${fa2_repo}"
  run_quality_backend fa3 "${REPO_ROOT}"
  python3 "${QUALITY_ANALYZER}" \
    --actions-results "${quality_dir}/actions" \
    --long-results "${quality_dir}/long" \
    --output "${quality_dir}/report.json" \
    --require-lpips \
    | tee "${quality_dir}/analysis.log"
  printf 'FA3_QUALITY_PASS=true\n' > "${LOCAL_RESULTS}/FA3_QUALITY_PASS"
else
  readonly NSYS_URL="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb"
  readonly NSYS_SHA256="b896cb2b9586ddf617c363a43bababad0a015dff4c77d8f0fbb9c26144056a69"
  readonly NSYS_DIR="${LOCAL_ROOT}/nsight-systems"
  readonly NSYS_DEB="${LOCAL_ROOT}/nsight-systems-cli.deb"
  mkdir -p "${NSYS_DIR}"
  if ! command -v nsys >/dev/null; then
    curl --fail --location --retry 3 --output "${NSYS_DEB}" "${NSYS_URL}"
    printf '%s  %s\n' "${NSYS_SHA256}" "${NSYS_DEB}" | sha256sum --check -
    dpkg-deb --extract "${NSYS_DEB}" "${NSYS_DIR}"
    NSYS_BIN="$(find "${NSYS_DIR}" -type f -name nsys -perm -111 -print -quit)"
    [[ -n "${NSYS_BIN}" ]]
    export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
  fi
  nsys --version | tee "${LOCAL_RESULTS}/nsys-version.txt"
  nsys status -e | tee "${LOCAL_RESULTS}/nsys-status.txt"

  nsys_dir="${LOCAL_RESULTS}/nsys"
  mkdir -p "${nsys_dir}"
  nsys_session="minwm-${MINWM_RUN_ID}"
  nsys launch \
    --session-new="${nsys_session}" \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -- \
    "${server_command[@]}" \
    --enable-layerwise-nvtx-marker \
    > "${nsys_dir}/server.log" 2>&1 &
  server_pid=$!
  wait_for_server "${nsys_dir}/server.log"
  assert_server_contract "${nsys_dir}/server.log"

  python3 "${PROFILE_CLIENT}" \
    --cases "${CASES}" \
    --case "${CASE}" \
    --profile-name "${MINWM_GPU_SKU,,}-local-taehv-nsys-protocol-smoke" \
    --sink-size 8 \
    --kv-cache-num-frames 32 \
    --warmup-chunks "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \
    --measured-chunks "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}" \
    --timeout 3600 \
    --output "${nsys_dir}/protocol-smoke.json" \
    > "${nsys_dir}/protocol-smoke-client.log" 2>&1
  assert_profile_result \
    "${nsys_dir}/protocol-smoke.json" \
    "${MINWM_PROTOCOL_SMOKE_WARMUP_CHUNKS}" \
    "${MINWM_PROTOCOL_SMOKE_MEASURED_CHUNKS}"
  assert_runtime_alignment "${nsys_dir}/server.log"
  grep -F 'Preloading TAEHV decoder weights' "${nsys_dir}/server.log" \
    > "${nsys_dir}/taehv-load-evidence.txt"
  no_offload_fit_marker="${nsys_dir}/NO_OFFLOAD_PROTOCOL_FIT_PASS.json"
  record_no_offload_protocol_fit \
    "${nsys_dir}/protocol-smoke.json" \
    "${nsys_dir}/server.log" \
    "${no_offload_fit_marker}"
  assert_no_offload_protocol_fit "${no_offload_fit_marker}"

  capture_log_start_line="$(( $(wc -l < "${nsys_dir}/server.log") + 1 ))"
  nsys start \
    --session="${nsys_session}" \
    --output="${nsys_dir}/minwm-local-taehv" \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --sample=none
  capture_active=1
  python3 "${PROFILE_CLIENT}" \
    --cases "${CASES}" \
    --case "${CASE}" \
    --profile-name "${MINWM_GPU_SKU,,}-local-taehv-main-segment-nsys" \
    --sink-size 8 \
    --kv-cache-num-frames 32 \
    --warmup-chunks 8 \
    --measured-chunks 8 \
    --timeout 3600 \
    --output "${nsys_dir}/throughput.json" \
    > "${nsys_dir}/profile-client.log" 2>&1
  assert_profile_result "${nsys_dir}/throughput.json" 8 8
  timeout --kill-after=5s 20s \
    nsys stop --session="${nsys_session}"
  capture_active=0
  capture_log_end_line="$(wc -l < "${nsys_dir}/server.log")"
  sed -n "${capture_log_start_line},${capture_log_end_line}p" \
    "${nsys_dir}/server.log" > "${nsys_dir}/capture-server.log"
  python3 - "${nsys_dir}/capture-server.log" \
    "${nsys_dir}/capture-chunks.json" <<'PY'
import json
import sys

chunks = []
for line in open(sys.argv[1]):
    marker = "realtime_trace "
    if marker not in line:
        continue
    event = json.loads(line.split(marker, 1)[1])
    if event.get("event") == "server.chunk_complete":
        chunks.append({
            "chunk_index": int(event["chunk_index"]),
            "frame_shape": event.get("frame_shape"),
            "num_frames": int(event["num_frames"]),
            "server_epoch_ms": int(event["server_epoch_ms"]),
        })
assert [item["chunk_index"] for item in chunks] == list(range(16)), chunks
assert all(item["frame_shape"] == [704, 1248, 3] for item in chunks), chunks
assert chunks[0]["num_frames"] == 17
assert all(item["num_frames"] == 16 for item in chunks[1:])
with open(sys.argv[2], "w") as handle:
    json.dump({
        "captured_chunks": chunks,
        "measured_chunk_indices": list(range(8, 16)),
        "warmup_chunk_indices": list(range(8)),
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  stop_server

  report="${nsys_dir}/minwm-local-taehv.nsys-rep"
  sqlite="${nsys_dir}/minwm-local-taehv.sqlite"
  [[ -f "${report}" ]]
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "${report}" \
    > "${nsys_dir}/stats.txt"
  nsys stats --report nvtx_gpu_proj_sum "${report}" \
    > "${nsys_dir}/nvtx-gpu-proj.txt" || true
  nsys export --type=sqlite --output="${sqlite}" --force-overwrite=true "${report}"
  python3 - "${sqlite}" "${nsys_dir}/metrics.json" \
    "${nsys_dir}/sqlite-schema.json" <<'PY'
import json
import os
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
sqlite_master = cur.execute(
    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
).fetchall()
tables = {row[1] for row in sqlite_master if row[0] == "table"}
columns = {
    table: [row[1] for row in cur.execute(f'PRAGMA table_info("{table}")')]
    for table in sorted(tables)
}
with open(sys.argv[3], "w") as handle:
    json.dump(
        {"columns": columns, "sqlite_master": sqlite_master},
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")

required_tables = {
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "ENUM_NSYS_EVENT_TYPE",
    "GPU_METRICS",
    "NVTX_EVENTS",
    "StringIds",
    "TARGET_INFO_GPU_METRICS",
}
assert required_tables <= tables, {
    "missing_tables": sorted(required_tables - tables),
    "schema_path": sys.argv[3],
}
required_columns = {
    "CUPTI_ACTIVITY_KIND_KERNEL": {"deviceId", "end", "start"},
    "ENUM_NSYS_EVENT_TYPE": {"id", "name"},
    "GPU_METRICS": {"metricId", "timestamp", "typeId", "value"},
    "NVTX_EVENTS": {"end", "eventType", "start", "text", "textId"},
    "StringIds": {"id", "value"},
    "TARGET_INFO_GPU_METRICS": {"metricId", "metricName"},
}
missing_columns = {
    table: sorted(expected - set(columns[table]))
    for table, expected in required_columns.items()
    if expected - set(columns[table])
}
assert not missing_columns, {
    "missing_columns": missing_columns,
    "schema_path": sys.argv[3],
}

stage_name = "stage_MinWMCausalDMDDenoisingStage"
paired_event_kinds = (
    "NvtxPushPopRange",
    "NvtxStartEndRange",
)
right_partial_event_kinds = ("NvtxPushRange", "NvtxStartRange")
left_partial_event_kinds = ("NvtxPopRange", "NvtxEndRange")
all_target_event_kinds = (
    *paired_event_kinds,
    *right_partial_event_kinds,
    *left_partial_event_kinds,
)
stage_discovery_rows = cur.execute(
    """
    SELECT et.name,COALESCE(NULLIF(n.text,''),s.value),COUNT(*),
           SUM(n.start<0),SUM(n.end IS NULL)
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id=n.textId
    JOIN ENUM_NSYS_EVENT_TYPE AS et ON et.id=n.eventType
    WHERE COALESCE(NULLIF(n.text,''),s.value) LIKE 'stage_%'
    GROUP BY et.name,COALESCE(NULLIF(n.text,''),s.value)
    ORDER BY COALESCE(NULLIF(n.text,''),s.value),et.name
    """
).fetchall()
def target_range_evidence(label):
    status_rows = cur.execute(
        """
        WITH target AS (
          SELECT n.start,n.end,et.name AS event_kind,
                 CASE
                   WHEN et.name IN ('NvtxPopRange','NvtxEndRange') OR n.start < 0
                     THEN 'partial_left'
                   WHEN et.name IN ('NvtxPushRange','NvtxStartRange') OR n.end IS NULL
                     THEN 'partial_right'
                   WHEN n.end <= n.start THEN 'invalid'
                   WHEN et.name IN ('NvtxPushPopRange','NvtxStartEndRange')
                     THEN 'complete'
                   ELSE 'invalid'
                 END AS range_status
          FROM NVTX_EVENTS AS n
          LEFT JOIN StringIds AS s ON s.id=n.textId
          JOIN ENUM_NSYS_EVENT_TYPE AS et ON et.id=n.eventType
          WHERE et.name IN (?,?,?,?,?,?)
            AND COALESCE(NULLIF(n.text,''),s.value)=?
        )
        SELECT range_status,COUNT(*) FROM target GROUP BY range_status
        ORDER BY range_status
        """,
        (*all_target_event_kinds, label),
    ).fetchall()
    counts = {status: count for status, count in status_rows}
    for status in ("complete", "invalid", "partial_left", "partial_right"):
        counts.setdefault(status, 0)
    rows = cur.execute(
        """
        SELECT n.start,n.end,COALESCE(NULLIF(n.text,''),s.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id=n.textId
        JOIN ENUM_NSYS_EVENT_TYPE AS et ON et.id=n.eventType
        WHERE et.name IN (?,?)
          AND COALESCE(NULLIF(n.text,''),s.value)=?
          AND n.start>=0 AND n.end IS NOT NULL AND n.end>n.start
        ORDER BY n.start
        """,
        (*paired_event_kinds, label),
    ).fetchall()
    return counts, rows


expected_range_status_counts = {
    "complete": 16,
    "invalid": 0,
    "partial_left": 0,
    "partial_right": 0,
}
range_status_counts, stage_rows = target_range_evidence(stage_name)
assert range_status_counts == expected_range_status_counts, range_status_counts
stage_names = {row[2] for row in stage_rows}
assert stage_names == {stage_name}, stage_names
assert all(start < end for start, end, _ in stage_rows)
assert all(
    stage_rows[index][1] <= stage_rows[index + 1][0]
    for index in range(len(stage_rows) - 1)
)
candidate_evidence_required = (
    os.environ["MINWM_REQUIRE_CANDIDATE_EVIDENCE"] == "true"
)
action_name = "minwm_action_residual_prepare_once_per_chunk"
action_range_status_counts = None
action_rows = []
if candidate_evidence_required:
    action_range_status_counts, action_rows = target_range_evidence(action_name)
    assert action_range_status_counts == expected_range_status_counts, (
        action_range_status_counts
    )
    assert len(action_rows) == len(stage_rows)
    for stage, action in zip(stage_rows, action_rows):
        assert stage[0] <= action[0] < action[1] <= stage[1], (stage, action)
stage_kernel_counts = [
    cur.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start>=? AND end<=? AND end>start",
        (start, end),
    ).fetchone()[0]
    for start, end, _ in stage_rows
]
assert all(count > 0 for count in stage_kernel_counts), stage_kernel_counts
gpu_metrics = cur.execute("SELECT COUNT(*) FROM GPU_METRICS").fetchone()[0]
kernels = cur.execute(
    "SELECT deviceId, COUNT(*), SUM(end-start) "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId ORDER BY deviceId"
).fetchall()
assert gpu_metrics > 0, "GPU_METRICS is empty; SYS_ADMIN/counter capture failed"
assert len(kernels) == 1, kernels
kernel_min, kernel_max = cur.execute(
    "SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
).fetchone()
assert kernel_min is not None and kernel_max is not None
names = {}
for metric_id in (3, 5, 18, 19):
    row = cur.execute(
        "SELECT metricName FROM TARGET_INFO_GPU_METRICS "
        "WHERE metricId=? LIMIT 1", (metric_id,)
    ).fetchone()
    names[metric_id] = row[0] if row else f"metric_{metric_id}"
def metrics_for_window(start, end):
    metrics = {}
    for type_id, metric_id, mean_value, samples in cur.execute(
        "SELECT typeId,metricId,AVG(value),COUNT(*) FROM GPU_METRICS "
        "WHERE metricId IN (3,5,18,19) AND timestamp BETWEEN ? AND ? "
        "GROUP BY typeId,metricId ORDER BY typeId,metricId",
        (start, end),
    ):
        metrics.setdefault(str(type_id), {})[names[metric_id]] = {
            "mean": mean_value,
            "samples": samples,
        }
    return metrics

capture_start = stage_rows[0][0]
capture_end = stage_rows[-1][1]
measured_start = stage_rows[8][0]
result = {
    "capture_chunk_count": 16,
    "candidate_evidence_required": candidate_evidence_required,
    "denoise_stage_name": stage_name,
    "denoise_stage_range_status_counts": range_status_counts,
    "denoise_stage_ranges": [
        {
            "chunk_index": index,
            "duration_ms": (end - start) / 1e6,
            "end": end,
            "fully_contained_kernel_count": stage_kernel_counts[index],
            "start": start,
        }
        for index, (start, end, _) in enumerate(stage_rows)
    ],
    "gpu_metric_samples": gpu_metrics,
    "gpu_metrics_all_chunks": metrics_for_window(capture_start, capture_end),
    "gpu_metrics_measured_chunks": metrics_for_window(measured_start, capture_end),
    "kernels": [
        {"device_id": row[0], "count": row[1], "total_ms": row[2] / 1e6}
        for row in kernels
    ],
    "measured_chunk_indices": list(range(8, 16)),
    "action_prepare_once_per_chunk": {
        "name": action_name,
        "range_status_counts": action_range_status_counts,
        "ranges": [
            {
                "chunk_index": index,
                "duration_ms": (end - start) / 1e6,
                "end": end,
                "start": start,
            }
            for index, (start, end, _) in enumerate(action_rows)
        ],
    },
    "nvtx_stage_discovery": [
        {
            "event_kind": row[0],
            "label": row[1],
            "ranges": row[2],
            "start_before_capture": row[3],
            "end_missing": row[4],
        }
        for row in stage_discovery_rows
    ],
    "sqlite_schema_path": sys.argv[3],
    "warmup_chunk_indices": list(range(8)),
}
with open(sys.argv[2], "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
fi

python3 - <<'PY' | tee "${LOCAL_RESULTS}/runtime-final.json"
import importlib.metadata
import json
import torch

packages = ("diffusers", "torch", "transformers")
print(json.dumps({
    "cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0),
    "packages": {name: importlib.metadata.version(name) for name in packages},
    "torch": torch.__version__,
}, indent=2, sort_keys=True))
PY
printf 'status=0\nrun_id=%s\nmode=%s\n' \
  "${MINWM_RUN_ID}" "${MINWM_PROFILE_MODE}" \
  > "${LOCAL_RESULTS}/RUN_COMPLETE"
archive_results
printf 'MINWM_SINGLE_GPU_TAEHV24_COMPLETE mode=%s results=%s\n' \
  "${MINWM_PROFILE_MODE}" "${REMOTE_RESULTS}"
cp "${LOCAL_RESULTS}/RUN_COMPLETE" "${REMOTE_RESULTS}/SUCCESS"
trap - EXIT INT TERM
