#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_CONTAINER_IMAGE:?set MINWM_CONTAINER_IMAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}"
MODEL_DIR="${WORK_ROOT}/sglang-model"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/s4-qkv-measurement"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
CASE_ID="${MINWM_CASE_ID:-00_forward_080_pottery_720p}"
SP_DEGREES="${MINWM_S0_SP_DEGREES:-2 4}"
QKV_LANES="${MINWM_S4_QKV_LANES:-control candidate}"
OFF_WARMUP_CHUNKS="${MINWM_S0_OFF_WARMUP_CHUNKS:-20}"
OFF_MEASURED_CHUNKS="${MINWM_S0_OFF_MEASURED_CHUNKS:-200}"
PROFILE_PRECONDITION_CHUNKS="${MINWM_S0_PROFILE_PRECONDITION_CHUNKS:-20}"
PROFILE_DISCARD_CHUNKS="${MINWM_S0_PROFILE_DISCARD_CHUNKS:-1}"
PROFILE_MEASURED_CHUNKS="${MINWM_S0_PROFILE_MEASURED_CHUNKS:-10}"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"
NSYS_URL="${MINWM_NSYS_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb}"
NSYS_ROOT="${WORK_ROOT}/nsight-systems"
NSYS_DEB="${WORK_ROOT}/nsight-systems-cli.deb"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if ! [[ "${OFF_WARMUP_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${OFF_MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PROFILE_PRECONDITION_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PROFILE_DISCARD_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${PROFILE_MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "All warmup/measurement chunk counts must be positive integers" >&2
  exit 2
fi
if (( PROFILE_MEASURED_CHUNKS < 10 )); then
  echo "Nsight capture requires at least 10 stable measured chunks" >&2
  exit 2
fi
if [[ -n "${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-}" ]]; then
  echo "SGLANG_DIFFUSION_TORCH_PROFILER_DIR must be unset during Nsight capture" >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite existing S4 measurement attempt: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export SGLANG_REALTIME_TRACE_SYNC_CUDA=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR

GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
ALLOCATED_GPU_COUNT="${MINWM_ALLOCATED_GPU_COUNT:-$(nvidia-smi -L | wc -l | xargs)}"
{
  echo "sglang_commit=${SGLANG_GIT_REF}"
  echo "minwm_commit=${MINWM_GIT_REF}"
  echo "container_image=${MINWM_CONTAINER_IMAGE}"
  echo "gpu_model=${GPU_MODEL}"
  echo "allocated_gpu_count=${ALLOCATED_GPU_COUNT}"
  echo "sp_degrees=${SP_DEGREES}"
  echo "qkv_lanes=${QKV_LANES}"
  echo "qkv_control=MINWM_FUSED_QKV_PROJECTION=0"
  echo "qkv_candidate=MINWM_FUSED_QKV_PROJECTION=1"
  echo "off_window=${OFF_WARMUP_CHUNKS}+${OFF_MEASURED_CHUNKS}"
  echo "nsys_window=${PROFILE_PRECONDITION_CHUNKS} precondition + ${PROFILE_DISCARD_CHUNKS} discarded + ${PROFILE_MEASURED_CHUNKS} stable"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "torch_profiler_concurrent=false"
  echo "invalid_attempt_policy=preserve-in-place-with-path-size-sha256-marker"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RESULT_ROOT}/contract.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.txt"

server_pid=""
monitor_pid=""
nsys_session=""
qkv_lane=""
qkv_fused_flag=""

wait_for_server() {
  local pid="$1" log_path="$2"
  for _ in $(seq 1 300); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

stop_server() {
  if [[ -n "${server_pid}" ]]; then
    pkill -TERM -f "sglang serve --model-path ${MODEL_DIR}.*--port 30000" \
      2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}

mark_invalid_attempt() {
  local status="$1"
  python3 - "${RESULT_ROOT}" "${status}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
status = int(sys.argv[2])
invalid_dir = root / "invalid"
invalid_dir.mkdir(parents=True, exist_ok=True)
artifacts = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or invalid_dir in path.parents:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    artifacts.append(
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
marker = {
    "reason": f"runner_exit_status_{status}",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "attempt_root": str(root),
    "artifacts": artifacts,
    "recoverability": (
        "All partial evidence remains in place under this attempt root; "
        "aggregators must exclude directories containing invalid/."
    ),
}
(invalid_dir / "attempt-invalid.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n"
)
PY
}

cleanup() {
  local status="${1:-0}"
  set +e
  if [[ -n "${nsys_session}" ]]; then
    nsys stop --session="${nsys_session}" 2>/dev/null || true
    nsys_session=""
  fi
  stop_server
  if (( status != 0 )); then
    mark_invalid_attempt "${status}"
  fi
}

on_exit() {
  local status=$?
  cleanup "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_args() {
  local degree="$1"
  printf '%s\0' \
    sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --vae-config.use-parallel-decode true \
    --attention-backend fa \
    --performance-mode speed \
    --num-gpus "${degree}" \
    --tp-size 1 \
    --sp-degree "${degree}" \
    --ulysses-degree "${degree}" \
    --ring-degree 1 \
    --enable-cfg-parallel false \
    --enable-torch-compile false \
    --warmup-mode off \
    --port 30000
}

read_server_args() {
  local degree="$1"
  mapfile -d '' -t SERVER_ARGS < <(server_args "${degree}")
}

start_server() {
  local degree="$1" log_path="$2" telemetry_path="$3"
  read_server_args "${degree}"
  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_FUSED_QKV_PROJECTION="${qkv_fused_flag}" \
    "${SERVER_ARGS[@]}" > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${log_path}"
  (
    while kill -0 "${server_pid}" 2>/dev/null; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,clocks.sm,pstate,power.draw,temperature.gpu,memory.used \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${telemetry_path}" &
  monitor_pid=$!
}

client_common_args() {
  local degree="$1" output="$2" profile_name="$3" run_id="$4"
  printf '%s\0' \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --output "${output}" \
    --profile-name "${profile_name}" \
    --run-id "${run_id}" \
    --sglang-commit "${SGLANG_GIT_REF}" \
    --minwm-commit "${MINWM_GIT_REF}" \
    --container-image "${MINWM_CONTAINER_IMAGE}" \
    --gpu-model "${GPU_MODEL}" \
    --gpu-count "${degree}" \
    --allocated-gpu-count "${ALLOCATED_GPU_COUNT}" \
    --sp-degree "${degree}" \
    --precision bf16 \
    --fast-lane \
    --checkpoint-id global_step_003200/ema_student/model.pt \
    --checkpoint-step 3200 \
    --kv-cache-num-frames "${KV_CACHE_NUM_FRAMES}" \
    --require-complete-stage-trace
}

run_client() {
  local degree="$1" output="$2" profile_name="$3" run_id="$4"
  shift 4
  mapfile -d '' -t CLIENT_ARGS < <(
    client_common_args "${degree}" "${output}" "${profile_name}" "${run_id}"
  )
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    "${CLIENT_ARGS[@]}" "$@" | tee "${output%.json}.log"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${output}"
  python3 - "${output}" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
expected = record["workload"]["measured_chunks"]
if record["mode"] == "profiler_off":
    container = record["metrics"]["profiler_off"]
    paths = {
        "scheduler_chunk_wall_ms": container["scheduler_chunk_wall_ms"],
        "dit_wall_ms": container["dit_wall_ms"],
        "vae_wall_ms": container["vae_wall_ms"],
    }
else:
    profiler_on = record["metrics"]["profiler_on"]
    observed = profiler_on["observed_wall_with_profiler_overhead"]
    paths = {
        "observed.scheduler_chunk_wall_ms": observed["scheduler_chunk_wall_ms"],
        "observed.dit_wall_ms": observed["dit_wall_ms"],
        "observed.vae_wall_ms": observed["vae_wall_ms"],
        "dit_cuda_ms": profiler_on["dit_cuda_ms"],
        "vae_cuda_ms": profiler_on["vae_cuda_ms"],
    }
for name, metric in paths.items():
    if metric.get("status") != "available":
        raise AssertionError(f"{name} must be available: {metric}")
    count = metric.get("value", {}).get("count")
    if count != expected:
        raise AssertionError(f"{name} count={count}, expected {expected}")
print(f"complete latency counts: mode={record['mode']} count={expected}")
PY
}

aggregate_repeats() {
  local lane_dir="$1"
  shift
  python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate "$@" \
    --output "${lane_dir}/repeat-summary.json"
}

install_nsys() {
  if ! command -v nsys >/dev/null; then
    mkdir -p "${NSYS_ROOT}"
    curl --fail --location --retry 3 --output "${NSYS_DEB}" "${NSYS_URL}"
    sha256sum "${NSYS_DEB}" | tee "${RESULT_ROOT}/nsys-package-sha256.txt"
    dpkg-deb --extract "${NSYS_DEB}" "${NSYS_ROOT}"
    NSYS_BIN="$(find "${NSYS_ROOT}" -type f -name nsys -perm -111 -print -quit)"
    [[ -n "${NSYS_BIN}" ]]
    export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
  fi
  nsys --version | tee "${RESULT_ROOT}/nsys-version.txt"
  nsys status -e 2>&1 | tee "${RESULT_ROOT}/nsys-status.txt" || true
}

run_profiler_off() {
  local degree="$1" lane_dir="$2"
  start_server \
    "${degree}" \
    "${lane_dir}/profiler-off-server.log" \
    "${lane_dir}/profiler-off-gpu-telemetry.csv"
  local repeat_paths=()
  for repeat in 1 2; do
    local output="${lane_dir}/profiler-off-repeat${repeat}.json"
    run_client \
      "${degree}" "${output}" "bf16-${qkv_lane}-sp${degree}" \
      "${MINWM_RUN_ID}-${qkv_lane}-sp${degree}-off-r${repeat}" \
      --measurement-mode profiler_off \
      --warmup-chunks "${OFF_WARMUP_CHUNKS}" \
      --measured-chunks "${OFF_MEASURED_CHUNKS}"
    repeat_paths+=("${output}")
  done
  aggregate_repeats "${lane_dir}" "${repeat_paths[@]}"
  if ! python3 - "${lane_dir}/repeat-summary.json" <<'PY'
import json, sys
raise SystemExit(not json.load(open(sys.argv[1]))["acceptance"]["passes_cv_target"])
PY
  then
    echo "CV target missed after two repeats; collecting an automatic third repeat"
    local output="${lane_dir}/profiler-off-repeat3.json"
    run_client \
      "${degree}" "${output}" "bf16-${qkv_lane}-sp${degree}" \
      "${MINWM_RUN_ID}-${qkv_lane}-sp${degree}-off-r3" \
      --measurement-mode profiler_off \
      --warmup-chunks "${OFF_WARMUP_CHUNKS}" \
      --measured-chunks "${OFF_MEASURED_CHUNKS}"
    repeat_paths+=("${output}")
    python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
      "${repeat_paths[@]}" \
      --noise-explanation \
      "Automatic third repeat collected; inspect profiler-off-gpu-telemetry.csv for clocks, power, temperature, and utilization." \
      --output "${lane_dir}/repeat-summary.json"
  fi
  stop_server
}

run_profiler_on() {
  local degree="$1" lane_dir="$2"
  local profile_dir="${lane_dir}/profiler-on"
  local session="minwm-s4-${MINWM_RUN_ID}-${qkv_lane}-sp${degree}"
  local report="${profile_dir}/sp${degree}.nsys-rep"
  local sqlite="${profile_dir}/sp${degree}.sqlite"
  local status_log="${profile_dir}/nsys-capture-status.log"
  mkdir -p "${profile_dir}"
  read_server_args "${degree}"
  SERVER_ARGS+=(--enable-layerwise-nvtx-marker)
  if [[ -e "${report}" || -e "${sqlite}" ]]; then
    echo "Refusing to overwrite existing Nsight evidence in ${profile_dir}" >&2
    exit 2
  fi

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_FUSED_QKV_PROJECTION="${qkv_fused_flag}" \
  nsys launch \
    --session-new="${session}" \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -- "${SERVER_ARGS[@]}" > "${profile_dir}/server.log" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${profile_dir}/server.log"

  run_client \
    "${degree}" "${profile_dir}/precondition-warmup.json" \
    "bf16-${qkv_lane}-sp${degree}-precondition" \
    "${MINWM_RUN_ID}-${qkv_lane}-sp${degree}-precondition" \
    --measurement-mode profiler_off \
    --warmup-chunks "$((PROFILE_PRECONDITION_CHUNKS - 1))" \
    --measured-chunks 1

  local gpu_devices
  gpu_devices="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
      | awk -v count="${degree}" 'NR <= count' \
      | paste -sd, -
  )"
  [[ -n "${gpu_devices}" ]]
  nsys_session="${session}"
  if nsys start \
    --session="${session}" \
    --output="${profile_dir}/sp${degree}" \
    --gpu-metrics-devices="${gpu_devices}" \
    --gpu-metrics-frequency=10000 \
    --sample=none > >(tee "${status_log}") 2>&1; then
    echo "gpu_metrics_start=success" | tee -a "${status_log}"
  else
    echo "gpu_metrics_start=failed; retrying CUDA/NVTX capture without GPU metrics" \
      | tee -a "${status_log}"
    nsys start \
      --session="${session}" \
      --output="${profile_dir}/sp${degree}" \
      --sample=none 2>&1 | tee -a "${status_log}"
  fi

  run_client \
    "${degree}" "${profile_dir}/client.json" \
    "bf16-${qkv_lane}-sp${degree}-nsys" \
    "${MINWM_RUN_ID}-${qkv_lane}-sp${degree}-nsys" \
    --measurement-mode profiler_on \
    --precondition-warmup-chunks "${PROFILE_PRECONDITION_CHUNKS}" \
    --warmup-chunks "${PROFILE_DISCARD_CHUNKS}" \
    --measured-chunks "${PROFILE_MEASURED_CHUNKS}"
  nsys stop --session="${session}" 2>&1 | tee -a "${status_log}"
  nsys_session=""
  stop_server

  [[ -f "${report}" ]]
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "${report}" \
    > "${profile_dir}/nsys-stats.txt"
  nsys export \
    --type=sqlite \
    --output="${sqlite}" \
    --force-overwrite=true \
    "${report}"
  python3 "${SCRIPT_DIR}/measurement_tool.py" merge-nsys \
    --result "${profile_dir}/client.json" \
    --sqlite "${sqlite}" \
    --status-log "${status_log}" \
    --output "${profile_dir}/measurement.json"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${profile_dir}/measurement.json"
}

install_nsys
read -r -a degrees <<< "${SP_DEGREES}"
read -r -a qkv_lanes <<< "${QKV_LANES}"
for degree in "${degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(2|4)$ ]]; then
    echo "S0 accepts SP degree 2 or 4, got ${degree}" >&2
    exit 2
  fi
  for qkv_lane in "${qkv_lanes[@]}"; do
    case "${qkv_lane}" in
      control) qkv_fused_flag=0 ;;
      candidate) qkv_fused_flag=1 ;;
      *)
        echo "S4 accepts QKV lane control or candidate, got ${qkv_lane}" >&2
        exit 2
        ;;
    esac
    lane_dir="${RESULT_ROOT}/${qkv_lane}/sp${degree}"
    mkdir -p "${lane_dir}"
    {
      echo "qkv_lane=${qkv_lane}"
      echo "MINWM_FUSED_QKV_PROJECTION=${qkv_fused_flag}"
      echo "sp_degree=${degree}"
    } > "${lane_dir}/qkv-contract.txt"
    run_profiler_off "${degree}" "${lane_dir}"
    run_profiler_on "${degree}" "${lane_dir}"
  done
done

python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lanes = {}
for qkv_dir in sorted(root.iterdir()):
    if not qkv_dir.is_dir():
        continue
    degrees = {}
    for lane_dir in sorted(qkv_dir.glob("sp*")):
        repeat = json.loads((lane_dir / "repeat-summary.json").read_text())
        profile = json.loads((lane_dir / "profiler-on/measurement.json").read_text())
        degrees[lane_dir.name] = {
            "profiler_off": repeat,
            "profiler_on": profile,
        }
    lanes[qkv_dir.name] = degrees
summary = {
    "schema_version": "minwm-realtime-s4-qkv-ab/v1",
    "lanes": lanes,
}
(root / "s4-qkv-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    qkv_lane: {
        degree: {
            "off_cv_pass": value["profiler_off"]["acceptance"]["passes_cv_target"],
            "gpu_metrics": {
                name: metric["status"]
                for name, metric in value["profiler_on"]["metrics"]["profiler_on"]["gpu_metrics"].items()
            },
        }
        for degree, value in degrees.items()
    }
    for qkv_lane, degrees in lanes.items()
}, indent=2, sort_keys=True))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/complete.txt"
echo "MINWM_S4_QKV_MEASUREMENT_COMPLETE results=${RESULT_ROOT}"
