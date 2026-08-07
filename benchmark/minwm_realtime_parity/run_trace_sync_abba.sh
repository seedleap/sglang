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
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/trace-sync-abba"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
CASE_ID="${MINWM_CASE_ID:-00_forward_080_pottery_720p}"
SP_DEGREES="${MINWM_TRACE_SYNC_SP_DEGREES:-2 4}"
REPEAT_NUMBERS="${MINWM_TRACE_SYNC_REPEATS:-1 2}"
WARMUP_CHUNKS="${MINWM_TRACE_SYNC_WARMUP_CHUNKS:-20}"
MEASURED_CHUNKS="${MINWM_TRACE_SYNC_MEASURED_CHUNKS:-200}"
KV_CACHE_NUM_FRAMES="${MINWM_TRACE_SYNC_KV_CACHE_NUM_FRAMES:-45}"

read -r -a degrees <<< "${SP_DEGREES}"
read -r -a repeats <<< "${REPEAT_NUMBERS}"
if (( ${#repeats[@]} != 2 )) \
  || ! [[ "${repeats[0]}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${repeats[1]}" =~ ^[1-9][0-9]*$ ]] \
  || [[ "${repeats[0]}" == "${repeats[1]}" ]]; then
  echo "MINWM_TRACE_SYNC_REPEATS requires two distinct positive integers" >&2
  exit 2
fi

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if [[ "${WARMUP_CHUNKS}" != "20" || ! "${MEASURED_CHUNKS}" =~ ^[0-9]+$ ]] \
  || (( MEASURED_CHUNKS < 200 )); then
  echo "Trace-sync headline requires exactly 20 warmup and at least 200 measured" >&2
  exit 2
fi
if [[ -n "${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-}" ]]; then
  echo "SGLANG_DIFFUSION_TORCH_PROFILER_DIR must be unset for headline A/B" >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite pre-existing result root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
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
  echo "repeats=${REPEAT_NUMBERS}"
  echo "window=${WARMUP_CHUNKS}+${MEASURED_CHUNKS}"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "order=control${repeats[0]},candidate${repeats[0]},candidate${repeats[1]},control${repeats[1]}"
  echo "control=SGLANG_REALTIME_TRACE_SYNC_CUDA=1"
  echo "candidate=SGLANG_REALTIME_TRACE_SYNC_CUDA=0"
  echo "torch_profiler_concurrent=false"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RESULT_ROOT}/contract.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.txt"

server_pid=""
monitor_pid=""
failure_scope="${RESULT_ROOT}"

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

mark_failed_attempt() {
  local status="$1"
  python3 "${SCRIPT_DIR}/measurement_tool.py" mark-invalid \
    --root "${failure_scope}" \
    --reason "trace-sync ABBA runner exited non-zero (status=${status})" \
    --marker "${failure_scope}/invalid-marker-$(date --utc +%Y%m%dT%H%M%SZ).json" \
    || true
}

on_exit() {
  local status="$?"
  trap - EXIT INT TERM
  stop_server
  if (( status != 0 )); then
    mark_failed_attempt "${status}"
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local degree="$1" sync_cuda="$2" log_path="$3" telemetry_path="$4"
  mapfile -d '' -t SERVER_ARGS < <(server_args "${degree}")
  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  SGLANG_REALTIME_TRACE_SYNC_CUDA="${sync_cuda}" \
    "${SERVER_ARGS[@]}" > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${log_path}"
  (
    sample=0
    while kill -0 "${server_pid}" 2>/dev/null; do
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu,clocks.sm,pstate,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        | sed "s/^/${sample},/" || true
      sample=$((sample + 1))
      sleep 1
    done
  ) > "${telemetry_path}" &
  monitor_pid=$!
}

summarize_telemetry() {
  local telemetry_path="$1" output="$2" degree="$3"
  python3 - "${telemetry_path}" "${output}" "${degree}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

source, output, degree = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
memory_by_sample = defaultdict(dict)
with source.open(newline="") as stream:
    for row in csv.reader(stream):
        if len(row) < 3:
            continue
        sample, device, memory = int(row[0]), int(row[1]), float(row[2])
        if device < degree:
            memory_by_sample[sample][device] = memory
complete = [
    (sample, sum(devices.values()))
    for sample, devices in sorted(memory_by_sample.items())
    if len(devices) == degree
]
if not complete:
    raise SystemExit("no complete active-GPU telemetry samples")
values = [value for _sample, value in complete]
tail = values[-min(20, len(values)):]
middle_start = len(values) // 2
middle = values[middle_start:middle_start + min(20, len(values) - middle_start)]
summary = {
    "sample_count": len(values),
    "active_gpu_count": degree,
    "peak_active_memory_mib": max(values),
    "middle_active_memory_mib_mean": statistics.fmean(middle),
    "tail_active_memory_mib_mean": statistics.fmean(tail),
    "tail_minus_middle_memory_mib": statistics.fmean(tail) - statistics.fmean(middle),
    "source": str(source),
}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
}

verify_trace_status() {
  local log_path="$1" expected="$2"
  python3 - "${log_path}" "${expected}" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
expected = sys.argv[2]
needle = f'"component":"minwm_denoising"'
status = f'"cuda_timing_status":"{expected}"'
matching = [line for line in text.splitlines() if needle in line and status in line]
if not matching:
    raise SystemExit(f"missing minwm_denoising cuda_timing_status={expected}")
print(f"trace_status={expected} matching_events={len(matching)}")
PY
}

run_one() {
  local degree="$1" arm="$2" repeat="$3"
  local sync_cuda expected_status
  if [[ "${arm}" == "control" ]]; then
    sync_cuda=1
    expected_status=available
  else
    sync_cuda=0
    expected_status=disabled
  fi
  local lane_dir="${RESULT_ROOT}/sp${degree}"
  local prefix="${lane_dir}/${arm}-repeat${repeat}"
  mkdir -p "${lane_dir}"
  failure_scope="${lane_dir}"
  start_server "${degree}" "${sync_cuda}" "${prefix}-server.log" "${prefix}-gpu-telemetry.csv"
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --output "${prefix}.json" \
    --profile-name "bf16-fast-sp${degree}-trace-sync-${arm}" \
    --run-id "${MINWM_RUN_ID}-sp${degree}-${arm}-r${repeat}" \
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
    --server-trace-sync-cuda "${sync_cuda}" \
    --require-complete-stage-trace \
    --measurement-mode profiler_off \
    --warmup-chunks "${WARMUP_CHUNKS}" \
    --measured-chunks "${MEASURED_CHUNKS}" \
    | tee "${prefix}-client.log"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${prefix}.json"
  stop_server
  summarize_telemetry \
    "${prefix}-gpu-telemetry.csv" \
    "${prefix}-telemetry-summary.json" \
    "${degree}"
  verify_trace_status "${prefix}-server.log" "${expected_status}" \
    | tee "${prefix}-trace-status.txt"
}

for degree in "${degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(2|4)$ ]]; then
    echo "Trace-sync A/B accepts SP degree 2 or 4, got ${degree}" >&2
    exit 2
  fi
  run_one "${degree}" control "${repeats[0]}"
  run_one "${degree}" candidate "${repeats[0]}"
  run_one "${degree}" candidate "${repeats[1]}"
  run_one "${degree}" control "${repeats[1]}"
  python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
    "${RESULT_ROOT}/sp${degree}/control-repeat${repeats[0]}.json" \
    "${RESULT_ROOT}/sp${degree}/control-repeat${repeats[1]}.json" \
    --output "${RESULT_ROOT}/sp${degree}/control-summary.json"
  python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
    "${RESULT_ROOT}/sp${degree}/candidate-repeat${repeats[0]}.json" \
    "${RESULT_ROOT}/sp${degree}/candidate-repeat${repeats[1]}.json" \
    --output "${RESULT_ROOT}/sp${degree}/candidate-summary.json"
done

failure_scope="${RESULT_ROOT}"
python3 "${SCRIPT_DIR}/compare_trace_sync_abba.py" \
  --root "${RESULT_ROOT}" \
  --sp-degrees "${degrees[@]}" \
  --repeats "${repeats[@]}" \
  --output "${RESULT_ROOT}/comparison-summary.json"
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/complete.txt"
echo "MINWM_TRACE_SYNC_ABBA_COMPLETE results=${RESULT_ROOT}"
