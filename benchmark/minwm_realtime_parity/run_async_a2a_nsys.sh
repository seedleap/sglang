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
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
CASE_ID="${MINWM_CASE_ID:-00_forward_080_pottery_720p}"
SP_DEGREE="${MINWM_ASYNC_A2A_NSYS_SP_DEGREE:-2}"
PRECONDITION_CHUNKS="${MINWM_ASYNC_A2A_NSYS_PRECONDITION_CHUNKS:-20}"
DISCARD_CHUNKS="${MINWM_ASYNC_A2A_NSYS_DISCARD_CHUNKS:-1}"
MEASURED_CHUNKS="${MINWM_ASYNC_A2A_NSYS_MEASURED_CHUNKS:-10}"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"
A2A_BACKEND="${MINWM_ASYNC_A2A_BENCH_BACKEND:-process_group}"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/async-a2a-nsys"
NSYS_URL="${MINWM_NSYS_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb}"
NSYS_ROOT="${WORK_ROOT}/nsight-systems"
NSYS_DEB="${WORK_ROOT}/nsight-systems-cli.deb"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if [[ "${SP_DEGREE}" != "2" ]]; then
  echo "Initial async A2A Nsight attribution requires SP2" >&2
  exit 2
fi
if ! [[ "${PRECONDITION_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${DISCARD_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Nsight chunk counts must be positive integers" >&2
  exit 2
fi
if (( MEASURED_CHUNKS < 10 )); then
  echo "Nsight attribution requires at least 10 stable chunks" >&2
  exit 2
fi
if [[ -n "${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-}" ]]; then
  echo "SGLANG_DIFFUSION_TORCH_PROFILER_DIR must be unset" >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite Nsight attempt: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}" "${NSYS_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export SGLANG_REALTIME_TRACE_SYNC_CUDA=0
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
  echo "active_gpu_count=${SP_DEGREE}"
  echo "sp_degree=${SP_DEGREE}"
  echo "resolution=1248x704"
  echo "precondition_chunks=${PRECONDITION_CHUNKS}"
  echo "discard_chunks=${DISCARD_CHUNKS}"
  echo "stable_chunks=${MEASURED_CHUNKS}"
  echo "a2a_backend=${A2A_BACKEND}"
  echo "candidate_output_a2a=disabled"
  echo "trace=cuda,nvtx"
  echo "trace_fork_before_exec=true"
  echo "cuda_graph_trace=node"
  echo "gpu_metrics_devices=all"
  echo "gpu_metrics_frequency=10000"
  echo "torch_profiler_concurrent=false"
  echo "trace_sync_cuda=0"
  echo "profiler_wall_headline_eligible=false"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RESULT_ROOT}/contract.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.txt"

if ! command -v nsys >/dev/null; then
  curl --fail --location --retry 3 --output "${NSYS_DEB}" "${NSYS_URL}"
  sha256sum "${NSYS_DEB}" | tee "${RESULT_ROOT}/nsys-package-sha256.txt"
  dpkg-deb --extract "${NSYS_DEB}" "${NSYS_ROOT}"
  NSYS_BIN="$(find "${NSYS_ROOT}" -type f -name nsys -perm -111 -print -quit)"
  [[ -n "${NSYS_BIN}" ]]
  export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
fi
nsys --version | tee "${RESULT_ROOT}/nsys-version.txt"
nsys status -e 2>&1 | tee "${RESULT_ROOT}/nsys-status.txt" || true
nsys profile --gpu-metrics-devices=help 2>&1 \
  | tee "${RESULT_ROOT}/nsys-gpu-metrics-devices.txt" || true

server_pid=""
nsys_session=""
failure_scope="${RESULT_ROOT}"

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
}

cleanup() {
  if [[ -n "${nsys_session}" ]]; then
    nsys stop --session="${nsys_session}" 2>/dev/null || true
    nsys_session=""
  fi
  stop_server
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  cleanup
  if (( status != 0 )) && [[ -d "${failure_scope}" ]]; then
    local timestamp
    timestamp="$(date --utc +%Y%m%dT%H%M%S%NZ)"
    python3 "${SCRIPT_DIR}/measurement_tool.py" mark-invalid \
      --root "${failure_scope}" \
      --reason "async A2A Nsight runner exited non-zero (status=${status})" \
      --marker "${failure_scope}/invalid-marker-${timestamp}.json" || true
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_args() {
  printf '%s\0' \
    sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --vae-config.use-parallel-decode true \
    --attention-backend fa \
    --performance-mode speed \
    --num-gpus "${SP_DEGREE}" \
    --tp-size 1 \
    --sp-degree "${SP_DEGREE}" \
    --ulysses-degree "${SP_DEGREE}" \
    --ring-degree 1 \
    --enable-cfg-parallel false \
    --enable-torch-compile false \
    --warmup-mode off \
    --port 30000
}

client_common_args() {
  local output="$1" profile_name="$2" run_id="$3"
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
    --gpu-count "${SP_DEGREE}" \
    --allocated-gpu-count "${ALLOCATED_GPU_COUNT}" \
    --sp-degree "${SP_DEGREE}" \
    --precision bf16 \
    --fast-lane \
    --checkpoint-id global_step_003200/ema_student/model.pt \
    --checkpoint-step 3200 \
    --kv-cache-num-frames "${KV_CACHE_NUM_FRAMES}" \
    --require-complete-stage-trace
}

run_client() {
  local output="$1" profile_name="$2" run_id="$3"
  shift 3
  mapfile -d '' -t CLIENT_ARGS < <(
    client_common_args "${output}" "${profile_name}" "${run_id}"
  )
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    "${CLIENT_ARGS[@]}" "$@" | tee "${output%.json}.log"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${output}"
}

readarray -d '' -t SERVER_ARGS < <(server_args)
for lane in baseline candidate; do
  lane_dir="${RESULT_ROOT}/${lane}"
  failure_scope="${lane_dir}"
  mkdir -p "${lane_dir}"
  session="minwm-${MINWM_RUN_ID}-${lane}"
  report="${lane_dir}/${lane}.nsys-rep"
  sqlite="${lane_dir}/${lane}.sqlite"
  status_log="${lane_dir}/nsys-capture-status.log"
  async_flag=0
  if [[ "${lane}" == "candidate" ]]; then
    async_flag=1
  fi

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_ASYNC_A2A="${async_flag}" \
  MINWM_ASYNC_A2A_OUTPUT=0 \
  MINWM_ASYNC_A2A_BACKEND="${A2A_BACKEND}" \
  SGLANG_REALTIME_NSYS_WARMUP_CHUNKS="${DISCARD_CHUNKS}" \
  SGLANG_REALTIME_NSYS_MEASURED_CHUNKS="${MEASURED_CHUNKS}" \
  nsys launch \
    --session-new="${session}" \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -- "${SERVER_ARGS[@]}" > "${lane_dir}/server.log" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${lane_dir}/server.log"

  run_client \
    "${lane_dir}/precondition.json" \
    "${lane}-sp${SP_DEGREE}-precondition" \
    "${MINWM_RUN_ID}-${lane}-precondition" \
    --measurement-mode profiler_off \
    --warmup-chunks "$((PRECONDITION_CHUNKS - 1))" \
    --measured-chunks 1

  nsys_session="${session}"
  if nsys start \
    --session="${session}" \
    --output="${lane_dir}/${lane}" \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --sample=none > >(tee "${status_log}") 2>&1; then
    echo "gpu_metrics_start=success" | tee -a "${status_log}"
  else
    echo "gpu_metrics_start=failed; retrying CUDA/NVTX without GPU metrics" \
      | tee -a "${status_log}"
    nsys start \
      --session="${session}" \
      --output="${lane_dir}/${lane}" \
      --sample=none 2>&1 | tee -a "${status_log}"
  fi

  run_client \
    "${lane_dir}/profile-client.json" \
    "${lane}-sp${SP_DEGREE}-nsys" \
    "${MINWM_RUN_ID}-${lane}-nsys" \
    --measurement-mode profiler_off \
    --precondition-warmup-chunks "${PRECONDITION_CHUNKS}" \
    --warmup-chunks "${DISCARD_CHUNKS}" \
    --measured-chunks "${MEASURED_CHUNKS}"
  nsys stop --session="${session}" 2>&1 | tee -a "${status_log}"
  nsys_session=""
  stop_server

  [[ -f "${report}" ]]
  nsys export --type=sqlite --output="${sqlite}" "${report}"
  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "${report}" \
    > "${lane_dir}/nsys-stats.txt"
  python3 "${SCRIPT_DIR}/async_a2a_nsys_metrics.py" analyze \
    --sqlite "${sqlite}" \
    --client "${lane_dir}/profile-client.json" \
    --lane "${lane}" \
    --status-log "${status_log}" \
    --output "${lane_dir}/a2a-metrics.json" \
    | tee "${lane_dir}/a2a-metrics.log"
done

failure_scope="${RESULT_ROOT}"
python3 "${SCRIPT_DIR}/async_a2a_nsys_metrics.py" compare \
  --baseline "${RESULT_ROOT}/baseline/a2a-metrics.json" \
  --candidate "${RESULT_ROOT}/candidate/a2a-metrics.json" \
  --output "${RESULT_ROOT}/comparison.json" \
  | tee "${RESULT_ROOT}/comparison.log"
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/completed-utc.txt"
echo "MINWM_ASYNC_A2A_NSYS_COMPLETE results=${RESULT_ROOT}"
