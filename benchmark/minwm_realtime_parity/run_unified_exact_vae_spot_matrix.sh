#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_MATRIX_ID:?set MINWM_MATRIX_ID}"
: "${MINWM_EXPECTED_GPU_SUBSTRING:?set MINWM_EXPECTED_GPU_SUBSTRING}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${SCRIPT_DIR}/aws_b200_entrypoint.sh"
TARGET_FPS="${MINWM_TARGET_FPS:-24}"
PARITY_MAX_ABSOLUTE_ERROR="${MINWM_PARITY_MAX_ABSOLUTE_ERROR:-8}"
PARITY_MIN_PSNR_DB="${MINWM_PARITY_MIN_PSNR_DB:-58}"
SP_DEGREES="${MINWM_SP_DEGREES:-1 2 4}"
VAE_GPU_INDICES="${MINWM_VAE_GPU_INDICES:-${MINWM_VAE_GPU_INDEX:-7}}"
VAE_PARALLEL_SIZE="${MINWM_VAE_PARALLEL_SIZE:-1}"
EXPECTED_GPU_COUNT="${MINWM_EXPECTED_GPU_COUNT:-8}"
ALLOW_SHARED_VAE_GPU="${MINWM_ALLOW_SHARED_VAE_GPU:-false}"
DEDICATED_VAE_CUDA_STREAM="${MINWM_DEDICATED_VAE_CUDA_STREAM:-false}"
REPETITIONS="${MINWM_MATRIX_REPETITIONS:-1}"
ENABLE_CUDA_MPS="${MINWM_ENABLE_CUDA_MPS:-false}"
WARMUP_CHUNKS="${MINWM_THROUGHPUT_WARMUP_CHUNKS:-20}"
MEASURED_CHUNKS="${MINWM_THROUGHPUT_MEASURED_CHUNKS:-200}"
CASES_PATH="${MINWM_THROUGHPUT_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
THROUGHPUT_CASE="${MINWM_THROUGHPUT_CASE:-00_forward_080_pottery_720p}"
CONTRACT_SIZES="${MINWM_CONTRACT_SIZES:-832x480}"
SHM_ROOT="${MINWM_REALTIME_VAE_SHM_DIR:-/dev/shm/sglang-realtime-vae}"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_MATRIX_ID}"
ARCHIVE_ROOT="${MINWM_ARCHIVE_ROOT:-}"
SETUP_RUN="${MINWM_MATRIX_ID}-setup"
MODEL_DIR="/work/minwm-realtime/${SETUP_RUN}/sglang-model"

mkdir -p "${RESULT_ROOT}" "${SHM_ROOT}"
exec > >(tee -a "${RESULT_ROOT}/${MINWM_MATRIX_ID}-job.log") 2>&1

case "${TARGET_FPS}" in
  ''|*[!0-9.]*|.*|*.*.*)
    echo "MINWM_TARGET_FPS must be a positive number" >&2
    exit 2
    ;;
esac
python3 - "${TARGET_FPS}" <<'PY'
import sys

if float(sys.argv[1]) <= 0:
    raise SystemExit("MINWM_TARGET_FPS must be positive")
PY
if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MINWM_MATRIX_REPETITIONS must be a positive integer" >&2
  exit 2
fi

read -r -a requested_degrees <<< "${SP_DEGREES}"
if ! [[ "${VAE_PARALLEL_SIZE}" =~ ^[1-8]$ ]]; then
  echo "MINWM_VAE_PARALLEL_SIZE must be an integer from 1 through 8" >&2
  exit 2
fi
IFS=',' read -r -a vae_gpu_indices <<< "${VAE_GPU_INDICES}"
if (( ${#vae_gpu_indices[@]} != VAE_PARALLEL_SIZE )); then
  echo "MINWM_VAE_GPU_INDICES count must equal MINWM_VAE_PARALLEL_SIZE" >&2
  exit 2
fi
for degree in "${requested_degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(1|2|4)$ ]]; then
    echo "MINWM_SP_DEGREES supports only 1, 2, and 4 in the single-node overlap matrix" >&2
    exit 2
  fi
  for vae_gpu_index in "${vae_gpu_indices[@]}"; do
    if ! [[ "${vae_gpu_index}" =~ ^[0-7]$ ]]; then
      echo "MINWM_VAE_GPU_INDICES must contain GPU indices 0 through 7" >&2
      exit 2
    fi
    if [[ "${ALLOW_SHARED_VAE_GPU}" != "true" ]] && (( vae_gpu_index < degree )); then
      echo "SP${degree} overlaps reserved VAE GPU index ${vae_gpu_index}" >&2
      exit 2
    fi
  done
done

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} != EXPECTED_GPU_COUNT )); then
  echo "expected an isolated ${EXPECTED_GPU_COUNT}-GPU node, found ${#gpu_names[@]} GPUs" >&2
  exit 1
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ "${gpu_name}" != *"${MINWM_EXPECTED_GPU_SUBSTRING}"* ]]; then
    echo "expected GPU containing ${MINWM_EXPECTED_GPU_SUBSTRING}, found ${gpu_name}" >&2
    exit 1
  fi
done

nvidia-smi -q > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-nvidia-smi.txt"
nvidia-smi topo -m > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-topology.txt"
nvidia-smi topo -p2p r > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-p2p-read.txt" 2>&1 || true
nvidia-smi topo -p2p w > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-p2p-write.txt" 2>&1 || true
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-cpu-numa.txt"
python3 - <<'PY' > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-software.json"
import json, platform, torch
print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                  "torch": torch.__version__, "cuda": torch.version.cuda}, sort_keys=True))
PY
{
  echo "matrix=${MINWM_MATRIX_ID}"
  echo "pod=${POD_NAME:-unknown}"
  echo "node=${NODE_NAME:-unknown}"
  echo "capacity=spot"
  echo "gpu_request=${EXPECTED_GPU_COUNT}"
  echo "gpu_expected=${MINWM_EXPECTED_GPU_SUBSTRING}"
  echo "sglang=${SGLANG_GIT_REF:-unknown}"
  echo "minwm=${MINWM_GIT_REF:-unknown}"
  echo "target_fps=${TARGET_FPS}"
  echo "sp_degrees=${SP_DEGREES}"
  echo "vae_gpu_indices=${VAE_GPU_INDICES}"
  echo "vae_parallel_size=${VAE_PARALLEL_SIZE}"
  echo "allow_shared_vae_gpu=${ALLOW_SHARED_VAE_GPU}"
  echo "dedicated_vae_cuda_stream=${DEDICATED_VAE_CUDA_STREAM}"
  echo "cuda_mps=${ENABLE_CUDA_MPS}"
  echo "repetitions=${REPETITIONS}"
  echo "warmup_chunks=${WARMUP_CHUNKS}"
  echo "measured_chunks=${MEASURED_CHUNKS}"
  echo "cases=${CASES_PATH}"
  echo "case=${THROUGHPUT_CASE}"
  date -Iseconds
} > "${RESULT_ROOT}/${MINWM_MATRIX_ID}-provenance.txt"

echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_START lane=setup timestamp=$(date -Iseconds)"
env \
  CUDA_VISIBLE_DEVICES=0 \
  MINWM_RUN_ID="${SETUP_RUN}" \
  MINWM_BENCHMARK_MODE=setup_only \
  bash "${ENTRYPOINT}"
echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_END lane=setup status=0 timestamp=$(date -Iseconds)"

denoiser_pid=""
monitor_pid=""
vae_pid=""
mps_started=false

persist_results() {
  [[ -n "${ARCHIVE_ROOT}" ]] || return 0
  local archive_dir="${ARCHIVE_ROOT%/}/${MINWM_MATRIX_ID}"
  mkdir -p "${archive_dir}"
  cp -a "${RESULT_ROOT}/." "${archive_dir}/"
  date -Iseconds > "${archive_dir}/last-flush.txt"
}

stop_denoiser() {
  if [[ -n "${denoiser_pid}" ]]; then
    kill "${denoiser_pid}" 2>/dev/null || true
    wait "${denoiser_pid}" 2>/dev/null || true
    denoiser_pid=""
  fi
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}

cleanup() {
  stop_denoiser
  if [[ -n "${vae_pid}" ]]; then
    kill "${vae_pid}" 2>/dev/null || true
    wait "${vae_pid}" 2>/dev/null || true
    vae_pid=""
  fi
  if [[ "${mps_started}" == "true" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    mps_started=false
  fi
  persist_results || true
}
trap cleanup EXIT INT TERM

if [[ "${ENABLE_CUDA_MPS}" == "true" ]]; then
  if ! command -v nvidia-cuda-mps-control >/dev/null; then
    echo "MINWM_ENABLE_CUDA_MPS=true but nvidia-cuda-mps-control is unavailable" >&2
    exit 1
  fi
  export CUDA_MPS_PIPE_DIRECTORY="${RESULT_ROOT}/mps-pipe"
  export CUDA_MPS_LOG_DIRECTORY="${RESULT_ROOT}/mps-log"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d
  mps_started=true
fi

wait_for_health() {
  local port="$1" pid="$2" log_path="$3"
  for _ in $(seq 1 300); do
    if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  echo "server on port ${port} did not become healthy" >&2
  tail -300 "${log_path}" >&2
  return 1
}

visible_gpus() {
  local degree="$1"
  seq -s, 0 "$((degree - 1))"
}

start_denoiser() {
  local mode="$1" degree="$2" lane_dir="$3"
  local remote_args=()
  if [[ "${mode}" == "remote" ]]; then
    remote_args+=(
      --realtime-vae-backend exact_remote
      --realtime-vae-worker-url ws://127.0.0.1:31000/v1/realtime_vae/decode
      --realtime-vae-transport auto
      --realtime-vae-shared-memory-dir "${SHM_ROOT}"
    )
  fi
  CUDA_VISIBLE_DEVICES="$(visible_gpus "${degree}")" \
  MINWM_ATTENTION_IMPL=dense \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=false \
  MINWM_NATIVE_COMPONENTS= \
  MINWM_PARITY_DETERMINISTIC=1 \
  MINWM_DETERMINISTIC_ATTENTION=true \
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
  SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=0 \
  MINWM_ROOT=/workspace/minWM \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
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
      --realtime-session-idle-timeout-s 1800 \
      "${remote_args[@]}" \
      --port 30000 \
      > "${lane_dir}/server.log" 2>&1 &
  denoiser_pid=$!
  wait_for_health 30000 "${denoiser_pid}" "${lane_dir}/server.log"
  (
    while kill -0 "${denoiser_pid}" 2>/dev/null; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${lane_dir}/gpu-utilization.csv" &
  monitor_pid=$!
}

run_lane() {
  local mode="$1" degree="$2" profile="$3"
  local lane_dir="${RESULT_ROOT}/${profile}"
  mkdir -p "${lane_dir}"
  echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_START lane=${profile} timestamp=$(date -Iseconds)"
  start_denoiser "${mode}" "${degree}" "${lane_dir}"
  set +e
  PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/benchmark_rtx6000_contract.py" \
    --output "${lane_dir}/throughput.json" \
    --server-log "${lane_dir}/server.log" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --profile-name "${profile}" \
    --sglang-git-ref "${SGLANG_GIT_REF:-unknown}" \
    --sizes ${CONTRACT_SIZES} \
    | tee "${lane_dir}/client.log"
  local status=${PIPESTATUS[0]}
  set -e
  stop_denoiser
  if (( status != 0 )); then
    persist_results || true
    echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_END lane=${profile} status=${status} timestamp=$(date -Iseconds)" >&2
    tail -300 "${lane_dir}/server.log" >&2
    return "${status}"
  fi
  date -Iseconds > "${lane_dir}/COMPLETE"
  persist_results
  echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_END lane=${profile} status=0 timestamp=$(date -Iseconds)"
}

run_lane local 1 local-sp1-warmup
for repetition in $(seq 1 "${REPETITIONS}"); do
  run_lane local 1 "local-sp1-r${repetition}"
done

vae_log="${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-worker.log"
vae_launcher=(python3)
if (( VAE_PARALLEL_SIZE > 1 )); then
  vae_launcher=(torchrun --standalone --nproc-per-node "${VAE_PARALLEL_SIZE}")
fi
vae_stream_args=()
if [[ "${DEDICATED_VAE_CUDA_STREAM}" == "true" ]]; then
  vae_stream_args+=(--dedicated-cuda-stream)
fi
CUDA_VISIBLE_DEVICES="${VAE_GPU_INDICES}" \
MINWM_NATIVE_COMPONENTS= \
PYTHONPATH=/workspace/sglang/python \
  "${vae_launcher[@]}" -m sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server \
    --decoder-backend exact \
    --vae-path "${MODEL_DIR}/vae" \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --num-gpus "${VAE_PARALLEL_SIZE}" \
    --tp-size 1 \
    --sp-degree "${VAE_PARALLEL_SIZE}" \
    --ulysses-degree "${VAE_PARALLEL_SIZE}" \
    --ring-degree 1 \
    --enable-cfg-parallel false \
    --attention-backend fa \
    --performance-mode speed \
    --enable-torch-compile false \
    --warmup-mode off \
    --max-sessions 1 \
    "${vae_stream_args[@]}" \
    --shared-memory-dir "${SHM_ROOT}" \
    --host 127.0.0.1 \
    --port 31000 \
    > "${vae_log}" 2>&1 &
vae_pid=$!
wait_for_health 31000 "${vae_pid}" "${vae_log}"
curl --fail --silent http://127.0.0.1:31000/health \
  | tee "${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-health.json"
python3 - "${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-health.json" "${VAE_PARALLEL_SIZE}" <<'PY'
import json
import sys

health = json.load(open(sys.argv[1]))
assert health["decoder_backend"] == "exact", health
assert health["decoder_fidelity"] == "exact", health
assert health["max_sessions"] == 1, health
assert health["encoded_frames_per_batch"] == 16, health
assert health["decode_parallel_size"] == int(sys.argv[2]), health
PY

target_met=false
parity_met=false
selected_degree=""
for degree in "${requested_degrees[@]}"; do
  profile="exact-remote-sp${degree}"
  curl --fail --silent http://127.0.0.1:31000/metrics \
    > "${RESULT_ROOT}/${profile}-vae-metrics-before.prom"
  for repetition in $(seq 1 "${REPETITIONS}"); do
    run_lane remote "${degree}" "${profile}-r${repetition}"
  done
  curl --fail --silent http://127.0.0.1:31000/metrics \
    > "${RESULT_ROOT}/${profile}-vae-metrics-after.prom"
  if [[ "${degree}" == "1" ]]; then
    parity_met="$(python3 - "${RESULT_ROOT}/local-sp1-r1/throughput.json" "${RESULT_ROOT}/${profile}-r1/throughput.json" <<'PY'
import json
import sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
print("true" if [x["payload_sha256"] for x in a["results"]] == [x["payload_sha256"] for x in b["results"]] else "false")
PY
)"
  fi
  fps="$(python3 - "${RESULT_ROOT}" "${profile}" "${REPETITIONS}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
values = [
    json.loads((root / f"{sys.argv[2]}-r{i}" / "throughput.json").read_text())
    ["results"][0]["steady_client_fps"]
    for i in range(1, int(sys.argv[3]) + 1)
]
print(statistics.median(values))
PY
)"
  if python3 - "${fps}" "${TARGET_FPS}" <<'PY'
import sys

raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
  then
    target_met=true
    selected_degree="${degree}"
    echo "MINWM_UNIFIED_EXACT_TARGET_MET sp=${degree} fps=${fps} target=${TARGET_FPS}"
    break
  fi
  echo "MINWM_UNIFIED_EXACT_TARGET_MISS sp=${degree} fps=${fps} target=${TARGET_FPS}"
done

python3 - \
  "${RESULT_ROOT}" "${TARGET_FPS}" "${target_met}" "${parity_met}" \
  "${selected_degree}" \
  "${MINWM_MATRIX_ID}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lanes = {}
for result_path in sorted(root.glob("*/throughput.json")):
    result = json.loads(result_path.read_text())
    lanes[result["profile_name"]] = result["results"]
summary = {
    "schema_version": "minwm-unified-exact-spot-matrix/v1",
    "matrix_id": sys.argv[6],
    "target_fps": float(sys.argv[2]),
    "target_met": sys.argv[3] == "true",
    "exact_sp1_bitwise_equal": sys.argv[4] == "true",
    "exact_sp1_numerical_parity": sys.argv[4] == "true",
    "selected_sp_degree": int(sys.argv[5]) if sys.argv[5] else None,
    "lanes": lanes,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
persist_results

if [[ "${target_met}" != "true" ]]; then
  echo "MINWM_UNIFIED_EXACT_TARGET_NOT_MET tested_sp=${SP_DEGREES} target=${TARGET_FPS}"
fi
echo "MINWM_UNIFIED_EXACT_MATRIX_COMPLETE results=${RESULT_ROOT}"
if [[ "${parity_met}" != "true" ]]; then
  echo "MINWM_UNIFIED_EXACT_PARITY_FAILED local and exact-remote SP1 numerical gate failed" >&2
  exit 2
fi
