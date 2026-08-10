#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_MATRIX_ID:?set MINWM_MATRIX_ID}"
: "${MINWM_EXPECTED_GPU_SUBSTRING:?set MINWM_EXPECTED_GPU_SUBSTRING}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${SCRIPT_DIR}/aws_b200_entrypoint.sh"
TARGET_FPS="${MINWM_TARGET_FPS:-24}"
SP_DEGREES="${MINWM_SP_DEGREES:-1 2 4}"
VAE_GPU_INDEX="${MINWM_VAE_GPU_INDEX:-7}"
WARMUP_CHUNKS="${MINWM_THROUGHPUT_WARMUP_CHUNKS:-20}"
MEASURED_CHUNKS="${MINWM_THROUGHPUT_MEASURED_CHUNKS:-200}"
CASES_PATH="${MINWM_THROUGHPUT_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
THROUGHPUT_CASE="${MINWM_THROUGHPUT_CASE:-00_forward_080_pottery_720p}"
SHM_ROOT="${MINWM_REALTIME_VAE_SHM_DIR:-/dev/shm/sglang-realtime-vae}"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_MATRIX_ID}"
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

read -r -a requested_degrees <<< "${SP_DEGREES}"
for degree in "${requested_degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(1|2|4)$ ]]; then
    echo "MINWM_SP_DEGREES supports only 1, 2, and 4 in the single-node overlap matrix" >&2
    exit 2
  fi
  if (( degree > VAE_GPU_INDEX )); then
    echo "SP${degree} overlaps the reserved VAE GPU index ${VAE_GPU_INDEX}" >&2
    exit 2
  fi
done

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} != 8 )); then
  echo "expected an isolated eight-GPU p5/p6 node, found ${#gpu_names[@]} GPUs" >&2
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
{
  echo "matrix=${MINWM_MATRIX_ID}"
  echo "pod=${POD_NAME:-unknown}"
  echo "node=${NODE_NAME:-unknown}"
  echo "capacity=spot"
  echo "gpu_request=8"
  echo "gpu_expected=${MINWM_EXPECTED_GPU_SUBSTRING}"
  echo "sglang=${SGLANG_GIT_REF:-unknown}"
  echo "minwm=${MINWM_GIT_REF:-unknown}"
  echo "target_fps=${TARGET_FPS}"
  echo "sp_degrees=${SP_DEGREES}"
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
}
trap cleanup EXIT INT TERM

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
  PYTHONPATH="${SCRIPT_DIR}" python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --cases "${CASES_PATH}" \
    --output "${lane_dir}/throughput.json" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --case "${THROUGHPUT_CASE}" \
    --profile-name "${profile}" \
    --warmup-chunks "${WARMUP_CHUNKS}" \
    --measured-chunks "${MEASURED_CHUNKS}" \
    --kv-cache-num-frames 45 \
    --save-first-measured-frame \
    | tee "${lane_dir}/client.log"
  local status=${PIPESTATUS[0]}
  set -e
  stop_denoiser
  if (( status != 0 )); then
    echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_END lane=${profile} status=${status} timestamp=$(date -Iseconds)" >&2
    tail -300 "${lane_dir}/server.log" >&2
    return "${status}"
  fi
  echo "MINWM_UNIFIED_EXACT_MATRIX_LANE_END lane=${profile} status=0 timestamp=$(date -Iseconds)"
}

run_lane local 1 local-sp1

vae_log="${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-worker.log"
CUDA_VISIBLE_DEVICES="${VAE_GPU_INDEX}" \
MINWM_NATIVE_COMPONENTS= \
PYTHONPATH=/workspace/sglang/python \
  python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server \
    --decoder-backend exact \
    --vae-path "${MODEL_DIR}/vae" \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --num-gpus 1 \
    --attention-backend fa \
    --performance-mode speed \
    --enable-torch-compile false \
    --warmup-mode off \
    --max-sessions 1 \
    --shared-memory-dir "${SHM_ROOT}" \
    --host 127.0.0.1 \
    --port 31000 \
    > "${vae_log}" 2>&1 &
vae_pid=$!
wait_for_health 31000 "${vae_pid}" "${vae_log}"
curl --fail --silent http://127.0.0.1:31000/health \
  | tee "${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-health.json"
python3 - "${RESULT_ROOT}/${MINWM_MATRIX_ID}-exact-vae-health.json" <<'PY'
import json
import sys

health = json.load(open(sys.argv[1]))
assert health["decoder_backend"] == "exact", health
assert health["decoder_fidelity"] == "exact", health
assert health["max_sessions"] == 1, health
PY

target_met=false
parity_met=false
selected_degree=""
for degree in "${requested_degrees[@]}"; do
  profile="exact-remote-sp${degree}"
  run_lane remote "${degree}" "${profile}"
  if [[ "${degree}" == "1" ]]; then
    python3 - \
      "${RESULT_ROOT}/local-sp1/throughput.json" \
      "${RESULT_ROOT}/${profile}/throughput.json" \
      "${RESULT_ROOT}/local-vs-remote-sp1-parity.json" <<'PY'
import json
import math
import sys
from pathlib import Path

local = json.load(open(sys.argv[1]))
remote = json.load(open(sys.argv[2]))
local_frames = local.get("measured_frame_sha256", {})
remote_frames = remote.get("measured_frame_sha256", {})
all_frame_keys = sorted(set(local_frames) | set(remote_frames))
first_differing_frame = next(
    (key for key in all_frame_keys if local_frames.get(key) != remote_frames.get(key)),
    None,
)
local_first_frame = Path(sys.argv[1]).with_name("first-measured-frame.rgb").read_bytes()
remote_first_frame = Path(sys.argv[2]).with_name("first-measured-frame.rgb").read_bytes()
if len(local_first_frame) != len(remote_first_frame):
    raise SystemExit("local and remote first measured RGB frame lengths differ")
different_bytes = 0
absolute_error_sum = 0
squared_error_sum = 0
max_absolute_error = 0
for local_value, remote_value in zip(local_first_frame, remote_first_frame):
    error = abs(local_value - remote_value)
    different_bytes += error != 0
    absolute_error_sum += error
    squared_error_sum += error * error
    max_absolute_error = max(max_absolute_error, error)
mean_absolute_error = absolute_error_sum / len(local_first_frame)
mean_squared_error = squared_error_sum / len(local_first_frame)
psnr_db = (
    math.inf
    if mean_squared_error == 0
    else 20 * math.log10(255) - 10 * math.log10(mean_squared_error)
)
summary = {
    "local_payload_sha256": local["measured_payload_sha256"],
    "remote_payload_sha256": remote["measured_payload_sha256"],
    "bitwise_equal": local["measured_payload_sha256"]
    == remote["measured_payload_sha256"],
    "frame_hashes_equal": local_frames == remote_frames,
    "first_differing_frame": first_differing_frame,
    "first_frame_num_bytes": len(local_first_frame),
    "first_frame_different_byte_fraction": different_bytes / len(local_first_frame),
    "first_frame_mean_absolute_error": mean_absolute_error,
    "first_frame_max_absolute_error": max_absolute_error,
    "first_frame_psnr_db": psnr_db,
    "local_client_fps": local["client"]["steady_received_fps_ratio_of_sums"],
    "remote_client_fps": remote["client"]["steady_received_fps_ratio_of_sums"],
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
    parity_met="$(python3 - "${RESULT_ROOT}/local-vs-remote-sp1-parity.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
print("true" if summary["bitwise_equal"] else "false")
PY
)"
  fi
  fps="$(python3 - "${RESULT_ROOT}/${profile}/throughput.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
print(result["client"]["steady_received_fps_ratio_of_sums"])
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
    lanes[result["profile_name"]] = {
        "client_fps": result["client"]["steady_received_fps_ratio_of_sums"],
        "client_interarrival_ms": result["client"]["steady_payload_interarrival_ms"],
        "first_payload_ms": result["client"][
            "init_send_complete_to_first_payload_complete_ms"
        ],
        "payload_sha256": result["measured_payload_sha256"],
        "server": result["server"],
    }
summary = {
    "schema_version": "minwm-unified-exact-spot-matrix/v1",
    "matrix_id": sys.argv[6],
    "target_fps": float(sys.argv[2]),
    "target_met": sys.argv[3] == "true",
    "exact_sp1_bitwise_equal": sys.argv[4] == "true",
    "selected_sp_degree": int(sys.argv[5]) if sys.argv[5] else None,
    "lanes": lanes,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ "${target_met}" != "true" ]]; then
  echo "MINWM_UNIFIED_EXACT_TARGET_NOT_MET tested_sp=${SP_DEGREES} target=${TARGET_FPS}"
fi
echo "MINWM_UNIFIED_EXACT_MATRIX_COMPLETE results=${RESULT_ROOT}"
if [[ "${parity_met}" != "true" ]]; then
  echo "MINWM_UNIFIED_EXACT_PARITY_FAILED local and exact-remote SP1 payloads differ" >&2
  exit 2
fi
