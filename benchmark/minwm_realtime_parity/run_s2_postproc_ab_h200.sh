#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID to the unique Pod/attempt staging id}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT to the unique Pod/attempt directory}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_CONTAINER_IMAGE:?set MINWM_CONTAINER_IMAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_RUN_ID="${MINWM_RUN_ID}"
MODEL_DIR="/work/minwm-realtime/${STAGE_RUN_ID}/sglang-model"
QUALITY_ROOT="${MINWM_RESULTS_ROOT%/}/${STAGE_RUN_ID}/s2-quality"
CASES="${SCRIPT_DIR}/cases_720p_compile_smoke.json"
CASE_ID="00_forward_080_pottery_720p"
BASELINE_RUN_ID="${STAGE_RUN_ID}-self-post-baseline"
FAST_RUN_ID="${STAGE_RUN_ID}-self-post-fast"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"

[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
[[ "${KV_CACHE_NUM_FRAMES}" == "45" ]] || {
  echo "S2 steady-state A/B requires rolling-window cache=45" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
grep -q -- '--require-complete-stage-trace' "${SCRIPT_DIR}/run_s0_measurement.sh"

mkdir -p "${QUALITY_ROOT}"
{
  echo "sglang_commit=${SGLANG_GIT_REF}"
  echo "s0_tooling_commit=59aa68a382"
  echo "minwm_commit=${MINWM_GIT_REF}"
  echo "container_image=${MINWM_CONTAINER_IMAGE}"
  echo "stage_run_id=${STAGE_RUN_ID}"
  echo "baseline_run_id=${BASELINE_RUN_ID}"
  echo "fast_run_id=${FAST_RUN_ID}"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${QUALITY_ROOT}/contract.txt"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]]; then
    pkill -TERM -f "sglang serve --model-path ${MODEL_DIR}.*--port 30000" \
      2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT INT TERM

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

run_quality_lane() {
  local lane="$1" fast="$2" prefix="$3"
  local lane_dir="${QUALITY_ROOT}/${lane}"
  local log_path="${lane_dir}/server.log"
  local dump_dir="${lane_dir}/parity-dumps"
  mkdir -p "${lane_dir}" "${dump_dir}"
  MINWM_FUSE_SELF_ATTN_POST_FAST="${fast}" \
  MINWM_PARITY_DUMP_DIR="${dump_dir}" \
  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --vae-config.use-parallel-decode true \
      --attention-backend fa \
      --performance-mode speed \
      --num-gpus 2 \
      --tp-size 1 \
      --sp-degree 2 \
      --ulysses-degree 2 \
      --ring-degree 1 \
      --enable-cfg-parallel false \
      --enable-torch-compile false \
      --warmup-mode off \
      --port 30000 > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${log_path}"
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --results "${QUALITY_ROOT}" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix "${prefix}" \
    --engine-name "sglang-minwm-s2-${lane}" \
    --kv-cache-num-frames "${KV_CACHE_NUM_FRAMES}" \
    | tee "${lane_dir}/client.log"
  stop_server
}

run_quality_lane baseline false self_post_baseline
run_quality_lane fast true self_post_fast
python3 "${SCRIPT_DIR}/compare_s2_postproc_quality.py" \
  --results "${QUALITY_ROOT}" \
  --case "${CASE_ID}" \
  --thresholds "${SCRIPT_DIR}/thresholds.json" \
  --baseline-dumps "${QUALITY_ROOT}/baseline/parity-dumps" \
  --candidate-dumps "${QUALITY_ROOT}/fast/parity-dumps" \
  --output "${QUALITY_ROOT}/quality-comparison.json" \
  | tee "${QUALITY_ROOT}/quality-comparison.log"

link_model_for_run() {
  local run_id="$1"
  local work_root="/work/minwm-realtime/${run_id}"
  mkdir -p "${work_root}"
  [[ ! -e "${work_root}/sglang-model" ]]
  ln -s "${MODEL_DIR}" "${work_root}/sglang-model"
}

run_measurement_lane() {
  local run_id="$1" fast="$2"
  link_model_for_run "${run_id}"
  MINWM_RUN_ID="${run_id}" \
  MINWM_FUSE_SELF_ATTN_POST_FAST="${fast}" \
  MINWM_S0_SP_DEGREES="2 4" \
  MINWM_S0_OFF_WARMUP_CHUNKS=20 \
  MINWM_S0_OFF_MEASURED_CHUNKS=200 \
  MINWM_S0_PROFILE_PRECONDITION_CHUNKS=20 \
  MINWM_S0_PROFILE_DISCARD_CHUNKS=1 \
  MINWM_S0_PROFILE_MEASURED_CHUNKS=10 \
  MINWM_S0_KV_CACHE_NUM_FRAMES=45 \
    bash "${SCRIPT_DIR}/run_s0_measurement.sh"
}

run_measurement_lane "${BASELINE_RUN_ID}" false
run_measurement_lane "${FAST_RUN_ID}" true

python3 - "${MINWM_RESULTS_ROOT}" "${BASELINE_RUN_ID}" "${FAST_RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_id, fast_id = sys.argv[2:]
records = {}
for degree in (2, 4):
    lanes = {}
    for name, run_id in (("baseline", baseline_id), ("fast", fast_id)):
        lane_root = root / run_id / "s0-measurement" / f"sp{degree}"
        repeat = json.loads((lane_root / "repeat-summary.json").read_text())
        profile = json.loads(
            (lane_root / "profiler-on/measurement.json").read_text()
        )
        lanes[name] = {"profiler_off": repeat, "profiler_on": profile}
    deltas = {}
    for metric in (
        "client_fps",
        "scheduler_fps",
        "scheduler_chunk_wall_ms",
        "dit_wall_ms",
        "vae_wall_ms",
    ):
        baseline = lanes["baseline"]["profiler_off"]["metrics"][metric]["mean"]
        fast = lanes["fast"]["profiler_off"]["metrics"][metric]["mean"]
        deltas[metric] = {
            "baseline": baseline,
            "fast": fast,
            "relative_change": fast / baseline - 1,
        }
    records[f"sp{degree}"] = {"lanes": lanes, "headline_deltas": deltas}
report = {
    "schema_version": "minwm-s2-postproc-ab/v1",
    "baseline_run_id": baseline_id,
    "fast_run_id": fast_id,
    "lanes": records,
}
output = root / fast_id / "s0-measurement" / "s2-ab-comparison.json"
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    degree: value["headline_deltas"] for degree, value in records.items()
}, indent=2, sort_keys=True))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${QUALITY_ROOT}/complete.txt"
echo "MINWM_S2_POSTPROC_AB_COMPLETE results=${MINWM_RESULTS_ROOT}"
