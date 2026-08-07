#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_S3_PRODUCT_GIT_REF:?set MINWM_S3_PRODUCT_GIT_REF}"
: "${MINWM_S3_RUNNER_IMPL_GIT_REF:?set MINWM_S3_RUNNER_IMPL_GIT_REF}"
: "${MINWM_S0_TOOL_GIT_REF:?set MINWM_S0_TOOL_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_CONTAINER_IMAGE:?set MINWM_CONTAINER_IMAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}"
MODEL_DIR="${WORK_ROOT}/sglang-model"
RUN_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/s3-post-sp4-abba-off-only"
CASES="${SCRIPT_DIR}/cases_720p_compile_smoke.json"
CASE_ID="00_forward_080_pottery_720p"
SP_DEGREE=4
WARMUP_CHUNKS="${MINWM_S0_OFF_WARMUP_CHUNKS:-20}"
MEASURED_CHUNKS="${MINWM_S0_OFF_MEASURED_CHUNKS:-200}"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"
POSITIONS=(A1 B1 B2 A2)
LANES=(01 00 00 01)

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
git -C /workspace/sglang merge-base --is-ancestor \
  "${MINWM_S3_RUNNER_IMPL_GIT_REF}" "${SGLANG_GIT_REF}"
git -C /workspace/sglang merge-base --is-ancestor \
  "${MINWM_S0_TOOL_GIT_REF}" "${SGLANG_GIT_REF}"
if (( WARMUP_CHUNKS != 20 || MEASURED_CHUNKS != 200 )); then
  echo "S3 SP4 ABBA contract requires exactly 20 warmup + 200 measured" >&2
  exit 2
fi
if (( KV_CACHE_NUM_FRAMES != 45 )); then
  echo "S3 SP4 ABBA rolling-window contract requires KV cache 45" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

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
  echo "s3_product_commit=${MINWM_S3_PRODUCT_GIT_REF}"
  echo "s3_runner_impl_commit=${MINWM_S3_RUNNER_IMPL_GIT_REF}"
  echo "s0_tool_commit=${MINWM_S0_TOOL_GIT_REF}"
  echo "minwm_commit=${MINWM_GIT_REF}"
  echo "container_image=${MINWM_CONTAINER_IMAGE}"
  echo "gpu_model=${GPU_MODEL}"
  echo "active_gpu_count=${SP_DEGREE}"
  echo "allocated_gpu_count=${ALLOCATED_GPU_COUNT}"
  echo "sp_degree=${SP_DEGREE}"
  echo "position_order=${POSITIONS[*]}"
  echo "lane_order=${LANES[*]}"
  echo "off_window=${WARMUP_CHUNKS}+${MEASURED_CHUNKS}"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "torch_profiler_concurrent=false"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RUN_ROOT}/contract.txt"
nvidia-smi -q > "${RUN_ROOT}/nvidia-smi-q-start.txt"

server_pid=""
monitor_pid=""
INVALID_SCOPE="${RUN_ROOT}"

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

record_invalid_attempt() {
  local status="$1" scope="$2"
  python3 - "${scope}" "${status}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
status = int(sys.argv[2])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or "invalid" in path.relative_to(root).parts:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
marker = root / "invalid" / "attempt.json"
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps({
    "reason": "runner_exit_nonzero",
    "exit_status": status,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "evidence_location": str(root),
    "recoverability": "all listed evidence remains in place on the PVC",
    "files": files,
}, indent=2, sort_keys=True) + "\n")
print(f"marked invalid scope: {marker}")
PY
}

on_exit() {
  local status=$?
  trap - EXIT
  stop_server
  if (( status != 0 )); then
    record_invalid_attempt "${status}" "${INVALID_SCOPE}" || true
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set_lane() {
  case "$1" in
    00) export MINWM_FUSED_POST_A2A_ROPE_CACHE=0 ;;
    01) export MINWM_FUSED_POST_A2A_ROPE_CACHE=1 ;;
    *) echo "unsupported post-only lane $1" >&2; return 2 ;;
  esac
}

start_server() {
  local log_path="$1" telemetry_path="$2"
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
      --num-gpus "${SP_DEGREE}" \
      --tp-size 1 \
      --sp-degree "${SP_DEGREE}" \
      --ulysses-degree "${SP_DEGREE}" \
      --ring-degree 1 \
      --enable-cfg-parallel false \
      --enable-torch-compile false \
      --warmup-mode off \
      --port 30000 > "${log_path}" 2>&1 &
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

run_client() {
  local position="$1" lane="$2" output="$3"
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --output "${output}" \
    --profile-name "bf16-s3-post-sp4-abba-${position}-lane${lane}" \
    --run-id "${MINWM_RUN_ID}-sp4-${position}-lane${lane}" \
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
    --require-complete-stage-trace \
    --measurement-mode profiler_off \
    --warmup-chunks "${WARMUP_CHUNKS}" \
    --measured-chunks "${MEASURED_CHUNKS}" \
    | tee "${output%.json}.log"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${output}"
  python3 - "${output}" <<'PY'
import json
import sys

path = sys.argv[1]
record = json.load(open(path))
expected = record["workload"]["measured_chunks"]
if expected != 200:
    raise RuntimeError(f"{path}: measured_chunks={expected}, want 200")
for name in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"):
    metric = record["metrics"]["profiler_off"][name]
    if metric["status"] != "available" or metric["value"]["count"] != expected:
        raise RuntimeError(f"{path}: invalid {name}: {metric}")
print(f"count-ok {path} measured_chunks={expected}")
PY
}

PYTHONPATH=/workspace/sglang/python python3 -m pytest -q \
  /workspace/sglang/test/registered/jit/diffusion/test_minwm_ulysses_fused.py \
  | tee "${RUN_ROOT}/kernel-bitwise-tests.log"
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_ROOT}/PRECHECK_COMPLETE"

measurements="${RUN_ROOT}/measurements"
mkdir -p "${measurements}"
outputs=()
for index in "${!POSITIONS[@]}"; do
  position="${POSITIONS[${index}]}"
  lane="${LANES[${index}]}"
  set_lane "${lane}"
  position_root="${measurements}/${position}"
  mkdir -p "${position_root}"
  INVALID_SCOPE="${position_root}"
  {
    echo "position=${position}"
    echo "position_index=${index}"
    echo "lane=${lane}"
    echo "fused_post_a2a_rope_cache=${MINWM_FUSED_POST_A2A_ROPE_CACHE}"
    echo "independent_server=true"
    echo "sp_degree=${SP_DEGREE}"
    echo "off_window=${WARMUP_CHUNKS}+${MEASURED_CHUNKS}"
    echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
    echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  } | tee "${position_root}/contract.txt"
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${position_root}/POSITION_STARTED"
  start_server \
    "${position_root}/server.log" \
    "${position_root}/gpu-telemetry.csv"
  nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE \
    > "${position_root}/nvidia-smi-q-before.txt"
  output="${position_root}/profiler-off.json"
  run_client "${position}" "${lane}" "${output}"
  nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE \
    > "${position_root}/nvidia-smi-q-after.txt"
  outputs+=("${output}")
  stop_server
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${position_root}/POSITION_COMPLETE"
done

INVALID_SCOPE="${RUN_ROOT}"
python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
  "${measurements}/A1/profiler-off.json" \
  "${measurements}/A2/profiler-off.json" \
  --output "${measurements}/candidate-summary.json"
python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
  "${measurements}/B1/profiler-off.json" \
  "${measurements}/B2/profiler-off.json" \
  --output "${measurements}/baseline-summary.json"
python3 - "${measurements}" <<'PY' | tee "${measurements}/abba-summary.log"
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
order = (("A1", "01"), ("B1", "00"), ("B2", "00"), ("A2", "01"))
records = {name: json.loads((root / name / "profiler-off.json").read_text()) for name, _ in order}

def metrics(record):
    off = record["metrics"]["profiler_off"]
    return {
        "client_fps": off["client_fps"]["value"],
        "scheduler_fps": off["scheduler_fps"]["value"],
        "scheduler_chunk_wall_ms": off["scheduler_chunk_wall_ms"]["value"]["mean"],
        "dit_wall_ms": off["dit_wall_ms"]["value"]["mean"],
        "vae_wall_ms": off["vae_wall_ms"]["value"]["mean"],
    }

positions = {name: metrics(records[name]) for name, _ in order}
candidate = {key: statistics.fmean(positions[name][key] for name in ("A1", "A2")) for key in positions["A1"]}
baseline = {key: statistics.fmean(positions[name][key] for name in ("B1", "B2")) for key in positions["B1"]}
summary = {
    "order": [{"position": name, "lane": lane} for name, lane in order],
    "positions": positions,
    "candidate_mean": candidate,
    "baseline_mean": baseline,
    "candidate_delta_percent": {
        key: (candidate[key] / baseline[key] - 1.0) * 100.0 for key in candidate
    },
    "all_counts_exact_200": all(
        records[name]["metrics"]["profiler_off"][metric]["value"]["count"] == 200
        for name, _ in order
        for metric in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms")
    ),
}
(root / "abba-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

nvidia-smi -q > "${RUN_ROOT}/nvidia-smi-q-end.txt"
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_ROOT}/ABBA_COMPLETE"
echo "MINWM_S3_POST_SP4_ABBA_OFF_ONLY_COMPLETE results=${RUN_ROOT}"
