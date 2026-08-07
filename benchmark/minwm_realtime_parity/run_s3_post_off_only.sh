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
RUN_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/s3-post-off-only"
SHORT_CASES="${SCRIPT_DIR}/cases_720p_compile_smoke.json"
EVICTION_CASES="${SCRIPT_DIR}/cases_s3_720p_eviction.json"
CASE_ID="00_forward_080_pottery_720p"
OFF_WARMUP_CHUNKS="${MINWM_S0_OFF_WARMUP_CHUNKS:-20}"
OFF_MEASURED_CHUNKS="${MINWM_S0_OFF_MEASURED_CHUNKS:-200}"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${SHORT_CASES}" ]]
[[ -f "${EVICTION_CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if ! [[ "${OFF_WARMUP_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${OFF_MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "warmup and measured chunk counts must be positive integers" >&2
  exit 2
fi
if (( OFF_WARMUP_CHUNKS != 20 || OFF_MEASURED_CHUNKS != 200 )); then
  echo "S3 post-only contract requires exactly 20 warmup + 200 measured" >&2
  exit 2
fi
if (( KV_CACHE_NUM_FRAMES != 45 )); then
  echo "S3 post-only rolling-window contract requires KV cache 45" >&2
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
  echo "minwm_commit=${MINWM_GIT_REF}"
  echo "container_image=${MINWM_CONTAINER_IMAGE}"
  echo "gpu_model=${GPU_MODEL}"
  echo "allocated_gpu_count=${ALLOCATED_GPU_COUNT}"
  echo "sp_degrees=2 4"
  echo "lanes=00 01"
  echo "off_window=${OFF_WARMUP_CHUNKS}+${OFF_MEASURED_CHUNKS}"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "torch_profiler_concurrent=false"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RUN_ROOT}/contract.txt"
nvidia-smi -q > "${RUN_ROOT}/nvidia-smi-q.txt"

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
  local lane="$1"
  case "${lane}" in
    00) export MINWM_FUSED_POST_A2A_ROPE_CACHE=0 ;;
    01) export MINWM_FUSED_POST_A2A_ROPE_CACHE=1 ;;
    *) echo "unsupported post-only lane ${lane}" >&2; return 2 ;;
  esac
}

start_server() {
  local degree="$1" log_path="$2" telemetry_path="${3:-}"
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
      --num-gpus "${degree}" \
      --tp-size 1 \
      --sp-degree "${degree}" \
      --ulysses-degree "${degree}" \
      --ring-degree 1 \
      --enable-cfg-parallel false \
      --enable-torch-compile false \
      --warmup-mode off \
      --port 30000 > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_pid}" "${log_path}"
  if [[ -n "${telemetry_path}" ]]; then
    (
      while kill -0 "${server_pid}" 2>/dev/null; do
        nvidia-smi \
          --query-gpu=timestamp,index,utilization.gpu,clocks.sm,pstate,power.draw,temperature.gpu,memory.used \
          --format=csv,noheader,nounits || true
        sleep 1
      done
    ) > "${telemetry_path}" &
    monitor_pid=$!
  fi
}

run_parity_case() {
  local cases="$1" output_prefix="$2" output_root="$3" degree="$4" lane="$5"
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${cases}" \
    --case "${CASE_ID}" \
    --results "${output_root}" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix "${output_prefix}" \
    --engine-name "sglang-minwm-s3-post-sp${degree}-lane${lane}-${output_prefix}" \
    --warmup-runs 0 \
    --kv-cache-num-frames "${KV_CACHE_NUM_FRAMES}" \
    | tee "${output_root}/${output_prefix}-client.log"
}

PYTHONPATH=/workspace/sglang/python python3 -m pytest -q \
  /workspace/sglang/test/registered/jit/diffusion/test_minwm_ulysses_fused.py \
  | tee "${RUN_ROOT}/kernel-bitwise-tests.log"

for degree in 2 4; do
  for lane in 00 01; do
    set_lane "${lane}"
    parity_root="${RUN_ROOT}/parity/sp${degree}/lane${lane}"
    mkdir -p "${parity_root}/short" "${parity_root}/eviction"
    {
      echo "sglang_commit=${SGLANG_GIT_REF}"
      echo "sp_degree=${degree}"
      echo "lane=${lane}"
      echo "fused_post_a2a_rope_cache=${MINWM_FUSED_POST_A2A_ROPE_CACHE}"
      echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
    } | tee "${parity_root}/contract.txt"
    start_server "${degree}" "${parity_root}/server.log"
    run_parity_case "${SHORT_CASES}" short "${parity_root}/short" "${degree}" "${lane}"
    run_parity_case "${EVICTION_CASES}" eviction "${parity_root}/eviction" "${degree}" "${lane}"
    stop_server
  done
done

python3 - "${RUN_ROOT}/parity" <<'PY' | tee "${RUN_ROOT}/parity-summary.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for degree in (2, 4):
    summary[f"sp{degree}"] = {}
    for case_name in ("short", "eviction"):
        hashes = {}
        for lane in ("00", "01"):
            record = json.loads(
                (root / f"sp{degree}" / f"lane{lane}" / case_name / f"{case_name}_run.json").read_text()
            )
            hashes[lane] = record["cases"][0]["frames_sha256"]
        if hashes["00"] != hashes["01"]:
            raise RuntimeError(f"SP{degree} {case_name} post parity mismatch: {hashes}")
        summary[f"sp{degree}"][case_name] = {
            "frames_sha256": hashes["00"],
            "lanes": hashes,
            "bitwise_exact": True,
        }
(root / "parity-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_ROOT}/QUALITY_COMPLETE"
if [[ "${MINWM_S3_STOP_BEFORE_OFF_ONLY:-0}" == "1" \
  || -e "${RUN_ROOT}/STOP_BEFORE_OFF_ONLY" ]]; then
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_ROOT}/stopped-before-off-only.txt"
  echo "MINWM_S3_STOPPED_BEFORE_OFF_ONLY results=${RUN_ROOT}"
  exit 0
fi

run_measurement_client() {
  local degree="$1" lane="$2" output="$3" repeat="$4"
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --cases "${SHORT_CASES}" \
    --case "${CASE_ID}" \
    --output "${output}" \
    --profile-name "bf16-s3-post-sp${degree}-lane${lane}" \
    --run-id "${MINWM_RUN_ID}-sp${degree}-lane${lane}-off-r${repeat}" \
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
    --require-complete-stage-trace \
    --measurement-mode profiler_off \
    --warmup-chunks "${OFF_WARMUP_CHUNKS}" \
    --measured-chunks "${OFF_MEASURED_CHUNKS}" \
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

measurement_root="${RUN_ROOT}/measurements"
mkdir -p "${measurement_root}"
for degree in 2 4; do
  for lane in 00 01; do
    set_lane "${lane}"
    lane_root="${measurement_root}/sp${degree}/lane${lane}"
    mkdir -p "${lane_root}"
    INVALID_SCOPE="${lane_root}"
    {
      echo "sglang_commit=${SGLANG_GIT_REF}"
      echo "sp_degree=${degree}"
      echo "lane=${lane}"
      echo "fused_post_a2a_rope_cache=${MINWM_FUSED_POST_A2A_ROPE_CACHE}"
      echo "off_window=${OFF_WARMUP_CHUNKS}+${OFF_MEASURED_CHUNKS}"
      echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
    } | tee "${lane_root}/contract.txt"
    start_server \
      "${degree}" \
      "${lane_root}/profiler-off-server.log" \
      "${lane_root}/profiler-off-gpu-telemetry.csv"
    repeat_paths=()
    for repeat in 1 2; do
      output="${lane_root}/profiler-off-repeat${repeat}.json"
      run_measurement_client "${degree}" "${lane}" "${output}" "${repeat}"
      repeat_paths+=("${output}")
    done
    python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
      "${repeat_paths[@]}" \
      --output "${lane_root}/repeat-summary.json"
    if ! python3 - "${lane_root}/repeat-summary.json" <<'PY'
import json
import sys
raise SystemExit(not json.load(open(sys.argv[1]))["acceptance"]["passes_cv_target"])
PY
    then
      output="${lane_root}/profiler-off-repeat3.json"
      run_measurement_client "${degree}" "${lane}" "${output}" 3
      repeat_paths+=("${output}")
      python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
        "${repeat_paths[@]}" \
        --noise-explanation \
        "Automatic third repeat collected after the two-repeat CV target miss; inspect GPU telemetry." \
        --output "${lane_root}/repeat-summary.json"
    fi
    stop_server
    date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${lane_root}/LANE_COMPLETE"
    INVALID_SCOPE="${RUN_ROOT}"
  done
done

python3 - "${measurement_root}" <<'PY' | tee "${measurement_root}/post-off-summary.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lanes = {}
for degree in (2, 4):
    for lane in ("00", "01"):
        lane_root = root / f"sp{degree}" / f"lane{lane}"
        key = f"sp{degree}-lane{lane}"
        lanes[key] = json.loads((lane_root / "repeat-summary.json").read_text())
summary = {
    "schema_version": "minwm-s3-post-off-only/v1",
    "lanes": lanes,
}
(root / "post-off-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    key: {
        "passes_cv_target": value["acceptance"]["passes_cv_target"],
        "repeat_count": len(value["run_ids"]),
    }
    for key, value in lanes.items()
}, indent=2, sort_keys=True))
PY

echo "MINWM_S3_POST_OFF_ONLY_COMPLETE results=${RUN_ROOT}"
