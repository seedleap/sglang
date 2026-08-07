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
SP_DEGREES="${MINWM_ASYNC_A2A_SP_DEGREES:-2 4}"
A2A_SEQUENCE="${MINWM_ASYNC_A2A_SEQUENCE:-candidate baseline baseline candidate baseline candidate candidate baseline candidate baseline baseline candidate}"
A2A_BACKEND="${MINWM_ASYNC_A2A_BENCH_BACKEND:-process_group}"
A2A_EXPERIMENT="${MINWM_ASYNC_A2A_EXPERIMENT:-input_split}"
OUTPUT_TILES="${MINWM_ASYNC_A2A_OUTPUT_TILES:-1}"
MIN_LANE_SAMPLES="${MINWM_ASYNC_A2A_MIN_LANE_SAMPLES:-5}"
OFF_WARMUP_CHUNKS="${MINWM_S0_OFF_WARMUP_CHUNKS:-20}"
OFF_MEASURED_CHUNKS="${MINWM_S0_OFF_MEASURED_CHUNKS:-200}"
KV_CACHE_NUM_FRAMES="${MINWM_S0_KV_CACHE_NUM_FRAMES:-45}"
RESULT_KIND="async-a2a-abba"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/${RESULT_KIND}"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if ! [[ "${OFF_WARMUP_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${OFF_MEASURED_CHUNKS}" =~ ^[1-9][0-9]*$ \
  && "${MIN_LANE_SAMPLES}" =~ ^[1-9][0-9]*$ \
  && "${OUTPUT_TILES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Profiler-off warmup/measurement chunk counts must be positive integers" >&2
  exit 2
fi
if [[ "${A2A_EXPERIMENT}" != "input_split" \
  && "${A2A_EXPERIMENT}" != "output_tiled" ]]; then
  echo "MINWM_ASYNC_A2A_EXPERIMENT must be input_split or output_tiled" >&2
  exit 2
fi
if [[ "${A2A_EXPERIMENT}" == "output_tiled" && "${OUTPUT_TILES}" == "1" ]]; then
  echo "output_tiled experiment requires MINWM_ASYNC_A2A_OUTPUT_TILES > 1" >&2
  exit 2
fi
if [[ -n "${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-}" ]]; then
  echo "SGLANG_DIFFUSION_TORCH_PROFILER_DIR must be unset for profiler-off headline" >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite existing async A2A measurement attempt: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"

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
  echo "sp_degrees=${SP_DEGREES}"
  echo "a2a_sequence=${A2A_SEQUENCE}"
  echo "a2a_backend=${A2A_BACKEND}"
  echo "a2a_experiment=${A2A_EXPERIMENT}"
  echo "output_tiles=${OUTPUT_TILES}"
  echo "minimum_lane_samples=${MIN_LANE_SAMPLES}"
  echo "server_restart_per_position=true"
  echo "a2a_baseline=input=0,output=0"
  echo "a2a_candidate=${A2A_EXPERIMENT}"
  echo "measurement_mode=profiler_off_only"
  echo "trace_sync_cuda=0"
  echo "off_window=${OFF_WARMUP_CHUNKS}+${OFF_MEASURED_CHUNKS}"
  echo "nsys_status=disabled_pending_exact_window_contract"
  echo "kv_cache_num_frames=${KV_CACHE_NUM_FRAMES}"
  echo "torch_profiler_concurrent=false"
  echo "invalid_attempt_policy=preserve-in-place-with-scoped-path-size-sha256-marker"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RESULT_ROOT}/contract.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.txt"

server_pid=""
monitor_pid=""
a2a_lane=""
async_a2a_flag=""
input_a2a_flag=""
output_a2a_flag=""
invalid_scope="${RESULT_ROOT}"

configure_a2a_flags() {
  input_a2a_flag=0
  output_a2a_flag=0
  if [[ "${async_a2a_flag}" != "1" ]]; then
    return
  fi
  if [[ "${A2A_EXPERIMENT}" == "input_split" ]]; then
    input_a2a_flag=1
  else
    output_a2a_flag=1
  fi
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

mark_invalid_scope() {
  local status="$1"
  python3 - "${invalid_scope}" "${status}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
status = int(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
marker_path = root / "invalid-marker.json"
artifacts = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name.startswith("invalid-marker"):
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
    "scope_root": str(root),
    "artifacts": artifacts,
    "recoverability": (
        "All partial evidence remains in place under this failure scope; "
        "aggregators must exclude this scope without invalidating sibling lanes."
    ),
}
with marker_path.open("x") as output:
    output.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
PY
}

cleanup() {
  local status="${1:-0}"
  set +e
  stop_server
  if (( status != 0 )); then
    mark_invalid_scope "${status}"
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
  MINWM_ASYNC_A2A="${input_a2a_flag}" \
  MINWM_ASYNC_A2A_OUTPUT="${output_a2a_flag}" \
  MINWM_ASYNC_A2A_OUTPUT_TILES="${OUTPUT_TILES}" \
  MINWM_ASYNC_A2A_BACKEND="${A2A_BACKEND}" \
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
assert record["mode"] == "profiler_off", record["mode"]
container = record["metrics"]["profiler_off"]
paths = {
    "scheduler_chunk_wall_ms": container["scheduler_chunk_wall_ms"],
    "dit_wall_ms": container["dit_wall_ms"],
    "vae_wall_ms": container["vae_wall_ms"],
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
  mkdir -p "${lane_dir}"
  python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate "$@" \
    --output "${lane_dir}/repeat-summary.json"
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
      "${degree}" "${output}" "bf16-${a2a_lane}-sp${degree}" \
      "${MINWM_RUN_ID}-${a2a_lane}-sp${degree}-off-r${repeat}" \
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
      "${degree}" "${output}" "bf16-${a2a_lane}-sp${degree}" \
      "${MINWM_RUN_ID}-${a2a_lane}-sp${degree}-off-r3" \
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

record_compile_cache_state() {
  local output="$1"
  python3 - "${output}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

roots = {
    "inductor": Path(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR",
            "/root/.cache/sgl_diffusion/torch_compile_cache/inductor",
        )
    ),
    "triton": Path(
        os.environ.get(
            "TRITON_CACHE_DIR",
            "/root/.cache/sgl_diffusion/torch_compile_cache/triton",
        )
    ),
}
summary = {}
for name, root in roots.items():
    entries = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                entries.append(
                    f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}"
                )
    payload = "\n".join(entries).encode()
    summary[name] = {
        "path": str(root),
        "exists": root.exists(),
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry.split("\0")[1]) for entry in entries),
        "metadata_listing_sha256": hashlib.sha256(payload).hexdigest(),
    }
record = {
    "schema_version": "minwm-async-a2a-compile-cache-state/v1",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "roots": summary,
    "note": "Read-only metadata snapshot; cache contents were not cleared or rewritten.",
}
with Path(sys.argv[1]).open("x") as output_file:
    output_file.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

assert_server_stopped() {
  if pgrep -f "sglang serve --model-path ${MODEL_DIR}.*--port 30000" >/dev/null; then
    echo "Previous position server still exists after stop_server" >&2
    return 1
  fi
  if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
    echo "Previous position health endpoint is still live after stop_server" >&2
    return 1
  fi
}

run_profiler_off_position() {
  local degree="$1" position="$2" lane="$3" position_dir="$4"
  case "${lane}" in
    baseline) async_a2a_flag=0 ;;
    candidate) async_a2a_flag=1 ;;
    *)
      echo "async A2A ABBA accepts A2A lane baseline or candidate, got ${lane}" >&2
      exit 2
      ;;
  esac
  configure_a2a_flags
  a2a_lane="${lane}"
  invalid_scope="${position_dir}"
  mkdir -p "${position_dir}"
  {
    echo "position=${position}"
    echo "a2a_lane=${lane}"
    echo "MINWM_ASYNC_A2A=${input_a2a_flag}"
    echo "MINWM_ASYNC_A2A_OUTPUT=${output_a2a_flag}"
    echo "MINWM_ASYNC_A2A_OUTPUT_TILES=${OUTPUT_TILES}"
    echo "MINWM_ASYNC_A2A_BACKEND=${A2A_BACKEND}"
    echo "sp_degree=${degree}"
    echo "server_restart_boundary=before_and_after_this_position"
  } > "${position_dir}/async-a2a-contract.txt"
  record_compile_cache_state "${position_dir}/compile-cache-before.json"
  start_server \
    "${degree}" \
    "${position_dir}/profiler-off-server.log" \
    "${position_dir}/profiler-off-gpu-telemetry.csv"
  local output="${position_dir}/profiler-off.json"
  run_client \
    "${degree}" "${output}" "bf16-${lane}-sp${degree}-position${position}" \
    "${MINWM_RUN_ID}-sp${degree}-position${position}-${lane}" \
    --measurement-mode profiler_off \
    --warmup-chunks "${OFF_WARMUP_CHUNKS}" \
    --measured-chunks "${OFF_MEASURED_CHUNKS}"
  stop_server
  assert_server_stopped
  record_compile_cache_state "${position_dir}/compile-cache-after.json"
}

run_abba_measurements() {
  read -r -a abba_lanes <<< "${A2A_SEQUENCE}"
  if (( ${#abba_lanes[@]} % 4 != 0 )); then
    echo "MINWM_ASYNC_A2A_SEQUENCE must contain complete four-position ABBA/BAAB blocks" >&2
    exit 2
  fi
  local baseline_count=0
  local candidate_count=0
  local block_start
  for ((block_start = 0; block_start < ${#abba_lanes[@]}; block_start += 4)); do
    local block="${abba_lanes[*]:block_start:4}"
    local block_index=$((block_start / 4))
    local expected="candidate baseline baseline candidate"
    if (( block_index % 2 == 1 )); then
      expected="baseline candidate candidate baseline"
    fi
    if [[ "${block}" != "${expected}" ]]; then
      echo "block $((block_index + 1)) must alternate ABBA/BAAB; expected '${expected}', got '${block}'" >&2
      exit 2
    fi
  done
  local lane
  for lane in "${abba_lanes[@]}"; do
    case "${lane}" in
      baseline) baseline_count=$((baseline_count + 1)) ;;
      candidate) candidate_count=$((candidate_count + 1)) ;;
    esac
  done
  if (( baseline_count < MIN_LANE_SAMPLES || candidate_count < MIN_LANE_SAMPLES )); then
    echo "each lane needs at least ${MIN_LANE_SAMPLES} samples; baseline=${baseline_count}, candidate=${candidate_count}" >&2
    exit 2
  fi

  read -r -a degrees <<< "${SP_DEGREES}"
  for degree in "${degrees[@]}"; do
    if ! [[ "${degree}" =~ ^(2|4)$ ]]; then
      echo "S0 accepts SP degree 2 or 4, got ${degree}" >&2
      exit 2
    fi
    local sp_dir="${RESULT_ROOT}/sp${degree}"
    mkdir -p "${sp_dir}"
    local baseline_paths=()
    local candidate_paths=()
    local index=0
    for lane in "${abba_lanes[@]}"; do
      index=$((index + 1))
      local position
      position="$(printf '%02d' "${index}")"
      local position_dir="${sp_dir}/position-${position}-${lane}"
      run_profiler_off_position "${degree}" "${position}" "${lane}" "${position_dir}"
      if [[ "${lane}" == "baseline" ]]; then
        baseline_paths+=("${position_dir}/profiler-off.json")
      else
        candidate_paths+=("${position_dir}/profiler-off.json")
      fi
    done
    invalid_scope="${sp_dir}/summary"
    mkdir -p "${invalid_scope}"
    aggregate_repeats "${sp_dir}/summary/baseline" "${baseline_paths[@]}"
    aggregate_repeats "${sp_dir}/summary/candidate" "${candidate_paths[@]}"
  done

  invalid_scope="${RESULT_ROOT}/summary"
  mkdir -p "${invalid_scope}"
  python3 - "${RESULT_ROOT}" "${A2A_SEQUENCE}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
order = sys.argv[2].split()
metric_names = (
    "client_fps",
    "scheduler_fps",
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
)

def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    offset = (len(ordered) - 1) * q
    lower = int(offset)
    upper = min(lower + 1, len(ordered) - 1)
    weight = offset - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def summarize(values):
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "stdev": stdev,
        "cv_percent": stdev / mean * 100.0 if mean else 0.0,
    }

degrees = {}
for sp_dir in sorted(root.glob("sp*")):
    lanes = {}
    for lane in ("baseline", "candidate"):
        records = []
        for position_dir in sorted(sp_dir.glob(f"position-??-{lane}")):
            record = json.loads((position_dir / "profiler-off.json").read_text())
            metrics = record["metrics"]["profiler_off"]
            records.append(
                {
                    "position": int(position_dir.name.split("-")[1]),
                    "run_id": record["run_id"],
                    "metrics": {
                        "client_fps": metrics["client_fps"]["value"],
                        "scheduler_fps": metrics["scheduler_fps"]["value"],
                        "scheduler_chunk_wall_ms": metrics["scheduler_chunk_wall_ms"]["value"]["mean"],
                        "dit_wall_ms": metrics["dit_wall_ms"]["value"]["mean"],
                        "vae_wall_ms": metrics["vae_wall_ms"]["value"]["mean"],
                    },
                }
            )
        lanes[lane] = {
            "positions": records,
            "repeat_summary": json.loads(
                (sp_dir / "summary" / lane / "repeat-summary.json").read_text()
            ),
            "summary": {
                name: summarize([record["metrics"][name] for record in records])
                for name in metric_names
            },
        }
    deltas = {}
    for name in metric_names:
        baseline = lanes["baseline"]["summary"][name]["median"]
        candidate = lanes["candidate"]["summary"][name]["median"]
        raw_percent = (candidate / baseline - 1.0) * 100.0
        deltas[name] = {
            "basis": "median",
            "candidate_vs_baseline_percent": raw_percent,
            "performance_change_percent": (
                raw_percent if name.endswith("fps") else -raw_percent
            ),
        }
    degrees[sp_dir.name] = {"lanes": lanes, "deltas": deltas}

summary = {
    "schema_version": "minwm-realtime-async-a2a-abba/v1",
    "order": order,
    "server_restart_per_position": True,
    "compile_cache_policy": "preserve_and_snapshot_before_after_each_position",
    "degrees": degrees,
}
with (root / "async-a2a-abba-summary.json").open("x") as output:
    output.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/complete.txt"
  echo "MINWM_ASYNC_A2A_ABBA_COMPLETE results=${RESULT_ROOT}"
}

run_abba_measurements
