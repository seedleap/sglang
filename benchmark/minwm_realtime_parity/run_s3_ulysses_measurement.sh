#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN_ID="${MINWM_RUN_ID}"
BASE_WORK_ROOT="/work/minwm-realtime/${BASE_RUN_ID}"
MODEL_DIR="${BASE_WORK_ROOT}/sglang-model"
RUN_ROOT="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}/s3"
SHORT_CASES="${SCRIPT_DIR}/cases_720p_compile_smoke.json"
EVICTION_CASES="${SCRIPT_DIR}/cases_s3_720p_eviction.json"
CASE_ID="00_forward_080_pottery_720p"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${SHORT_CASES}" ]]
[[ -f "${EVICTION_CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
mkdir -p "${RUN_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export SGLANG_REALTIME_TRACE_SYNC_CUDA=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR

set_lane() {
  local lane="$1"
  case "${lane}" in
    00)
      export MINWM_FUSED_PRE_A2A_QK_NORM=0
      export MINWM_FUSED_POST_A2A_ROPE_CACHE=0
      ;;
    10)
      export MINWM_FUSED_PRE_A2A_QK_NORM=1
      export MINWM_FUSED_POST_A2A_ROPE_CACHE=0
      ;;
    01)
      export MINWM_FUSED_PRE_A2A_QK_NORM=0
      export MINWM_FUSED_POST_A2A_ROPE_CACHE=1
      ;;
    11)
      export MINWM_FUSED_PRE_A2A_QK_NORM=1
      export MINWM_FUSED_POST_A2A_ROPE_CACHE=1
      ;;
    *)
      echo "unsupported S3 lane ${lane}" >&2
      return 2
      ;;
  esac
}

server_pid=""
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

record_invalid_attempt() {
  local status="$1"
  python3 - "${RUN_ROOT}" "${status}" <<'PY'
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
    files.append(
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
marker = root / "invalid" / "attempt.json"
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(
    json.dumps(
        {
            "reason": "runner_exit_nonzero",
            "exit_status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_location": str(root),
            "recoverability": "all listed evidence remains in place on the PVC",
            "files": files,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"marked invalid attempt: {marker}")
PY
}

on_exit() {
  local status=$?
  trap - EXIT
  stop_server
  if (( status != 0 )); then
    record_invalid_attempt "${status}" || true
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local degree="$1" log_path="$2"
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
}

run_parity_case() {
  local cases="$1" output_prefix="$2" output_root="$3" degree="$4" lane="$5"
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${cases}" \
    --case "${CASE_ID}" \
    --results "${output_root}" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix "${output_prefix}" \
    --engine-name "sglang-minwm-s3-sp${degree}-lane${lane}-${output_prefix}" \
    --warmup-runs 0 \
    --kv-cache-num-frames 45 \
    | tee "${output_root}/${output_prefix}-client.log"
}

PYTHONPATH=/workspace/sglang/python python3 -m pytest -q \
  /workspace/sglang/test/registered/jit/diffusion/test_minwm_ulysses_fused.py \
  | tee "${RUN_ROOT}/kernel-bitwise-tests.log"

for degree in 2 4; do
  for lane in 00 10 01 11; do
    set_lane "${lane}"
    parity_root="${RUN_ROOT}/parity/sp${degree}/lane${lane}"
    mkdir -p "${parity_root}/short" "${parity_root}/eviction"
    {
      echo "sglang_commit=${SGLANG_GIT_REF}"
      echo "sp_degree=${degree}"
      echo "lane=${lane}"
      echo "fused_pre_a2a_qk_norm=${MINWM_FUSED_PRE_A2A_QK_NORM}"
      echo "fused_post_a2a_rope_cache=${MINWM_FUSED_POST_A2A_ROPE_CACHE}"
      echo "kv_cache_num_frames=45"
    } | tee "${parity_root}/contract.txt"
    start_server "${degree}" "${parity_root}/server.log"
    run_parity_case \
      "${SHORT_CASES}" short "${parity_root}/short" "${degree}" "${lane}"
    run_parity_case \
      "${EVICTION_CASES}" eviction "${parity_root}/eviction" "${degree}" "${lane}"
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
    sp_root = root / f"sp{degree}"
    summary[f"sp{degree}"] = {}
    for case_name, prefix in (("short", "short"), ("eviction", "eviction")):
        hashes = {}
        for lane in ("00", "10", "01", "11"):
            result = json.loads(
                (sp_root / f"lane{lane}" / case_name / f"{prefix}_run.json").read_text()
            )
            hashes[lane] = result["cases"][0]["frames_sha256"]
        if any(value != hashes["00"] for value in hashes.values()):
            raise RuntimeError(
                f"SP{degree} {case_name} parity mismatch: {hashes}"
            )
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

if [[ "${MINWM_S3_STOP_BEFORE_FORMAL_AB:-0}" == "1" \
  || -e "${RUN_ROOT}/STOP_BEFORE_FORMAL_AB" ]]; then
  date --utc +%Y-%m-%dT%H:%M:%SZ \
    | tee "${RUN_ROOT}/stopped-before-formal-ab.txt"
  echo "MINWM_S3_STOPPED_BEFORE_FORMAL_AB results=${RUN_ROOT}"
  exit 0
fi

measurement_root="${RUN_ROOT}/measurements"
mkdir -p "${measurement_root}"
for lane in 00 10 01 11; do
  set_lane "${lane}"
  lane_run_id="${BASE_RUN_ID}-lane${lane}"
  lane_work_root="/work/minwm-realtime/${lane_run_id}"
  mkdir -p "${lane_work_root}"
  ln -s "${MODEL_DIR}" "${lane_work_root}/sglang-model"
  lane_result_root="${measurement_root}/lane${lane}"
  mkdir -p "${lane_result_root}"
  {
    echo "sglang_commit=${SGLANG_GIT_REF}"
    echo "s0_tool_commit=b9240233b2438829cbd72ee3dfbc1d37ed675560"
    echo "lane=${lane}"
    echo "fused_pre_a2a_qk_norm=${MINWM_FUSED_PRE_A2A_QK_NORM}"
    echo "fused_post_a2a_rope_cache=${MINWM_FUSED_POST_A2A_ROPE_CACHE}"
    echo "kv_cache_num_frames=45"
  } | tee "${lane_result_root}/feature-contract.txt"
  MINWM_RUN_ID="${lane_run_id}" \
  MINWM_RESULTS_ROOT="${lane_result_root}" \
    bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  if ! command -v nsys >/dev/null; then
    nsys_bin="$(find "${lane_work_root}/nsight-systems" -type f -name nsys -perm -111 -print -quit)"
    [[ -n "${nsys_bin}" ]]
    export PATH="$(dirname "${nsys_bin}"):${PATH}"
  fi
done

python3 - "${measurement_root}" <<'PY' \
  | tee "${measurement_root}/s3-latency-count-assertions.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = [
    path
    for pattern in ("lane*/**/profiler-off-repeat*.json", "lane*/**/measurement.json")
    for path in sorted(root.glob(pattern))
    if "invalid" not in path.relative_to(root).parts
]
if not results:
    raise RuntimeError(f"no S3 measurement JSON found under {root}")
for result in results:
    record = json.loads(result.read_text())
    expected = record["workload"]["measured_chunks"]
    if record["mode"] == "profiler_off" and expected != 200:
        raise RuntimeError(f"{result}: profiler-off measured_chunks={expected}, want 200")
    sections = (
        ("profiler_off", "scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"),
        ("profiler_on", "dit_cuda_ms", "vae_cuda_ms"),
    )
    for section, *names in sections:
        metrics = record["metrics"].get(section)
        if metrics is None:
            continue
        for name in names:
            metric = metrics.get(name)
            if metric is None or metric.get("status") != "available":
                raise RuntimeError(f"{result}: {section}.{name} is not available")
            count = metric["value"].get("count")
            if count != expected:
                raise RuntimeError(
                    f"{result}: {section}.{name}.count={count}, want {expected}"
                )
    print(f"count-ok {result} measured_chunks={expected}")
PY

while IFS= read -r result; do
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${result}"
done < <(
  find "${measurement_root}" -type f \
    \( -name 'profiler-off-repeat*.json' -o -name measurement.json \) \
    -not -path '*/invalid/*' \
    | sort
)

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_ROOT}/complete.txt"
echo "MINWM_S3_ULYSSES_MEASUREMENT_COMPLETE results=${RUN_ROOT}"
