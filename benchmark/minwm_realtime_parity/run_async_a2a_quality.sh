#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_CONTAINER_IMAGE:?set MINWM_CONTAINER_IMAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/work/minwm-realtime/${MINWM_RUN_ID}/sglang-model"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/async-a2a-quality"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_5s.json}"
CASE_ID="${MINWM_CASE_ID:-00_forward_080_pottery_720p}"
SP_DEGREES="${MINWM_ASYNC_A2A_SP_DEGREES:-2 4}"
A2A_BACKEND="${MINWM_ASYNC_A2A_BENCH_BACKEND:-process_group}"
A2A_EXPERIMENT="${MINWM_ASYNC_A2A_EXPERIMENT:-input_split}"
OUTPUT_TILES="${MINWM_ASYNC_A2A_OUTPUT_TILES:-1}"
STABILITY_REQUESTS="${MINWM_ASYNC_A2A_STABILITY_REQUESTS:-10}"
server_pid=""
invalid_scope="${RESULT_ROOT}"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ "$(git -C /workspace/sglang rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
if ! [[ "${STABILITY_REQUESTS}" =~ ^[1-9][0-9]*$ \
  && "${OUTPUT_TILES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Stability requests and output tiles must be positive" >&2
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
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite existing async A2A quality attempt: ${RESULT_ROOT}" >&2
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

{
  echo "sglang_commit=${SGLANG_GIT_REF}"
  echo "minwm_commit=${MINWM_GIT_REF}"
  echo "container_image=${MINWM_CONTAINER_IMAGE}"
  echo "cases=${CASES}"
  echo "case_id=${CASE_ID}"
  echo "sp_degrees=${SP_DEGREES}"
  echo "a2a_backend=${A2A_BACKEND}"
  echo "a2a_experiment=${A2A_EXPERIMENT}"
  echo "output_tiles=${OUTPUT_TILES}"
  echo "candidate_stability_requests=${STABILITY_REQUESTS}"
  echo "resolution=1248x704"
  echo "chunks=8"
  echo "generated_frames=128"
  echo "seed=42"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} > "${RESULT_ROOT}/contract.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi-q.txt"

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 300); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
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

mark_invalid() {
  local status="$1"
  python3 - "${invalid_scope}" "${status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
(root / "invalid-marker.json").write_text(json.dumps({
    "exit_status": int(sys.argv[2]),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": str(root.resolve()),
    "recoverability": "Partial artifacts are intentionally preserved in place.",
}, indent=2, sort_keys=True) + "\n")
PY
}

on_exit() {
  local status=$?
  set +e
  stop_server
  if (( status != 0 )); then
    mark_invalid "${status}"
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_lane() {
  local degree="$1" lane="$2" async_flag="$3" output_prefix="$4"
  local warmup_runs="$5" result_dir="$6" dump_dir="$7"
  local input_a2a_flag=0 output_a2a_flag=0
  if [[ "${async_flag}" == "1" ]]; then
    if [[ "${A2A_EXPERIMENT}" == "input_split" ]]; then
      input_a2a_flag=1
    else
      output_a2a_flag=1
    fi
  fi
  local log_path="${result_dir}/${output_prefix}-server.log"
  invalid_scope="${result_dir}/${lane}"
  mkdir -p "${result_dir}" "${invalid_scope}"
  {
    echo "lane=${lane}"
    echo "MINWM_ASYNC_A2A=${input_a2a_flag}"
    echo "MINWM_ASYNC_A2A_OUTPUT=${output_a2a_flag}"
    echo "MINWM_ASYNC_A2A_OUTPUT_TILES=${OUTPUT_TILES}"
    echo "MINWM_ASYNC_A2A_BACKEND=${A2A_BACKEND}"
    echo "sp_degree=${degree}"
    echo "warmup_runs=${warmup_runs}"
  } > "${invalid_scope}/lane-contract.txt"

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_ASYNC_A2A="${input_a2a_flag}" \
  MINWM_ASYNC_A2A_OUTPUT="${output_a2a_flag}" \
  MINWM_ASYNC_A2A_OUTPUT_TILES="${OUTPUT_TILES}" \
  MINWM_ASYNC_A2A_BACKEND="${A2A_BACKEND}" \
  MINWM_PARITY_DUMP_DIR="${dump_dir}" \
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
  wait_for_server "${log_path}"

  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --results "${result_dir}" \
    --output-prefix "${output_prefix}" \
    --engine-name "sglang-minwm-async-a2a-${lane}-sp${degree}" \
    --warmup-runs "${warmup_runs}" \
    --kv-cache-num-frames 45
  stop_server
}

read -r -a degrees <<< "${SP_DEGREES}"
for degree in "${degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(2|4)$ ]]; then
    echo "quality runner accepts SP2 or SP4, got ${degree}" >&2
    exit 2
  fi
  sp_dir="${RESULT_ROOT}/sp${degree}"
  run_lane "${degree}" baseline 0 baseline 0 "${sp_dir}" \
    "${RESULT_ROOT}/layer-probes/baseline/sp${degree}"
  run_lane "${degree}" candidate 1 sglang "$((STABILITY_REQUESTS - 1))" \
    "${sp_dir}" "${RESULT_ROOT}/layer-probes/candidate/sp${degree}"

  python3 "${SCRIPT_DIR}/compare_results.py" \
    --cases "${CASES}" \
    --case "${CASE_ID}" \
    --results "${sp_dir}" \
    --profile bitwise
done

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 - \
  "${RESULT_ROOT}" "${CASE_ID}" "${SP_DEGREES}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
case_id = sys.argv[2]
degrees = sys.argv[3].split()
summary = {"video_bitwise": {}, "tensor_parity": {}}
required = {
    "self_q_norm_000.pt",
    "self_k_norm_000.pt",
    "self_q_roped_000.pt",
    "self_k_roped_000.pt",
    "self_attention_output_000.pt",
    "block0_output_000.pt",
    "output_proj_output_000.pt",
}

for degree in degrees:
    sp_name = f"sp{degree}"
    report = json.loads((root / sp_name / "report.json").read_text())
    case = next(item for item in report["cases"] if item["id"] == case_id)
    if not case["metrics"]["generated_frames"]["bitwise_equal"]:
        raise AssertionError(f"{sp_name}: baseline/candidate video is not bitwise exact")
    summary["video_bitwise"][sp_name] = True

    lane_roots = {}
    for lane in ("baseline", "candidate"):
        matches = sorted((root / "layer-probes" / lane / sp_name).glob(
            f"sglang/sp_{int(degree):02d}_rank_*"
        ))
        if len(matches) != int(degree):
            raise AssertionError(
                f"{sp_name}/{lane}: expected {degree} rank dumps, got {len(matches)}"
            )
        lane_roots[lane] = matches

    rank_summaries = {}
    for rank, (baseline_dir, candidate_dir) in enumerate(
        zip(lane_roots["baseline"], lane_roots["candidate"])
    ):
        baseline_names = {path.name for path in baseline_dir.glob("*.pt")}
        candidate_names = {path.name for path in candidate_dir.glob("*.pt")}
        missing_candidate = baseline_names - candidate_names
        if missing_candidate:
            raise AssertionError(
                f"{sp_name}/rank{rank}: candidate is missing baseline probes "
                f"{sorted(missing_candidate)}"
            )
        missing = required - baseline_names
        if missing:
            raise AssertionError(f"{sp_name}/rank{rank}: missing probes {sorted(missing)}")
        metrics = {}
        for name in sorted(baseline_names):
            baseline = torch.load(
                baseline_dir / name, map_location="cpu", weights_only=True
            )
            candidate = torch.load(
                candidate_dir / name, map_location="cpu", weights_only=True
            )
            if not isinstance(baseline, torch.Tensor) or not isinstance(
                candidate, torch.Tensor
            ):
                continue
            if baseline.shape != candidate.shape or baseline.dtype != candidate.dtype:
                raise AssertionError(f"{sp_name}/rank{rank}/{name}: tensor contract differs")
            equal = torch.equal(baseline, candidate)
            difference = baseline.float() - candidate.float()
            metrics[name] = {
                "shape": list(baseline.shape),
                "dtype": str(baseline.dtype),
                "bitwise_equal": bool(equal),
                "max_abs": float(difference.abs().max().item()),
                "rmse": float(difference.square().mean().sqrt().item()),
            }
            if not equal:
                raise AssertionError(f"{sp_name}/rank{rank}/{name}: not bitwise exact")
        rank_summaries[f"rank{rank}"] = {
            "candidate_extra_probes": sorted(candidate_names - baseline_names),
            "metrics": metrics,
        }
    summary["tensor_parity"][sp_name] = rank_summaries

(root / "async-a2a-quality-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "video_bitwise": summary["video_bitwise"],
    "tensor_probe_counts": {
        degree: {rank: len(record["metrics"]) for rank, record in ranks.items()}
        for degree, ranks in summary["tensor_parity"].items()
    },
}, indent=2, sort_keys=True))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/complete.txt"
echo "MINWM_ASYNC_A2A_QUALITY_COMPLETE results=${RESULT_ROOT}"
