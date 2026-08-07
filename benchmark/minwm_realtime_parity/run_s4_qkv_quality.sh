#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/work/minwm-realtime/${MINWM_RUN_ID}/sglang-model"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/s4-qkv-quality"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
COMPILE_CASES="${MINWM_COMPILE_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_two_chunk.json}"
CASE_ID="${MINWM_CASE_ID:-00_forward_080_pottery_720p}"
server_pid=""

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite existing S4 quality attempt: ${RESULT_ROOT}" >&2
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
trap stop_server EXIT INT TERM

run_lane() {
  local label="$1" fused="$2" num_gpus="$3" sp_degree="$4" tp_size="$5"
  local output_prefix="$6" result_dir="$7" dump_dir="$8" compile="$9"
  local quantization="${10:-}"
  local cases_path="${11:-${CASES}}"
  local log_path="${result_dir}/${output_prefix}-server.log"
  local quantization_args=()
  if [[ -n "${quantization}" ]]; then
    quantization_args=(--quantization "${quantization}")
  fi
  mkdir -p "${result_dir}"

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_FUSED_QKV_PROJECTION="${fused}" \
  MINWM_PARITY_DUMP_DIR="${dump_dir}" \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --vae-config.use-parallel-decode true \
    --attention-backend fa \
    --performance-mode speed \
    --num-gpus "${num_gpus}" \
    --tp-size "${tp_size}" \
    --sp-degree "${sp_degree}" \
    --ulysses-degree "${sp_degree}" \
    --ring-degree 1 \
    --enable-cfg-parallel false \
    --enable-torch-compile "${compile}" \
    --warmup-mode off \
    "${quantization_args[@]}" \
    --port 30000 > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${log_path}"

  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${cases_path}" \
    --case "${CASE_ID}" \
    --results "${result_dir}" \
    --output-prefix "${output_prefix}" \
    --engine-name "sglang-minwm-${label}" \
    --kv-cache-num-frames 45
  stop_server
}

run_tp2_existing_blocker() {
  local label="$1" fused="$2" output_prefix="$3" result_dir="$4"
  local log_path="${result_dir}/${output_prefix}-server.log"
  local client_log="${result_dir}/${output_prefix}-client.log"
  mkdir -p "${result_dir}"

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE=parallel \
  MINWM_FUSED_QKV_PROJECTION="${fused}" \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --vae-config.use-parallel-decode true \
    --attention-backend fa \
    --performance-mode speed \
    --num-gpus 2 \
    --tp-size 2 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --enable-cfg-parallel false \
    --enable-torch-compile false \
    --warmup-mode off \
    --port 30000 > "${log_path}" 2>&1 &
  server_pid=$!
  wait_for_server "${log_path}"

  set +e
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${COMPILE_CASES}" \
    --case "${CASE_ID}" \
    --results "${result_dir}" \
    --output-prefix "${output_prefix}" \
    --engine-name "sglang-minwm-${label}" \
    --kv-cache-num-frames 45 > >(tee "${client_log}") 2>&1
  local client_status=$?
  set -e
  stop_server

  if (( client_status == 0 )); then
    echo "TP2 unexpectedly passed; update the acceptance lane instead of hiding it" >&2
    return 1
  fi
  grep -F "'MinWMRMSNorm' object has no attribute 'variance_epsilon'" \
    "${client_log}"
  grep -F "MinWM QKV projection mode:" "${log_path}"
  printf '%s\n' "${client_status}" > "${result_dir}/${output_prefix}-client-exit-status.txt"
}

SP1_RESULTS="${RESULT_ROOT}/sp1"
run_lane control 0 1 1 1 baseline "${SP1_RESULTS}" \
  "${RESULT_ROOT}/layer-probes/control" false
run_lane candidate 1 1 1 1 sglang "${SP1_RESULTS}" \
  "${RESULT_ROOT}/layer-probes/candidate" false
run_lane candidate-replay 1 1 1 1 candidate_replay "${SP1_RESULTS}" "" false
run_lane candidate-compile 1 1 1 1 candidate_compile "${SP1_RESULTS}" "" true \
  "" "${COMPILE_CASES}"
run_tp2_existing_blocker tp2-control 0 tp2_control_blocked "${SP1_RESULTS}"
run_tp2_existing_blocker tp2-candidate 1 tp2_candidate_blocked "${SP1_RESULTS}"
run_lane candidate-fp8-fallback 1 1 1 1 candidate_fp8 "${SP1_RESULTS}" "" false fp8

python3 - "${SP1_RESULTS}" "${SGLANG_GIT_REF:-unknown}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
lanes = {}
for lane in ("tp2_control_blocked", "tp2_candidate_blocked"):
    lanes[lane] = {
        "client_exit_status": int(
            (root / f"{lane}-client-exit-status.txt").read_text().strip()
        ),
        "client_log": f"{lane}-client.log",
        "server_log": f"{lane}-server.log",
        "expected_error": (
            "'MinWMRMSNorm' object has no attribute 'variance_epsilon'"
        ),
    }
(root / "tp2-existing-s3-blocker.json").write_text(
    json.dumps(
        {
            "schema": "minwm-s4-tp2-existing-blocker/v1",
            "runner_commit": sys.argv[2],
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "Full-model TP2 is blocked before QKV comparison by unchanged "
                "MinWMRMSNorm tensor-parallel code; S4 does not modify S3."
            ),
            "lanes": lanes,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

grep -F "requested with quantized weights; using the compatible three-projection fallback" \
  "${SP1_RESULTS}/candidate_fp8-server.log"
grep -F "MinWM QKV projection mode: three-gemm-parity-fallback" \
  "${SP1_RESULTS}/candidate_fp8-server.log"

python3 "${SCRIPT_DIR}/compare_results.py" \
  --cases "${CASES}" \
  --case "${CASE_ID}" \
  --results "${SP1_RESULTS}" \
  --profile bf16_backend_candidate

SP4_RESULTS="${RESULT_ROOT}/sp4"
run_lane control-sp4 0 4 4 1 baseline "${SP4_RESULTS}" "" false
run_lane candidate-sp4 1 4 4 1 sglang "${SP4_RESULTS}" "" false

python3 "${SCRIPT_DIR}/compare_results.py" \
  --cases "${CASES}" \
  --case "${CASE_ID}" \
  --results "${SP4_RESULTS}" \
  --profile bf16_backend_candidate

SP2_RESULTS="${RESULT_ROOT}/sp2"
run_lane control-sp2 0 2 2 1 baseline "${SP2_RESULTS}" "" false
run_lane candidate-sp2 1 2 2 1 sglang "${SP2_RESULTS}" "" false
run_lane candidate-sp2-replay 1 2 2 1 candidate_replay "${SP2_RESULTS}" "" false

python3 "${SCRIPT_DIR}/compare_results.py" \
  --cases "${CASES}" \
  --case "${CASE_ID}" \
  --results "${SP2_RESULTS}" \
  --profile bf16_backend_candidate

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 - \
  "${RESULT_ROOT}" "${CASE_ID}" "${SCRIPT_DIR}/thresholds.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

from compare_results import evaluate, metric_block

root = Path(sys.argv[1])
case_id = sys.argv[2]
thresholds_path = Path(sys.argv[3])
thresholds = json.loads(thresholds_path.read_text())
fast_lane_profile = thresholds["profiles"]["bf16_backend_candidate"]

comparisons = {}
for degree in ("sp1", "sp2"):
    case_dir = root / degree / "cases" / case_id
    candidate = np.load(case_dir / "sglang.npy", allow_pickle=False)
    replay = np.load(case_dir / "candidate_replay.npy", allow_pickle=False)
    metrics = metric_block(candidate, replay)
    comparisons[f"{degree}_candidate_replay"] = metrics
    if not metrics["bitwise_equal"]:
        raise AssertionError(f"{degree} candidate replay is not bitwise deterministic")

sp1_case = root / "sp1" / "cases" / case_id
candidate = np.load(sp1_case / "sglang.npy", allow_pickle=False)
for prefix in ("candidate_compile",):
    value = np.load(sp1_case / f"{prefix}.npy", allow_pickle=False)
    reference = candidate[: value.shape[0]]
    metrics = metric_block(reference, value)
    passed, failures = evaluate({"generated_frames": metrics}, fast_lane_profile)
    comparisons[f"sp1_candidate_vs_{prefix}"] = {
        **metrics,
        "passed": passed,
        "failures": failures,
    }
    if not passed:
        raise AssertionError(f"{prefix} quality threshold failed: {failures}")

control_dump = next((root / "layer-probes/control").glob("sglang/sp_01_rank_00"))
candidate_dump = next((root / "layer-probes/candidate").glob("sglang/sp_01_rank_00"))
layer_metrics = {}
for control_path in sorted(control_dump.glob("*.pt")):
    candidate_path = candidate_dump / control_path.name
    if not candidate_path.exists():
        continue
    control = torch.load(control_path, map_location="cpu", weights_only=True)
    candidate_tensor = torch.load(
        candidate_path, map_location="cpu", weights_only=True
    )
    if not isinstance(control, torch.Tensor) or not isinstance(
        candidate_tensor, torch.Tensor
    ):
        continue
    if control.shape != candidate_tensor.shape:
        layer_metrics[control_path.name] = {
            "control_shape": list(control.shape),
            "candidate_shape": list(candidate_tensor.shape),
            "shape_equal": False,
        }
        continue
    difference = control.float() - candidate_tensor.float()
    layer_metrics[control_path.name] = {
        "shape": list(control.shape),
        "dtype": str(control.dtype),
        "bitwise_equal": bool(torch.equal(control, candidate_tensor)),
        "max_abs": float(difference.abs().max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
    }

for exact_name in (
    "self_q_input_000.pt",
    "self_q_weight.pt",
    "self_q_bias.pt",
):
    if not layer_metrics.get(exact_name, {}).get("bitwise_equal", False):
        raise AssertionError(f"load/input contract drifted at {exact_name}")

for probe_name in (
    "self_q_output_000.pt",
    "self_k_output_000.pt",
    "self_v_output_000.pt",
    "self_q_norm_000.pt",
    "self_k_norm_000.pt",
    "block0_output_000.pt",
    "output_proj_output_000.pt",
):
    if probe_name not in layer_metrics:
        raise AssertionError(f"required layer probe missing: {probe_name}")
    if not layer_metrics[probe_name].get("shape_equal", True):
        raise AssertionError(f"layer probe shape drifted: {probe_name}")

(root / "determinism-and-variant-metrics.json").write_text(
    json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
)
(root / "layer-probe-metrics.json").write_text(
    json.dumps(layer_metrics, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({
    "determinism": {
        name: metrics["bitwise_equal"]
        for name, metrics in comparisons.items()
        if name.endswith("candidate_replay")
    },
    "layer_probe_files": len(layer_metrics),
}, indent=2, sort_keys=True))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/complete.txt"
echo "MINWM_S4_QKV_QUALITY_COMPLETE results=${RESULT_ROOT}"
