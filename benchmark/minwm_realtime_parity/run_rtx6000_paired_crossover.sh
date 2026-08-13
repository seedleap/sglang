#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR="${2:-${MINWM_PAIR:-}}"
[[ "${1:-}" == "--pair" && "${PAIR}" =~ ^[ABCD]$ ]] || {
  echo "usage: $0 --pair A|B|C|D" >&2
  exit 2
}

: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_ARCHIVE_ROOT:?set MINWM_ARCHIVE_ROOT}"

RUN_ID="rtx6000-pair-${PAIR,,}-setup"
MODEL_DIR="/work/minwm-realtime/${RUN_ID}/sglang-model"
PAIR_ROOT="/work/minwm-paired/pair-${PAIR,,}"
CONFIG_PATH="${PAIR_ROOT}/paired.json"
mkdir -p "${PAIR_ROOT}" "${MINWM_ARCHIVE_ROOT}"

flush_setup() {
  local source="/s3/world-model/evals/minwm/realtime-vae/20260813/bounded-8gpu/setup/pair-${PAIR,,}"
  mkdir -p "${source}"
  cp -a "/s3/world-model/evals/minwm/realtime-vae/20260813/bounded-8gpu/setup-local/pair-${PAIR,,}/." "${source}/" 2>/dev/null || true
}
trap flush_setup EXIT INT TERM

# The measured 480p lower bound already exceeds a 96 GB card once the causal
# cache reaches chunk 55.  720p cannot be admitted from that lower bound.  Do
# not shorten the 1089-frame contract: preserve serial output as the fallback.
if [[ "${PAIR}" == "C" || "${PAIR}" == "D" ]]; then
  size=832x480
  [[ "${PAIR}" == "D" ]] && size=1248x704
  gate_dir="${MINWM_ARCHIVE_ROOT}/memory-gate-${size}"
  mkdir -p "${gate_dir}"
  set +e
  python3 "${SCRIPT_DIR}/check_vae_overlap_memory_gate.py" \
    --gpu 0 --denoiser-peak-mib 88750 --vae-peak-mib 7946 \
    --transient-mib 516 --safety-margin-mib 2048 \
    --output "${gate_dir}/admission.json"
  status=$?
  set -e
  if (( status != 0 )); then
    printf '%s\n' "serial_single_gpu_or_second_gpu; never shorten or drop frames" > "${gate_dir}/FALLBACK"
    date -Iseconds > "${gate_dir}/COMPLETE"
    exit 3
  fi
fi

setup_results="/s3/world-model/evals/minwm/realtime-vae/20260813/bounded-8gpu/setup-local/pair-${PAIR,,}"
env CUDA_VISIBLE_DEVICES=0 \
  MINWM_RUN_ID="${RUN_ID}" \
  MINWM_BENCHMARK_MODE=setup_only \
  MINWM_RESULTS_ROOT="${setup_results}" \
  bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"

size=832x480
[[ "${PAIR}" == "B" ]] && size=1248x704
case_name="pair-${PAIR,,}-${size}-eager-vs-cuda-graph"

python3 - "${CONFIG_PATH}" "${SGLANG_GIT_REF}" "${MODEL_DIR}" \
  "${MINWM_ARCHIVE_ROOT}" "${case_name}" "${size}" <<'PY'
import json, sys
from pathlib import Path

path, commit, model, artifacts, name, size = sys.argv[1:]
common = [
    "sglang", "serve", "--model-path", model,
    "--pipeline-class-name", "MinWMCausalDMDPipeline",
    "--attention-backend", "fa", "--performance-mode", "speed",
    "--num-gpus", "1", "--tp-size", "1", "--sp-degree", "1",
    "--ulysses-degree", "1", "--ring-degree", "1",
    "--enable-cfg-parallel", "false", "--enable-torch-compile", "false",
    "--warmup-mode", "off", "--realtime-session-idle-timeout-s", "1800",
    "--port", "{port}",
]
env = {
    "MINWM_ATTENTION_IMPL": "dense",
    "MINWM_PACKED_ATTENTION_DETERMINISTIC": "false",
    "MINWM_NATIVE_COMPONENTS": "",
    "MINWM_PARITY_DETERMINISTIC": "1",
    "MINWM_DETERMINISTIC_ATTENTION": "true",
    "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
    "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "false",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "MINWM_ROOT": "/workspace/minWM",
}
config = {
    "sglang_git_ref": commit,
    "nvme_root": str(Path(path).parent / "nvme"),
    "artifact_root": artifacts,
    "base_port": 31000,
    "paired_reps": 3,
    "warmup_chunks": 5,
    "measured_chunks": 69,
    "steady_start_chunk": 10,
    "calibration_chunks": 12,
    "concurrency_threshold": 0.02,
    "health_timeout": 1800,
    "gpu_slots": [
        {"gpu": 0, "cpu_set": "0-19", "numa_node": 0},
        {"gpu": 1, "cpu_set": "20-39", "numa_node": 0},
    ],
    "cases": [{
        "name": name, "size": size, "mode": "cuda_graph",
        "control": {"command": common + ["--enable-cuda-graph", "false"], "env": env},
        "candidate": {"command": common + ["--enable-cuda-graph", "true"], "env": env},
    }],
}
Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

python3 "${SCRIPT_DIR}/run_paired_crossover.py" --config "${CONFIG_PATH}" --dry-run \
  | tee "${MINWM_ARCHIVE_ROOT}/dry-run.json"
exec python3 "${SCRIPT_DIR}/run_paired_crossover.py" --config "${CONFIG_PATH}"
