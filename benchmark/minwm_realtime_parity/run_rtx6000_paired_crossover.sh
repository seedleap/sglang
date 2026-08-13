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
: "${MINWM_ARCHIVE_S3_URI:?set MINWM_ARCHIVE_S3_URI}"

RUN_ID="rtx6000-shared-a3d231ccdc-setup"
MODEL_DIR="/work/minwm-realtime/${RUN_ID}/sglang-model"
PAIR_ROOT="/work/minwm-paired/pair-${PAIR,,}"
CONFIG_PATH="${PAIR_ROOT}/paired.json"
LOCAL_ARTIFACT_ROOT="${PAIR_ROOT}/artifacts"
mkdir -p "/work/minwm-realtime" "${PAIR_ROOT}" "${MINWM_ARCHIVE_ROOT}" "${LOCAL_ARTIFACT_ROOT}"
aws s3 sync "${MINWM_ARCHIVE_S3_URI%/}/" "${LOCAL_ARTIFACT_ROOT}/" \
  --no-progress --only-show-errors || true

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

setup_results="/work/minwm-realtime/${RUN_ID}/setup-results"
did_full_setup=false
exec 9>"/work/minwm-realtime/.${RUN_ID}.lock"
flock -x 9
if [[ ! -f "/work/minwm-realtime/${RUN_ID}/SETUP_COMPLETE" ]]; then
  env CUDA_VISIBLE_DEVICES=0 \
    MINWM_RUN_ID="${RUN_ID}" \
    MINWM_BENCHMARK_MODE=setup_only \
    MINWM_RESULTS_ROOT="${setup_results}" \
    bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"
  date -Iseconds > "/work/minwm-realtime/${RUN_ID}/SETUP_COMPLETE"
  did_full_setup=true
fi
flock -u 9
if [[ "${did_full_setup}" != "true" ]]; then
  runtime_run_id="rtx6000-pair-${PAIR,,}-${SGLANG_GIT_REF:0:10}-runtime"
  env CUDA_VISIBLE_DEVICES=0 \
    MINWM_RUN_ID="${runtime_run_id}" \
    MINWM_REUSE_INPUT_RUN_ID="${RUN_ID}" \
    MINWM_REUSE_MODEL_RUN_ID="${RUN_ID}" \
    MINWM_BENCHMARK_MODE=setup_only \
    MINWM_RESULTS_ROOT="/work/minwm-realtime/${runtime_run_id}/setup-results" \
    bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"
fi

size=832x480
[[ "${PAIR}" == "B" ]] && size=1248x704
case_name="pair-${PAIR,,}-${size}-w32s8-eager-vs-cuda-graph"

python3 - "${CONFIG_PATH}" "${SGLANG_GIT_REF}" "${MODEL_DIR}" \
  "${LOCAL_ARTIFACT_ROOT}" "${MINWM_ARCHIVE_S3_URI}" "${case_name}" "${size}" "${PAIR}" <<'PY'
import json, os, sys
from pathlib import Path

path, commit, model, artifacts, archive, name, size, pair = sys.argv[1:]
available = sorted(os.sched_getaffinity(0))
offsets = {"A": 0, "B": 48, "C": 96, "D": 144}
offset = offsets[pair] if len(available) >= offsets[pair] + 40 else 0
lane0, lane1 = available[offset:offset + 20], available[offset + 20:offset + 40]
if len(lane1) != 20:
    raise RuntimeError(f"need 40 CPUs for a pair, found {len(available)}")

def cpu_set(cpus):
    return ",".join(str(cpu) for cpu in cpus)

cpu0, cpu1, numa = cpu_set(lane0), cpu_set(lane1), 0
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
    "upload_command": [
        "bash", "-lc",
        "if [[ -f {source}/COMPLETE ]]; then "
        "aws s3 cp {source} " + archive + "/{relative} --recursive "
        "--exclude COMPLETE --exclude UPLOADED.json --no-progress --only-show-errors && "
        "aws s3 cp {source}/COMPLETE " + archive + "/{relative}/COMPLETE "
        "--no-progress --only-show-errors && "
        "aws s3 cp {source}/result.json " + archive + "/{relative}/UPLOADED.json "
        "--no-progress --only-show-errors; else "
        "aws s3 cp {source} " + archive + "/{relative} --recursive "
        "--exclude INTERRUPTED.json --no-progress --only-show-errors && "
        "aws s3 cp {source}/INTERRUPTED.json " + archive + "/{relative}/INTERRUPTED.json "
        "--no-progress --only-show-errors; fi",
    ],
    "upload_file_command": [
        "aws", "s3", "cp", "{source}", archive + "/{relative}",
        "--no-progress", "--only-show-errors",
    ],
    "base_port": 31000,
    "paired_reps": 3,
    "warmup_chunks": 5,
    "measured_chunks": 69,
    "steady_start_chunk": 10,
    "calibration_chunks": 12,
    "concurrency_threshold": 0.02,
    "health_timeout": 1800,
    "gpu_slots": [
        {"gpu": 0, "cpu_set": cpu0, "numa_node": numa},
        {"gpu": 1, "cpu_set": cpu1, "numa_node": numa},
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
  | tee "${PAIR_ROOT}/dry-run.json"
aws s3 cp "${PAIR_ROOT}/dry-run.json" \
  "${MINWM_ARCHIVE_S3_URI%/}/dry-run-${SGLANG_GIT_REF:0:10}.json" \
  --no-progress --only-show-errors
exec python3 "${SCRIPT_DIR}/run_paired_crossover.py" --config "${CONFIG_PATH}"
