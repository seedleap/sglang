#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_ARCHIVE_S3_URI:?set MINWM_ARCHIVE_S3_URI}"

RUN_ID="rtx6000-taehv-shared-4220c8a"
MODEL_DIR="/work/minwm-realtime/${RUN_ID}/sglang-model"
PAIR_ROOT="/work/minwm-taehv-paired"
CONFIG_PATH="${PAIR_ROOT}/paired.json"
LOCAL_ARTIFACT_ROOT="${PAIR_ROOT}/artifacts"
TAEHV_ROOT="/work/minwm-taehv"
TAEHV_CHECKPOINT="${TAEHV_ROOT}/taew2_2.pth"
TAEHV_SHA256="d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"
TAEHV_REVISION="093b918971d59001a0bad6dfd6e0409b5e1752cf"
TAEHV_URL="https://raw.githubusercontent.com/madebyollin/taehv/${TAEHV_REVISION}/taew2_2.pth"
export TAEHV_SHA256 TAEHV_REVISION
mkdir -p "/work/minwm-realtime" "${PAIR_ROOT}" "${LOCAL_ARTIFACT_ROOT}" "${TAEHV_ROOT}"
aws s3 sync "${MINWM_ARCHIVE_S3_URI%/}/" "${LOCAL_ARTIFACT_ROOT}/" \
  --no-progress --only-show-errors || true

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
  runtime_run_id="rtx6000-taehv-${SGLANG_GIT_REF:0:10}-runtime"
  env CUDA_VISIBLE_DEVICES=0 \
    MINWM_RUN_ID="${runtime_run_id}" \
    MINWM_REUSE_INPUT_RUN_ID="${RUN_ID}" \
    MINWM_REUSE_MODEL_RUN_ID="${RUN_ID}" \
    MINWM_BENCHMARK_MODE=setup_only \
    MINWM_RESULTS_ROOT="/work/minwm-realtime/${runtime_run_id}/setup-results" \
    bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"
fi

python3 -m pip install --no-deps \
  "taehv @ git+https://github.com/madebyollin/taehv.git@${TAEHV_REVISION}" \
  --root-user-action=ignore
exec 8>"${TAEHV_ROOT}/.checkpoint.lock"
flock -x 8
if ! echo "${TAEHV_SHA256}  ${TAEHV_CHECKPOINT}" | sha256sum --check --status 2>/dev/null; then
  rm -f "${TAEHV_CHECKPOINT}.partial"
  python3 - "${TAEHV_URL}" "${TAEHV_CHECKPOINT}.partial" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  echo "${TAEHV_SHA256}  ${TAEHV_CHECKPOINT}.partial" | sha256sum --check -
  mv "${TAEHV_CHECKPOINT}.partial" "${TAEHV_CHECKPOINT}"
fi
flock -u 8
echo "${TAEHV_SHA256}  ${TAEHV_CHECKPOINT}" | sha256sum --check -

python3 - "${CONFIG_PATH}" "${SGLANG_GIT_REF}" "${MODEL_DIR}" \
  "${LOCAL_ARTIFACT_ROOT}" "${MINWM_ARCHIVE_S3_URI}" "${TAEHV_CHECKPOINT}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

path, commit, model, artifacts, archive, taehv_checkpoint = sys.argv[1:]
available = sorted(os.sched_getaffinity(0))
topology = subprocess.check_output(["lscpu", "-p=CPU,NODE"], text=True)
by_numa = {}
for line in topology.splitlines():
    if not line or line.startswith("#"):
        continue
    cpu, node = (int(value) for value in line.split(",")[:2])
    if cpu in available:
        by_numa.setdefault(node, []).append(cpu)
eligible = [(node, cpus) for node, cpus in sorted(by_numa.items()) if len(cpus) >= 20]
if len(eligible) >= 2:
    (numa0, lane0), (numa1, lane1) = eligible[:2]
    lane0, lane1 = lane0[:20], lane1[:20]
elif eligible and len(eligible[0][1]) >= 40:
    numa0, cpus = eligible[0]
    numa1, lane0, lane1 = numa0, cpus[:20], cpus[20:40]
elif len(available) >= 40:
    numa0 = numa1 = None
    lane0, lane1 = available[:20], available[20:40]
else:
    raise RuntimeError(f"need 40 CPUs for a pair, found {len(available)}")

def cpu_set(cpus):
    return ",".join(str(cpu) for cpu in cpus)

common = [
    "sglang", "serve", "--model-path", model,
    "--pipeline-class-name", "MinWMCausalDMDPipeline",
    "--attention-backend", "fa", "--performance-mode", "speed",
    "--num-gpus", "1", "--tp-size", "1", "--sp-degree", "1",
    "--ulysses-degree", "1", "--ring-degree", "1",
    "--enable-cfg-parallel", "false", "--enable-torch-compile", "false",
    "--enable-cuda-graph", "false", "--warmup-mode", "off",
    "--realtime-session-idle-timeout-s", "1800", "--port", "{port}",
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

def variant(command, backend):
    return {
        "command": command,
        "env": env,
        "required_log_patterns": [f'"decoder_backend": "{backend}"'],
    }

cases = []
for size in ("832x480", "1248x704"):
    cases.append({
        "name": f"taehv-local-{size}-w32s8-eager",
        "size": size,
        "mode": "eager",
        "control": variant(common, "causal_vae"),
        "candidate": variant(
            common + ["--vae-config.taehv-checkpoint-path", taehv_checkpoint],
            "taehv",
        ),
    })

config = {
    "sglang_git_ref": commit,
    "minwm_git_ref": os.environ["MINWM_GIT_REF"],
    "taehv_checkpoint_path": taehv_checkpoint,
    "taehv_checkpoint_sha256": os.environ["TAEHV_SHA256"],
    "taehv_revision": os.environ["TAEHV_REVISION"],
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
    "base_port": 32000,
    "paired_reps": 3,
    "warmup_chunks": 5,
    "measured_chunks": 69,
    "steady_start_chunk": 10,
    "calibration_chunks": 12,
    "concurrency_threshold": 0.02,
    "health_timeout": 1800,
    "gpu_slots": [
        {"gpu": 0, "cpu_set": cpu_set(lane0), "numa_node": numa0},
        {"gpu": 1, "cpu_set": cpu_set(lane1), "numa_node": numa1},
    ],
    "cases": cases,
}
Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

manifest_root="${LOCAL_ARTIFACT_ROOT}/environment"
mkdir -p "${manifest_root}"
git -C /workspace/sglang rev-parse HEAD > "${manifest_root}/sglang-git.txt"
git -C /workspace/minWM rev-parse HEAD > "${manifest_root}/minwm-git.txt"
sha256sum "${TAEHV_CHECKPOINT}" > "${manifest_root}/taehv-checkpoint-sha256.txt"
printf '%s\n' "${TAEHV_REVISION}" > "${manifest_root}/taehv-revision.txt"

python3 "${SCRIPT_DIR}/run_paired_crossover.py" --config "${CONFIG_PATH}" --dry-run \
  | tee "${PAIR_ROOT}/dry-run.json"
aws s3 cp "${PAIR_ROOT}/dry-run.json" \
  "${MINWM_ARCHIVE_S3_URI%/}/dry-run-${SGLANG_GIT_REF:0:10}.json" \
  --no-progress --only-show-errors
python3 "${SCRIPT_DIR}/run_paired_crossover.py" --config "${CONFIG_PATH}"
for size in 832x480 1248x704; do
  case_root="${LOCAL_ARTIFACT_ROOT}/taehv-local-${size}-w32s8-eager"
  quality_output="${case_root}/quality-comparison.json"
  python3 "${SCRIPT_DIR}/compare_taehv_samples.py" \
    --exact "${case_root}/rep-00/control/quality-samples/${size}" \
    --taehv "${case_root}/rep-00/candidate/quality-samples/${size}" \
    --output "${quality_output}"
  aws s3 cp "${quality_output}" \
    "${MINWM_ARCHIVE_S3_URI%/}/taehv-local-${size}-w32s8-eager/quality-comparison.json" \
    --no-progress --only-show-errors
done
