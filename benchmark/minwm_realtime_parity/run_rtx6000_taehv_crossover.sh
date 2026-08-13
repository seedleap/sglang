#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_ARCHIVE_S3_URI:?set MINWM_ARCHIVE_S3_URI}"
TAEHV_EXPERIMENT_MODE="${TAEHV_EXPERIMENT_MODE:-exact_vs_taehv}"
TAEHV_BASELINE_GIT_REF="${TAEHV_BASELINE_GIT_REF:-}"
export TAEHV_EXPERIMENT_MODE TAEHV_BASELINE_GIT_REF

RUN_ID="rtx6000-taehv-tianpeng-gap12-4220c8a"
INPUT_RUN_ID="rtx6000-taehv-shared-4220c8a"
MODEL_DIR="/work/minwm-realtime/${RUN_ID}/sglang-model"
PAIR_ROOT="/work/minwm-taehv-paired-tianpeng-gap12-${TAEHV_EXPERIMENT_MODE}"
CONFIG_PATH="${PAIR_ROOT}/paired.json"
LOCAL_ARTIFACT_ROOT="${PAIR_ROOT}/artifacts"
TAEHV_ROOT="/work/minwm-taehv"
TAEHV_CHECKPOINT="${TAEHV_ROOT}/taew2_2.pth"
TAEHV_SHA256="d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"
TAEHV_REVISION="093b918971d59001a0bad6dfd6e0409b5e1752cf"
TAEHV_URL="https://raw.githubusercontent.com/madebyollin/taehv/${TAEHV_REVISION}/taew2_2.pth"
ALIGNMENT_SOURCE_URL="https://leap-world-us-east-2.s3.us-east-2.amazonaws.com/world-model/sft/prompt_compare/detailmix_director_gap12_20260729_094145/inference-alignment/"
ALIGNMENT_S3_URI="s3://leap-world-us-east-2/world-model/sft/prompt_compare/detailmix_director_gap12_20260729_094145/inference-alignment"
ALIGNMENT_CACHE="/work/minwm-tianpeng-alignment/detailmix-director-gap12-20260729"
export TAEHV_SHA256 TAEHV_REVISION
mkdir -p "/work/minwm-realtime" "${PAIR_ROOT}" "${LOCAL_ARTIFACT_ROOT}" \
  "${TAEHV_ROOT}" "${ALIGNMENT_CACHE}"
aws s3 sync "${ALIGNMENT_S3_URI%/}/" "${ALIGNMENT_CACHE}/" \
  --exclude '*' \
  --include gap12.jsonl \
  --include input_manifest.json \
  --include run_manifest.json \
  --no-progress --only-show-errors
for alignment_file in gap12.jsonl input_manifest.json run_manifest.json; do
  [[ -s "${ALIGNMENT_CACHE}/${alignment_file}" ]]
done
ALIGNMENT_READ_URL="file://${ALIGNMENT_CACHE}/"
aws s3 sync "${MINWM_ARCHIVE_S3_URI%/}/" "${LOCAL_ARTIFACT_ROOT}/" \
  --no-progress --only-show-errors || true

setup_results="/work/minwm-realtime/${RUN_ID}/setup-results"
did_full_setup=false
exec 9>"/work/minwm-realtime/.${RUN_ID}.lock"
flock -x 9
if [[ ! -f "/work/minwm-realtime/${RUN_ID}/SETUP_COMPLETE" ]]; then
  reuse_input_env=()
  if [[ -f "/work/minwm-realtime/${INPUT_RUN_ID}/checkpoint/model.pt" \
    && -d "/work/minwm-realtime/${INPUT_RUN_ID}/pretrained/transformer" ]]; then
    reuse_input_env+=(MINWM_REUSE_INPUT_RUN_ID="${INPUT_RUN_ID}")
  fi
  env "${reuse_input_env[@]}" CUDA_VISIBLE_DEVICES=0 \
    MINWM_RUN_ID="${RUN_ID}" \
    MINWM_BENCHMARK_MODE=setup_only \
    MINWM_RESULTS_ROOT="${setup_results}" \
    MINWM_CONVERT_LOCAL_ATTN_SIZE=32 \
    MINWM_CONVERT_SINK_SIZE=8 \
    MINWM_CONVERT_WINDOW_SIZE=32 \
    MINWM_CONVERT_ROPE_POSITION_MODE=block_relative \
    MINWM_CONVERT_ROPE_MAX_FRAME_GAP=12 \
    MINWM_CONVERT_PROMPT_FIRST_FRAME_PIN_ENABLED=1 \
    bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"
  date -Iseconds > "/work/minwm-realtime/${RUN_ID}/SETUP_COMPLETE"
  did_full_setup=true
fi

manifest_root="${LOCAL_ARTIFACT_ROOT}/environment"
mkdir -p "${manifest_root}"
if [[ -f "/work/minwm-realtime/${INPUT_RUN_ID}/checkpoint/model.pt" ]]; then
  experiment_checkpoint="/work/minwm-realtime/${INPUT_RUN_ID}/checkpoint/model.pt"
else
  experiment_checkpoint="/work/minwm-realtime/${RUN_ID}/checkpoint/model.pt"
fi
experiment_checkpoint_sha256="$(sha256sum "${experiment_checkpoint}" | awk '{print $1}')"
printf '%s  %s\n' "${experiment_checkpoint_sha256}" "${experiment_checkpoint}" \
  > "${manifest_root}/experiment-checkpoint-sha256.txt"
alignment_provenance="${manifest_root}/alignment-provenance.json"
python3 "${SCRIPT_DIR}/tianpeng_runtime_alignment_gate.py" \
  --model-dir "${MODEL_DIR}" \
  --checkpoint-sha256 "${experiment_checkpoint_sha256}" \
  --alignment-url "${ALIGNMENT_READ_URL}" \
  --canonical-source-url "${ALIGNMENT_SOURCE_URL}" \
  --output "${alignment_provenance}"
aws s3 cp "${alignment_provenance}" \
  "${MINWM_ARCHIVE_S3_URI%/}/environment/alignment-provenance.json" \
  --no-progress --only-show-errors
alignment_complete="${manifest_root}/ALIGNMENT_COMPLETE"
printf 'pass\n' > "${alignment_complete}.partial"
mv "${alignment_complete}.partial" "${alignment_complete}"
aws s3 cp "${alignment_complete}" \
  "${MINWM_ARCHIVE_S3_URI%/}/environment/ALIGNMENT_COMPLETE" \
  --no-progress --only-show-errors
flock -u 9
if [[ "${did_full_setup}" != "true" ]]; then
  runtime_run_id="rtx6000-taehv-${SGLANG_GIT_REF:0:10}-runtime"
  runtime_input_run_id="${RUN_ID}"
  if [[ -f "/work/minwm-realtime/${INPUT_RUN_ID}/checkpoint/model.pt" \
    && -d "/work/minwm-realtime/${INPUT_RUN_ID}/pretrained/transformer" ]]; then
    runtime_input_run_id="${INPUT_RUN_ID}"
  fi
  env CUDA_VISIBLE_DEVICES=0 \
    MINWM_RUN_ID="${runtime_run_id}" \
    MINWM_REUSE_INPUT_RUN_ID="${runtime_input_run_id}" \
    MINWM_REUSE_MODEL_RUN_ID="${RUN_ID}" \
    MINWM_BENCHMARK_MODE=setup_only \
    MINWM_RESULTS_ROOT="/work/minwm-realtime/${runtime_run_id}/setup-results" \
    bash "${SCRIPT_DIR}/aws_b200_entrypoint.sh"
fi

if [[ "${TAEHV_EXPERIMENT_MODE}" == "memory_ab" ]]; then
  unit_test_log="${PAIR_ROOT}/test-realtime-vae.log"
  (
    cd /workspace/sglang
    PYTHONPATH=/workspace/sglang/python python3 -m pytest -q \
      python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
  ) 2>&1 | tee "${unit_test_log}"
  aws s3 cp "${unit_test_log}" \
    "${MINWM_ARCHIVE_S3_URI%/}/environment/test-realtime-vae.log" \
    --no-progress --only-show-errors
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
  "${LOCAL_ARTIFACT_ROOT}" "${MINWM_ARCHIVE_S3_URI}" "${TAEHV_CHECKPOINT}" \
  "${experiment_checkpoint_sha256}" "${ALIGNMENT_READ_URL}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

path, commit, model, artifacts, archive, taehv_checkpoint, checkpoint_sha256, alignment_url = (
    sys.argv[1:]
)
gpu_count = int(os.environ["MINWM_REQUESTED_GPUS"])
if gpu_count not in {1, 2}:
    raise RuntimeError(f"MINWM_REQUESTED_GPUS must be 1 or 2, got {gpu_count}")
available = sorted(os.sched_getaffinity(0))
topology = subprocess.check_output(["lscpu", "-p=CPU,NODE"], text=True)
by_numa = {}
for line in topology.splitlines():
    if not line or line.startswith("#"):
        continue
    cpu, node = (int(value) for value in line.split(",")[:2])
    if cpu in available:
        by_numa.setdefault(node, []).append(cpu)
lane_width = 12 if gpu_count == 1 else 20
eligible = [
    (node, cpus) for node, cpus in sorted(by_numa.items()) if len(cpus) >= lane_width
]
if gpu_count == 1 and eligible:
    numa0, lane0 = eligible[0]
    lanes = [(0, lane0[:lane_width], numa0)]
elif gpu_count == 1 and len(available) >= lane_width:
    lanes = [(0, available[:lane_width], None)]
elif len(eligible) >= 2:
    (numa0, lane0), (numa1, lane1) = eligible[:2]
    lane0, lane1 = lane0[:20], lane1[:20]
    lanes = [(0, lane0, numa0), (1, lane1, numa1)]
elif eligible and len(eligible[0][1]) >= 40:
    numa0, cpus = eligible[0]
    numa1, lane0, lane1 = numa0, cpus[:20], cpus[20:40]
    lanes = [(0, lane0, numa0), (1, lane1, numa1)]
elif len(available) >= 40:
    numa0 = numa1 = None
    lane0, lane1 = available[:20], available[20:40]
    lanes = [(0, lane0, numa0), (1, lane1, numa1)]
else:
    raise RuntimeError(f"insufficient CPUs for {gpu_count} GPU topology: {len(available)}")

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
    "MINWM_RUNTIME_ALIGNMENT_LOG": "1",
}
if os.environ["TAEHV_EXPERIMENT_MODE"] == "memory_ab":
    env["SGLANG_REALTIME_MEMORY_TRACE"] = "1"

alignment_pattern = (
    "MINWM_RUNTIME_ALIGNMENT local_attn_size=32 sink_size=8 window_size=32 "
    "rope_position_mode=block_relative rope_gap=12 "
    "prompt_first_frame_pin_enabled=True request_sink_size=8 "
    "request_window_size=32 allow_growth=False"
)

def variant(command, backend):
    return {
        "command": command,
        "env": env,
        "required_log_patterns": [
            f'"decoder_backend":"{backend}"',
            alignment_pattern,
        ],
    }

cases = []
for size in ("832x480", "1248x704"):
    if os.environ["TAEHV_EXPERIMENT_MODE"] == "memory_ab":
        baseline_ref = os.environ["TAEHV_BASELINE_GIT_REF"]
        if not baseline_ref:
            raise RuntimeError("TAEHV_BASELINE_GIT_REF is required for memory_ab")
        taehv_common = common + [
            "--vae-config.taehv-checkpoint-path", taehv_checkpoint
        ]
        baseline = variant(taehv_common, "taehv")
        baseline["env"] = {
            **env,
            "PYTHONPATH": "/workspace/sglang-baseline/python",
        }
        baseline["sglang_git_ref"] = baseline_ref

        fixed = variant(taehv_common, "taehv")
        fixed["env"] = {**env, "PYTHONPATH": "/workspace/sglang/python"}
        fixed["required_log_patterns"].extend(
            [
                "realtime_memory_checkpoint checkpoint=model_loaded",
                '"checkpoint":"first_image_vae_encode_gate"',
                '"checkpoint":"after_dit_cache_init"',
            ]
        )
        fixed_offload = variant(
            taehv_common + ["--vae-cpu-offload", "true"], "taehv"
        )
        fixed_offload["env"] = fixed["env"]
        fixed_offload["required_log_patterns"] = list(
            fixed["required_log_patterns"]
        )
        fixed_low_memory = variant(
            taehv_common
            + [
                "--vae-cpu-offload",
                "true",
                "--text-encoder-cpu-offload",
                "true",
            ],
            "taehv",
        )
        fixed_low_memory["env"] = fixed["env"]
        fixed_low_memory["required_log_patterns"] = [
            *fixed["required_log_patterns"],
            '"checkpoint":"before_text_encode"',
            '"checkpoint":"after_text_encode"',
        ]

        cases.extend(
            [
                {
                    "name": f"taehv-memory-a-tianpeng-gap12-{size}-eager",
                    "size": size,
                    "mode": "eager",
                    "paired_reps": 1,
                    "control": baseline,
                    "candidate": fixed,
                },
                {
                    "name": f"taehv-memory-b-tianpeng-gap12-{size}-eager",
                    "size": size,
                    "mode": "eager",
                    "paired_reps": 1,
                    "control": fixed,
                    "candidate": fixed_offload,
                },
                {
                    "name": f"taehv-memory-c-tianpeng-gap12-{size}-eager",
                    "size": size,
                    "mode": "eager",
                    "control": fixed_offload,
                    "candidate": fixed_low_memory,
                },
            ]
        )
    else:
        cases.append({
            "name": f"taehv-local-tianpeng-gap12-{size}-eager",
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
    "checkpoint_sha256": checkpoint_sha256,
    "alignment_url": alignment_url,
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
    "paired_reps": (
        3
        if os.environ["TAEHV_EXPERIMENT_MODE"] == "memory_ab"
        else (4 if gpu_count == 1 else 3)
    ),
    "warmup_chunks": 5,
    "measured_chunks": 69,
    "steady_start_chunk": 10,
    "calibration_chunks": 12,
    "concurrency_threshold": 0.02,
    "health_timeout": 1800,
    "telemetry_loop_ms": 100 if os.environ["TAEHV_EXPERIMENT_MODE"] == "memory_ab" else 1000,
    "gpu_slots": [
        {"gpu": gpu, "cpu_set": cpu_set(cpus), "numa_node": numa}
        for gpu, cpus, numa in lanes
    ],
    "cases": cases,
}
Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

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
if [[ "${TAEHV_EXPERIMENT_MODE}" == "exact_vs_taehv" ]]; then
  for size in 832x480 1248x704; do
    case_root="${LOCAL_ARTIFACT_ROOT}/taehv-local-tianpeng-gap12-${size}-eager"
    quality_output="${case_root}/quality-comparison.json"
    python3 "${SCRIPT_DIR}/compare_taehv_samples.py" \
      --exact "${case_root}/rep-00/control/quality-samples/${size}" \
      --taehv "${case_root}/rep-00/candidate/quality-samples/${size}" \
      --output "${quality_output}"
    aws s3 cp "${quality_output}" \
      "${MINWM_ARCHIVE_S3_URI%/}/taehv-local-tianpeng-gap12-${size}-eager/quality-comparison.json" \
      --no-progress --only-show-errors
  done
fi
