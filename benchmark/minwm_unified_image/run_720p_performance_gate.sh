#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINWM_GPU_FAMILY:?set MINWM_GPU_FAMILY to hopper or blackwell}"
: "${EXPECTED_SOURCE_COMMIT:?set EXPECTED_SOURCE_COMMIT}"
: "${EXPECTED_IMAGE_DIGEST:?set EXPECTED_IMAGE_DIGEST}"

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'invalid EXPECTED_SOURCE_COMMIT: %s\n' "${EXPECTED_SOURCE_COMMIT}" >&2
  exit 2
fi
if [[ ! "${EXPECTED_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'invalid EXPECTED_IMAGE_DIGEST: %s\n' "${EXPECTED_IMAGE_DIGEST}" >&2
  exit 2
fi

case "${MINWM_GPU_FAMILY}" in
  hopper)
    EXPECTED_GPU_NAME="NVIDIA H200"
    EXPECTED_CAPABILITY="9.0"
    EXPECTED_PACKED_BACKEND="fa3"
    BASELINE_SCHEDULER_FPS="7.89698680794534"
    MIN_SCHEDULER_FPS="7.6600772037069795"
    BASELINE_CLIENT_FPS="7.890654071296376"
    MIN_CLIENT_FPS="7.653934449157485"
    ;;
  blackwell)
    EXPECTED_GPU_NAME="NVIDIA B200"
    EXPECTED_CAPABILITY="10.0"
    EXPECTED_PACKED_BACKEND="fa4"
    BASELINE_SCHEDULER_FPS="14.395795592730778"
    MIN_SCHEDULER_FPS="13.963921724948854"
    BASELINE_CLIENT_FPS="14.376812178063995"
    MIN_CLIENT_FPS="13.945507812722074"
    ;;
  *)
    printf 'unsupported MINWM_GPU_FAMILY: %s\n' "${MINWM_GPU_FAMILY}" >&2
    exit 2
    ;;
esac

# Accepted profiler-off packed-fast BF16 measurements are recorded in
# benchmark/minwm_720p_attn_ffn_20260818/RESULTS.zh-CN.md. The release gate is
# a one-sided regression check: both scheduler and loopback-client FPS must
# retain at least 97% of the device baseline; improvements remain valid.

SOURCE_ROOT="/sgl-workspace/sglang"
WORK_ROOT="${MINWM_PERF_WORK_ROOT:-/work/minwm-720p-release-gate}"
CHECKPOINT_MOUNT="/s3/world-model/evals/minwm/checkpoint-staging/wan22-5B-stage3-dmd-30-0731-acfd622962df/global_step_001800/ema_student/model.pt"
DONOR_MOUNT="/s3/world-model/checkpoints/minWM/Wan2.2-TI2V-5B-from-diffusers"
FIRST_FRAME_MOUNT="/s3/world-model/eval/platform/eval_sets/minWM/testset100_v2/img/p02.png"
CHECKPOINT_URI="s3://leap-world-us-east-2/world-model/evals/minwm/checkpoint-staging/wan22-5B-stage3-dmd-30-0731-acfd622962df/global_step_001800/ema_student/model.pt"
CHECKPOINT_VERSION="9lr1pX__59kUFlsv6Q.NKFQO3EoRQY9C"
CHECKPOINT_BYTES="10007165995"
CHECKPOINT_CRC64="GLaCsoJZbwM="
CHECKPOINT_SOURCE_URI="s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/dmd/wan22-5B-stage3-dmd-30-0731-acfd622962df/global_step_001800/ema_student/model.pt"
CHECKPOINT_SOURCE_VERSION="ghDNCSQtQ2OJ1LnrxtMKLXtqNWDK0CsR"
CHECKPOINT_SOURCE_ETAG="14038072b7fc6a9fee92388a02dc2569-191"
CASES="${SOURCE_ROOT}/benchmark/minwm_realtime_parity/cases_720p_compile_smoke.json"
INPUT_CONTRACT="${SOURCE_ROOT}/benchmark/minwm_unified_image/inputs_720p.json"
CASE="00_forward_080_pottery_720p"
CHECKPOINT="${WORK_ROOT}/checkpoint/model.pt"
DONOR="${WORK_ROOT}/donor"
MODEL="${WORK_ROOT}/model"
RESULTS="${WORK_ROOT}/results"
FIRST_FRAME="${WORK_ROOT}/inputs/p02.png"
GATE_CASES="${RESULTS}/cases.local.json"
SERVER_LOG="${RESULTS}/server.log"
CLIENT_LOG="${RESULTS}/client.log"
THROUGHPUT="${RESULTS}/throughput.json"
GPU_LOG="${RESULTS}/gpu.csv"
RUNTIME_JSON="${RESULTS}/runtime.json"
WARMUP_CHUNKS=20
MEASURED_CHUNKS=100
KV_FRAMES=45

server_pid=""
monitor_pid=""
client_pid=""
wait_pid_down() {
  local pid="$1"
  local attempts="$2"
  for _ in $(seq 1 "${attempts}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

stop_client() {
  if [[ -z "${client_pid}" ]]; then
    return
  fi
  kill -TERM "${client_pid}" 2>/dev/null || true
  if ! wait_pid_down "${client_pid}" 40; then
    kill -KILL "${client_pid}" 2>/dev/null || true
  fi
  wait "${client_pid}" 2>/dev/null || true
  client_pid=""
}

stop_monitor() {
  if [[ -z "${monitor_pid}" ]]; then
    return
  fi
  kill -TERM "${monitor_pid}" 2>/dev/null || true
  if ! wait_pid_down "${monitor_pid}" 20; then
    kill -KILL "${monitor_pid}" 2>/dev/null || true
  fi
  wait "${monitor_pid}" 2>/dev/null || true
  monitor_pid=""
}

wait_for_server_down() {
  local attempts="${1:-20}"
  for _ in $(seq 1 "${attempts}"); do
    if ! curl --fail --silent --connect-timeout 1 --max-time 2 \
      http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

stop_server() {
  if [[ -z "${server_pid}" ]]; then
    return
  fi
  kill -TERM "${server_pid}" 2>/dev/null || true
  pkill -TERM -f "sglang serve --model-path ${MODEL}.*--port 30000" \
    2>/dev/null || true
  local needs_kill=0
  if ! wait_pid_down "${server_pid}" 120; then
    needs_kill=1
  fi
  if ! wait_for_server_down 5; then
    needs_kill=1
  fi
  if (( needs_kill )); then
    kill -KILL "${server_pid}" 2>/dev/null || true
    pkill -KILL -f "sglang serve --model-path ${MODEL}.*--port 30000" \
      2>/dev/null || true
    wait_pid_down "${server_pid}" 20 || true
    wait_for_server_down 10 || true
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}

cleanup() {
  local status=$?
  stop_client
  stop_monitor
  stop_server
  if (( status != 0 )); then
    printf 'MINWM_720P_PERFORMANCE_GATE_FAILED status=%d\n' "${status}" >&2
    if [[ -f "${SERVER_LOG}" ]]; then
      printf '%s\n' '--- server.log tail ---' >&2
      tail -400 "${SERVER_LOG}" >&2 || true
    fi
    if [[ -f "${CLIENT_LOG}" ]]; then
      printf '%s\n' '--- client.log tail ---' >&2
      tail -200 "${CLIENT_LOG}" >&2 || true
    fi
    if [[ -f "${RESULTS}/conversion.log" ]]; then
      printf '%s\n' '--- conversion.log tail ---' >&2
      tail -100 "${RESULTS}/conversion.log" >&2 || true
    fi
  fi
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "${WORK_ROOT}" == /work/* ]] || {
  printf 'MINWM_PERF_WORK_ROOT must be below /work: %s\n' "${WORK_ROOT}" >&2
  exit 2
}
if [[ -e "${WORK_ROOT}" ]] && find "${WORK_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  printf 'refusing non-empty work directory: %s\n' "${WORK_ROOT}" >&2
  exit 2
fi
mkdir -p \
  "${WORK_ROOT}/checkpoint" \
  "${WORK_ROOT}/inputs" \
  "${DONOR}" \
  "${MODEL}" \
  "${RESULTS}"

for path in \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py" \
  "${SOURCE_ROOT}/benchmark/minwm_realtime_parity/benchmark_realtime_throughput.py" \
  "${INPUT_CONTRACT}" \
  "${CASES}" \
  "${CHECKPOINT_MOUNT}" \
  "${FIRST_FRAME_MOUNT}"; do
  [[ -f "${path}" ]] || {
    printf 'required file is missing: %s\n' "${path}" >&2
    exit 1
  }
done
for component in text_encoder tokenizer vae scheduler; do
  [[ -e "${DONOR_MOUNT}/${component}" ]] || {
    printf 'donor component is missing: %s\n' "${DONOR_MOUNT}/${component}" >&2
    exit 1
  }
done

export EXPECTED_GPU_NAME EXPECTED_CAPABILITY EXPECTED_PACKED_BACKEND
export BASELINE_SCHEDULER_FPS MIN_SCHEDULER_FPS
export BASELINE_CLIENT_FPS MIN_CLIENT_FPS
export CHECKPOINT_URI CHECKPOINT_VERSION CHECKPOINT_BYTES CHECKPOINT_CRC64
export INPUT_CONTRACT
export CASES
export WARMUP_CHUNKS MEASURED_CHUNKS KV_FRAMES
export RUNTIME_JSON
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

expected_commit = os.environ["EXPECTED_SOURCE_COMMIT"]
actual_commit = os.environ.get("SGLANG_BUILD_COMMIT")
if actual_commit != expected_commit:
    raise RuntimeError(
        f"image source commit mismatch: expected {expected_commit}, got {actual_commit}"
    )
if os.environ.get("SGLANG_USE_SGL_FA3_KERNEL") != "0":
    raise RuntimeError("release image must set SGLANG_USE_SGL_FA3_KERNEL=0")
if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(
        "performance baseline requires torch 2.11.0+cu130 / CUDA 13.0, got "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"single-card gate requires exactly one visible GPU, got {torch.cuda.device_count()}"
    )
gpu_name = torch.cuda.get_device_name(0)
capability = ".".join(str(value) for value in torch.cuda.get_device_capability(0))
if gpu_name != os.environ["EXPECTED_GPU_NAME"]:
    raise RuntimeError(
        f"GPU mismatch: expected {os.environ['EXPECTED_GPU_NAME']}, got {gpu_name}"
    )
if capability != os.environ["EXPECTED_CAPABILITY"]:
    raise RuntimeError(
        f"compute capability mismatch: expected {os.environ['EXPECTED_CAPABILITY']}, got {capability}"
    )
input_contract_path = Path(os.environ["INPUT_CONTRACT"])
input_contract_bytes = input_contract_path.read_bytes()
input_contract = json.loads(input_contract_bytes)
cases_path = Path(os.environ["CASES"])
cases_bytes = cases_path.read_bytes()
cases_sha256 = hashlib.sha256(cases_bytes).hexdigest()
if cases_sha256 != input_contract["cases"]["sha256"]:
    raise RuntimeError(
        "720p cases file drifted from inputs_720p.json: "
        f"expected {input_contract['cases']['sha256']}, got {cases_sha256}"
    )
case_contract = json.loads(cases_bytes)["cases"][0]
checkpoint_contract = input_contract["checkpoint"]
claimed_checkpoint = {
    "uri": os.environ["CHECKPOINT_URI"],
    "version_id": os.environ["CHECKPOINT_VERSION"],
    "bytes": int(os.environ["CHECKPOINT_BYTES"]),
    "crc64nvme": os.environ["CHECKPOINT_CRC64"],
}
expected_checkpoint = {
    "uri": f"s3://{checkpoint_contract['bucket']}/{checkpoint_contract['key']}",
    "version_id": checkpoint_contract["version_id"],
    "bytes": int(checkpoint_contract["content_length"]),
    "crc64nvme": checkpoint_contract["checksum_crc64nvme"],
}
if claimed_checkpoint != expected_checkpoint:
    raise RuntimeError(
        "runner checkpoint constants drifted from inputs_720p.json: "
        f"runner={claimed_checkpoint}, manifest={expected_checkpoint}"
    )
payload = {
    "schema_version": "minwm-720p-image-performance-gate/v1",
    "source_commit": actual_commit,
    "requested_image_digest": os.environ["EXPECTED_IMAGE_DIGEST"],
    "image_tag": os.environ.get("SGLANG_IMAGE_TAG"),
    "gpu_family": os.environ["MINWM_GPU_FAMILY"],
    "gpu": gpu_name,
    "compute_capability": capability,
    "visible_gpu_count": torch.cuda.device_count(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "sglang_use_sgl_fa3_kernel": os.environ["SGLANG_USE_SGL_FA3_KERNEL"],
    "expected_packed_backend": os.environ["EXPECTED_PACKED_BACKEND"],
    "checkpoint": {
        **claimed_checkpoint,
        "sha256": checkpoint_contract["sha256"],
        "input_contract": str(input_contract_path),
        "input_contract_sha256": hashlib.sha256(input_contract_bytes).hexdigest(),
    },
    "request": {
        "case": "00_forward_080_pottery_720p",
        "cases_file": str(cases_path),
        "cases_sha256": cases_sha256,
        "prompt": case_contract["prompt"],
        "first_frame": case_contract["first_frame"],
        "action_weights": case_contract["action_weights"],
        "size": "1248x704",
        "kv_cache_num_frames": int(os.environ["KV_FRAMES"]),
        "warmup_chunks": int(os.environ["WARMUP_CHUNKS"]),
        "measured_chunks": int(os.environ["MEASURED_CHUNKS"]),
        "sp_degree": 1,
    },
    "execution": {
        "performance_mode": "speed",
        "attention_impl": "packed",
        "packed_attention_deterministic": False,
        "precision": "bf16",
        "quantization": None,
        "segment_compile": True,
        "whole_dit_compile": False,
        "cuda_graph": False,
        "cfg_parallel": False,
        "native_components": "",
        "cache_rotated_k": True,
        "precompute_cache_rope": True,
        "cache_packed_metadata": True,
    },
    "accepted_baseline": {
        "scheduler_fps": float(os.environ["BASELINE_SCHEDULER_FPS"]),
        "minimum_scheduler_fps": float(os.environ["MIN_SCHEDULER_FPS"]),
        "client_fps": float(os.environ["BASELINE_CLIENT_FPS"]),
        "minimum_client_fps": float(os.environ["MIN_CLIENT_FPS"]),
        "maximum_regression_fraction": 0.03,
    },
}
Path(os.environ["RUNTIME_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

actual_checkpoint_bytes="$(stat -c '%s' "${CHECKPOINT_MOUNT}")"
if [[ "${actual_checkpoint_bytes}" != "${CHECKPOINT_BYTES}" ]]; then
  printf 'checkpoint size mismatch: expected %s, got %s\n' \
    "${CHECKPOINT_BYTES}" "${actual_checkpoint_bytes}" >&2
  exit 1
fi

printf 'Staging immutable gs1800 inputs from the read-only S3 CSI mount\n'
cp "${CHECKPOINT_MOUNT}" "${CHECKPOINT}"
cp "${FIRST_FRAME_MOUNT}" "${FIRST_FRAME}"
for component in text_encoder tokenizer vae scheduler; do
  cp -a --no-preserve=ownership "${DONOR_MOUNT}/${component}" "${DONOR}/"
done
[[ "$(stat -c '%s' "${CHECKPOINT}")" == "${CHECKPOINT_BYTES}" ]]
sha256sum "${CHECKPOINT}" | tee "${RESULTS}/checkpoint-sha256.txt"
sha256sum "${FIRST_FRAME}" | tee "${RESULTS}/first-frame-sha256.txt"

python3 - \
  "${INPUT_CONTRACT}" \
  "${DONOR}" \
  "${RESULTS}/checkpoint-sha256.txt" \
  "${RESULTS}/first-frame-sha256.txt" \
  "${RESULTS}/local-input-contract.json" \
  "${CASES}" \
  "${FIRST_FRAME}" \
  "${GATE_CASES}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
donor_root = Path(sys.argv[2])
checkpoint_sha_path = Path(sys.argv[3])
first_frame_sha_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])
cases_path = Path(sys.argv[6])
first_frame_path = Path(sys.argv[7])
gate_cases_path = Path(sys.argv[8])
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
checkpoint_sha = checkpoint_sha_path.read_text().split()[0]
first_frame_sha = first_frame_sha_path.read_text().split()[0]
if checkpoint_sha != manifest["checkpoint"]["sha256"]:
    raise RuntimeError(
        "checkpoint SHA256 mismatch: "
        f"expected {manifest['checkpoint']['sha256']}, got {checkpoint_sha}"
    )
if first_frame_sha != manifest["first_frame"]["sha256"]:
    raise RuntimeError(
        "first-frame SHA256 mismatch: "
        f"expected {manifest['first_frame']['sha256']}, got {first_frame_sha}"
    )
expected_donor = {
    item["path"]: {
        "bytes": int(item["content_length"]),
        "sha256": item["sha256"],
    }
    for item in manifest["donor"]["files"]
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


actual_donor = {}
for path in donor_root.rglob("*"):
    if path.is_file():
        actual_donor[str(path.relative_to(donor_root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
if actual_donor != expected_donor:
    missing = sorted(set(expected_donor) - set(actual_donor))
    unexpected = sorted(set(actual_donor) - set(expected_donor))
    changed = {
        path: {"expected": expected_donor[path], "actual": actual_donor[path]}
        for path in set(expected_donor) & set(actual_donor)
        if expected_donor[path] != actual_donor[path]
    }
    raise RuntimeError(
        "donor inventory mismatch: "
        f"missing={missing}, unexpected={unexpected}, changed={changed}"
    )
result = {
    "schema_version": "minwm-720p-local-input-contract/v1",
    "status": "pass",
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "checkpoint_sha256": checkpoint_sha,
    "first_frame_sha256": first_frame_sha,
    "donor_file_count": len(actual_donor),
    "donor_bytes": sum(item["bytes"] for item in actual_donor.values()),
    "donor_sha256": dict(sorted(
        (path, item["sha256"]) for path, item in actual_donor.items()
    )),
}
gate_cases = json.loads(cases_path.read_text())
selected_case = gate_cases["cases"][0]
selected_case["first_frame"] = str(first_frame_path)
selected_case["first_frame_sha256"] = first_frame_sha
gate_cases_path.write_text(json.dumps(gate_cases, indent=2, sort_keys=True) + "\n")
result["gate_cases_sha256"] = hashlib.sha256(gate_cases_path.read_bytes()).hexdigest()
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

python3 "${SOURCE_ROOT}/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py" \
  --minwm-checkpoint "${CHECKPOINT}" \
  --donor-diffusers-dir "${DONOR}" \
  --output-dir "${MODEL}" \
  --link-donor \
  --source-uri "${CHECKPOINT_SOURCE_URI}" \
  --source-version-id "${CHECKPOINT_SOURCE_VERSION}" \
  --source-etag "${CHECKPOINT_SOURCE_ETAG}" \
  --local-attn-size -1 \
  --sink-size 0 \
  --sliding-window-num-frames 128 \
  --rope-position-mode absolute \
  --rope-max-frame-gap 1 \
  > "${RESULTS}/conversion.log"

python3 - "${MODEL}" "${CHECKPOINT_BYTES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "transformer/config.json").read_text())
manifest = json.loads((root / "minwm_conversion_manifest.json").read_text())
expected_config = {
    "local_attn_size": -1,
    "sink_size": 0,
    "sliding_window_num_frames": 128,
    "rope_position_mode": "absolute",
    "rope_max_frame_gap": 1,
    "prompt_first_frame_pin_enabled": False,
    "action_type": "primitive_token_residual",
}
for key, expected in expected_config.items():
    actual = config.get(key)
    if actual != expected:
        raise RuntimeError(f"converted model {key}: expected {expected!r}, got {actual!r}")
if manifest["source_checkpoint"]["local_size"] != int(sys.argv[2]):
    raise RuntimeError("converted model recorded the wrong checkpoint size")
if manifest["generator"]["action_type"] != "primitive_token_residual":
    raise RuntimeError("converted model recorded the wrong action type")
print(json.dumps({"converted_model_contract": expected_config}, sort_keys=True))
PY

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MINWM_S3_MOUNT=/s3
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

MINWM_ATTENTION_IMPL=packed \
MINWM_PACKED_ATTENTION_DETERMINISTIC=false \
MINWM_PARITY_DETERMINISTIC=0 \
MINWM_DETERMINISTIC_ATTENTION=false \
SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 \
MINWM_NATIVE_COMPONENTS= \
MINWM_SEGMENT_COMPILE=true \
MINWM_CACHE_ROTATED_K=true \
MINWM_PRECOMPUTE_CACHE_ROPE=true \
MINWM_CACHE_PACKED_METADATA=true \
MINWM_NVTX_BLOCK_PHASES=0 \
  sglang serve \
    --model-path "${MODEL}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --attention-backend fa \
    --performance-mode speed \
    --num-gpus 1 \
    --sp-degree 1 \
    --enable-cfg-parallel false \
    --enable-torch-compile false \
    --enable-cuda-graph false \
    --warmup-mode off \
    --realtime-session-idle-timeout-s 1800 \
    --port 30000 \
    > "${SERVER_LOG}" 2>&1 &
server_pid=$!

(
  while kill -0 "${server_pid}" 2>/dev/null; do
    nvidia-smi -i 0 \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "${GPU_LOG}" &
monitor_pid=$!

server_ready=0
for _ in $(seq 1 900); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    printf 'server exited before becoming healthy\n' >&2
    exit 1
  fi
  if curl --fail --silent --connect-timeout 1 --max-time 2 \
    http://127.0.0.1:30000/health >/dev/null; then
    server_ready=1
    break
  fi
  sleep 2
done
if [[ "${server_ready}" != "1" ]]; then
  printf 'server did not become healthy before timeout\n' >&2
  exit 1
fi

for evidence in \
  'Applying performance_mode=speed' \
  'Attention backends for transformer: fa' \
  'MinWM execution profile: attention_impl=packed packed_deterministic=False segment_compile=True cache_rotated_k=True precompute_cache_rope=True cache_packed_metadata=True nvtx_block_phases=False'; do
  if ! grep -Fq "${evidence}" "${SERVER_LOG}"; then
    printf 'server contract evidence is missing before measurement: %s\n' \
      "${evidence}" >&2
    exit 1
  fi
done

python3 "${SOURCE_ROOT}/benchmark/minwm_realtime_parity/benchmark_realtime_throughput.py" \
  --cases "${GATE_CASES}" \
  --case "${CASE}" \
  --output "${THROUGHPUT}" \
  --profile-name packed-fast-bf16-release-gate \
  --warmup-chunks "${WARMUP_CHUNKS}" \
  --measured-chunks "${MEASURED_CHUNKS}" \
  --kv-cache-num-frames "${KV_FRAMES}" \
  --save-first-measured-frame \
  --timeout 2400 \
  > "${CLIENT_LOG}" 2>&1 &
client_pid=$!

# The packed backend is selected on the first attention call. Inspect that
# first announcement while the one and only measurement session is still
# running, so a fallback cannot consume an entire 120-chunk release run.
backend_selected=0
for _ in $(seq 1 1200); do
  if grep -Fq \
    "MinWM packed-varlen attention backend=${EXPECTED_PACKED_BACKEND} device=cuda:0" \
    "${SERVER_LOG}"; then
    backend_selected=1
    break
  fi
  if grep -Fq 'MinWM packed-varlen attention backend=' "${SERVER_LOG}"; then
    printf 'packed attention selected an unexpected backend\n' >&2
    exit 1
  fi
  if ! kill -0 "${client_pid}" 2>/dev/null; then
    printf 'client exited before the packed backend was announced\n' >&2
    wait "${client_pid}" || true
    client_pid=""
    exit 1
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    printf 'server exited before the packed backend was announced\n' >&2
    exit 1
  fi
  sleep 1
done
if [[ "${backend_selected}" != "1" ]]; then
  printf 'packed backend was not announced before timeout\n' >&2
  exit 1
fi

wait "${client_pid}"
client_pid=""

stop_monitor
stop_server

export SERVER_LOG THROUGHPUT GPU_LOG RESULTS
python3 - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


throughput_path = Path(os.environ["THROUGHPUT"])
server_log_path = Path(os.environ["SERVER_LOG"])
gpu_log_path = Path(os.environ["GPU_LOG"])
throughput = json.loads(throughput_path.read_text())
server_text = server_log_path.read_text(errors="replace")

warmup = int(os.environ["WARMUP_CHUNKS"])
measured_count = int(os.environ["MEASURED_CHUNKS"])
measured_indices = set(range(warmup, warmup + measured_count))
if throughput.get("profile_name") != "packed-fast-bf16-release-gate":
    raise RuntimeError("throughput profile name drifted")
if int(throughput.get("warmup_chunks", -1)) != warmup:
    raise RuntimeError("throughput warmup count drifted")
if int(throughput.get("measured_chunks", -1)) != measured_count:
    raise RuntimeError("throughput measured count drifted")
if int(throughput.get("measured_frames", -1)) != measured_count * 16:
    raise RuntimeError("throughput measured frame count drifted")
contract = throughput.get("comparison_contract", {})
expected_contract = {
    "case": "00_forward_080_pottery_720p",
    "action_type": "primitive_token_residual",
    "action_label": 9,
    "seed": 42,
    "size": "1248x704",
    "steps": 4,
    "guidance_scale": 0.0,
    "latent_frames_per_chunk": 4,
    "generated_pixel_frames_per_steady_chunk": 16,
    "kv_cache_num_frames": 45,
}
for key, expected in expected_contract.items():
    if contract.get(key) != expected:
        raise RuntimeError(
            f"request contract {key}: expected {expected!r}, got {contract.get(key)!r}"
        )

profile_evidence = (
    "MinWM execution profile: attention_impl=packed "
    "packed_deterministic=False segment_compile=True cache_rotated_k=True "
    "precompute_cache_rope=True cache_packed_metadata=True nvtx_block_phases=False"
)
backend_evidence = (
    "MinWM packed-varlen attention backend="
    f"{os.environ['EXPECTED_PACKED_BACKEND']} device=cuda:0"
)
for evidence in (
    "Applying performance_mode=speed",
    "Attention backends for transformer: fa",
    profile_evidence,
    backend_evidence,
):
    if evidence not in server_text:
        raise RuntimeError(f"missing server contract evidence: {evidence}")

scheduler_ms_by_chunk = {}
for line in server_text.splitlines():
    marker = "realtime_trace "
    offset = line.find(marker)
    if offset < 0:
        continue
    try:
        event = json.loads(line[offset + len(marker) :])
    except json.JSONDecodeError:
        continue
    if event.get("event") != "server.chunk_complete":
        continue
    chunk_index = event.get("chunk_index")
    if chunk_index not in measured_indices:
        continue
    if chunk_index in scheduler_ms_by_chunk:
        raise RuntimeError(f"duplicate scheduler event for measured chunk {chunk_index}")
    scheduler_ms = event.get("scheduler_forward_ms")
    if scheduler_ms is None:
        raise RuntimeError(f"measured chunk {chunk_index} lacks scheduler_forward_ms")
    scheduler_ms_by_chunk[int(chunk_index)] = float(scheduler_ms)
if set(scheduler_ms_by_chunk) != measured_indices:
    missing = sorted(measured_indices - set(scheduler_ms_by_chunk))
    raise RuntimeError(
        f"expected {measured_count} scheduler events, got {len(scheduler_ms_by_chunk)}; "
        f"missing={missing[:12]}"
    )

measured_frames = int(throughput["measured_frames"])
scheduler_ms_sum = sum(scheduler_ms_by_chunk.values())
scheduler_fps = measured_frames / (scheduler_ms_sum / 1000.0)
client_fps = float(throughput["client"]["steady_received_fps_ratio_of_sums"])
minimum_scheduler_fps = float(os.environ["MIN_SCHEDULER_FPS"])
minimum_client_fps = float(os.environ["MIN_CLIENT_FPS"])
if not math.isfinite(scheduler_fps) or scheduler_fps <= 0:
    raise RuntimeError(f"invalid scheduler FPS: {scheduler_fps!r}")
if not math.isfinite(client_fps) or client_fps <= 0:
    raise RuntimeError(f"invalid client FPS: {client_fps!r}")
if scheduler_fps < minimum_scheduler_fps:
    raise RuntimeError(
        f"scheduler FPS regression: {scheduler_fps:.9f} < {minimum_scheduler_fps:.9f}"
    )
if client_fps < minimum_client_fps:
    raise RuntimeError(
        f"client FPS regression: {client_fps:.9f} < {minimum_client_fps:.9f}"
    )
relative_gap = abs(client_fps / scheduler_fps - 1.0)
if relative_gap > 0.01:
    raise RuntimeError(
        f"client/scheduler FPS diverged by {relative_gap:.3%}, exceeding 1%"
    )

memory_values = []
for line in gpu_log_path.read_text(errors="replace").splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 3:
        continue
    try:
        memory_values.append(float(fields[2]))
    except ValueError:
        pass

checkpoint_sha = Path(os.environ["RESULTS"], "checkpoint-sha256.txt").read_text().split()[0]
first_frame_sha = Path(os.environ["RESULTS"], "first-frame-sha256.txt").read_text().split()[0]
local_input_contract = json.loads(
    Path(os.environ["RESULTS"], "local-input-contract.json").read_text()
)
runtime_contract = json.loads(Path(os.environ["RUNTIME_JSON"]).read_text())
result = {
    "schema_version": "minwm-720p-image-performance-gate-result/v1",
    "status": "pass",
    "source_commit": os.environ["EXPECTED_SOURCE_COMMIT"],
    "requested_image_digest": os.environ["EXPECTED_IMAGE_DIGEST"],
    "gpu_family": os.environ["MINWM_GPU_FAMILY"],
    "gpu": os.environ["EXPECTED_GPU_NAME"],
    "packed_backend": os.environ["EXPECTED_PACKED_BACKEND"],
    "lane": "packed-fast-bf16",
    "scheduler": {
        "fps": scheduler_fps,
        "baseline_fps": float(os.environ["BASELINE_SCHEDULER_FPS"]),
        "minimum_fps": minimum_scheduler_fps,
        "change_vs_baseline_percent": 100.0
        * (scheduler_fps / float(os.environ["BASELINE_SCHEDULER_FPS"]) - 1.0),
        "measured_chunks": len(scheduler_ms_by_chunk),
        "scheduler_forward_ms_sum": scheduler_ms_sum,
        "scheduler_forward_ms_by_chunk": {
            str(index): scheduler_ms_by_chunk[index]
            for index in sorted(scheduler_ms_by_chunk)
        },
    },
    "client": {
        "fps": client_fps,
        "baseline_fps": float(os.environ["BASELINE_CLIENT_FPS"]),
        "minimum_fps": minimum_client_fps,
        "change_vs_baseline_percent": 100.0
        * (client_fps / float(os.environ["BASELINE_CLIENT_FPS"]) - 1.0),
        "steady_window_seconds": throughput["client"]["steady_window_seconds"],
        "steady_payload_interarrival_ms": throughput["client"][
            "steady_payload_interarrival_ms"
        ],
        "relative_gap_vs_scheduler": relative_gap,
    },
    "peak_gpu_memory_mib": max(memory_values) if memory_values else None,
    "checkpoint_sha256": checkpoint_sha,
    "first_frame_sha256": first_frame_sha,
    "input_contract": local_input_contract,
    "runtime_contract": runtime_contract,
    "measured_payload_sha256": throughput["measured_payload_sha256"],
    "throughput_json_sha256": sha256(throughput_path),
    "server_log_sha256": sha256(server_log_path),
    "backend_evidence": backend_evidence,
    "execution_profile_evidence": profile_evidence,
}
evidence_path = Path(os.environ["RESULTS"], "gate-evidence.json")
evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("MINWM_720P_PERFORMANCE_GATE_PASS")
print(json.dumps(result, indent=2, sort_keys=True))
PY

printf '%s\n' '--- selected server contract evidence ---'
grep -E \
  'Applying performance_mode=speed|Attention backends for transformer|MinWM execution profile|MinWM packed-varlen attention backend' \
  "${SERVER_LOG}"
printf '%s\n' '--- client timing summary ---'
cat "${CLIENT_LOG}"
