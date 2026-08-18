#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${MINWM_MODEL_DIR:?setup entrypoint must export MINWM_MODEL_DIR}"
: "${MINWM_CHECKPOINT:?setup entrypoint must export MINWM_CHECKPOINT}"
: "${MINWM_PRETRAINED_DIR:?setup entrypoint must export MINWM_PRETRAINED_DIR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARITY_DIR="$(cd -- "${SCRIPT_DIR}/../minwm_realtime_parity" && pwd)"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/attn-ffn-720p"
LOCAL_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}/attn-ffn-720p"
CASES="${MINWM_CASES_PATH:-${PARITY_DIR}/cases_720p_compile_smoke.json}"
CASE="${MINWM_CASE:-00_forward_080_pottery_720p}"
WARMUP_CHUNKS="${MINWM_MATRIX_WARMUP_CHUNKS:-20}"
MEASURED_CHUNKS="${MINWM_MATRIX_MEASURED_CHUNKS:-100}"
KV_FRAMES="${MINWM_MATRIX_KV_FRAMES:-45}"
GPU_FAMILY="${MINWM_GPU_FAMILY:?set MINWM_GPU_FAMILY to hopper or blackwell}"
STATIC_TRANSFORMER="${LOCAL_ROOT}/static-fp8-transformer"
CALIBRATION="${LOCAL_ROOT}/static-fp8-calibration.json"
NSYS_URL="${MINWM_NSYS_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb}"
SAGE3_SHA="${MINWM_SAGE3_SHA:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"

[[ "${GPU_FAMILY}" =~ ^(hopper|blackwell)$ ]]
[[ -f "${CASES}" ]]
[[ -f "${MINWM_MODEL_DIR}/minwm_conversion_manifest.json" ]]
mkdir -p "${RESULT_ROOT}/lanes" "${RESULT_ROOT}/profiles" "${LOCAL_ROOT}"

python3 - "${RESULT_ROOT}/contract.json" <<'PY'
import importlib.util
import json
import os
import platform
import subprocess
import sys

import torch

gpu_family = os.environ["MINWM_GPU_FAMILY"]
compute_capability = tuple(torch.cuda.get_device_capability(0))
if gpu_family == "hopper" and compute_capability[0] != 9:
    raise RuntimeError(
        f"MINWM_GPU_FAMILY=hopper requires SM9x, got {compute_capability}"
    )
if gpu_family == "blackwell" and compute_capability[0] < 10:
    raise RuntimeError(
        f"MINWM_GPU_FAMILY=blackwell requires SM10x+, got {compute_capability}"
    )

payload = {
    "schema_version": "minwm-720p-attn-ffn-contract/v1",
    "run_id": os.environ["MINWM_RUN_ID"],
    "gpu_family": gpu_family,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(compute_capability),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "platform": platform.platform(),
    "sglang_git_sha": subprocess.check_output(
        ["git", "-C", "/workspace/sglang", "rev-parse", "HEAD"], text=True
    ).strip(),
    "minwm_git_sha": subprocess.check_output(
        ["git", "-C", "/workspace/minWM", "rev-parse", "HEAD"], text=True
    ).strip(),
    "image_digest": os.environ.get("MINWM_IMAGE_DIGEST"),
    "checkpoint_uri": os.environ.get("MINWM_CHECKPOINT_S3_EAST"),
    "checkpoint_version_id": os.environ.get("MINWM_CHECKPOINT_VERSION_EAST"),
    "checkpoint_bytes": int(os.environ["MINWM_CHECKPOINT_BYTES"]),
    "checkpoint_crc64": os.environ.get("MINWM_CHECKPOINT_CRC64"),
    "request": {
        "size": "1248x704",
        "case": os.environ.get("MINWM_CASE", "00_forward_080_pottery_720p"),
        "kv_cache_num_frames": int(os.environ.get("MINWM_MATRIX_KV_FRAMES", "45")),
        "warmup_chunks": int(os.environ.get("MINWM_MATRIX_WARMUP_CHUNKS", "20")),
        "measured_chunks": int(os.environ.get("MINWM_MATRIX_MEASURED_CHUNKS", "100")),
        "sp_degree": 1,
        "whole_dit_compile": False,
    },
    "comparison": {
        "performance_reference_lane": "packed-fast-bf16",
        "quality_reference_lane": "packed-det-bf16",
        "speed_execution_profile": {
            "performance_mode": "speed",
            "attention_impl": "packed",
            "packed_attention_deterministic": False,
            "native_components": "",
            "segment_compile": True,
            "whole_dit_compile": False,
            "cuda_graph": False,
            "cache_rotated_k": True,
            "precompute_cache_rope": True,
            "cache_packed_metadata": True,
            "quantization": None,
        },
    },
    "packages_before_optional_install": {
        name: importlib.util.find_spec(name) is not None
        for name in ("flash_attn", "flash_attn_interface", "sageattention", "sageattn3")
    },
}
with open(sys.argv[1], "w") as output:
    json.dump(payload, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi.txt"
nvidia-smi topo -m > "${RESULT_ROOT}/topology.txt"
cp "${MINWM_MODEL_DIR}/minwm_conversion_manifest.json" "${RESULT_ROOT}/"
cp "${MINWM_MODEL_DIR}/transformer/config.json" "${RESULT_ROOT}/transformer-config.json"

install_sage_backend() {
  if [[ "${MINWM_INSTALL_SAGE:-1}" != "1" ]]; then
    return
  fi
  if [[ "${GPU_FAMILY}" == "hopper" ]]; then
    if python3 -c 'import sageattention' >/dev/null 2>&1; then
      return
    fi
    set +e
    MAX_JOBS="${MAX_JOBS:-16}" python3 -m pip install \
      'sageattention==2.2.0' --no-build-isolation \
      > "${RESULT_ROOT}/sage-install.log" 2>&1
    local status=$?
    set -e
    echo "${status}" > "${RESULT_ROOT}/sage-install.status"
    return
  fi
  if python3 -c 'import sageattn3' >/dev/null 2>&1; then
    return
  fi
  local source="${LOCAL_ROOT}/SageAttention"
  set +e
  if [[ ! -d "${source}/.git" ]]; then
    python3 - "${source}" <<'PY'
import pathlib, shutil, sys
target = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path("/work/minwm-realtime").resolve()
if root not in target.parents:
    raise SystemExit(f"refusing to replace non-work path: {target}")
if target.exists():
    shutil.rmtree(target)
PY
    git clone --filter=blob:none https://github.com/thu-ml/SageAttention.git "${source}" \
      > "${RESULT_ROOT}/sage-install.log" 2>&1
  fi
  git -C "${source}" fetch origin "${SAGE3_SHA}" \
      >> "${RESULT_ROOT}/sage-install.log" 2>&1 \
    && git -C "${source}" checkout --detach "${SAGE3_SHA}" \
      >> "${RESULT_ROOT}/sage-install.log" 2>&1 \
    && MAX_JOBS="${MAX_JOBS:-16}" python3 -m pip install \
      "${source}/sageattention3_blackwell" --no-build-isolation \
      >> "${RESULT_ROOT}/sage-install.log" 2>&1
  local status=$?
  set -e
  echo "${status}" > "${RESULT_ROOT}/sage-install.status"
}

prepare_static_fp8() {
  if [[ -f "${STATIC_TRANSFORMER}/minwm_static_fp8_manifest.json" ]]; then
    return
  fi
  local calibration_results="${LOCAL_ROOT}/calibration-baseline"
  mkdir -p "${calibration_results}"
  python3 -m pytest -q \
    /workspace/sglang/python/sglang/multimodal_gen/test/unit/test_minwm_static_fp8_transformer.py \
    | tee "${RESULT_ROOT}/static-fp8-routing-test.log"
  MINWM_PARITY_DETERMINISTIC=1 \
  MINWM_DETERMINISTIC_ATTENTION=true \
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
    python3 "${PARITY_DIR}/run_minwm_baseline.py" \
      --cases "${CASES}" \
      --case "${CASE}" \
      --minwm-root /workspace/minWM \
      --checkpoint "${MINWM_CHECKPOINT}" \
      --pretrained-dir "${MINWM_PRETRAINED_DIR}" \
      --config "${MINWM_CONFIG}" \
      --results "${calibration_results}" \
      --local-attn-size -1 \
      --sink-size 0 \
      --fp8-calibration-output "${CALIBRATION}" \
      | tee "${RESULT_ROOT}/static-fp8-calibration.log"
  python3 -m sglang.multimodal_gen.tools.build_minwm_static_fp8_transformer \
    --input-dir "${MINWM_MODEL_DIR}/transformer" \
    --calibration "${CALIBRATION}" \
    --output-dir "${STATIC_TRANSFORMER}" \
    --activation-margin "${MINWM_STATIC_FP8_MARGIN:-1.05}" \
    --module-scope ffn \
    --overwrite \
    | tee "${RESULT_ROOT}/static-fp8-build.log"
  cp "${CALIBRATION}" "${RESULT_ROOT}/"
  cp "${STATIC_TRANSFORMER}/minwm_static_fp8_manifest.json" "${RESULT_ROOT}/"
}

server_pid=""
monitor_pid=""
wait_for_server_down() {
  for _ in $(seq 1 120); do
    if ! curl --fail --silent http://127.0.0.1:30000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

stop_server() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  pkill -TERM -f "sglang serve --model-path ${MINWM_MODEL_DIR}.*--port 30000" \
    2>/dev/null || true
  if ! wait_for_server_down; then
    [[ -n "${server_pid}" ]] && kill -KILL "${server_pid}" 2>/dev/null || true
    pkill -KILL -f "sglang serve --model-path ${MINWM_MODEL_DIR}.*--port 30000" \
      2>/dev/null || true
    wait_for_server_down || true
  fi
  if [[ -n "${server_pid}" ]]; then
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 600); do
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

write_status() {
  local path="$1" state="$2" code="$3"
  python3 - "${path}" "${state}" "${code}" <<'PY'
import json, sys, time
with open(sys.argv[1], "w") as output:
    json.dump({"status": sys.argv[2], "exit_code": int(sys.argv[3]), "timestamp": time.time()}, output, indent=2, sort_keys=True)
    output.write("\n")
PY
}

backend_available() {
  local backend="$1"
  case "${backend}" in
    sage_attn)
      python3 -c 'from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import SageAttentionBackend; from sageattention import sageattn' \
        >/dev/null 2>&1
      ;;
    sage_attn_3)
      python3 -c 'from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3 import SageAttention3Backend; from sageattn3 import sageattn3_blackwell' \
        >/dev/null 2>&1
      ;;
    *) return 0 ;;
  esac
}

requested_backend_selected() {
  local backend="$1" log_path="$2"
  case "${backend}" in
    sage_attn)
      grep -Eq \
        "Using sage_attn attention backend|Attention backends for .*: .*sage_attn" \
        "${log_path}"
      ;;
    sage_attn_3)
      grep -Eq \
        "Using sage_attn_3 attention backend|Attention backends for .*: .*sage_attn_3" \
        "${log_path}"
      ;;
    *) return 0 ;;
  esac
}

run_lane() {
  local name="$1" impl="$2" deterministic="$3" backend="$4"
  local quantization="$5" transformer_path="$6" compile="$7"
  local lane_dir="${RESULT_ROOT}/lanes/${name}"
  mkdir -p "${lane_dir}"
  if [[ -f "${lane_dir}/throughput.json" ]]; then
    echo "MINWM_MATRIX_RESUME_SKIP lane=${name}"
    return 0
  fi
  python3 - "${lane_dir}/lane.json" "${name}" "${impl}" "${deterministic}" \
    "${backend}" "${quantization}" "${transformer_path}" "${compile}" <<'PY'
import json, sys
keys = ("name", "attention_impl", "deterministic", "attention_backend", "quantization", "transformer_path", "whole_dit_compile")
values = sys.argv[2:]
payload = dict(zip(keys, values))
payload["deterministic"] = payload["deterministic"] == "true"
payload["whole_dit_compile"] = payload["whole_dit_compile"] == "true"
with open(sys.argv[1], "w") as output:
    json.dump(payload, output, indent=2, sort_keys=True)
    output.write("\n")
PY
  if ! backend_available "${backend}"; then
    echo "requested backend ${backend} is unavailable" > "${lane_dir}/skip-reason.txt"
    write_status "${lane_dir}/status.json" skipped 0
    return 0
  fi
  if [[ -n "${transformer_path}" && ! -f "${transformer_path}/minwm_static_fp8_manifest.json" ]]; then
    echo "static transformer is unavailable" > "${lane_dir}/skip-reason.txt"
    write_status "${lane_dir}/status.json" skipped 0
    return 0
  fi

  local deterministic_env=0
  [[ "${deterministic}" == "true" ]] && deterministic_env=1
  local quantization_args=()
  local transformer_args=()
  [[ -n "${quantization}" ]] && quantization_args=(--quantization "${quantization}")
  [[ -n "${transformer_path}" ]] && transformer_args=(--transformer-path "${transformer_path}")
  if ! wait_for_server_down; then
    stop_server
  fi
  wait_for_server_down || {
    echo "port 30000 is still occupied before lane ${name}" >&2
    return 1
  }
  echo "MINWM_MATRIX_LANE_START lane=${name} timestamp=$(date -Iseconds)"
  write_status "${lane_dir}/status.json" running 0
  MINWM_ATTENTION_IMPL="${impl}" \
  MINWM_PACKED_ATTENTION_DETERMINISTIC="${deterministic}" \
  MINWM_PARITY_DETERMINISTIC="${deterministic_env}" \
  MINWM_DETERMINISTIC_ATTENTION="${deterministic}" \
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${deterministic_env}" \
  MINWM_NATIVE_COMPONENTS= \
  MINWM_SEGMENT_COMPILE=true \
  MINWM_CACHE_ROTATED_K=true \
  MINWM_PRECOMPUTE_CACHE_ROPE=true \
  MINWM_CACHE_PACKED_METADATA=true \
  MINWM_NVTX_BLOCK_PHASES=0 \
    sglang serve \
      --model-path "${MINWM_MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --attention-backend "${backend}" \
      --performance-mode speed \
      --num-gpus 1 \
      --sp-degree 1 \
      --enable-cfg-parallel false \
      "${quantization_args[@]}" \
      "${transformer_args[@]}" \
      --enable-torch-compile "${compile}" \
      --enable-cuda-graph false \
      --warmup-mode off \
      --realtime-session-idle-timeout-s 1800 \
      --port 30000 \
      > "${lane_dir}/server.log" 2>&1 &
  server_pid=$!
  (
    while kill -0 "${server_pid}" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${lane_dir}/gpu.csv" &
  monitor_pid=$!
  local status=0
  if wait_for_server "${lane_dir}/server.log"; then
    if ! requested_backend_selected "${backend}" "${lane_dir}/server.log"; then
      echo "requested backend ${backend} was not selected; refusing fallback result" \
        > "${lane_dir}/skip-reason.txt"
      stop_server
      write_status "${lane_dir}/status.json" skipped 0
      return 0
    fi
    set +e
    python3 "${PARITY_DIR}/benchmark_realtime_throughput.py" \
      --cases "${CASES}" \
      --case "${CASE}" \
      --output "${lane_dir}/throughput.json" \
      --profile-name "${name}" \
      --warmup-chunks "${WARMUP_CHUNKS}" \
      --measured-chunks "${MEASURED_CHUNKS}" \
      --kv-cache-num-frames "${KV_FRAMES}" \
      --save-first-measured-frame \
      > "${lane_dir}/client.log" 2>&1
    status=$?
    set -e
  else
    status=1
  fi
  stop_server
  if (( status == 0 )); then
    write_status "${lane_dir}/status.json" success 0
  else
    write_status "${lane_dir}/status.json" failed "${status}"
    tail -300 "${lane_dir}/server.log" >&2
  fi
  echo "MINWM_MATRIX_LANE_END lane=${name} status=${status} timestamp=$(date -Iseconds)"
  return 0
}

install_sage_backend

run_lane packed-det-bf16 packed true fa "" "" false
run_lane packed-fast-bf16 packed false fa "" "" false
run_lane dense-fa-bf16 dense false fa "" "" false
run_lane dense-sdpa-bf16 dense false torch_sdpa "" "" false
if [[ "${GPU_FAMILY}" == "hopper" ]]; then
  run_lane dense-cross-sage2-bf16 dense false sage_attn "" "" false
else
  run_lane dense-cross-sage3-bf16 dense false sage_attn_3 "" "" false
fi
run_lane dense-fa-online-fp8 dense false fa fp8 "" false
if [[ "${MINWM_RUN_COMPILE_LANES:-0}" == "1" ]]; then
  run_lane dense-fa-bf16-compile dense false fa "" "" true
fi
prepare_static_fp8
run_lane packed-fast-static-ffn-fp8 packed false fa "" "${STATIC_TRANSFORMER}" false
if [[ "${MINWM_RUN_COMPILE_LANES:-0}" == "1" ]]; then
  run_lane packed-fast-static-ffn-fp8-compile packed false fa "" "${STATIC_TRANSFORMER}" true
fi
run_lane dense-fa-static-ffn-fp8 dense false fa "" "${STATIC_TRANSFORMER}" false
if [[ "${MINWM_RUN_COMPILE_LANES:-0}" == "1" ]]; then
  run_lane dense-fa-static-ffn-fp8-compile dense false fa "" "${STATIC_TRANSFORMER}" true
fi

python3 "${SCRIPT_DIR}/summarize_matrix.py" "${RESULT_ROOT}" \
  --performance-reference packed-fast-bf16 \
  --quality-reference packed-det-bf16 \
  | tee "${RESULT_ROOT}/summary.log"

if [[ "${MINWM_RUN_NSYS:-1}" != "1" ]]; then
  echo "MINWM_ATTN_FFN_MATRIX_COMPLETE results=${RESULT_ROOT}"
  exit 0
fi

install_nsys() {
  if command -v nsys >/dev/null; then
    return
  fi
  local root="${LOCAL_ROOT}/nsight-systems" deb="${LOCAL_ROOT}/nsight-systems.deb"
  mkdir -p "${root}"
  curl --fail --location --retry 3 --output "${deb}" "${NSYS_URL}"
  sha256sum "${deb}" | tee "${RESULT_ROOT}/nsys-package-sha256.txt"
  dpkg-deb --extract "${deb}" "${root}"
  local binary
  binary="$(find "${root}" -type f -name nsys -perm -111 -print -quit)"
  [[ -n "${binary}" ]]
  export PATH="$(dirname "${binary}"):${PATH}"
}

run_nsys_profile() {
  local name="$1"
  local source_lane="${RESULT_ROOT}/lanes/${name}"
  local profile_dir="${RESULT_ROOT}/profiles/${name}"
  [[ -f "${source_lane}/throughput.json" ]] || return 0
  [[ -f "${profile_dir}/phase-summary.json" ]] && return 0
  mkdir -p "${profile_dir}"
  rm -f \
    "${profile_dir}/trace.nsys-rep" \
    "${profile_dir}/trace.qdstrm" \
    "${profile_dir}/trace.sqlite"
  local lane_config=()
  mapfile -t lane_config < <(
    python3 - "${source_lane}/lane.json" <<'PY'
import json, sys
lane = json.load(open(sys.argv[1]))
for key in (
    "attention_impl",
    "deterministic",
    "attention_backend",
    "quantization",
    "transformer_path",
    "whole_dit_compile",
):
    value = lane.get(key, "")
    if isinstance(value, bool):
        value = str(value).lower()
    print(value)
PY
  )
  local impl="${lane_config[0]}"
  local deterministic="${lane_config[1]}"
  local backend="${lane_config[2]}"
  local quantization="${lane_config[3]}"
  local transformer_path="${lane_config[4]}"
  local compile="${lane_config[5]}"
  if [[ "${compile}" == "true" ]]; then
    echo "whole-DiT compiled lanes are intentionally excluded from phase profiling" \
      > "${profile_dir}/skip-reason.txt"
    return 0
  fi
  local session="minwm-${GPU_FAMILY}-$RANDOM"
  local deterministic_env=0
  [[ "${deterministic}" == "true" ]] && deterministic_env=1
  local quantization_args=()
  local transformer_args=()
  [[ -n "${quantization}" ]] && quantization_args=(--quantization "${quantization}")
  [[ -n "${transformer_path}" ]] && transformer_args=(--transformer-path "${transformer_path}")
  local server_log="${profile_dir}/server.log"
  if ! wait_for_server_down; then
    stop_server
  fi
  wait_for_server_down || {
    echo "port 30000 is still occupied before profile ${name}" >&2
    return 1
  }
  MINWM_ATTENTION_IMPL="${impl}" \
  MINWM_PACKED_ATTENTION_DETERMINISTIC="${deterministic}" \
  MINWM_PARITY_DETERMINISTIC="${deterministic_env}" \
  MINWM_DETERMINISTIC_ATTENTION="${deterministic}" \
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${deterministic_env}" \
  MINWM_NATIVE_COMPONENTS= \
  MINWM_SEGMENT_COMPILE=true \
  MINWM_CACHE_ROTATED_K=true \
  MINWM_PRECOMPUTE_CACHE_ROPE=true \
  MINWM_CACHE_PACKED_METADATA=true \
  MINWM_NVTX_BLOCK_PHASES=1 \
    nsys launch \
      --session-new="${session}" \
      --trace=cuda,nvtx \
      --trace-fork-before-exec=true \
      --cuda-graph-trace=node \
      -- \
      sglang serve \
        --model-path "${MINWM_MODEL_DIR}" \
        --pipeline-class-name MinWMCausalDMDPipeline \
        --attention-backend "${backend}" \
        --performance-mode speed \
        --num-gpus 1 \
        --sp-degree 1 \
        --enable-cfg-parallel false \
        "${quantization_args[@]}" \
        "${transformer_args[@]}" \
        --enable-torch-compile false \
        --enable-cuda-graph false \
        --warmup-mode off \
        --realtime-session-idle-timeout-s 1800 \
        --port 30000 \
        > "${server_log}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_log}"
  if ! requested_backend_selected "${backend}" "${server_log}"; then
    stop_server
    echo "requested backend ${backend} was not selected; refusing fallback trace" \
      > "${profile_dir}/failed.txt"
    return 0
  fi

  python3 "${PARITY_DIR}/benchmark_realtime_throughput.py" \
    --cases "${CASES}" --case "${CASE}" \
    --output "${profile_dir}/throwaway.json" \
    --profile-name "${name}-throwaway" \
    --warmup-chunks 5 --measured-chunks 5 \
    --kv-cache-num-frames "${KV_FRAMES}" \
    > "${profile_dir}/throwaway-client.log" 2>&1

  local progress="${LOCAL_ROOT}/${name}-progress.json"
  rm -f "${progress}"
  python3 "${PARITY_DIR}/benchmark_realtime_throughput.py" \
    --cases "${CASES}" --case "${CASE}" \
    --output "${profile_dir}/profile-client.json" \
    --profile-name "${name}-nsys" \
    --warmup-chunks 16 \
    --measured-chunks "${MINWM_PROFILE_MEASURED_CHUNKS:-10}" \
    --kv-cache-num-frames "${KV_FRAMES}" \
    --progress-file "${progress}" \
    > "${profile_dir}/profile-client.log" 2>&1 &
  local client_pid=$!
  local reached=0
  for _ in $(seq 1 1800); do
    if [[ -f "${progress}" ]] && python3 - "${progress}" <<'PY'
import json, os, sys
trigger = int(os.environ.get("MINWM_PROFILE_TRIGGER_CHUNK", "8"))
raise SystemExit(0 if json.load(open(sys.argv[1]))["last_completed_chunk"] >= trigger else 1)
PY
    then
      reached=1
      break
    fi
    kill -0 "${client_pid}" 2>/dev/null || break
    sleep 0.25
  done
  if [[ "${reached}" != "1" ]]; then
    wait "${client_pid}" || true
    stop_server
    echo "profile client did not reach steady KV window" > "${profile_dir}/failed.txt"
    return 0
  fi
  nsys start \
    --session="${session}" \
    --output="${profile_dir}/trace" \
    --gpu-metrics-devices="${MINWM_PROFILE_GPU_METRICS_DEVICES:-cuda-visible}" \
    --gpu-metrics-frequency=10000 \
    --sample=none
  local client_status=0
  wait "${client_pid}" || client_status=$?
  nsys stop --session="${session}" || true
  stop_server
  if (( client_status != 0 )); then
    echo "profile client failed with ${client_status}" > "${profile_dir}/failed.txt"
    return 0
  fi
  local report="${profile_dir}/trace.nsys-rep"
  local sqlite="${profile_dir}/trace.sqlite"
  [[ -f "${report}" ]]
  nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum "${report}" \
    > "${profile_dir}/nsys-stats.txt" || true
  nsys export --type sqlite --output "${sqlite}" --force-overwrite=true "${report}"
  python3 "${SCRIPT_DIR}/analyze_nsys_phases.py" "${sqlite}" \
    --min-dit-block-ranges 5 \
    --output "${profile_dir}/phase-summary.json" \
    | tee "${profile_dir}/phase-summary.log"
}

install_nsys
nsys --version | tee "${RESULT_ROOT}/nsys-version.txt"
nsys status -e | tee "${RESULT_ROOT}/nsys-status.txt" || true
run_nsys_profile packed-fast-bf16
run_nsys_profile dense-fa-bf16
mapfile -t profile_winners < <(
  python3 - "${RESULT_ROOT}/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
print(summary.get("best_eager_lane_raw_speed") or "")
print(summary.get("best_eager_lane_quality_screened") or "")
PY
)
profiled_lanes="|dense-fa-bf16|"
for winner_lane in "${profile_winners[@]}"; do
  if [[ -n "${winner_lane}" && "${profiled_lanes}" != *"|${winner_lane}|"* ]]; then
    run_nsys_profile "${winner_lane}"
    profiled_lanes+="${winner_lane}|"
  fi
done

echo "MINWM_ATTN_FFN_MATRIX_COMPLETE results=${RESULT_ROOT}"
