#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

phase="${H3_PHASE:?H3_PHASE is required}"
run_id="${H3_RUN_ID:?H3_RUN_ID is required}"
hardware="${H3_HARDWARE:?H3_HARDWARE is required}"
gpu_count="${H3_GPU_COUNT:?H3_GPU_COUNT is required}"
result_dir="/work/results/${run_id}"
server_log="${result_dir}/server.log"
telemetry_csv="${result_dir}/nvidia-smi.csv"
server_pid=""
telemetry_pid=""

die() {
  echo "error: $*" >&2
  exit 2
}

cleanup() {
  local status=$?
  set +e
  if [[ -n "${telemetry_pid}" ]]; then
    kill -TERM "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${server_pid}" ]]; then
    kill -TERM "${server_pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      kill -0 "${server_pid}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -KILL "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
  if (( status != 0 )) && [[ -f "${server_log}" ]]; then
    tail -n 300 "${server_log}" >&2 || true
  fi
  printf 'MINIMAX_H3_RUN_STATUS={"ok":%s,"exit_code":%d,"run_id":"%s","hardware":"%s","phase":"%s"}\n' \
    "$([[ ${status} -eq 0 ]] && echo true || echo false)" \
    "${status}" "${run_id}" "${hardware}" "${phase}"
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 143' TERM INT

case "${phase}" in
  attention-probe|e2e) ;;
  *) die "H3_PHASE must be attention-probe or e2e" ;;
esac
case "${hardware}" in
  b200|b300) ;;
  *) die "H3_HARDWARE must be b200 or b300" ;;
esac
[[ "${gpu_count}" =~ ^[1-9][0-9]*$ ]] || die "H3_GPU_COUNT must be positive"
visible_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
[[ "${visible_gpus}" == "${gpu_count}" ]] \
  || die "expected ${gpu_count} visible GPUs, found ${visible_gpus}"

mkdir -p "${result_dir}"
nvidia-smi -q > "${result_dir}/nvidia-smi-q.txt"
export CARGO_TARGET_DIR=/root/.cache/sglang-rust-target
python3 -m pip install -e "python[diffusion]" --root-user-action=ignore

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME=/root/.cache/xdg
export TORCH_HOME=/root/.cache/torch

# The pinned image predates this main checkout. Editable installation upgrades
# flashinfer-python to the version pinned by python/pyproject.toml, so align the
# separate CUDA 13 cubin/JIT wheel before importing SGLang.
python3 -m pip install \
  --force-reinstall \
  --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 \
  "flashinfer-jit-cache==0.6.15.post1+cu130" \
  --root-user-action=ignore
python3 -m pip freeze --all > "${result_dir}/python-environment.txt"

if [[ "${phase}" == "attention-probe" ]]; then
  [[ "${gpu_count}" == "1" ]] || die "attention-probe must request one GPU"
  PYTHONPATH=python python3 -m pytest -q \
    python/sglang/multimodal_gen/test/unit/test_minimax_h3_causal_attention.py \
    | tee "${result_dir}/unit-tests.txt"
  PYTHONPATH=python python3 \
    benchmark/minimax_h3_causal/attention_probe.py \
    | tee "${result_dir}/attention-probe.json"
  exit 0
fi

tp_size="${H3_TP_SIZE:?H3_TP_SIZE is required for e2e}"
ulysses_degree="${H3_ULYSSES_DEGREE:?H3_ULYSSES_DEGREE is required for e2e}"
causal_mode="${H3_CAUSAL_MODE:-flex}"
mask_cache="${H3_MASK_CACHE:-true}"
model_id="${H3_MODEL_ID:-MiniMaxAI/MiniMax-H3}"
model_revision="${H3_MODEL_REVISION:?H3_MODEL_REVISION is required for e2e}"
port="${H3_PORT:-30010}"
[[ "${gpu_count}" == "8" ]] || die "e2e matrix must request eight GPUs"
[[ $((tp_size * ulysses_degree)) -eq "${gpu_count}" ]] \
  || die "TP * Ulysses must equal H3_GPU_COUNT"
case "${causal_mode}" in
  off|flex) ;;
  *) die "H3_CAUSAL_MODE must be off or flex for full-resolution e2e" ;;
esac
case "${mask_cache}" in
  true|false) ;;
  *) die "H3_MASK_CACHE must be true or false" ;;
esac

telemetry_loop() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >> "${telemetry_csv}" || true
    sleep 1
  done
}
telemetry_loop &
telemetry_pid=$!

attention_config="minimax_h3_causal_mode=${causal_mode},minimax_h3_causal_block_frames=3,minimax_h3_causal_sink_frames=4,minimax_h3_causal_window_frames=20,minimax_h3_causal_cache_block_mask=${mask_cache}"
server_cmd=(
  sglang serve
  --model-path "${model_id}"
  --revision "${model_revision}"
  --model-variant fl2va
  --num-gpus "${gpu_count}"
  --tp-size "${tp_size}"
  --ulysses-degree "${ulysses_degree}"
  --ring-degree 1
  --performance-mode speed
  --dit-precision bf16
  --enable-torch-compile false
  --attention-backend-config "${attention_config}"
  --host 127.0.0.1
  --port "${port}"
)
printf '%q ' "${server_cmd[@]}" > "${result_dir}/server-command.txt"
printf '\n' >> "${result_dir}/server-command.txt"
"${server_cmd[@]}" > "${server_log}" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 540); do
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    die "SGLang server exited before readiness"
  fi
  if curl --fail --silent --show-error \
    "http://127.0.0.1:${port}/v1/models" \
    --output "${result_dir}/models.json"; then
    ready=true
    break
  fi
  sleep 10
done
[[ "${ready}" == "true" ]] || die "SGLang server was not ready within 90 minutes"

topology="tp${tp_size}-u${ulysses_degree}-${hardware}"
if [[ "${causal_mode}" == "off" ]]; then
  variant=noncausal
elif [[ "${mask_cache}" == "true" ]]; then
  variant=causal-flex-mask-cache
else
  variant=causal-flex-no-mask-cache
fi
mkdir -p "${result_dir}/videos"
read -r -a nfe_values <<< "${H3_NFE:-3}"
python3 benchmark/minimax_h3_causal/run_matrix.py \
  --base-url "http://127.0.0.1:${port}" \
  --model "${model_id}" \
  --model-revision "${model_revision}" \
  --topology "${topology}" \
  --variant "${variant}" \
  --nfe "${nfe_values[@]}" \
  --warmup "${H3_WARMUP:-1}" \
  --repeats "${H3_REPEATS:-3}" \
  --seconds "${H3_SECONDS:-5}" \
  --first-frame-uri \
    "file:///workspace/sglang/examples/assets/example_image.png" \
  --output "${result_dir}/matrix.jsonl" \
  --video-dir "${result_dir}/videos"

for video in "${result_dir}"/videos/*.mp4; do
  sha256sum "${video}"
  ffprobe -v error -show_streams -show_format -of json "${video}"
done
echo "MINIMAX_H3_RESULT_DIR=${result_dir}"
