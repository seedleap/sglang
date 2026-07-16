#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:-/results/capacity-smoke-720p}
gpu_total=8
gpus_per_server=2
server_count=$((gpu_total / gpus_per_server))
model_id=${MODEL_ID:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}
model_revision=${MODEL_REVISION:-59cccf49f2d2dd27418ae7a04b82b10868d455c2}
messages_path=${MESSAGES_PATH:-/opt/bench/smoke-messages.jsonl.gz}
image_urls_path=${IMAGE_URLS_PATH:-/opt/bench/smoke-image-urls.json}
put_urls_path=${PUT_URLS_PATH:-}
server_cache_root=${SERVER_CACHE_ROOT:-${results_root}/server-cache}
resume=${RESUME:-false}
stream_upload=${STREAM_UPLOAD:-false}
upload_workers=${UPLOAD_WORKERS:-16}

mkdir -p "${results_root}"
server_pids=()

stop_servers() {
  local pid
  for pid in "${server_pids[@]:-}"; do
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
  done
  sleep 15
  for pid in "${server_pids[@]:-}"; do
    kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  done
  server_pids=()
}
trap stop_servers EXIT INT TERM

wait_for_server() {
  local port=$1
  local pid=$2
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 5
  done
  return 1
}

nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,driver_version --format=csv \
  > "${results_root}/gpu-inventory.csv"
python3 -m pip show flash-attn-4 kernels sglang-kernel torch \
  > "${results_root}/runtime-packages.txt"

urls=()
ports=()
for ((server_index = 0; server_index < server_count; server_index++)); do
  first_gpu=$((server_index * gpus_per_server))
  last_gpu=$((first_gpu + gpus_per_server - 1))
  devices=$(seq -s, "${first_gpu}" "${last_gpu}")
  port=$((30000 + server_index * 100))
  ports+=("${port}")
  urls+=("ws://127.0.0.1:${port}/v1/realtime_video/generate")
  cache_dir="${server_cache_root}/${server_index}"
  mkdir -p "${cache_dir}"

  CUDA_VISIBLE_DEVICES="${devices}" \
  NCCL_PROTO=Simple \
  PYTHONUNBUFFERED=1 \
  SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES=60 \
  SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW=true \
  SGLANG_DIFFUSION_CACHE_ROOT="${cache_dir}" \
  SGLANG_DIFFUSION_STAGE_LOGGING=true \
  setsid sglang serve \
    --model-path "${model_id}" \
    --revision "${model_revision}" \
    --pipeline-class-name LingBotWorldCausalDMDPipeline \
    --num-gpus "${gpus_per_server}" \
    --performance-mode speed \
    --tp-size 1 \
    --sp-degree "${gpus_per_server}" \
    --ulysses-degree "${gpus_per_server}" \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload false \
    --vae-config.use-parallel-decode true \
    --vae-config.parallel-decode-mode spatial \
    --enable-torch-compile false \
    --enable-layerwise-nvtx-marker false \
    --attention-backend-config lingbot_causal_fa_num_splits=0 \
    --master-port "$((port + 5))" \
    --host 127.0.0.1 \
    --port "${port}" \
    > "${results_root}/server-${server_index}.log" 2>&1 &
  server_pids+=("$!")
done

for ((server_index = 0; server_index < server_count; server_index++)); do
  if ! wait_for_server "${ports[$server_index]}" "${server_pids[$server_index]}"; then
    tail -n 300 "${results_root}/server-${server_index}.log" >&2 || true
    exit 1
  fi
done

nvidia-smi dmon -s pucvmet -d 1 -o DT > "${results_root}/nvidia-dmon.log" 2>&1 &
dmon_pid=$!
urls_csv=$(IFS=,; echo "${urls[*]}")

set +e
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi
stream_upload_pid=""
stream_upload_done="${results_root}/stream-upload-generation-finished"
if [[ "${stream_upload}" == "true" && -n "${put_urls_path}" ]]; then
  rm -f "${stream_upload_done}"
  python3 /opt/bench/upload_progress.py \
    --progress "${results_root}/cases/progress.jsonl" \
    --put-urls "${put_urls_path}" \
    --done-file "${stream_upload_done}" \
    --output "${results_root}/upload-summary.json" \
    --workers "${upload_workers}" \
    > "${results_root}/upload.log" 2>&1 &
  stream_upload_pid=$!
fi
python3 /opt/bench/benchmark_evalset.py \
  --messages "${messages_path}" \
  --image-urls "${image_urls_path}" \
  --urls "${urls_csv}" \
  --gpu-count "${gpu_total}" \
  --width 1280 \
  --height 720 \
  --fps 24 \
  --warmup-chunks 3 \
  --output-dir "${results_root}/cases" \
  --output "${results_root}/cases/summary.json" \
  "${resume_args[@]}" \
  2>&1 | tee "${results_root}/benchmark.log"
benchmark_status=${PIPESTATUS[0]}
set -e

upload_status=0
if [[ -n "${stream_upload_pid}" ]]; then
  touch "${stream_upload_done}"
  set +e
  wait "${stream_upload_pid}"
  upload_status=$?
  set -e
  tail -n 80 "${results_root}/upload.log" || true
elif [[ "${benchmark_status}" -eq 0 && -n "${put_urls_path}" ]]; then
  set +e
  python3 /opt/bench/upload_outputs.py \
    --summary "${results_root}/cases/summary.json" \
    --put-urls "${put_urls_path}" \
    --output "${results_root}/upload-summary.json" \
    2>&1 | tee "${results_root}/upload.log"
  upload_status=${PIPESTATUS[0]}
  set -e
fi

kill -TERM "${dmon_pid}" >/dev/null 2>&1 || true
wait "${dmon_pid}" >/dev/null 2>&1 || true
echo "${benchmark_status}" > "${results_root}/exit-code"
echo "${upload_status}" > "${results_root}/upload-exit-code"
touch "${results_root}/finished"
if [[ "${benchmark_status}" -ne 0 ]]; then
  exit "${benchmark_status}"
fi
exit "${upload_status}"
