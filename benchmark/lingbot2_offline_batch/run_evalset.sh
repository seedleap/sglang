#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:-/results/testset100-v2-480p-16fps}
gpu_total=${GPU_TOTAL:-8}
gpus_per_server=${GPUS_PER_SERVER:-2}
server_count=$((gpu_total / gpus_per_server))
model_id=${MODEL_ID:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}
model_revision=${MODEL_REVISION:-59cccf49f2d2dd27418ae7a04b82b10868d455c2}
messages_path=${MESSAGES_PATH:-/opt/bench/messages.jsonl.gz}
image_urls_path=${IMAGE_URLS_PATH:-/opt/bench/image_urls.json}
priority_messages_path=${PRIORITY_MESSAGES_PATH:-/opt/bench/thirdperson50-messages.jsonl.gz}
priority_image_urls_path=${PRIORITY_IMAGE_URLS_PATH:-/opt/bench/thirdperson50-image-urls.json}
smoke_regex=${SMOKE_REGEX:-G1_short_w_p02$|G2_p08$|G4_p02$|G5_p08$}

if (( gpu_total % gpus_per_server != 0 )); then
  echo "GPU_TOTAL must be divisible by GPUS_PER_SERVER" >&2
  exit 2
fi

mkdir -p "${results_root}" "${results_root}/server-cache"
server_pids=()

stop_servers() {
  local pid
  if (( ${#server_pids[@]} == 0 )); then
    return 0
  fi
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

urls=()
ports=()
startup_start=$SECONDS
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv \
  > "${results_root}/gpu-inventory.csv"

for ((server_index = 0; server_index < server_count; server_index++)); do
  first_gpu=$((server_index * gpus_per_server))
  last_gpu=$((first_gpu + gpus_per_server - 1))
  devices=$(seq -s, "${first_gpu}" "${last_gpu}")
  port=$((30000 + server_index * 100))
  ports+=("${port}")
  urls+=("ws://127.0.0.1:${port}/v1/realtime_video/generate")
  cache_dir="${results_root}/server-cache/${server_index}"
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
echo "$((SECONDS - startup_start))" > "${results_root}/server-startup-seconds"

nvidia-smi dmon -s pucvmet -d 1 -o DT \
  > "${results_root}/nvidia-dmon.log" 2>&1 &
dmon_pid=$!
urls_csv=$(IFS=,; echo "${urls[*]}")

if [[ ! -f ${results_root}/priority-thirdperson50-finished ]]; then
  python3 /opt/bench/benchmark_evalset.py \
    --messages "${priority_messages_path}" \
    --image-urls "${priority_image_urls_path}" \
    --urls "${urls_csv}" \
    --gpu-count "${gpu_total}" \
    --width 832 \
    --height 480 \
    --fps 16 \
    --warmup-chunks 3 \
    --output-dir "${results_root}/priority-thirdperson50" \
    --output "${results_root}/priority-thirdperson50/summary.json" \
    2>&1 | tee "${results_root}/priority-thirdperson50.log"
  touch "${results_root}/priority-thirdperson50-finished"
fi

if [[ ! -f ${results_root}/smoke-finished ]]; then
  python3 /opt/bench/benchmark_evalset.py \
    --messages "${messages_path}" \
    --image-urls "${image_urls_path}" \
    --urls "${urls_csv}" \
    --gpu-count "${gpu_total}" \
    --width 832 \
    --height 480 \
    --fps 16 \
    --warmup-chunks 0 \
    --sample-id-regex "${smoke_regex}" \
    --output-dir "${results_root}/smoke" \
    --output "${results_root}/smoke/summary.json" \
    2>&1 | tee "${results_root}/smoke.log"
  touch "${results_root}/smoke-finished"
fi

set +e
python3 /opt/bench/benchmark_evalset.py \
  --messages "${messages_path}" \
  --image-urls "${image_urls_path}" \
  --urls "${urls_csv}" \
  --gpu-count "${gpu_total}" \
  --width 832 \
  --height 480 \
  --fps 16 \
  --warmup-chunks 0 \
  --output-dir "${results_root}/full" \
  --output "${results_root}/full/summary.json" \
  2>&1 | tee "${results_root}/full.log"
eval_status=${PIPESTATUS[0]}
set -e

kill -TERM "${dmon_pid}" >/dev/null 2>&1 || true
wait "${dmon_pid}" >/dev/null 2>&1 || true
echo "${eval_status}" > "${results_root}/eval-exit-code"
touch "${results_root}/eval-finished"
exit "${eval_status}"
