#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:-/results/b300-thirdperson50-480p-16fps}
gpu_total=${GPU_TOTAL:-8}
topologies=${TOPOLOGIES:-"2 1"}
model_id=${MODEL_ID:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}
model_revision=${MODEL_REVISION:-59cccf49f2d2dd27418ae7a04b82b10868d455c2}
messages_path=${MESSAGES_PATH:-/opt/bench/thirdperson50-messages.jsonl.gz}
image_urls_path=${IMAGE_URLS_PATH:-/opt/bench/thirdperson50-image-urls.json}
width=${WIDTH:-832}
height=${HEIGHT:-480}
fps=${FPS:-16}
warmup_chunks=${WARMUP_CHUNKS:-3}

mkdir -p "${results_root}"
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

run_topology() {
  local gpus_per_server=$1
  local server_count=$((gpu_total / gpus_per_server))
  local label="${server_count}x${gpus_per_server}gpu"
  local topology_dir="${results_root}/${label}"
  local ports=()
  local urls=()
  local server_index

  mkdir -p "${topology_dir}"
  stop_servers
  nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap --format=csv \
    > "${topology_dir}/gpu-inventory.csv"

  for ((server_index = 0; server_index < server_count; server_index++)); do
    local first_gpu=$((server_index * gpus_per_server))
    local last_gpu=$((first_gpu + gpus_per_server - 1))
    local devices
    devices=$(seq -s, "${first_gpu}" "${last_gpu}")
    local port=$((30000 + server_index * 100))
    ports+=("${port}")
    urls+=("ws://127.0.0.1:${port}/v1/realtime_video/generate")
    mkdir -p "${topology_dir}/cache-${server_index}"

    CUDA_VISIBLE_DEVICES="${devices}" \
    NCCL_PROTO=Simple \
    PYTHONUNBUFFERED=1 \
    SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES=60 \
    SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW=true \
    SGLANG_DIFFUSION_CACHE_ROOT="${topology_dir}/cache-${server_index}" \
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
      > "${topology_dir}/server-${server_index}.log" 2>&1 &
    server_pids+=("$!")
  done

  for ((server_index = 0; server_index < server_count; server_index++)); do
    if ! wait_for_server "${ports[$server_index]}" "${server_pids[$server_index]}"; then
      tail -n 300 "${topology_dir}/server-${server_index}.log" >&2 || true
      stop_servers
      return 1
    fi
  done

  nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "${topology_dir}/nvidia-dmon.log" 2>&1 &
  local dmon_pid=$!
  local urls_csv
  urls_csv=$(IFS=,; echo "${urls[*]}")

  set +e
  python3 /opt/bench/benchmark_evalset.py \
    --messages "${messages_path}" \
    --image-urls "${image_urls_path}" \
    --urls "${urls_csv}" \
    --gpu-count "${gpu_total}" \
    --width "${width}" \
    --height "${height}" \
    --fps "${fps}" \
    --warmup-chunks "${warmup_chunks}" \
    --output-dir "${topology_dir}/thirdperson50" \
    --output "${topology_dir}/thirdperson50/summary.json" \
    2>&1 | tee "${topology_dir}/thirdperson50.log"
  local benchmark_status=${PIPESTATUS[0]}
  set -e

  kill -TERM "${dmon_pid}" >/dev/null 2>&1 || true
  wait "${dmon_pid}" >/dev/null 2>&1 || true
  echo "${benchmark_status}" > "${topology_dir}/exit-code"
  stop_servers
  return "${benchmark_status}"
}

overall_status=0
for gpus_per_server in ${topologies}; do
  if ! run_topology "${gpus_per_server}"; then
    echo "Topology failed: $((gpu_total / gpus_per_server))x${gpus_per_server}gpu" >&2
    overall_status=1
  fi
done
exit "${overall_status}"
