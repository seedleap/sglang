#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:-/results}
gpu_total=${GPU_TOTAL:-8}
videos=${VIDEOS:-8}
topologies=${TOPOLOGIES:-"8 4 2 1"}
chunks=${CHUNKS:-32}
warmup_chunks=${WARMUP_CHUNKS:-3}
frames_per_chunk=${FRAMES_PER_CHUNK:-9}
size=${SIZE:-832x480}
fps=${FPS:-25}
perspective_mode=${PERSPECTIVE_MODE:-mixed}
model_id=${MODEL_ID:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}
model_revision=${MODEL_REVISION:-59cccf49f2d2dd27418ae7a04b82b10868d455c2}
first_frame_args=()
if [[ -n ${FIRST_FRAME_FIRST_PERSON:-} ]]; then
  first_frame_args+=(--first-frame-first-person "${FIRST_FRAME_FIRST_PERSON}")
fi
if [[ -n ${FIRST_FRAME_THIRD_PERSON:-} ]]; then
  first_frame_args+=(--first-frame-third-person "${FIRST_FRAME_THIRD_PERSON}")
fi

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
  local start_ts
  local urls=()
  local ports=()
  local server_index

  mkdir -p "${topology_dir}/videos"
  stop_servers
  start_ts=$SECONDS
  nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv \
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
      return 1
    fi
  done
  local startup_sec=$((SECONDS - start_ts))

  nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "${topology_dir}/nvidia-dmon.log" 2>&1 &
  local dmon_pid=$!

  local urls_csv
  urls_csv=$(IFS=,; echo "${urls[*]}")
  set +e
  python3 /opt/bench/benchmark_batch.py \
    --urls "${urls_csv}" \
    --gpu-count "${gpu_total}" \
    --videos "${videos}" \
    --chunks "${chunks}" \
    --warmup-chunks "${warmup_chunks}" \
    --frames-per-chunk "${frames_per_chunk}" \
    --size "${size}" \
    --fps "${fps}" \
    --perspective-mode "${perspective_mode}" \
    "${first_frame_args[@]}" \
    --server-startup-sec "${startup_sec}" \
    --output-dir "${topology_dir}/videos" \
    --output "${topology_dir}/summary.json" \
    > "${topology_dir}/benchmark.log" 2>&1
  local benchmark_status=$?
  set -e

  kill -TERM "${dmon_pid}" >/dev/null 2>&1 || true
  wait "${dmon_pid}" >/dev/null 2>&1 || true
  cat "${topology_dir}/benchmark.log"
  stop_servers
  return "${benchmark_status}"
}

overall_status=0
for gpus_per_server in ${topologies}; do
  if (( gpu_total % gpus_per_server != 0 )); then
    echo "Skipping invalid topology: ${gpu_total}/${gpus_per_server}" >&2
    overall_status=1
    continue
  fi
  if ! run_topology "${gpus_per_server}"; then
    echo "Topology failed: $((gpu_total / gpus_per_server))x${gpus_per_server}gpu" >&2
    overall_status=1
  fi
done

python3 /opt/bench/summarize_topologies.py \
  --results-root "${results_root}" \
  --output "${results_root}/comparison.json"
exit "${overall_status}"
