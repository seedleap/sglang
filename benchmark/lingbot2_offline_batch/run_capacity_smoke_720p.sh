#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:-/results/capacity-smoke-720p}
gpu_total=${SGLANG_VIDEO_GPU_TOTAL:-${GPU_TOTAL:-8}}
gpus_per_server=${SGLANG_VIDEO_GPUS_PER_SERVER:-${GPUS_PER_SERVER:-2}}
topology=${SGLANG_VIDEO_TOPOLOGY:-}
case "${topology}" in
  "") ;;
  4x2) gpus_per_server=2 ;;
  8x1) gpus_per_server=1 ;;
  *)
    echo "unsupported SGLANG_VIDEO_TOPOLOGY: ${topology}" >&2
    exit 2
    ;;
esac
if ! [[ "${gpu_total}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${gpus_per_server}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_TOTAL and GPUS_PER_SERVER must be positive integers" >&2
  exit 2
fi
if (( gpu_total % gpus_per_server != 0 )); then
  echo "GPU_TOTAL=${gpu_total} is not divisible by GPUS_PER_SERVER=${gpus_per_server}" >&2
  exit 2
fi
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
width=${SGLANG_VIDEO_WIDTH:-${WIDTH:-1280}}
height=${SGLANG_VIDEO_HEIGHT:-${HEIGHT:-704}}
fps=${SGLANG_VIDEO_FPS:-${FPS:-24}}
ws_close_timeout=${SGLANG_VIDEO_WS_CLOSE_TIMEOUT:-${WS_CLOSE_TIMEOUT:-10}}
server_stop_grace_seconds=${SGLANG_VIDEO_SERVER_STOP_GRACE_SECONDS:-5}
taehv_checkpoint_path=${TAEHV_CHECKPOINT_PATH:-}
case_limit=${SGLANG_VIDEO_CASE_LIMIT:-}

mkdir -p "${results_root}"
runner_started_epoch=$(date +%s)
printf '{"gpu_total":%s,"gpus_per_server":%s,"server_count":%s,"topology":"%s"}\n' \
  "${gpu_total}" "${gpus_per_server}" "${server_count}" \
  "${topology:-${server_count}x${gpus_per_server}}" \
  > "${results_root}/topology.json"
server_pids=()

stop_servers() {
  local pid
  for pid in "${server_pids[@]:-}"; do
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
  done
  sleep "${server_stop_grace_seconds}"
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

taehv_enabled=false
taehv_args=()
taehv_checkpoint_sha256=""
taehv_package_version=""
if [[ -n "${taehv_checkpoint_path}" ]]; then
  if [[ ! -r "${taehv_checkpoint_path}" ]]; then
    echo "TAEHV checkpoint is not readable: ${taehv_checkpoint_path}" >&2
    exit 2
  fi
  python3 -c 'import taehv' || {
    echo "TAEHV package is unavailable" >&2
    exit 2
  }
  taehv_enabled=true
  taehv_checkpoint_sha256=$(sha256sum "${taehv_checkpoint_path}" | awk '{print $1}')
  taehv_package_version=$(python3 - <<'PY'
from importlib.metadata import version

print(version("taehv"))
PY
)
  taehv_args=(--vae-config.taehv-checkpoint-path "${taehv_checkpoint_path}")
fi
python3 - \
  "${results_root}/taehv-runtime.json" \
  "${taehv_enabled}" \
  "${taehv_checkpoint_path}" \
  "${taehv_checkpoint_sha256}" \
  "${taehv_package_version}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "enabled": sys.argv[2] == "true",
            "checkpoint_path": sys.argv[3] or None,
            "checkpoint_sha256": sys.argv[4] or None,
            "package_version": sys.argv[5] or None,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

urls=()
ports=()
server_start_epoch=$(date +%s)
for ((server_index = 0; server_index < server_count; server_index++)); do
  first_gpu=$((server_index * gpus_per_server))
  last_gpu=$((first_gpu + gpus_per_server - 1))
  devices=$(seq -s, "${first_gpu}" "${last_gpu}")
  port=$((30000 + server_index * 100))
  ports+=("${port}")
  urls+=("ws://127.0.0.1:${port}/v1/realtime_video/generate")
  cache_dir="${server_cache_root}/gpus-per-server-${gpus_per_server}/server-${server_index}"
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
    "${taehv_args[@]}" \
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
server_ready_epoch=$(date +%s)
server_startup_seconds=$((server_ready_epoch - server_start_epoch))
printf '%s\n' "${server_startup_seconds}" > "${results_root}/server-startup-seconds"

nvidia-smi dmon -s pucvmet -d 1 -o DT > "${results_root}/nvidia-dmon.log" 2>&1 &
dmon_pid=$!
urls_csv=$(IFS=,; echo "${urls[*]}")

set +e
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi
case_limit_args=()
if [[ -n "${case_limit}" ]]; then
  if ! [[ "${case_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SGLANG_VIDEO_CASE_LIMIT must be a positive integer" >&2
    exit 2
  fi
  case_limit_args+=(--limit "${case_limit}")
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
benchmark_started_epoch=$(date +%s)
python3 /opt/bench/benchmark_evalset.py \
  --messages "${messages_path}" \
  --image-urls "${image_urls_path}" \
  --urls "${urls_csv}" \
  --gpu-count "${gpu_total}" \
  --width "${width}" \
  --height "${height}" \
  --fps "${fps}" \
  --warmup-chunks 3 \
  --close-timeout "${ws_close_timeout}" \
  --output-dir "${results_root}/cases" \
  --output "${results_root}/cases/summary.json" \
  "${case_limit_args[@]}" \
  "${resume_args[@]}" \
  2>&1 | tee "${results_root}/benchmark.log"
benchmark_status=${PIPESTATUS[0]}
benchmark_finished_epoch=$(date +%s)
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
upload_finished_epoch=$(date +%s)

kill -TERM "${dmon_pid}" >/dev/null 2>&1 || true
wait "${dmon_pid}" >/dev/null 2>&1 || true
runner_finished_epoch=$(date +%s)
python3 - \
  "${results_root}/lifecycle.json" \
  "${runner_started_epoch}" \
  "${server_start_epoch}" \
  "${server_ready_epoch}" \
  "${benchmark_started_epoch}" \
  "${benchmark_finished_epoch}" \
  "${upload_finished_epoch}" \
  "${runner_finished_epoch}" <<'PY'
import json
import sys
from pathlib import Path

labels = (
    "runner_started_epoch",
    "server_start_epoch",
    "server_ready_epoch",
    "benchmark_started_epoch",
    "benchmark_finished_epoch",
    "upload_finished_epoch",
    "runner_finished_epoch",
)
values = dict(zip(labels, (int(value) for value in sys.argv[2:])))
values["server_startup_sec"] = (
    values["server_ready_epoch"] - values["server_start_epoch"]
)
values["benchmark_wall_sec"] = (
    values["benchmark_finished_epoch"] - values["benchmark_started_epoch"]
)
values["upload_and_finalize_sec"] = (
    values["runner_finished_epoch"] - values["benchmark_finished_epoch"]
)
values["runner_wall_sec"] = (
    values["runner_finished_epoch"] - values["runner_started_epoch"]
)
Path(sys.argv[1]).write_text(
    json.dumps(values, indent=2) + "\n", encoding="utf-8"
)
PY
echo "${benchmark_status}" > "${results_root}/exit-code"
echo "${upload_status}" > "${results_root}/upload-exit-code"
touch "${results_root}/finished"
if [[ "${benchmark_status}" -ne 0 ]]; then
  exit "${benchmark_status}"
fi
exit "${upload_status}"
