#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

RELEASED_PUBLIC_WEB_HOST="116.62.150.115"
SSH_HOST="${SSH_HOST:?set SSH_HOST=root@<new-aliyun-host> before deploying}"
REMOTE_DIR="${REMOTE_DIR:-/root/zing-realtime}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-${REMOTE_DIR}/start_remote.sh}"
REMOTE_OVERLAY="${REMOTE_OVERLAY:-${REMOTE_DIR}/sglang-main-overlay.tar.gz}"
PUBLIC_WEB_HOST="${PUBLIC_WEB_HOST:-${SSH_HOST#*@}}"
PUBLIC_GATEWAY_PORT="${PUBLIC_GATEWAY_PORT:-18080}"
PUBLIC_GATEWAY_BASE_URL="${PUBLIC_GATEWAY_BASE_URL:-http://${PUBLIC_WEB_HOST}:${PUBLIC_GATEWAY_PORT}}"
START_GPU_WORKERS="${START_GPU_WORKERS:-true}"
SKIP_IMAGE_PULL="${SKIP_IMAGE_PULL:-false}"
ALIYUN_ECS_RAM_ROLE="${ALIYUN_ECS_RAM_ROLE:-ZingRealtimeEcsPullRole}"
DENOISER_ATTENTION_BACKEND="${DENOISER_ATTENTION_BACKEND:-fa}"
DENOISER_ATTENTION_IMPL="${DENOISER_ATTENTION_IMPL:-packed}"
DENOISER_CACHE_ROTATED_K="${DENOISER_CACHE_ROTATED_K:-false}"
DENOISER_PERFORMANCE_MODE="${DENOISER_PERFORMANCE_MODE:-speed}"
DENOISER_ENABLE_CUDA_GRAPH="${DENOISER_ENABLE_CUDA_GRAPH:-false}"
DENOISER_PYTORCH_CUDA_ALLOC_CONF="${DENOISER_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REALTIME_WORKER_MAX_CONSUMED_AGE_S="${REALTIME_WORKER_MAX_CONSUMED_AGE_S:-120}"
REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS:-18}"
H264_LIVE_EDGE_TARGET_MS="${H264_LIVE_EDGE_TARGET_MS:-500}"
H264_LIVE_EDGE_SEEK_THRESHOLD_MS="${H264_LIVE_EDGE_SEEK_THRESHOLD_MS:-900}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

main() {
  if [[ "${SSH_HOST}" == *"${RELEASED_PUBLIC_WEB_HOST}"* || "${PUBLIC_WEB_HOST}" == "${RELEASED_PUBLIC_WEB_HOST}" ]]; then
    echo "${RELEASED_PUBLIC_WEB_HOST} was released; set SSH_HOST/PUBLIC_WEB_HOST to the new Aliyun host." >&2
    exit 1
  fi

  local archive
  archive="$(mktemp -t sglang-aliyun-overlay.XXXXXX.tar.gz)"
  trap "rm -f '${archive}'" EXIT

  log "packaging latest main-based realtime code overlay"
  COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --exclude='._*' \
    --exclude='.DS_Store' \
    -C "${ROOT}" \
    -czf "${archive}" \
    python/sglang/multimodal_gen

  log "uploading deployment artifacts to ${SSH_HOST}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${SSH_HOST}" \
    "mkdir -p '${REMOTE_DIR}'"
  scp -q -o BatchMode=yes "${archive}" "${SSH_HOST}:${REMOTE_OVERLAY}"
  scp -q -o BatchMode=yes "${SCRIPT_DIR}/start_remote.sh" \
    "${SSH_HOST}:${REMOTE_SCRIPT}"

  log "starting direct-H264 deployment"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "chmod +x '${REMOTE_SCRIPT}' && \
     CODE_OVERLAY_ARCHIVE='${REMOTE_OVERLAY}' \
     PUBLIC_WEB_HOST='${PUBLIC_WEB_HOST}' \
     PUBLIC_GATEWAY_PORT='${PUBLIC_GATEWAY_PORT}' \
     PUBLIC_GATEWAY_BASE_URL='${PUBLIC_GATEWAY_BASE_URL}' \
     START_GPU_WORKERS='${START_GPU_WORKERS}' \
     SKIP_IMAGE_PULL='${SKIP_IMAGE_PULL}' \
     ALIYUN_ECS_RAM_ROLE='${ALIYUN_ECS_RAM_ROLE}' \
     DENOISER_ATTENTION_BACKEND='${DENOISER_ATTENTION_BACKEND}' \
     DENOISER_ATTENTION_IMPL='${DENOISER_ATTENTION_IMPL}' \
     DENOISER_CACHE_ROTATED_K='${DENOISER_CACHE_ROTATED_K}' \
     DENOISER_PERFORMANCE_MODE='${DENOISER_PERFORMANCE_MODE}' \
     DENOISER_ENABLE_CUDA_GRAPH='${DENOISER_ENABLE_CUDA_GRAPH}' \
     DENOISER_PYTORCH_CUDA_ALLOC_CONF='${DENOISER_PYTORCH_CUDA_ALLOC_CONF}' \
     REALTIME_WORKER_MAX_CONSUMED_AGE_S='${REALTIME_WORKER_MAX_CONSUMED_AGE_S}' \
     REALTIME_TARGET_FPS='${REALTIME_TARGET_FPS}' \
     H264_LIVE_EDGE_TARGET_MS='${H264_LIVE_EDGE_TARGET_MS}' \
     H264_LIVE_EDGE_SEEK_THRESHOLD_MS='${H264_LIVE_EDGE_SEEK_THRESHOLD_MS}' \
     exec '${REMOTE_SCRIPT}'"

  log "WebUI: http://${PUBLIC_WEB_HOST}/?mode=i2v&playback=smooth_timeline"
  log "Gateway: ${PUBLIC_GATEWAY_BASE_URL}"
}

main "$@"
