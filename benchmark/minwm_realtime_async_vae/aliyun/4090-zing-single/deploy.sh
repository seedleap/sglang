#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

SSH_HOST="${SSH_HOST:-root@116.62.150.115}"
REMOTE_DIR="${REMOTE_DIR:-/root/zing-realtime}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-${REMOTE_DIR}/start_remote.sh}"
REMOTE_OVERLAY="${REMOTE_OVERLAY:-${REMOTE_DIR}/sglang-main-overlay.tar.gz}"
PUBLIC_WEB_HOST="${PUBLIC_WEB_HOST:-116.62.150.115}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

main() {
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
     SKIP_IMAGE_PULL='true' \
     exec '${REMOTE_SCRIPT}'"

  log "WebUI: http://${PUBLIC_WEB_HOST}/?mode=i2v&playback=smooth_timeline"
}

main "$@"
