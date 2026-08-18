#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s hopper|blackwell SOURCE_SHA40 sha256:IMAGE_DIGEST\n' "$0" >&2
  exit 2
fi

FAMILY="$1"
SOURCE_SHA="$2"
IMAGE_DIGEST="$3"
if [[ ! "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'invalid source SHA: %s\n' "${SOURCE_SHA}" >&2
  exit 2
fi
if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'invalid image digest: %s\n' "${IMAGE_DIGEST}" >&2
  exit 2
fi
case "${FAMILY}" in
  hopper) DOCUMENT_NUMBER=1 ;;
  blackwell) DOCUMENT_NUMBER=2 ;;
  *)
    printf 'unsupported family: %s\n' "${FAMILY}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/gpu_probe_jobs.template.yaml"
SHORT_SHA="${SOURCE_SHA:0:12}"
DIGEST_HEX="${IMAGE_DIGEST#sha256:}"
SHORT_DIGEST="${DIGEST_HEX:0:12}"

awk -v wanted="${DOCUMENT_NUMBER}" '
  BEGIN { document = 1 }
  /^---$/ { document += 1; next }
  document == wanted { print }
' "${TEMPLATE}" | sed \
  -e "s/__SOURCE_SHA40__/${SOURCE_SHA}/g" \
  -e "s/__SOURCE_SHA12__/${SHORT_SHA}/g" \
  -e "s/__IMAGE_DIGEST12__/${SHORT_DIGEST}/g" \
  -e "s/__IMAGE_DIGEST__/${IMAGE_DIGEST}/g"
