#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?set RUNNER_COMMIT to the immutable benchmark commit}"
: "${NODE_NAME:?set NODE_NAME to the acquired RTX node}"
GPU_COUNT="${GPU_COUNT:-1}"
APPLY="${APPLY:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/k8s/minwm_rtx6000_taehv_local_job.template.yaml"
OUTPUT="${RENDER_OUTPUT:-/tmp/minwm-rtx6000-taehv-${RUNNER_COMMIT:0:10}.yaml}"
case "${GPU_COUNT}" in
  1) CPU_REQUEST=12; CPU_LIMIT=14; MEMORY_REQUEST=108Gi; MEMORY_LIMIT=114Gi ;;
  2) CPU_REQUEST=40; CPU_LIMIT=44; MEMORY_REQUEST=220Gi; MEMORY_LIMIT=240Gi ;;
  *) echo "GPU_COUNT must be 1 or 2" >&2; exit 2 ;;
esac
sed -e "s/__RUNNER_COMMIT__/${RUNNER_COMMIT}/g" \
    -e "s/__NODE_NAME__/${NODE_NAME}/g" \
    -e "s/__GPU_COUNT__/${GPU_COUNT}/g" \
    -e "s/__CPU_REQUEST__/${CPU_REQUEST}/g" \
    -e "s/__CPU_LIMIT__/${CPU_LIMIT}/g" \
    -e "s/__MEMORY_REQUEST__/${MEMORY_REQUEST}/g" \
    -e "s/__MEMORY_LIMIT__/${MEMORY_LIMIT}/g" \
    "${TEMPLATE}" > "${OUTPUT}"
kubectl --context leap-world-east apply --dry-run=server -f "${OUTPUT}" >/dev/null
if [[ "${APPLY}" == "true" ]]; then
  kubectl --context leap-world-east apply -f "${OUTPUT}"
else
  echo "dry-run manifest=${OUTPUT}"
fi
