#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?set RUNNER_COMMIT to the immutable benchmark commit}"
APPLY="${APPLY:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/k8s/minwm_rtx6000_taehv_local_job.template.yaml"
OUTPUT="${RENDER_OUTPUT:-/tmp/minwm-rtx6000-taehv-${RUNNER_COMMIT:0:10}.yaml}"
sed -e "s/__RUNNER_COMMIT__/${RUNNER_COMMIT}/g" "${TEMPLATE}" > "${OUTPUT}"
kubectl --context leap-world-east apply --dry-run=server -f "${OUTPUT}" >/dev/null
if [[ "${APPLY}" == "true" ]]; then
  kubectl --context leap-world-east apply -f "${OUTPUT}"
else
  echo "dry-run manifest=${OUTPUT}"
fi
