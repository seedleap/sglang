#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?set RUNNER_COMMIT to the immutable paired-runner commit}"
PAIRS="${PAIRS:-A B C D}"
APPLY="${APPLY:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/k8s/minwm_rtx6000_pair_job.template.yaml"
RENDER_DIR="${RENDER_DIR:-/tmp/minwm-rtx6000-pairs-${RUNNER_COMMIT:0:10}}"
mkdir -p "${RENDER_DIR}"

for pair in ${PAIRS}; do
  [[ "${pair}" =~ ^[ABCD]$ ]] || { echo "invalid pair ${pair}" >&2; exit 2; }
  lower="$(tr '[:upper:]' '[:lower:]' <<<"${pair}")"
  output="${RENDER_DIR}/pair-${lower}.yaml"
  sed -e "s/__PAIR__/${pair}/g" -e "s/__PAIR_LOWER__/${lower}/g" \
      -e "s/__RUNNER_COMMIT__/${RUNNER_COMMIT}/g" "${TEMPLATE}" > "${output}"
  kubectl --context leap-world-east apply --dry-run=server -f "${output}" >/dev/null
  if [[ "${APPLY}" == "true" ]]; then
    kubectl --context leap-world-east apply -f "${output}"
  else
    echo "dry-run pair=${pair} manifest=${output}"
  fi
done
