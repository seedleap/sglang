#!/usr/bin/env bash
set -euo pipefail

: "${KUBE_CONTEXT:=leap-world-us-east-2}"
: "${NAMESPACE:=minwm-zing-webrtc-lab}"
: "${MEDIA_HOST:=127.0.0.1}"

ROOT="$(git rev-parse --show-toplevel)"
IMAGES_FILE="${ROOT}/benchmark/minwm_zing_webrtc/.env.images"
[[ -f "${IMAGES_FILE}" ]] || {
  echo "Missing ${IMAGES_FILE}; run build_and_push.sh first" >&2
  exit 1
}
# shellcheck disable=SC1090
source "${IMAGES_FILE}"
: "${GPU_IMAGE:?GPU_IMAGE missing}"
: "${CPU_IMAGE:?CPU_IMAGE missing}"
: "${WEBUI_IMAGE:?WEBUI_IMAGE missing}"
: "${MEDIAMTX_IMAGE:?MEDIAMTX_IMAGE missing}"

render() {
  local destination=$1
  kubectl --context "${KUBE_CONTEXT}" kustomize \
    "${ROOT}/benchmark/minwm_zing_webrtc/k8s" | \
    sed \
      -e "s#REPLACE_WITH_GPU_IMAGE#${GPU_IMAGE}#g" \
      -e "s#REPLACE_WITH_CPU_IMAGE#${CPU_IMAGE}#g" \
      -e "s#REPLACE_WITH_WEBUI_IMAGE#${WEBUI_IMAGE}#g" \
      -e "s#REPLACE_WITH_MEDIAMTX_IMAGE#${MEDIAMTX_IMAGE}#g" \
      -e "s#REPLACE_WITH_MEDIA_PUBLIC_HOST#${MEDIA_HOST}#g" \
      >"${destination}"
}

wait_for_hostname() {
  local service=$1
  local hostname=""
  for _ in $(seq 1 120); do
    hostname="$(kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
      get service "${service}" \
      -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
    if [[ -n "${hostname}" ]]; then
      printf '%s' "${hostname}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${service} hostname" >&2
  return 1
}

manifest="$(mktemp)"
trap 'rm -f "${manifest}"' EXIT
render "${manifest}"
kubectl --context "${KUBE_CONTEXT}" apply -f "${manifest}"

MEDIA_HOST="$(wait_for_hostname zing-webrtc-media-public)"
APP_HOST="$(wait_for_hostname zing-webrtc-webui-public)"
render "${manifest}"
kubectl --context "${KUBE_CONTEXT}" apply -f "${manifest}"
kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
  rollout restart deployment/zing-webrtc-mediamtx
kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
  rollout status deployment/zing-webrtc-mediamtx --timeout=10m
kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
  rollout status deployment/zing-webrtc-webui --timeout=10m

cat >"${ROOT}/benchmark/minwm_zing_webrtc/.env.deployment" <<EOF
APP_URL=http://${APP_HOST}/
WEBRTC_BENCHMARK_URL=http://${APP_HOST}/webrtc-benchmark.html
MEDIA_URL=http://${MEDIA_HOST}/
EOF
cat "${ROOT}/benchmark/minwm_zing_webrtc/.env.deployment"
