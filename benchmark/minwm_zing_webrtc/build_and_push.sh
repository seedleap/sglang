#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-2}"
: "${ECR_REPOSITORY:=leap-world/minwm-realtime}"
: "${GPU_RUNTIME_IMAGE:=829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime@sha256:8939f754f788afdb80e0da33d7b082b0a80ebaa2305de875430ed99aa93e1eec}"
: "${CPU_RUNTIME_IMAGE:=829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime@sha256:dfcd763bd51e8035b2147b0f9b27d092636b5773534711954e556f01d7b36c51}"
: "${WEBUI_RUNTIME_IMAGE:=829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime@sha256:f99dfb4b93505c0e5bc78308f1ed62e4ef766b86f333a750a6479268e4d4aae4}"

ROOT="$(git rev-parse --show-toplevel)"
GIT_SHA="$(git rev-parse --short=12 HEAD)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPOSITORY_URI="${REGISTRY}/${ECR_REPOSITORY}"
TAG_PREFIX="zing-webrtc-${GIT_SHA}"

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

docker buildx build --platform linux/amd64 --push \
  -f "${ROOT}/benchmark/minwm_realtime_async_vae/docker/Dockerfile.gpu-code-overlay" \
  --build-arg "GPU_RUNTIME_IMAGE=${GPU_RUNTIME_IMAGE}" \
  --build-arg "GIT_SHA=${GIT_SHA}" \
  -t "${REPOSITORY_URI}:${TAG_PREFIX}-gpu" "${ROOT}"

docker buildx build --platform linux/amd64 --push \
  -f "${ROOT}/benchmark/minwm_realtime_async_vae/docker/Dockerfile.cpu-code-overlay" \
  --build-arg "CPU_RUNTIME_IMAGE=${CPU_RUNTIME_IMAGE}" \
  --build-arg "SOURCE_REVISION=${GIT_SHA}" \
  -t "${REPOSITORY_URI}:${TAG_PREFIX}-cpu" "${ROOT}"

docker buildx build --platform linux/amd64 --push \
  -f "${ROOT}/benchmark/minwm_zing_webrtc/docker/Dockerfile.webui-bridge" \
  --build-arg "WEBUI_RUNTIME_IMAGE=${WEBUI_RUNTIME_IMAGE}" \
  --build-arg "SOURCE_REVISION=${GIT_SHA}" \
  -t "${REPOSITORY_URI}:${TAG_PREFIX}-webui" "${ROOT}"

docker buildx build --platform linux/amd64 --push \
  -f "${ROOT}/benchmark/minwm_zing_webrtc/docker/Dockerfile.mediamtx" \
  -t "${REPOSITORY_URI}:${TAG_PREFIX}-mediamtx" "${ROOT}"

OUTPUT="${ROOT}/benchmark/minwm_zing_webrtc/.env.images"
: >"${OUTPUT}"
for ROLE in gpu cpu webui mediamtx; do
  TAG="${TAG_PREFIX}-${ROLE}"
  DIGEST="$(aws ecr describe-images \
    --region "${AWS_REGION}" \
    --repository-name "${ECR_REPOSITORY}" \
    --image-ids "imageTag=${TAG}" \
    --query 'imageDetails[0].imageDigest' \
    --output text)"
  printf '%s_IMAGE=%s@%s\n' \
    "$(printf '%s' "${ROLE}" | tr '[:lower:]' '[:upper:]')" \
    "${REPOSITORY_URI}" "${DIGEST}" >>"${OUTPUT}"
done
printf 'Wrote immutable image references to %s\n' "${OUTPUT}"
