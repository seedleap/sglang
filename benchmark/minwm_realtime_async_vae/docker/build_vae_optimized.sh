#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${ECR_REPOSITORY:?set ECR_REPOSITORY, for example leap-world/minwm-realtime}"

PYTHON_IMAGE_DIGEST="${PYTHON_IMAGE_DIGEST:-public.ecr.aws/docker/library/python@sha256:d657ab0ade19f404a6ccc883ab399540de667aff751748ce23c07330c5a89e64}"
if ! [[ "${PYTHON_IMAGE_DIGEST}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "base image must be pinned by sha256 digest: ${PYTHON_IMAGE_DIGEST}" >&2
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPOSITORY_URI="${REGISTRY}/${ECR_REPOSITORY}"
TAG="${IMAGE_TAG:-vae-optimized-${GIT_SHA}}"

aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${ECR_REPOSITORY}" >/dev/null
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

docker buildx build \
  --platform linux/amd64 \
  --progress=plain \
  --provenance=false \
  --file "${ROOT}/benchmark/minwm_realtime_async_vae/docker/Dockerfile.vae-optimized" \
  --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE_DIGEST}" \
  --label "org.opencontainers.image.revision=${GIT_SHA}" \
  --tag "${REPOSITORY_URI}:${TAG}" \
  --push \
  "${ROOT}"

DIGEST="$(aws ecr describe-images \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPOSITORY}" \
  --image-ids imageTag="${TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
printf '%s@%s\n' "${REPOSITORY_URI}" "${DIGEST}"
