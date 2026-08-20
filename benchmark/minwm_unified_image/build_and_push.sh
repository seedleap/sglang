#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-2}"
ECR_REPOSITORY="${ECR_REPOSITORY:-leap-world/minwm-runtime}"
ECR_CACHE_REPOSITORY="${ECR_CACHE_REPOSITORY:-leap-world/minwm-runtime-buildcache}"
EXPECTED_AWS_ACCOUNT="${EXPECTED_AWS_ACCOUNT:-829115578968}"
BUILDER_NAME="${BUILDER_NAME:-minwm-builder}"
CACHE_TAG="${CACHE_TAG:-linux-amd64}"
CUDA_BASE_IMAGE="nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e"

for command_name in aws cmp cp curl docker git grep jq sha256sum tar uname; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "${command_name}" >&2
    exit 1
  }
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  printf 'the release builder must be native Linux x86_64\n' >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SHORT_SHA="${GIT_SHA:0:12}"
BUILD_ID="${BUILD_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "${BUILD_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'BUILD_ID contains unsupported tag characters: %s\n' "${BUILD_ID}" >&2
  exit 1
fi
RESUME_IMAGE_TAG="${RESUME_IMAGE_TAG:-}"
if [[ -n "${RESUME_IMAGE_TAG}" ]]; then
  if [[ -n "${IMAGE_TAG:-}" && "${IMAGE_TAG}" != "${RESUME_IMAGE_TAG}" ]]; then
    printf 'IMAGE_TAG and RESUME_IMAGE_TAG disagree\n' >&2
    exit 1
  fi
  IMAGE_TAG="${RESUME_IMAGE_TAG}"
  RESUMED=true
else
  IMAGE_TAG="${IMAGE_TAG:-minwm-cu130-torch211-${SHORT_SHA}-${BUILD_ID}}"
  RESUMED=false
fi
if [[ ! "${IMAGE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  printf 'IMAGE_TAG is not a valid OCI tag: %s\n' "${IMAGE_TAG}" >&2
  exit 1
fi
BUILD_SOURCE="https://github.com/seedleap/sglang"
BUILD_URL="${BUILD_SOURCE}/commit/${GIT_SHA}"

if ! git -C "${REPO_ROOT}" diff --quiet || \
   ! git -C "${REPO_ROOT}" diff --cached --quiet; then
  printf 'refusing to build from tracked local changes; commit them first\n' >&2
  exit 1
fi
for required_path in \
  benchmark/minwm_unified_image/build_and_push.sh \
  python/kernels.lock \
  python/sglang/multimodal_gen/tools/minwm_dependency_check.py \
  python/sglang/multimodal_gen/tools/minwm_image_runtime_probe.py \
  python/sglang/multimodal_gen/tools/minwm_profile_launcher.py; do
  git -C "${REPO_ROOT}" cat-file -e "${GIT_SHA}:${required_path}" || {
    printf 'required release file is not tracked by %s: %s\n' \
      "${GIT_SHA}" "${required_path}" >&2
    exit 1
  }
done
if ! git ls-remote "${BUILD_SOURCE}.git" | \
  awk -v expected="${GIT_SHA}" '$1 == expected { found = 1 } END { exit !found }'; then
  printf 'source commit is not reachable from %s: %s\n' \
    "${BUILD_SOURCE}" "${GIT_SHA}" >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [[ "${ACCOUNT_ID}" != "${EXPECTED_AWS_ACCOUNT}" ]]; then
  printf 'unexpected AWS account: got %s, expected %s\n' \
    "${ACCOUNT_ID}" "${EXPECTED_AWS_ACCOUNT}" >&2
  exit 1
fi

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPOSITORY_URI="${REGISTRY}/${ECR_REPOSITORY}"
CACHE_REPOSITORY_URI="${REGISTRY}/${ECR_CACHE_REPOSITORY}"
IMAGE_URI="${REPOSITORY_URI}:${IMAGE_TAG}"
LOCAL_IMAGE="minwm-unified-local:${SHORT_SHA}-${BUILD_ID}"

RELEASE_REPOSITORY_JSON="$(aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${ECR_REPOSITORY}" \
  --output json)"
CACHE_REPOSITORY_JSON="$(aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${ECR_CACHE_REPOSITORY}" \
  --output json)"
if [[ "$(jq -r '.repositories[0].imageTagMutability' \
  <<<"${RELEASE_REPOSITORY_JSON}")" != "IMMUTABLE" ]]; then
  printf 'release repository must use immutable tags: %s\n' \
    "${ECR_REPOSITORY}" >&2
  exit 1
fi
if [[ "$(jq -r '.repositories[0].imageTagMutability' \
  <<<"${CACHE_REPOSITORY_JSON}")" != "MUTABLE" ]]; then
  printf 'cache repository must use mutable tags: %s\n' \
    "${ECR_CACHE_REPOSITORY}" >&2
  exit 1
fi
if [[ "$(jq -r '.repositories[0].imageScanningConfiguration.scanOnPush' \
  <<<"${RELEASE_REPOSITORY_JSON}")" != "true" ]]; then
  printf 'release repository must enable scanOnPush: %s\n' \
    "${ECR_REPOSITORY}" >&2
  exit 1
fi

EXISTING_IMAGE_COUNT="$(aws ecr batch-get-image \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPOSITORY}" \
  --image-ids "imageTag=${IMAGE_TAG}" \
  --query 'length(images)' \
  --output text)"
if [[ "${RESUMED}" == "true" ]]; then
  if [[ "${EXISTING_IMAGE_COUNT}" != "1" ]]; then
    printf 'resume tag must already exist exactly once: %s\n' "${IMAGE_URI}" >&2
    exit 1
  fi
elif [[ "${EXISTING_IMAGE_COUNT}" != "0" ]]; then
  printf 'refusing to overwrite existing image tag: %s\n' "${IMAGE_URI}" >&2
  exit 1
fi

docker info >/dev/null
BUILDER_INSPECT="$(docker buildx inspect "${BUILDER_NAME}" --bootstrap)"
if ! grep -Eq '^Driver:[[:space:]]+docker-container$' \
  <<<"${BUILDER_INSPECT}" || \
   ! grep -Eq '^Platforms:.*linux/amd64([,[:space:]]|$)' \
  <<<"${BUILDER_INSPECT}"; then
  printf 'builder %s must use docker-container and support linux/amd64\n' \
    "${BUILDER_NAME}" >&2
  exit 1
fi
SOURCE_DOCKER_CONFIG="${DOCKER_CONFIG:-${HOME}/.docker}"
DOCKER_CONFIG_DIR="$(mktemp -d /tmp/minwm-unified-docker-config.XXXXXX)"
if [[ -d "${SOURCE_DOCKER_CONFIG}" ]]; then
  cp -a "${SOURCE_DOCKER_CONFIG}/." "${DOCKER_CONFIG_DIR}/"
fi
export DOCKER_CONFIG="${DOCKER_CONFIG_DIR}"
docker buildx inspect "${BUILDER_NAME}" >/dev/null
CONTEXT_DIR=""
cleanup() {
  if [[ -n "${LOCAL_IMAGE:-}" ]]; then
    docker image rm --force "${LOCAL_IMAGE}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${IMMUTABLE_IMAGE:-}" ]]; then
    docker image rm --force "${IMMUTABLE_IMAGE}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CONTEXT_DIR:-}" && \
        "${CONTEXT_DIR}" == /tmp/minwm-unified-image-context.* ]]; then
    rm -rf -- "${CONTEXT_DIR}"
  fi
  if [[ -n "${DOCKER_CONFIG_DIR:-}" && \
        "${DOCKER_CONFIG_DIR}" == /tmp/minwm-unified-docker-config.* ]]; then
    rm -rf -- "${DOCKER_CONFIG_DIR}"
  fi
}
trap cleanup EXIT
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

CONTEXT_DIR="$(mktemp -d /tmp/minwm-unified-image-context.XXXXXX)"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d /tmp/minwm-unified-image-output.XXXXXX)}"
METADATA_FILE="${OUTPUT_DIR}/build-metadata.json"

mkdir -p "${OUTPUT_DIR}"
git -C "${REPO_ROOT}" archive "${GIT_SHA}" | tar -x -C "${CONTEXT_DIR}"

COMMON_BUILD_ARGS=(
  --builder "${BUILDER_NAME}"
  --file "${CONTEXT_DIR}/docker/Dockerfile"
  --target runtime
  --platform linux/amd64
  --build-arg "CUDA_VERSION=13.0.1"
  --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}"
  --build-arg "BUILD_TYPE=all"
  --build-arg "BRANCH_TYPE=local"
  --build-arg "GRACE_BLACKWELL=0"
  --build-arg "INSTALL_FLASHINFER_JIT_CACHE=1"
  --build-arg "REQUIRE_KERNELS_DOWNLOAD=1"
  --build-arg "SGLANG_EXCLUDE_MOVIEPY=1"
  --build-arg "SGLANG_EXCLUDE_NIXL=1"
  --build-arg "SGLANG_MINWM_SM120_FA4=1"
  --build-arg "SGLANG_MINWM_TAEHV=1"
  --build-arg "SGLANG_USE_SGL_FA3_KERNEL=0"
  --build-arg "SGLANG_BUILD_COMMIT=${GIT_SHA}"
  --build-arg "SGLANG_BUILD_SOURCE=${BUILD_SOURCE}"
  --build-arg "SGLANG_BUILD_URL=${BUILD_URL}"
  --build-arg "SGLANG_IMAGE_TAG=${IMAGE_URI}"
  --cache-from "type=registry,ref=${CACHE_REPOSITORY_URI}:${CACHE_TAG}"
  --cache-from "type=registry,ref=${REGISTRY}/leap-world/minwm-training:buildcache-linux-amd64"
)

BUILT_DIGEST=""
if [[ "${RESUMED}" == "false" ]]; then
  # Build and validate locally before any release tag is written.
  docker buildx build \
    "${COMMON_BUILD_ARGS[@]}" \
    --cache-to "type=registry,ref=${CACHE_REPOSITORY_URI}:${CACHE_TAG},mode=max,oci-mediatypes=true,image-manifest=true" \
    --provenance=false \
    --sbom=false \
    --tag "${LOCAL_IMAGE}" \
    --load \
    "${CONTEXT_DIR}"

  docker run --rm --entrypoint python3 "${LOCAL_IMAGE}" \
    -m sglang.multimodal_gen.tools.minwm_dependency_check \
    | tee "${OUTPUT_DIR}/pip-check-pre-push.txt"
  docker run --rm --entrypoint python3 "${LOCAL_IMAGE}" \
    -m sglang.multimodal_gen.tools.minwm_image_runtime_probe \
    --software-only \
    --expected-source-commit "${GIT_SHA}" \
    | tee "${OUTPUT_DIR}/software-contract-pre-push.json"

  # A unique tag in an immutable repository makes the registry write atomic.
  docker buildx build \
    "${COMMON_BUILD_ARGS[@]}" \
    --provenance=mode=max \
    --sbom=true \
    --metadata-file "${METADATA_FILE}" \
    --tag "${IMAGE_URI}" \
    --push \
    "${CONTEXT_DIR}"

  BUILT_DIGEST="$(jq -r '."containerimage.digest"' "${METADATA_FILE}")"
fi
RESOLVED_DIGEST=""
for resolve_attempt in 1 2 3 4 5 6; do
  RESOLVED_DIGEST="$(aws ecr batch-get-image \
    --region "${AWS_REGION}" \
    --repository-name "${ECR_REPOSITORY}" \
    --image-ids "imageTag=${IMAGE_TAG}" \
    --query 'images[0].imageId.imageDigest' \
    --output text 2>/dev/null || true)"
  if [[ "${RESOLVED_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    break
  fi
  sleep 5
done
if [[ ! "${RESOLVED_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'ECR returned invalid digest: %s\n' "${RESOLVED_DIGEST}" >&2
  exit 1
fi
if [[ "${RESUMED}" == "false" && "${BUILT_DIGEST}" != "${RESOLVED_DIGEST}" ]]; then
  printf 'build/ECR digest mismatch: build=%s ECR=%s\n' \
    "${BUILT_DIGEST}" "${RESOLVED_DIGEST}" >&2
  exit 1
fi

IMMUTABLE_IMAGE="${REPOSITORY_URI}@${RESOLVED_DIGEST}"
for pull_attempt in 1 2 3; do
  if docker pull "${IMMUTABLE_IMAGE}" >/dev/null; then
    break
  fi
  if [[ ${pull_attempt} -eq 3 ]]; then
    printf 'failed to pull immutable image after %s attempts\n' \
      "${pull_attempt}" >&2
    exit 1
  fi
  sleep 5
done
docker run --rm --entrypoint python3 "${IMMUTABLE_IMAGE}" \
  -m sglang.multimodal_gen.tools.minwm_dependency_check \
  | tee "${OUTPUT_DIR}/pip-check.txt"
docker run --rm --entrypoint python3 "${IMMUTABLE_IMAGE}" \
  -m sglang.multimodal_gen.tools.minwm_image_runtime_probe \
  --software-only \
  --expected-source-commit "${GIT_SHA}" \
  | tee "${OUTPUT_DIR}/software-contract.json"
docker run --rm --entrypoint python3 "${IMMUTABLE_IMAGE}" -m pip freeze \
  >"${OUTPUT_DIR}/pip-freeze.txt"
docker run --rm --entrypoint cat "${IMMUTABLE_IMAGE}" \
  /root/.cache/sglang/kernels.lock >"${OUTPUT_DIR}/kernels.lock"
cmp "${CONTEXT_DIR}/python/kernels.lock" "${OUTPUT_DIR}/kernels.lock"

docker buildx imagetools inspect --raw "${IMMUTABLE_IMAGE}" \
  >"${OUTPUT_DIR}/oci-index.json"
AMD64_DIGEST="$(jq -r '
  .manifests[]
  | select(.platform.os == "linux" and .platform.architecture == "amd64")
  | .digest
' "${OUTPUT_DIR}/oci-index.json")"
if [[ ! "${AMD64_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'OCI index does not contain exactly one linux/amd64 manifest\n' >&2
  exit 1
fi
ATTESTATION_DIGEST="$(jq -r --arg subject "${AMD64_DIGEST}" '
  .manifests[]
  | select(.annotations["vnd.docker.reference.type"] == "attestation-manifest")
  | select(.annotations["vnd.docker.reference.digest"] == $subject)
  | .digest
' "${OUTPUT_DIR}/oci-index.json")"
if [[ ! "${ATTESTATION_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'OCI index does not contain one attestation for %s\n' \
    "${AMD64_DIGEST}" >&2
  exit 1
fi
docker buildx imagetools inspect --raw \
  "${REPOSITORY_URI}@${ATTESTATION_DIGEST}" \
  >"${OUTPUT_DIR}/attestation-manifest.json"
PROVENANCE_PREDICATE="$(jq -r '
  .layers[]
  | .annotations["in-toto.io/predicate-type"]
  | select(startswith("https://slsa.dev/provenance/"))
' "${OUTPUT_DIR}/attestation-manifest.json")"
PROVENANCE_LAYER="$(jq -r --arg predicate "${PROVENANCE_PREDICATE}" '
  .layers[]
  | select(.annotations["in-toto.io/predicate-type"] == $predicate)
  | .digest
' "${OUTPUT_DIR}/attestation-manifest.json")"
SBOM_PREDICATE="https://spdx.dev/Document"
SBOM_LAYER="$(jq -r --arg predicate "${SBOM_PREDICATE}" '
  .layers[]
  | select(.annotations["in-toto.io/predicate-type"] == $predicate)
  | .digest
' "${OUTPUT_DIR}/attestation-manifest.json")"
if [[ ! "${PROVENANCE_LAYER}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'attestation manifest does not contain SLSA provenance\n' >&2
  exit 1
fi
if [[ ! "${SBOM_LAYER}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'attestation manifest does not contain an SPDX SBOM\n' >&2
  exit 1
fi

validate_attestation_statement() {
  local layer_digest="$1"
  local predicate="$2"
  local output_file="$3"
  local download_url
  local actual_digest

  download_url="$(aws ecr get-download-url-for-layer \
    --region "${AWS_REGION}" \
    --repository-name "${ECR_REPOSITORY}" \
    --layer-digest "${layer_digest}" \
    --query downloadUrl \
    --output text)"
  curl --fail --location --retry 3 --silent --show-error \
    "${download_url}" >"${output_file}"
  actual_digest="sha256:$(sha256sum "${output_file}" | awk '{print $1}')"
  if [[ "${actual_digest}" != "${layer_digest}" ]]; then
    printf 'attestation layer digest mismatch: got %s, expected %s\n' \
      "${actual_digest}" "${layer_digest}" >&2
    exit 1
  fi
  jq -e \
    --arg predicate "${predicate}" \
    --arg subject "${AMD64_DIGEST#sha256:}" \
    '(
      ._type | startswith("https://in-toto.io/Statement/")
    ) and (
      .predicateType == $predicate
    ) and any(
      .subject[]?; .digest.sha256 == $subject
    )' "${output_file}" >/dev/null
}

validate_attestation_statement \
  "${PROVENANCE_LAYER}" \
  "${PROVENANCE_PREDICATE}" \
  "${OUTPUT_DIR}/provenance.intoto.json"
validate_attestation_statement \
  "${SBOM_LAYER}" \
  "${SBOM_PREDICATE}" \
  "${OUTPUT_DIR}/sbom.spdx.intoto.json"

aws ecr batch-get-image \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPOSITORY}" \
  --image-ids "imageTag=${IMAGE_TAG}" \
  --accepted-media-types application/vnd.oci.image.index.v1+json \
  --output json \
  >"${OUTPUT_DIR}/ecr-image.json"

jq -n \
  --arg schema_version "minwm-unified-image-build/v1" \
  --arg source_commit "${GIT_SHA}" \
  --arg image_tag "${IMAGE_URI}" \
  --arg image_digest "${RESOLVED_DIGEST}" \
  --arg immutable_image "${IMMUTABLE_IMAGE}" \
  --arg build_url "${BUILD_URL}" \
  --arg cuda_base_image "${CUDA_BASE_IMAGE}" \
  --argjson resumed "${RESUMED}" \
  '{
    schema_version: $schema_version,
    source_commit: $source_commit,
    image_tag: $image_tag,
    image_digest: $image_digest,
    immutable_image: $immutable_image,
    build_url: $build_url,
    cuda_base_image: $cuda_base_image,
    resumed: $resumed,
    status: "software-validated"
  }' >"${OUTPUT_DIR}/build.json"

printf 'MINWM_IMAGE=%s\n' "${IMMUTABLE_IMAGE}" \
  >"${OUTPUT_DIR}/image.env"
printf 'Unified MinWM image candidate: %s\n' "${IMMUTABLE_IMAGE}"
printf 'Build evidence: %s\n' "${OUTPUT_DIR}"
