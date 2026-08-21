#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-2}"
RELEASE_REPOSITORY="${RELEASE_REPOSITORY:-leap-world/minwm-runtime}"
CACHE_REPOSITORY="${CACHE_REPOSITORY:-leap-world/minwm-runtime-buildcache}"
EXPECTED_AWS_ACCOUNT="${EXPECTED_AWS_ACCOUNT:-829115578968}"

for command_name in aws jq; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "${command_name}" >&2
    exit 1
  }
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [[ "${ACCOUNT_ID}" != "${EXPECTED_AWS_ACCOUNT}" ]]; then
  printf 'unexpected AWS account: got %s, expected %s\n' \
    "${ACCOUNT_ID}" "${EXPECTED_AWS_ACCOUNT}" >&2
  exit 1
fi

ensure_repository() {
  local repository_name="$1"
  local mutability="$2"
  local scan_on_push="$3"
  local repository_json
  local describe_status

  set +e
  repository_json="$(aws ecr describe-repositories \
    --region "${AWS_REGION}" \
    --repository-names "${repository_name}" \
    --output json 2>&1)"
  describe_status=$?
  set -e
  if [[ ${describe_status} -eq 0 ]]; then
    local actual_mutability
    actual_mutability="$(jq -r '.repositories[0].imageTagMutability' \
      <<<"${repository_json}")"
    if [[ "${actual_mutability}" != "${mutability}" ]]; then
      printf 'repository %s is %s, expected %s; refusing to reconfigure it\n' \
        "${repository_name}" "${actual_mutability}" "${mutability}" >&2
      exit 1
    fi
    if [[ "$(jq -r '.repositories[0].imageScanningConfiguration.scanOnPush' \
      <<<"${repository_json}")" != "${scan_on_push}" ]]; then
      printf 'repository %s has an unexpected scanOnPush setting\n' \
        "${repository_name}" >&2
      exit 1
    fi
    if [[ "$(jq -r '.repositories[0].encryptionConfiguration.encryptionType' \
      <<<"${repository_json}")" != "AES256" ]]; then
      printf 'repository %s must use AES256 encryption\n' \
        "${repository_name}" >&2
      exit 1
    fi
    return
  fi
  if [[ "${repository_json}" != *RepositoryNotFoundException* ]]; then
    printf '%s\n' "${repository_json}" >&2
    exit "${describe_status}"
  fi

  aws ecr create-repository \
    --region "${AWS_REGION}" \
    --repository-name "${repository_name}" \
    --image-tag-mutability "${mutability}" \
    --image-scanning-configuration "scanOnPush=${scan_on_push}" \
    --encryption-configuration encryptionType=AES256 \
    --tags Key=Project,Value=minwm Key=ManagedBy,Value=sglang-unified-image \
    --output json \
    >/dev/null
}

ensure_repository "${RELEASE_REPOSITORY}" IMMUTABLE true
ensure_repository "${CACHE_REPOSITORY}" MUTABLE false

CACHE_LIFECYCLE_POLICY='{
  "rules": [{
    "rulePriority": 1,
    "description": "Expire superseded untagged build-cache manifests",
    "selection": {
      "tagStatus": "untagged",
      "countType": "sinceImagePushed",
      "countUnit": "days",
      "countNumber": 7
    },
    "action": {"type": "expire"}
  }]
}'
set +e
CURRENT_LIFECYCLE_JSON="$(aws ecr get-lifecycle-policy \
  --region "${AWS_REGION}" \
  --repository-name "${CACHE_REPOSITORY}" \
  --output json 2>&1)"
LIFECYCLE_STATUS=$?
set -e
if [[ ${LIFECYCLE_STATUS} -eq 0 ]]; then
  CURRENT_LIFECYCLE="$(jq -r '.lifecyclePolicyText' \
    <<<"${CURRENT_LIFECYCLE_JSON}" | jq -S -c .)"
  EXPECTED_LIFECYCLE="$(jq -S -c . <<<"${CACHE_LIFECYCLE_POLICY}")"
  if [[ "${CURRENT_LIFECYCLE}" != "${EXPECTED_LIFECYCLE}" ]]; then
    printf 'cache lifecycle differs from the managed contract; refusing to overwrite\n' \
      >&2
    exit 1
  fi
elif [[ "${CURRENT_LIFECYCLE_JSON}" == *LifecyclePolicyNotFoundException* ]]; then
  aws ecr put-lifecycle-policy \
    --region "${AWS_REGION}" \
    --repository-name "${CACHE_REPOSITORY}" \
    --lifecycle-policy-text "${CACHE_LIFECYCLE_POLICY}" \
    --output json \
    >/dev/null
else
  printf '%s\n' "${CURRENT_LIFECYCLE_JSON}" >&2
  exit "${LIFECYCLE_STATUS}"
fi

printf 'release repository: %s.dkr.ecr.%s.amazonaws.com/%s (IMMUTABLE)\n' \
  "${ACCOUNT_ID}" "${AWS_REGION}" "${RELEASE_REPOSITORY}"
printf 'cache repository:   %s.dkr.ecr.%s.amazonaws.com/%s (MUTABLE)\n' \
  "${ACCOUNT_ID}" "${AWS_REGION}" "${CACHE_REPOSITORY}"
