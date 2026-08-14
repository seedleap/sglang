#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_S3_URI:?set MODEL_S3_URI to the immutable release root}"
: "${MODEL_NAME:?set MODEL_NAME}"
: "${MODEL_VERSION:?set MODEL_VERSION}"
: "${MODEL_RELEASE_ID:?set MODEL_RELEASE_ID}"
: "${MODEL_SOURCE_REVISION:?set MODEL_SOURCE_REVISION}"

AWS_REGION="${AWS_REGION:-us-east-2}"
CANARY_DESTINATION="${CANARY_DESTINATION:-/model-cache/model}"
CANARY_LOCK_PATH="${CANARY_LOCK_PATH:-/model-cache/.download.lock}"
CANARY_CONCURRENCY="${CANARY_CONCURRENCY:-128}"
CANARY_PART_SIZE_MIB="${CANARY_PART_SIZE_MIB:-16}"
SGLANG_ROOT="${SGLANG_ROOT:-/opt/sglang}"
DOWNLOADER="${SGLANG_ROOT}/benchmark/minwm_realtime_async_vae/download_model_artifact.py"

if [[ "${CANARY_DESTINATION}" == "/" || "${CANARY_LOCK_PATH}" == "/" ]]; then
  echo "canary paths must not be the filesystem root" >&2
  exit 2
fi
if [[ -e "${CANARY_DESTINATION}" || -e "${CANARY_LOCK_PATH}" ]]; then
  echo "canary requires an empty destination and a fresh lock path" >&2
  exit 2
fi
test -f "${DOWNLOADER}"
mkdir -p "$(dirname "${CANARY_DESTINATION}")" "$(dirname "${CANARY_LOCK_PATH}")"

COMMON_ARGS=(
  --model-s3-uri "${MODEL_S3_URI}"
  --model-name "${MODEL_NAME}"
  --model-version "${MODEL_VERSION}"
  --model-release-id "${MODEL_RELEASE_ID}"
  --expected-revision "${MODEL_SOURCE_REVISION}"
  --destination "${CANARY_DESTINATION}"
  --lock-path "${CANARY_LOCK_PATH}"
  --region "${AWS_REGION}"
  --concurrency "${CANARY_CONCURRENCY}"
  --part-size-mib "${CANARY_PART_SIZE_MIB}"
)

cold_result="$(python3 "${DOWNLOADER}" "${COMMON_ARGS[@]}")"
warm_result="$(python3 "${DOWNLOADER}" "${COMMON_ARGS[@]}")"

python3 - "${cold_result}" "${warm_result}" \
  "${MODEL_NAME}" "${MODEL_VERSION}" "${MODEL_RELEASE_ID}" <<'PY'
import json
import sys

cold = json.loads(sys.argv[1])
warm = json.loads(sys.argv[2])
expected = {
    "model_name": sys.argv[3],
    "model_version": sys.argv[4],
    "release_id": sys.argv[5],
}
for name, result in (("cold", cold), ("warm", warm)):
    if result.get("backend") != "awscrt":
        raise SystemExit(f"{name} stage did not use AWS CRT")
    for field, value in expected.items():
        if result.get(field) != value:
            raise SystemExit(f"{name} stage {field} mismatch")
if cold.get("cache_hit") is not False:
    raise SystemExit("cold stage unexpectedly reported a cache hit")
if warm.get("cache_hit") is not True:
    raise SystemExit("warm restart did not validate and reuse the cache")
print(json.dumps({"cold": cold, "warm": warm, "verified": True}, sort_keys=True))
PY

test -f "${CANARY_DESTINATION}/artifact-manifest.json"
test -f "${CANARY_DESTINATION}/info.json"
test -f "${CANARY_DESTINATION}/_READY"
