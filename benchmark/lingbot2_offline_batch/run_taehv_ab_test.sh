#!/usr/bin/env bash
set -euo pipefail

results_root=${RESULTS_ROOT:?RESULTS_ROOT is required}
variant=${TAEHV_AB_VARIANT:?TAEHV_AB_VARIANT is required}
source_s3_uri=${TAEHV_AB_SOURCE_S3_URI:?TAEHV_AB_SOURCE_S3_URI is required}
output_s3_uri=${TAEHV_AB_OUTPUT_S3_URI:?TAEHV_AB_OUTPUT_S3_URI is required}
case_limit=${SGLANG_VIDEO_CASE_LIMIT:-100}
input_dir="${results_root}/input"

case "${variant}" in
  baseline|taehv) ;;
  *)
    echo "TAEHV_AB_VARIANT must be baseline or taehv" >&2
    exit 2
    ;;
esac

mkdir -p "${results_root}"
python3 /opt/bench/prepare_taehv_ab_inputs.py \
  --source-s3-uri "${source_s3_uri}" \
  --output-dir "${input_dir}" \
  --limit "${case_limit}"
cp "${input_dir}/fixture.jsonl" "${results_root}/fixture.jsonl"
cp "${input_dir}/fixture-metadata.json" "${results_root}/fixture-metadata.json"

export MESSAGES_PATH="${input_dir}/messages.jsonl"
export IMAGE_URLS_PATH="${input_dir}/image-urls.json"
export RESUME=false
export STREAM_UPLOAD=false

set +e
/opt/bench/run_capacity_smoke_720p.sh
benchmark_status=$?
set -e

set +e
python3 /opt/bench/upload_taehv_ab_results.py \
  --root "${results_root}" \
  --s3-uri "${output_s3_uri}"
upload_status=$?
set -e

if [[ "${benchmark_status}" -ne 0 ]]; then
  exit "${benchmark_status}"
fi
exit "${upload_status}"
