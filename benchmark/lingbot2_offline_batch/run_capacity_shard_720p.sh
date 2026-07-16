#!/usr/bin/env bash
set -euo pipefail

shard_index=${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}
printf -v shard "%02d" "${shard_index}"
fsx_batch_name=${FSX_BATCH_NAME:-third_person_all_1000x5_720p_129f_20260715}
fsx_eval_root=${FSX_EVAL_ROOT:-/fsx/world-model/eval/lingbot2/eval_sets/minWM}
work_dir="/inputs/shard-${shard}"
results_root="${fsx_eval_root}/${fsx_batch_name}/shard-${shard}"
mkdir -p "${work_dir}" "${results_root}"

eval "$(python3 - "${shard_index}" <<'PY'
import json, shlex, sys
manifest = json.load(open('/opt/bench/artifact-urls.json'))
index = int(sys.argv[1])
row = manifest['shards'][index]
values = {
    'IMAGE_URLS_GET': manifest['image_urls']['get_url'],
    'MESSAGES_GET': row['messages']['get_url'],
    'PUT_URLS_GET': row['put_urls']['get_url'],
    'BENCHMARK_SUMMARY_PUT': row['benchmark_summary_put_url'],
    'UPLOAD_SUMMARY_PUT': row['upload_summary_put_url'],
}
for key, value in values.items():
    print(f'export {key}={shlex.quote(value)}')
PY
)"

curl -fsSL "${IMAGE_URLS_GET}" -o "${work_dir}/image-urls.json"
curl -fsSL "${MESSAGES_GET}" -o "${work_dir}/messages.jsonl.gz"
curl -fsSL "${PUT_URLS_GET}" -o "${work_dir}/put-urls.json"

export RESULTS_ROOT="${results_root}"
export MESSAGES_PATH="${work_dir}/messages.jsonl.gz"
export IMAGE_URLS_PATH="${work_dir}/image-urls.json"
export PUT_URLS_PATH="${work_dir}/put-urls.json"
export SERVER_CACHE_ROOT="/server-cache/shard-${shard}"
export RESUME=true

set +e
/opt/bench/run_capacity_smoke_720p.sh
run_status=$?
set -e

if [[ -f "${results_root}/cases/summary.json" ]]; then
  curl -fsS -X PUT \
    -H 'Content-Type: application/json' \
    --upload-file "${results_root}/cases/summary.json" \
    "${BENCHMARK_SUMMARY_PUT}"
fi
if [[ -f "${results_root}/upload-summary.json" ]]; then
  curl -fsS -X PUT \
    -H 'Content-Type: application/json' \
    --upload-file "${results_root}/upload-summary.json" \
    "${UPLOAD_SUMMARY_PUT}"
fi
exit "${run_status}"
