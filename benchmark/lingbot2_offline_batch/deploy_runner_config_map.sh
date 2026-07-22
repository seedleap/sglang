#!/usr/bin/env bash
set -euo pipefail

# Publish an immutable, revision-named runner bundle. The controller mounts this
# bundle read-only at /opt/bench, so GPU Jobs do not clone the repository or
# install Python packages at runtime.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
namespace=${SGLANG_VIDEO_NAMESPACE:-default}
context=${SGLANG_VIDEO_KUBECTL_CONTEXT:-leap-world-aws03-usw2}
revision=$(git -C "${repo_root}" rev-parse --short=12 HEAD)
name=${1:-sglang-video-runner-${revision}}

files=(
  run_t2i_video_batch.py
  t2i_video_batch.py
  prepare_capacity_smoke_720p.py
  thirdperson_actions.py
  run_capacity_smoke_720p.sh
  benchmark_evalset.py
  trajs.jsonl.gz
)

for file in "${files[@]}"; do
  test -f "${repo_root}/benchmark/lingbot2_offline_batch/${file}" || {
    echo "missing runner file: ${file}" >&2
    exit 1
  }
done

if kubectl --context "${context}" -n "${namespace}" get configmap "${name}" >/dev/null 2>&1; then
  echo "ConfigMap already exists and is versioned/immutable: ${name}" >&2
  exit 1
fi

args=()
for file in "${files[@]}"; do
  args+=("--from-file=${file}=${repo_root}/benchmark/lingbot2_offline_batch/${file}")
done

kubectl --context "${context}" -n "${namespace}" create configmap "${name}" \
  "${args[@]}"
kubectl --context "${context}" -n "${namespace}" patch configmap "${name}" \
  --type merge --patch '{"immutable":true}'
printf '%s\n' "${name}"
