#!/usr/bin/env bash
set -euo pipefail

image=${1:?usage: build_video_runner_image.sh <ecr-image:tag>}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_revision=$(git -C "${repo_root}" rev-parse HEAD)

docker buildx build \
  --platform linux/amd64 \
  --file "${repo_root}/benchmark/lingbot2_offline_batch/Dockerfile.video-runner" \
  --build-arg "SOURCE_REVISION=${source_revision}" \
  --tag "${image}" \
  --push \
  "${repo_root}"
