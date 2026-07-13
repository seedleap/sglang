#!/usr/bin/env bash
set -euo pipefail

base_image="lmsysorg/sglang:dev@sha256:8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7"
runtime="codex-lingbot2-runtime"
source_root=/opt/lingbot2/sglang
results_root=/opt/lingbot2/results
cache_root=/opt/lingbot2/cache

test "$(cat "${source_root}/.source_sha")" = afc619cc79a2960f9cab53b3823d904672f3c5c0
mkdir -p "${results_root}" "${cache_root}"
docker rm -f "${runtime}" >/dev/null 2>&1 || true

docker run --detach \
  --name "${runtime}" \
  --gpus all \
  --network host \
  --ipc host \
  --cap-add SYS_ADMIN \
  --ulimit memlock=-1 \
  --volume "${source_root}:/workspace/sglang" \
  --volume "/opt/lingbot2/benchmark_realtime.py:/workspace/benchmark_realtime.py:ro" \
  --volume "${results_root}:/results" \
  --volume "${cache_root}:/root/.cache" \
  --workdir /workspace/sglang \
  --env NCCL_PROTO=Simple \
  --env PYTHONUNBUFFERED=1 \
  --env SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES=60 \
  --env SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW=true \
  --env SGLANG_DIFFUSION_CACHE_ROOT=/results/sgl-diffusion-cache \
  --env SGLANG_DIFFUSION_STAGE_LOGGING=true \
  --entrypoint sleep \
  "${base_image}" infinity

docker exec "${runtime}" bash -lc '
  set -euo pipefail
  test "$(cat .source_sha)" = afc619cc79a2960f9cab53b3823d904672f3c5c0
  python3 -m pip install -e "python[diffusion]" --root-user-action=ignore
  python3 -m pip install \
    --force-reinstall \
    --no-deps \
    --index-url https://flashinfer.ai/whl/cu130 \
    "flashinfer-jit-cache==0.6.12+cu130"
  pytest -q \
    python/sglang/multimodal_gen/test/unit/test_flash_attention_num_splits.py \
    python/sglang/multimodal_gen/test/unit/test_usp_benchmark_bypass.py
'
