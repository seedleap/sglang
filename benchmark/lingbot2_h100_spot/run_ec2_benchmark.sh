#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: $0 RESULT_LABEL}"
runtime=codex-lingbot2-runtime

docker exec "${runtime}" python3 /workspace/benchmark_realtime.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --chunks 60 \
  --warmup-chunks 20 \
  --capture-chunk 20 \
  --capture-chunk 21 \
  --capture-dir /results \
  --output "/results/${label}.json"
