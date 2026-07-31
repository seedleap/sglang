#!/usr/bin/env bash
set -euo pipefail

mode="${1:-a2a-on}"
num_splits="${2:-2}"
case "${mode}" in
  a2a-on)
    bypass_usp_a2a=false
    ;;
  zero-a2a)
    bypass_usp_a2a=true
    ;;
  *)
    echo "usage: $0 [a2a-on|zero-a2a] [fa-num-splits]" >&2
    exit 2
    ;;
esac
if ! [[ "${num_splits}" =~ ^[0-9]+$ ]]; then
  echo "fa-num-splits must be a non-negative integer" >&2
  exit 2
fi

runtime="codex-lingbot2-runtime"
results_root=/opt/lingbot2/results

test "$(docker inspect --format '{{.State.Running}}' "${runtime}")" = true
mkdir -p "${results_root}"
docker exec "${runtime}" bash -lc \
  "pkill -TERM -f '[s]glang' >/dev/null 2>&1 || true"
sleep 5
docker exec "${runtime}" bash -lc \
  "pkill -KILL -f '[s]glang' >/dev/null 2>&1 || true"
log_name="server-${mode}-split${num_splits}.log"
rm -f "${results_root}/${log_name}"
rm -rf "${results_root}/perf-${mode}-split${num_splits}"

docker exec --detach \
  --env SGLANG_DIFFUSION_BENCHMARK_BYPASS_USP_A2A="${bypass_usp_a2a}" \
  --env SGLANG_PERF_LOG_DIR="/results/perf-${mode}-split${num_splits}" \
  "${runtime}" bash -lc "exec sglang serve \
    --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
    --revision 59cccf49f2d2dd27418ae7a04b82b10868d455c2 \
    --pipeline-class-name LingBotWorldCausalDMDPipeline \
    --num-gpus 8 \
    --performance-mode speed \
    --tp-size 1 \
    --sp-degree 8 \
    --ulysses-degree 8 \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload false \
    --vae-config.use-parallel-decode true \
    --vae-config.parallel-decode-mode spatial \
    --vae-config.taehv-checkpoint-path /opt/taehv/taew2_1.pth \
    --enable-torch-compile false \
    --enable-layerwise-nvtx-marker false \
    --attention-backend-config lingbot_causal_fa_num_splits=${num_splits} \
    --host 0.0.0.0 \
    --port 30000 > /results/${log_name} 2>&1"
