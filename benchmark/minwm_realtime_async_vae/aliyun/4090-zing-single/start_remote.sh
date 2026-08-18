#!/usr/bin/env bash
set -Eeuo pipefail

ALIYUN_REGION="${ALIYUN_REGION:-cn-beijing}"
ALIYUN_ZONE="${ALIYUN_ZONE:-cn-beijing-i}"
OSS_ENDPOINT="${OSS_ENDPOINT:-oss-cn-beijing-internal.aliyuncs.com}"
OSS_BUCKET="${OSS_BUCKET:-seedleap-sglang-rtx6000-beijing-20260813}"
OSS_MODEL_URI="${OSS_MODEL_URI:-oss://${OSS_BUCKET}/world-model/minwm/serving-artifacts/wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2/gs3200-ema-student-v1/model/}"
CODE_OVERLAY_OSS_URI="${CODE_OVERLAY_OSS_URI:?set CODE_OVERLAY_OSS_URI}"

ACR_INSTANCE_ID="${ACR_INSTANCE_ID:-cri-ghpj9pt8jwhxdk0e}"
ACR_REGISTRY="${ACR_REGISTRY:-loopit-registry-bj-registry.cn-beijing.cr.aliyuncs.com}"
ACR_REPOSITORY="${ACR_REPOSITORY:-minwm/sglang-minwm-realtime}"
GPU_IMAGE="${GPU_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:gpu-c7ae70a65a2d}"
GATEWAY_IMAGE="${GATEWAY_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:gateway-c7ae70a65a2d}"
WEBUI_IMAGE="${WEBUI_IMAGE:-${GPU_IMAGE}}"
CONTROL_IMAGE="${CONTROL_IMAGE:-${GPU_IMAGE}}"

MODEL_ID="${MODEL_ID:-wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2}"
MODEL_REVISION="${MODEL_REVISION:-gs3200-ema-student-v1}"
VAE_FINGERPRINT="${VAE_FINGERPRINT:-taew2_2-d053e216}"
MODEL_DIR="${MODEL_DIR:-/data/zing-realtime/model-cache/zing/model}"
BASE_DIR="${BASE_DIR:-/data/zing-realtime}"
DOCKER_NETWORK="${DOCKER_NETWORK:-zing-realtime}"
PUBLIC_WEB_PORT="${PUBLIC_WEB_PORT:-80}"

UI_CONFIG_JSON="${UI_CONFIG_JSON:-{\"generationModes\":[\"i2v\"],\"defaultGenerationMode\":\"i2v\",\"modelSlots\":[\"minwm\"],\"lockModelSlots\":true,\"size\":\"832x480\",\"targetFps\":24,\"sessionMaxLifetimeSeconds\":70,\"playbackAckEnabled\":false,\"h264WebSocketEnabled\":true,\"h264CompressedBitrateKbps\":3000,\"h264CompressedCrf\":20,\"h264CompressedPreset\":\"fast\",\"h264CompressedGopSeconds\":2,\"h264CompressedVbvBufferMs\":250,\"h264WebSocketLiveEdgeTargetMs\":80,\"h264WebSocketSeekThresholdMs\":260,\"singleExperience\":false,\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"size\":\"832x480\",\"targetFps\":24,\"sinkSize\":8,\"windowFrames\":32,\"continuous\":true,\"h264StartupDropFrames\":0}}}}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

configure_aliyun_cli() {
  require_cmd aliyun
  if [[ -n "${ALIYUN_ACCESS_KEY_ID:-}" && -n "${ALIYUN_ACCESS_KEY_SECRET:-}" ]]; then
    aliyun configure set \
      --profile default \
      --mode AK \
      --region "${ALIYUN_REGION}" \
      --access-key-id "${ALIYUN_ACCESS_KEY_ID}" \
      --access-key-secret "${ALIYUN_ACCESS_KEY_SECRET}" >/dev/null
  fi
}

ensure_data_mount() {
  mkdir -p /data "${BASE_DIR}"
  local root_source
  root_source="$(findmnt -n -o SOURCE / || true)"
  local candidate=""
  while read -r name type size mountpoint; do
    [[ "${type}" == "disk" ]] || continue
    [[ -z "${mountpoint}" ]] || continue
    [[ "/dev/${name}" != "${root_source}" ]] || continue
    if (( size >= 150 * 1024 * 1024 * 1024 )); then
      candidate="/dev/${name}"
      break
    fi
  done < <(lsblk -b -dn -o NAME,TYPE,SIZE,MOUNTPOINT)

  if [[ -n "${candidate}" ]] && ! findmnt -n /data >/dev/null 2>&1; then
    log "mounting data disk ${candidate} on /data"
    if ! blkid "${candidate}" >/dev/null 2>&1; then
      mkfs.ext4 -F "${candidate}"
    fi
    mount "${candidate}" /data
    local uuid
    uuid="$(blkid -s UUID -o value "${candidate}")"
    if [[ -n "${uuid}" ]] && ! grep -q "${uuid}" /etc/fstab; then
      printf 'UUID=%s /data ext4 defaults,nofail 0 2\n' "${uuid}" >>/etc/fstab
    fi
  fi

  mkdir -p "${BASE_DIR}" "${BASE_DIR}/logs" "${BASE_DIR}/worker-epochs" \
    "${BASE_DIR}/code-overlay" "${BASE_DIR}/model-cache" /data/docker
}

configure_docker_data_root() {
  require_cmd docker
  mkdir -p /etc/docker /data/docker
  local current_root=""
  current_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ "${current_root}" == "/data/docker" ]]; then
    return
  fi
  log "configuring Docker data-root=/data/docker"
  systemctl stop docker || true
  python3 - <<'PY'
import json
from pathlib import Path

path = Path("/etc/docker/daemon.json")
if path.exists():
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        data = {}
else:
    data = {}
data["data-root"] = "/data/docker"
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker >/dev/null || true
  fi
  systemctl start docker
}

oss_cp() {
  aliyun oss cp "$@" --region "${ALIYUN_REGION}" --endpoint "${OSS_ENDPOINT}" -f
}

download_code_overlay() {
  local archive="${BASE_DIR}/code-overlay/current.tar.gz"
  local extract_dir="${BASE_DIR}/code-overlay/current"
  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"
  log "downloading code overlay"
  oss_cp "${CODE_OVERLAY_OSS_URI}" "${archive}"
  tar -xzf "${archive}" -C "${extract_dir}"
  find "${extract_dir}" \( -name '._*' -o -name '.DS_Store' \) -delete
  test -f "${extract_dir}/python/sglang/multimodal_gen/apps/realtime_webui/app.js"
}

download_model() {
  if [[ -f "${MODEL_DIR}/_READY" ]]; then
    log "model cache already ready: ${MODEL_DIR}"
    return
  fi
  log "downloading model artifact to ${MODEL_DIR}"
  rm -rf "${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
  aliyun oss cp "${OSS_MODEL_URI}" "${MODEL_DIR}/" \
    --region "${ALIYUN_REGION}" \
    --endpoint "${OSS_ENDPOINT}" \
    --recursive \
    --update \
    --jobs 16 \
    --parallel 8 \
    -f
  test -f "${MODEL_DIR}/_READY"
}

login_acr() {
  log "logging in to ACR"
  local auth_json user token
  auth_json="$(aliyun cr GetAuthorizationToken --RegionId "${ALIYUN_REGION}" --InstanceId "${ACR_INSTANCE_ID}")"
  user="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["TempUsername"])' <<<"${auth_json}")"
  token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["AuthorizationToken"])' <<<"${auth_json}")"
  printf '%s' "${token}" | docker login --username "${user}" --password-stdin "${ACR_REGISTRY}" >/dev/null
}

pull_images() {
  log "pulling runtime images"
  docker pull "${GPU_IMAGE}"
  docker pull "${WEBUI_IMAGE}"
}

stop_old_containers() {
  log "stopping old zing-realtime containers"
  local gpu
  local names=(
    zing-coordinator
    zing-vae
    zing-vae-heartbeat
    zing-gateway
    zing-webui
  )
  for gpu in 1 2 3 4 5; do
    names+=("zing-denoiser-${gpu}" "zing-denoiser-${gpu}-heartbeat")
  done
  # Remove only legacy Zing workers from the GPUs reserved for the other team.
  # No new worker is started on GPU 6 or 7 below.
  names+=(
    zing-denoiser-6 zing-denoiser-6-heartbeat
    zing-denoiser-7 zing-denoiser-7-heartbeat
  )
  docker rm -f "${names[@]}" >/dev/null 2>&1 || true
  docker network inspect "${DOCKER_NETWORK}" >/dev/null 2>&1 || \
    docker network create "${DOCKER_NETWORK}" >/dev/null
}

common_mounts=()

build_common_mounts() {
  common_mounts=(
    -v "${BASE_DIR}/code-overlay/current/python/sglang/multimodal_gen:/opt/sglang/python/sglang/multimodal_gen:ro"
    -v "${BASE_DIR}/logs:/logs"
  )
}

start_coordinator() {
  log "starting coordinator"
  docker run -d --name zing-coordinator --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    "${common_mounts[@]}" \
    --entrypoint python \
    "${CONTROL_IMAGE}" \
    -m sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server \
    --host=0.0.0.0 \
    --port=18081 \
    --backend=memory \
    --ttl-s=30 \
    --worker-ttl-s=15 \
    --wait-timeout-s=10 \
    --candidate-limit=64 \
    --denoiser-capacity-limit=1 \
    --vae-capacity-limit=16 \
    >"${BASE_DIR}/logs/coordinator.container"
}

start_vae() {
  log "starting async TAEHV VAE on GPU 0"
  docker run -d --name zing-vae --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    --gpus "device=0" \
    -e PYTHONUNBUFFERED=1 \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${GPU_IMAGE}" \
    -m sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server \
      --decoder-backend=taehv \
      --checkpoint-path=/opt/taehv/taew2_2.pth \
      --device=cuda \
      --dtype=bfloat16 \
      --max-sessions=16 \
      --queue-depth-per-session=1 \
      --encoded-frames-per-batch=1 \
      --encode-workers=4 \
      --max-message-mb=64 \
      --host=0.0.0.0 \
      --port=18082 \
    >"${BASE_DIR}/logs/vae.container"

  docker run -d --name zing-vae-heartbeat --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${GPU_IMAGE}" \
    -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat \
      --coordinator-url=http://zing-coordinator:18081 \
      --health-url=http://zing-vae:18082/health \
      --state-url=http://zing-vae:18082/v1/realtime_worker/state \
      --worker-id=zing-vae-0 \
      --worker-epoch-file=/worker-epoch/vae-0 \
      --role=vae \
      --endpoint=ws://zing-vae:18082/v1/realtime_vae/decode \
      --reservation-endpoint=http://zing-vae:18082/v1/realtime_worker \
      --az="${ALIYUN_ZONE}" \
      --capacity=16 \
      --model-revision=all \
      --vae-fingerprint="${VAE_FINGERPRINT}" \
      --interval-s=5 \
    >"${BASE_DIR}/logs/vae-heartbeat.container"
}

start_denoiser() {
  local index="$1"
  local gpu="$2"
  local name="zing-denoiser-${index}"
  log "starting ${name} on GPU ${gpu}"
  docker run -d --name "${name}" --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    --gpus "device=${gpu}" \
    -e PYTHONUNBUFFERED=1 \
    -e SGLANG_DISABLE_PDEATHSIG=1 \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e OPENBLAS_NUM_THREADS=4 \
    -e NUMEXPR_NUM_THREADS=4 \
    -e VECLIB_MAXIMUM_THREADS=4 \
    -e TOKENIZERS_PARALLELISM=false \
    -e WORKER_EPOCH_FILE="/worker-epoch/denoiser-${index}" \
    -e MINWM_ATTENTION_IMPL=dense \
    -e MINWM_PACKED_ATTENTION_DETERMINISTIC=false \
    -e MINWM_NATIVE_COMPONENTS= \
    -e SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false \
    -e NCCL_DEBUG=WARN \
    -e NCCL_PROTO=Simple \
    -v "${MODEL_DIR}:/work/model:ro" \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${GPU_IMAGE}" \
    -m sglang.multimodal_gen.runtime.launch_server \
      --model-path=/work/model \
      --pipeline-class-name=MinWMCausalDMDPipeline \
      --attention-backend=torch_sdpa \
      --performance-mode=speed \
      --num-gpus=1 \
      --tp-size=1 \
      --sp-degree=1 \
      --ulysses-degree=1 \
      --ring-degree=1 \
      --enable-cuda-graph \
      --enable-cfg-parallel=false \
      --enable-torch-compile=false \
      --warmup-mode=off \
      --batching-max-size=1 \
      --batching-delay-ms=2 \
      --realtime-max-sessions=1 \
      --realtime-max-sessions-per-worker=1 \
      --realtime-vae-backend=taehv_remote \
      --realtime-vae-transport=websocket \
      --realtime-session-idle-timeout-s=90 \
      --realtime-session-max-lifetime-s=70 \
      --realtime-admission-wait-s=10 \
      --realtime-causal-sink-size=8 \
      --realtime-causal-kv-cache-num-frames=32 \
      --vae-config.taehv-checkpoint-path=/opt/taehv/taew2_2.pth \
      --vae-cpu-offload=true \
      --host=0.0.0.0 \
      --port=30000 \
    >"${BASE_DIR}/logs/${name}.container"

  docker run -d --name "${name}-heartbeat" --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${GPU_IMAGE}" \
    -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat \
      --coordinator-url=http://zing-coordinator:18081 \
      --health-url=http://${name}:30000/health \
      --state-url=http://${name}:30000/v1/realtime_worker/state \
      --worker-id="${name}" \
      --worker-epoch-file="/worker-epoch/denoiser-${index}" \
      --role=denoiser \
      --endpoint=ws://${name}:30000/v1/realtime_video/generate \
      --reservation-endpoint=http://${name}:30000/v1/realtime_worker \
      --az="${ALIYUN_ZONE}" \
      --capacity=1 \
      --model-revision="${MODEL_ID}" \
      --vae-fingerprint="${VAE_FINGERPRINT}" \
      --interval-s=5 \
    >"${BASE_DIR}/logs/${name}-heartbeat.container"
}

start_gateway() {
  log "starting gateway"
  docker run -d --name zing-gateway --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${CONTROL_IMAGE}" \
      -m sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server \
      --host=0.0.0.0 \
      --port=18080 \
      --coordinator-url=http://zing-coordinator:18081 \
      --model-revision="${MODEL_ID}" \
      --vae-fingerprint="${VAE_FINGERPRINT}" \
      --internal-output-url=ws://zing-gateway:18080/v1/internal/realtime_output \
      --output-queue-depth=64 \
      --output-enqueue-timeout-s=0 \
      --output-drain-timeout-s=70 \
      --lease-renew-interval-s=10 \
      --release-grace-s=0.5 \
      --max-admission-waiters=64 \
      --ui-config-json="${UI_CONFIG_JSON}" \
    >"${BASE_DIR}/logs/gateway.container"
}

start_webui() {
  log "starting webui on host port ${PUBLIC_WEB_PORT}"
  docker run -d --name zing-webui --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -p "${PUBLIC_WEB_PORT}:18080" \
    -e PYTHONUNBUFFERED=1 \
    -e WEBUI_PORT=18080 \
    -e REALTIME_UPSTREAM_HTTP=http://zing-gateway:18080 \
    -e REALTIME_UPSTREAM_WS=ws://zing-gateway:18080 \
    -e MINWM_UPSTREAM_HTTP=http://zing-gateway:18080/backends/minwm \
    -e MINWM_UPSTREAM_WS=ws://zing-gateway:18080/backends/minwm \
    -e VIDEO_PROMPT_REWRITE_PROVIDER=local \
    -e REALTIME_UI_CONFIG_JSON="${UI_CONFIG_JSON}" \
    "${common_mounts[@]}" \
    --entrypoint python \
    "${WEBUI_IMAGE}" \
    /opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui/server.py \
    >"${BASE_DIR}/logs/webui.container"
}

wait_http() {
  local url="$1"
  local deadline=$((SECONDS + 900))
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 5
  done
}

wait_container_http() {
  local name="$1"
  local url="$2"
  local deadline=$((SECONDS + 900))
  until docker exec "${name}" python3 -c \
    "import urllib.request; urllib.request.urlopen('${url}', timeout=2).read()" \
    >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "timed out waiting for ${name}: ${url}" >&2
      docker logs --tail 120 "${name}" 2>&1 || true
      return 1
    fi
    sleep 5
  done
}

show_failed_logs() {
  local name
  for name in $(docker ps -a --filter 'name=^/zing-' --format '{{.Names}}'); do
    if [[ "$(docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)" != "true" ]]; then
      log "${name} is not running; recent logs follow"
      docker logs --tail 120 "${name}" 2>&1 || true
    fi
  done
}

print_status() {
  log "container status"
  docker ps --filter 'name=^/zing-' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  log "gpu status"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
}

main() {
  configure_aliyun_cli
  ensure_data_mount
  configure_docker_data_root
  download_code_overlay
  download_model
  login_acr
  pull_images
  stop_old_containers
  build_common_mounts
  start_coordinator
  wait_container_http zing-coordinator http://127.0.0.1:18081/healthz
  start_vae
  wait_container_http zing-vae http://127.0.0.1:18082/health
  # GPU 6 and 7 are reserved for another team workload on this shared host.
  for gpu in 1 2 3 4 5; do
    start_denoiser "${gpu}" "${gpu}"
  done
  for gpu in 1 2 3 4 5; do
    wait_container_http "zing-denoiser-${gpu}" http://127.0.0.1:30000/health
  done
  start_gateway
  wait_container_http zing-gateway http://127.0.0.1:18080/healthz
  start_webui
  if ! wait_http "http://127.0.0.1:${PUBLIC_WEB_PORT}/runtime-config.js"; then
    show_failed_logs
    exit 1
  fi
  print_status
  log "webui: http://8.147.109.68/?mode=i2v&playback=smooth_timeline"
}

main "$@"
