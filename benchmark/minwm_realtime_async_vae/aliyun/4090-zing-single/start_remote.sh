#!/usr/bin/env bash
set -Eeuo pipefail

ALIYUN_REGION="${ALIYUN_REGION:-cn-wulanchabu}"
ALIYUN_ZONE="${ALIYUN_ZONE:-cn-wulanchabu-a}"
OSS_REGION="${OSS_REGION:-cn-beijing}"
OSS_ENDPOINT="${OSS_ENDPOINT:-oss-cn-beijing.aliyuncs.com}"
OSS_BUCKET="${OSS_BUCKET:-seedleap-sglang-rtx6000-beijing-20260813}"
OSS_MODEL_URI="${OSS_MODEL_URI:-oss://${OSS_BUCKET}/world-model/minwm/serving-artifacts/wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2/gs3200-ema-student-v1/model/}"
CODE_OVERLAY_OSS_URI="${CODE_OVERLAY_OSS_URI:-}"
CODE_OVERLAY_ARCHIVE="${CODE_OVERLAY_ARCHIVE:-}"

ACR_INSTANCE_ID="${ACR_INSTANCE_ID:-cri-ghpj9pt8jwhxdk0e}"
ACR_REGION="${ACR_REGION:-cn-beijing}"
ACR_REGISTRY="${ACR_REGISTRY:-loopit-registry-bj-registry.cn-beijing.cr.aliyuncs.com}"
ACR_REPOSITORY="${ACR_REPOSITORY:-minwm/sglang-minwm-realtime}"
DEFAULT_RUNTIME_IMAGE="${DEFAULT_RUNTIME_IMAGE:-829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-runtime:minwm-r3-20260820}"
DENOISER_IMAGE="${DENOISER_IMAGE:-${DEFAULT_RUNTIME_IMAGE}}"
VAE_IMAGE="${VAE_IMAGE:-${DENOISER_IMAGE}}"
GATEWAY_IMAGE="${GATEWAY_IMAGE:-${DENOISER_IMAGE}}"
WEBUI_IMAGE="${WEBUI_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:webui-prompt-rewriter-20260818}"
CONTROL_IMAGE="${CONTROL_IMAGE:-${DENOISER_IMAGE}}"
AUTO_DERIVE_VAE_FFMPEG_IMAGE="${AUTO_DERIVE_VAE_FFMPEG_IMAGE:-true}"
VAE_FFMPEG_IMAGE="${VAE_FFMPEG_IMAGE:-}"
AWS_ECR_REGION="${AWS_ECR_REGION:-us-east-2}"
ECR_REGISTRY="${ECR_REGISTRY:-829115578968.dkr.ecr.us-east-2.amazonaws.com}"
SKIP_ECR_LOGIN="${SKIP_ECR_LOGIN:-false}"
ALIYUN_ECS_RAM_ROLE="${ALIYUN_ECS_RAM_ROLE:-}"
RELEASED_PUBLIC_WEB_HOST="${RELEASED_PUBLIC_WEB_HOST:-116.62.150.115}"

MODEL_ID="${MODEL_ID:-wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2}"
MODEL_REVISION="${MODEL_REVISION:-gs3200-ema-student-v1}"
VAE_FINGERPRINT="${VAE_FINGERPRINT:-taew2_2-d053e216}"
MODEL_DIR="${MODEL_DIR:-/data/zing-realtime/model-cache/zing/model}"
BASE_DIR="${BASE_DIR:-/data/zing-realtime}"
TAEHV_HOST_DIR="${TAEHV_HOST_DIR:-${BASE_DIR}/taehv}"
TAEHV_HOST_CHECKPOINT_PATH="${TAEHV_HOST_CHECKPOINT_PATH:-${TAEHV_HOST_DIR}/taew2_2.pth}"
CODE_OVERLAY_RELEASE_ID="${CODE_OVERLAY_RELEASE_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
CODE_OVERLAY_RELEASES_DIR="${CODE_OVERLAY_RELEASES_DIR:-${BASE_DIR}/code-overlay/releases}"
ACTIVE_CODE_OVERLAY_DIR="${ACTIVE_CODE_OVERLAY_DIR:-${CODE_OVERLAY_RELEASES_DIR}/${CODE_OVERLAY_RELEASE_ID}}"
CODE_OVERLAY_CURRENT_LINK="${CODE_OVERLAY_CURRENT_LINK:-${BASE_DIR}/code-overlay/current}"
CONTAINER_PYTHONPATH="${CONTAINER_PYTHONPATH:-/opt/sglang/python}"
WEBUI_SECRET_DIR="${WEBUI_SECRET_DIR:-${BASE_DIR}/secrets/realtime-webui}"
WEBUI_GENERATED_DIR="${WEBUI_GENERATED_DIR:-${BASE_DIR}/realtime_webui_generated}"
WEBUI_PROXY_ENV_FILE="${WEBUI_PROXY_ENV_FILE:-${WEBUI_SECRET_DIR}/proxy.env}"
DOCKER_NETWORK="${DOCKER_NETWORK:-zing-realtime}"
PUBLIC_WEB_PORT="${PUBLIC_WEB_PORT:-80}"
PUBLIC_WEB_HOST="${PUBLIC_WEB_HOST:?set PUBLIC_WEB_HOST to the new public host/IP}"
PUBLIC_GATEWAY_PORT="${PUBLIC_GATEWAY_PORT:-18080}"
PUBLIC_GATEWAY_BASE_URL="${PUBLIC_GATEWAY_BASE_URL:-http://${PUBLIC_WEB_HOST}:${PUBLIC_GATEWAY_PORT}}"
USE_DEDICATED_DATA_DISK="${USE_DEDICATED_DATA_DISK:-false}"
START_GPU_WORKERS="${START_GPU_WORKERS:-true}"
DEPLOY_STRATEGY="${DEPLOY_STRATEGY:-rolling}"
ROLLING_DRAIN_TIMEOUT_S="${ROLLING_DRAIN_TIMEOUT_S:-120}"
ROLLING_IDLE_TIMEOUT_S="${ROLLING_IDLE_TIMEOUT_S:-180}"
ROLLING_POLL_INTERVAL_S="${ROLLING_POLL_INTERVAL_S:-2}"
ROLLING_RESTART_COORDINATOR="${ROLLING_RESTART_COORDINATOR:-false}"
ROLLING_RESTART_VAE="${ROLLING_RESTART_VAE:-true}"
ROLLING_RESTART_DENOISERS="${ROLLING_RESTART_DENOISERS:-true}"
ROLLING_RESTART_GATEWAY="${ROLLING_RESTART_GATEWAY:-true}"
ROLLING_RESTART_WEBUI="${ROLLING_RESTART_WEBUI:-true}"
DENOISER_START_MODE="${DENOISER_START_MODE:-profile}"
MINWM_PROFILE="${MINWM_PROFILE:-auto}"
PROFILE_MODEL_PATH="${PROFILE_MODEL_PATH:-/models/minwm-tianpeng-gap12}"
PROFILE_MODEL_HOST_DIR="${PROFILE_MODEL_HOST_DIR:-${MODEL_DIR}}"
PROFILE_MODEL_MOUNT_ENABLED="${PROFILE_MODEL_MOUNT_ENABLED:-true}"
PROFILE_TAEHV_CHECKPOINT_PATH="${PROFILE_TAEHV_CHECKPOINT_PATH:-/models/taehv/taew2_2.pth}"
LEGACY_MODEL_CONTAINER_PATH="${LEGACY_MODEL_CONTAINER_PATH:-/work/model}"
LEGACY_TAEHV_CHECKPOINT_PATH="${LEGACY_TAEHV_CHECKPOINT_PATH:-/opt/taehv/taew2_2.pth}"
VAE_TAEHV_CHECKPOINT_PATH="${VAE_TAEHV_CHECKPOINT_PATH:-${PROFILE_TAEHV_CHECKPOINT_PATH}}"
DENOISER_ATTENTION_BACKEND="${DENOISER_ATTENTION_BACKEND:-fa}"
DENOISER_ATTENTION_IMPL="${DENOISER_ATTENTION_IMPL:-packed}"
DENOISER_CACHE_ROTATED_K="${DENOISER_CACHE_ROTATED_K:-false}"
DENOISER_PERFORMANCE_MODE="${DENOISER_PERFORMANCE_MODE:-speed}"
DENOISER_ENABLE_CUDA_GRAPH="${DENOISER_ENABLE_CUDA_GRAPH:-false}"
DENOISER_PYTORCH_CUDA_ALLOC_CONF="${DENOISER_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REALTIME_WORKER_MAX_CONSUMED_AGE_S="${REALTIME_WORKER_MAX_CONSUMED_AGE_S:-120}"
REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS:-24}"
H264_LIVE_EDGE_TARGET_MS="${H264_LIVE_EDGE_TARGET_MS:-500}"
H264_LIVE_EDGE_SEEK_THRESHOLD_MS="${H264_LIVE_EDGE_SEEK_THRESHOLD_MS:-900}"
GATEWAY_OUTPUT_QUEUE_MAX_MESSAGES="${GATEWAY_OUTPUT_QUEUE_MAX_MESSAGES:-4096}"
GATEWAY_OUTPUT_QUEUE_MAX_BYTES="${GATEWAY_OUTPUT_QUEUE_MAX_BYTES:-67108864}"

UI_CONFIG_JSON="${UI_CONFIG_JSON:-{\"generationModes\":[\"i2v\"],\"defaultGenerationMode\":\"i2v\",\"modelSlots\":[\"minwm\"],\"lockModelSlots\":true,\"size\":\"832x480\",\"targetFps\":${REALTIME_TARGET_FPS},\"sessionMaxLifetimeSeconds\":70,\"playbackAckEnabled\":false,\"h264WebSocketEnabled\":true,\"h264DirectGatewayEnabled\":true,\"h264WebSocketBaseUrl\":\"${PUBLIC_GATEWAY_BASE_URL}\",\"h264CompressedBitrateKbps\":3000,\"h264CompressedCrf\":20,\"h264CompressedPreset\":\"fast\",\"h264CompressedGopSeconds\":2,\"h264CompressedVbvBufferMs\":250,\"h264WebSocketLiveEdgeTargetMs\":${H264_LIVE_EDGE_TARGET_MS},\"h264WebSocketSeekThresholdMs\":${H264_LIVE_EDGE_SEEK_THRESHOLD_MS},\"singleExperience\":false,\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"size\":\"832x480\",\"targetFps\":${REALTIME_TARGET_FPS},\"sinkSize\":8,\"windowFrames\":32,\"continuous\":true,\"h264StartupDropFrames\":0}}}}"

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
    if [[ -n "${ALIYUN_STS_TOKEN:-}" ]]; then
      aliyun configure set \
        --profile default \
        --mode StsToken \
        --region "${ALIYUN_REGION}" \
        --access-key-id "${ALIYUN_ACCESS_KEY_ID}" \
        --access-key-secret "${ALIYUN_ACCESS_KEY_SECRET}" \
        --sts-token "${ALIYUN_STS_TOKEN}" >/dev/null
      return
    fi
    aliyun configure set \
      --profile default \
      --mode AK \
      --region "${ALIYUN_REGION}" \
      --access-key-id "${ALIYUN_ACCESS_KEY_ID}" \
      --access-key-secret "${ALIYUN_ACCESS_KEY_SECRET}" >/dev/null
  elif [[ -n "${ALIYUN_ECS_RAM_ROLE}" ]]; then
    aliyun configure set \
      --profile default \
      --mode EcsRamRole \
      --region "${ALIYUN_REGION}" \
      --ram-role-name "${ALIYUN_ECS_RAM_ROLE}" >/dev/null
  fi
}

ensure_data_mount() {
  mkdir -p /data "${BASE_DIR}"
  if [[ "${USE_DEDICATED_DATA_DISK}" != "true" ]]; then
    log "using the root filesystem for /data; dedicated-disk discovery is disabled"
    mkdir -p "${BASE_DIR}" "${BASE_DIR}/logs" "${BASE_DIR}/worker-epochs" \
      "${BASE_DIR}/code-overlay" "${CODE_OVERLAY_RELEASES_DIR}" "${BASE_DIR}/model-cache" \
      "${TAEHV_HOST_DIR}" "${WEBUI_SECRET_DIR}" "${WEBUI_GENERATED_DIR}" /data/docker
    return
  fi
  local root_source
  root_source="$(findmnt -n -o SOURCE / || true)"
  local candidate=""
  while read -r name type size mountpoint; do
    [[ "${type}" == "disk" ]] || continue
    [[ -z "${mountpoint}" ]] || continue
    [[ "/dev/${name}" != "${root_source}" ]] || continue
    if lsblk -nr -o MOUNTPOINT "/dev/${name}" | grep -q '[^[:space:]]'; then
      continue
    fi
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
    "${BASE_DIR}/code-overlay" "${CODE_OVERLAY_RELEASES_DIR}" "${BASE_DIR}/model-cache" \
    "${TAEHV_HOST_DIR}" "${WEBUI_SECRET_DIR}" "${WEBUI_GENERATED_DIR}" /data/docker
}

ensure_taehv_checkpoint() {
  if [[ -s "${TAEHV_HOST_CHECKPOINT_PATH}" ]]; then
    log "TAEHV checkpoint ready: ${TAEHV_HOST_CHECKPOINT_PATH}"
    return
  fi

  log "seeding TAEHV checkpoint at ${TAEHV_HOST_CHECKPOINT_PATH}"
  local source_path=""
  source_path="$(find /data/docker/rootfs/overlayfs -path '*/opt/taehv/taew2_2.pth' -type f -print -quit 2>/dev/null || true)"
  if [[ -z "${source_path}" ]]; then
    echo "could not find taew2_2.pth in existing Docker layers; set TAEHV_HOST_CHECKPOINT_PATH to a valid checkpoint" >&2
    return 1
  fi
  mkdir -p "${TAEHV_HOST_DIR}"
  cp -f "${source_path}" "${TAEHV_HOST_CHECKPOINT_PATH}"
  chmod 0644 "${TAEHV_HOST_CHECKPOINT_PATH}"
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
  aliyun oss cp "$@" --region "${OSS_REGION}" --endpoint "${OSS_ENDPOINT}" -f
}

download_code_overlay() {
  local archive="${BASE_DIR}/code-overlay/${CODE_OVERLAY_RELEASE_ID}.tar.gz"
  local extract_dir="${ACTIVE_CODE_OVERLAY_DIR}"
  mkdir -p "${BASE_DIR}/code-overlay" "${CODE_OVERLAY_RELEASES_DIR}"
  if [[ -e "${extract_dir}" ]]; then
    rm -rf "${extract_dir}"
  fi
  mkdir -p "${extract_dir}"
  if [[ -n "${CODE_OVERLAY_ARCHIVE}" ]]; then
    log "installing local code overlay ${CODE_OVERLAY_ARCHIVE}"
    cp "${CODE_OVERLAY_ARCHIVE}" "${archive}"
  elif [[ -n "${CODE_OVERLAY_OSS_URI}" ]]; then
    log "downloading code overlay"
    oss_cp "${CODE_OVERLAY_OSS_URI}" "${archive}"
  else
    echo "set CODE_OVERLAY_ARCHIVE or CODE_OVERLAY_OSS_URI" >&2
    exit 1
  fi
  tar -xzf "${archive}" -C "${extract_dir}"
  find "${extract_dir}" \( -name '._*' -o -name '.DS_Store' \) -delete
  mkdir -p "${extract_dir}/python/sglang/multimodal_gen/apps/realtime_webui_generated"
  test -f "${extract_dir}/python/sglang/multimodal_gen/apps/realtime_webui/app.js"
  printf '%s\n' "${extract_dir}" >"${BASE_DIR}/code-overlay/current-release"
  if [[ -L "${CODE_OVERLAY_CURRENT_LINK}" || ! -e "${CODE_OVERLAY_CURRENT_LINK}" ]]; then
    ln -sfn "${extract_dir}" "${CODE_OVERLAY_CURRENT_LINK}"
  else
    log "leaving existing code overlay directory in place: ${CODE_OVERLAY_CURRENT_LINK}; active release is ${extract_dir}"
  fi
}

download_model() {
  if [[ "${DENOISER_START_MODE}" == "profile" && "${PROFILE_MODEL_MOUNT_ENABLED}" != "true" ]]; then
    log "skipping OSS model download; profile mode uses image-baked model at ${PROFILE_MODEL_PATH}"
    return
  fi
  if [[ -f "${MODEL_DIR}/_READY" ]]; then
    log "model cache already ready: ${MODEL_DIR}"
    return
  fi
  log "downloading model artifact to ${MODEL_DIR}"
  rm -rf "${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
  aliyun oss cp "${OSS_MODEL_URI}" "${MODEL_DIR}/" \
    --region "${OSS_REGION}" \
    --endpoint "${OSS_ENDPOINT}" \
    --recursive \
    --update \
    --jobs 16 \
    --parallel 8 \
    -f
  test -f "${MODEL_DIR}/_READY"
}

image_uses_registry() {
  local registry="$1"
  local image
  for image in "${DENOISER_IMAGE}" "${VAE_IMAGE}" "${CONTROL_IMAGE}" "${WEBUI_IMAGE}" "${GATEWAY_IMAGE}"; do
    [[ "${image}" == "${registry}/"* ]] && return 0
  done
  return 1
}

login_acr() {
  if ! image_uses_registry "${ACR_REGISTRY}"; then
    log "no ACR images configured; skipping ACR login"
    return
  fi
  log "logging in to ACR"
  local auth_json user token
  auth_json="$(aliyun cr GetAuthorizationToken --RegionId "${ACR_REGION}" --InstanceId "${ACR_INSTANCE_ID}")"
  user="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["TempUsername"])' <<<"${auth_json}")"
  token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["AuthorizationToken"])' <<<"${auth_json}")"
  printf '%s' "${token}" | docker login --username "${user}" --password-stdin "${ACR_REGISTRY}" >/dev/null
}

login_ecr() {
  if ! image_uses_registry "${ECR_REGISTRY}"; then
    log "no ECR images configured; skipping ECR login"
    return
  fi
  if [[ "${SKIP_ECR_LOGIN}" == "true" ]]; then
    log "skipping ECR login because SKIP_ECR_LOGIN=true"
    return
  fi
  if ! command -v aws >/dev/null 2>&1; then
    log "aws CLI is unavailable on the remote host; relying on existing Docker ECR auth"
    return
  fi
  log "logging in to ECR ${ECR_REGISTRY}"
  aws ecr get-login-password --region "${AWS_ECR_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}" >/dev/null
}

pull_images() {
  if [[ "${SKIP_IMAGE_PULL:-false}" == "true" ]]; then
    log "skipping runtime image pulls; using locally verified images"
    return
  fi

  log "pulling runtime images"
  local image
  local pulled=" "
  for image in "${DENOISER_IMAGE}" "${VAE_IMAGE}" "${CONTROL_IMAGE}" "${GATEWAY_IMAGE}" "${WEBUI_IMAGE}"; do
    [[ -n "${image}" ]] || continue
    if [[ "${pulled}" == *" ${image} "* ]]; then
      continue
    fi
    docker pull "${image}"
    pulled+="${image} "
  done
}

vae_ffmpeg_image_tag() {
  local image="$1"
  if [[ -n "${VAE_FFMPEG_IMAGE}" ]]; then
    printf '%s\n' "${VAE_FFMPEG_IMAGE}"
    return
  fi
  if [[ "${image}" == *@* ]]; then
    local digest_hash
    digest_hash="$(printf '%s' "${image}" | sha256sum | awk '{print substr($1,1,12)}')"
    printf '%s/%s:vae-ffmpeg-local-%s\n' "${ACR_REGISTRY}" "${ACR_REPOSITORY}" "${digest_hash}"
    return
  fi
  if [[ "${image##*/}" == *:* ]]; then
    printf '%s-ffmpeg-local\n' "${image}"
  else
    printf '%s:ffmpeg-local\n' "${image}"
  fi
}

image_has_ffmpeg() {
  local image="$1"
  docker run --rm --entrypoint /bin/sh "${image}" -lc \
    'command -v ffmpeg >/dev/null 2>&1' >/dev/null 2>&1
}

ensure_vae_ffmpeg_image() {
  if [[ "${START_GPU_WORKERS}" != "true" || "${AUTO_DERIVE_VAE_FFMPEG_IMAGE}" != "true" ]]; then
    return
  fi
  if image_has_ffmpeg "${VAE_IMAGE}"; then
    log "VAE image already contains ffmpeg: ${VAE_IMAGE}"
    return
  fi

  local target_image
  target_image="$(vae_ffmpeg_image_tag "${VAE_IMAGE}")"
  if image_has_ffmpeg "${target_image}"; then
    log "using existing local VAE ffmpeg image: ${target_image}"
    VAE_IMAGE="${target_image}"
    return
  fi

  log "VAE image ${VAE_IMAGE} does not contain ffmpeg; deriving local image ${target_image}"
  local cid
  cid="$(docker create --entrypoint /bin/sh "${VAE_IMAGE}" -lc \
    'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*')"
  if ! docker start -a "${cid}"; then
    docker rm -f "${cid}" >/dev/null 2>&1 || true
    echo "failed to derive VAE ffmpeg image from ${VAE_IMAGE}" >&2
    return 1
  fi
  docker commit "${cid}" "${target_image}" >/dev/null
  docker rm -f "${cid}" >/dev/null 2>&1 || true
  if ! image_has_ffmpeg "${target_image}"; then
    echo "derived image ${target_image} still does not provide ffmpeg" >&2
    return 1
  fi
  VAE_IMAGE="${target_image}"
}

ensure_docker_network() {
  docker network inspect "${DOCKER_NETWORK}" >/dev/null 2>&1 || \
    docker network create "${DOCKER_NETWORK}" >/dev/null
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
    zing-h264-bridge
    zing-h264ws-bridge
    zing-webui-h264-bridge
    torch-cu128-test
  )
  for gpu in 1 2 3 4 5 6 7; do
    names+=("zing-denoiser-${gpu}" "zing-denoiser-${gpu}-heartbeat")
  done
  docker rm -f "${names[@]}" >/dev/null 2>&1 || true
  ensure_docker_network
}

common_mounts=()

build_common_mounts() {
  common_mounts=(
    -v "${ACTIVE_CODE_OVERLAY_DIR}/python/sglang/multimodal_gen:/opt/sglang/python/sglang/multimodal_gen:ro"
    -v "${TAEHV_HOST_DIR}:/models/taehv:ro"
    -v "${BASE_DIR}/logs:/logs"
  )
  if [[ "${DENOISER_START_MODE}" == "profile" && "${PROFILE_MODEL_MOUNT_ENABLED}" == "true" ]]; then
    common_mounts+=(-v "${PROFILE_MODEL_HOST_DIR}:${PROFILE_MODEL_PATH}:ro")
  fi
}

start_coordinator() {
  log "starting coordinator"
  docker run -d --name zing-coordinator --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
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
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${VAE_IMAGE}" \
    -m sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server \
      --decoder-backend=taehv \
      --checkpoint-path="${VAE_TAEHV_CHECKPOINT_PATH}" \
      --device=cuda \
      --dtype=bfloat16 \
      --max-sessions=16 \
      --max-consumed-age-s="${REALTIME_WORKER_MAX_CONSUMED_AGE_S}" \
      --queue-depth-per-session=1 \
      --encoded-frames-per-batch=1 \
      --encode-workers=4 \
      --direct-h264-output \
      --direct-h264-trigger-output-format=jpeg \
      --h264-fps="${REALTIME_TARGET_FPS}" \
      --h264-threads=2 \
      --h264-preset=fast \
      --h264-profile=main \
      --h264-crf=20 \
      --h264-bitrate-kbps=3000 \
      --h264-vbv-buffer-ms=250 \
      --h264-gop-seconds=2 \
      --h264-max-queued-frames=24 \
      --h264-max-frame-age-ms=250 \
      --h264-live-edge-frames=6 \
      --h264-startup-drop-frames=0 \
      --max-message-mb=64 \
      --host=0.0.0.0 \
      --port=18082 \
    >"${BASE_DIR}/logs/vae.container"

  docker run -d --name zing-vae-heartbeat --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${CONTROL_IMAGE}" \
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
  local gpu_devices="$2"
  local name="zing-denoiser-${index}"
  local launch_module=""
  local model_mount_args=()
  local launch_args=()

  case "${DENOISER_START_MODE}" in
    profile)
      log "starting ${name} with profile=${MINWM_PROFILE} (SP1) on GPU ${gpu_devices}"
      launch_module="sglang.multimodal_gen.tools.minwm_profile_launcher"
      launch_args=(
        --profile "${MINWM_PROFILE}"
        --taehv-checkpoint-path "${PROFILE_TAEHV_CHECKPOINT_PATH}"
        --
        --model-path "${PROFILE_MODEL_PATH}"
        --batching-max-size 1
        --batching-delay-ms 2
        --realtime-max-sessions 1
        --realtime-max-sessions-per-worker 1
        --realtime-vae-backend taehv_remote
        --realtime-vae-transport websocket
        --realtime-session-idle-timeout-s 90
        --realtime-session-max-lifetime-s 70
        --realtime-worker-max-consumed-age-s "${REALTIME_WORKER_MAX_CONSUMED_AGE_S}"
        --realtime-admission-wait-s 10
        --host 0.0.0.0
        --port 30000
      )
      ;;
    legacy)
      log "starting ${name} with legacy launch_server (SP1) on GPU ${gpu_devices}"
      launch_module="sglang.multimodal_gen.runtime.launch_server"
      model_mount_args=(-v "${MODEL_DIR}:${LEGACY_MODEL_CONTAINER_PATH}:ro")
      launch_args=(
        --model-path "${LEGACY_MODEL_CONTAINER_PATH}"
        --pipeline-class-name MinWMCausalDMDPipeline
        --attention-backend "${DENOISER_ATTENTION_BACKEND}"
        --performance-mode "${DENOISER_PERFORMANCE_MODE}"
        --num-gpus 1
        --tp-size 1
        --sp-degree 1
        --ulysses-degree 1
        --ring-degree 1
        --enable-cuda-graph "${DENOISER_ENABLE_CUDA_GRAPH}"
        --enable-cfg-parallel false
        --enable-torch-compile false
        --warmup-mode off
        --batching-max-size 1
        --batching-delay-ms 2
        --realtime-max-sessions 1
        --realtime-max-sessions-per-worker 1
        --realtime-vae-backend taehv_remote
        --realtime-vae-transport websocket
        --realtime-session-idle-timeout-s 90
        --realtime-session-max-lifetime-s 70
        --realtime-worker-max-consumed-age-s "${REALTIME_WORKER_MAX_CONSUMED_AGE_S}"
        --realtime-admission-wait-s 10
        --realtime-causal-sink-size 8
        --realtime-causal-kv-cache-num-frames 32
        --vae-config.taehv-checkpoint-path "${LEGACY_TAEHV_CHECKPOINT_PATH}"
        --vae-cpu-offload true
        --host 0.0.0.0
        --port 30000
      )
      ;;
    *)
      echo "unsupported DENOISER_START_MODE=${DENOISER_START_MODE}; expected profile or legacy" >&2
      exit 1
      ;;
  esac

  docker run -d --name "${name}" --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    --gpus "device=${gpu_devices}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${DENOISER_PYTORCH_CUDA_ALLOC_CONF}" \
    -e SGLANG_DISABLE_PDEATHSIG=1 \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e OPENBLAS_NUM_THREADS=4 \
    -e NUMEXPR_NUM_THREADS=4 \
    -e VECLIB_MAXIMUM_THREADS=4 \
    -e TOKENIZERS_PARALLELISM=false \
    -e WORKER_EPOCH_FILE="/worker-epoch/denoiser-${index}" \
    -e MINWM_ATTENTION_IMPL="${DENOISER_ATTENTION_IMPL}" \
    -e MINWM_CACHE_ROTATED_K="${DENOISER_CACHE_ROTATED_K}" \
    -e MINWM_PACKED_ATTENTION_DETERMINISTIC=false \
    -e MINWM_NATIVE_COMPONENTS= \
    -e SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false \
    -e NCCL_DEBUG=WARN \
    -e NCCL_PROTO=Simple \
    "${model_mount_args[@]}" \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${DENOISER_IMAGE}" \
    -m "${launch_module}" \
    "${launch_args[@]}" \
    >"${BASE_DIR}/logs/${name}.container"

  docker run -d --name "${name}-heartbeat" --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    -v "${BASE_DIR}/worker-epochs:/worker-epoch" \
    "${common_mounts[@]}" \
    --entrypoint python3 \
    "${CONTROL_IMAGE}" \
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
    -p "${PUBLIC_GATEWAY_PORT}:18080" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
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
      --output-queue-max-messages="${GATEWAY_OUTPUT_QUEUE_MAX_MESSAGES}" \
      --output-queue-max-bytes="${GATEWAY_OUTPUT_QUEUE_MAX_BYTES}" \
      --output-drain-timeout-s=70 \
      --lease-renew-interval-s=10 \
      --release-grace-s=0.5 \
      --max-admission-waiters=64 \
      --ui-config-json="${UI_CONFIG_JSON}" \
    >"${BASE_DIR}/logs/gateway.container"
}

start_webui() {
  log "starting webui on host port ${PUBLIC_WEB_PORT}"
  local proxy_env_args=()
  if [[ -f "${WEBUI_PROXY_ENV_FILE}" ]]; then
    proxy_env_args=(--env-file "${WEBUI_PROXY_ENV_FILE}")
  fi
  docker run -d --name zing-webui --restart unless-stopped \
    --network "${DOCKER_NETWORK}" \
    -p "${PUBLIC_WEB_PORT}:18080" \
    "${proxy_env_args[@]}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
    -e WEBUI_PORT=18080 \
    -e REALTIME_UPSTREAM_HTTP=http://zing-gateway:18080 \
    -e REALTIME_UPSTREAM_WS=ws://zing-gateway:18080 \
    -e MINWM_UPSTREAM_HTTP=http://zing-gateway:18080/backends/minwm \
    -e MINWM_UPSTREAM_WS=ws://zing-gateway:18080/backends/minwm \
    -e VIDEO_PROMPT_REWRITE_PROVIDER=local \
    -e VIDEO_PROMPT_REWRITE_CREDENTIALS=/run/secrets/realtime-webui/prompt-rewriter-vertex.json \
    -e CREATE_WORLD_IMAGE_CONFIG=/run/secrets/realtime-webui/world-image-model-config.json \
    -e REALTIME_UI_CONFIG_JSON="${UI_CONFIG_JSON}" \
    "${common_mounts[@]}" \
    -v "${WEBUI_SECRET_DIR}:/run/secrets/realtime-webui:ro" \
    -v "${WEBUI_GENERATED_DIR}:/opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui_generated" \
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

container_running() {
  local name="$1"
  [[ "$(docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)" == "true" ]]
}

drain_worker_container() {
  local name="$1"
  local port="$2"
  if ! container_running "${name}"; then
    return 0
  fi

  log "draining ${name}"
  if ! docker exec "${name}" python3 - "${port}" "${ROLLING_DRAIN_TIMEOUT_S}" <<'PY'
import json
import sys
import time
import urllib.request

port = sys.argv[1]
timeout_s = float(sys.argv[2])
body = json.dumps({"deadline_unix_s": time.time() + timeout_s}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/realtime_worker/drain",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
urllib.request.urlopen(request, timeout=5).read()
PY
  then
    if worker_container_idle "${name}" "${port}" >/dev/null 2>&1; then
      log "${name} did not accept a drain request, but it is already idle; continuing"
      return 0
    fi
    return 1
  fi
}

worker_container_idle() {
  local name="$1"
  local port="$2"
  if ! container_running "${name}"; then
    return 0
  fi

  docker exec "${name}" python3 - "${port}" <<'PY'
import json
import sys
import urllib.request

port = sys.argv[1]
state = json.loads(
    urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/realtime_worker/state",
        timeout=3,
    ).read()
)
busy_keys = (
    "active_sessions",
    "reserved_sessions",
    "runnable_sessions",
    "blocked_sessions",
    "queued_sessions",
    "pending_sessions",
    "pending_inputs",
    "pending_outputs",
    "input_queue_size",
    "output_queue_size",
    "queue_size",
    "queue_depth",
)
busy = {key: int(state.get(key) or 0) for key in busy_keys if key in state}
if any(value != 0 for value in busy.values()):
    print(json.dumps({"lifecycle": state.get("lifecycle"), "busy": busy}, sort_keys=True))
    sys.exit(1)
PY
}

wait_worker_idle() {
  local name="$1"
  local port="$2"
  local deadline=$((SECONDS + ROLLING_IDLE_TIMEOUT_S))
  until worker_container_idle "${name}" "${port}" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "timed out waiting for ${name} to become idle before rolling restart" >&2
      worker_container_idle "${name}" "${port}" || true
      docker logs --tail 120 "${name}" 2>&1 || true
      return 1
    fi
    sleep "${ROLLING_POLL_INTERVAL_S}"
  done
}

wait_all_workers_idle() {
  local index
  wait_worker_idle zing-vae 18082
  for index in 1 2 3 4 5 6 7; do
    wait_worker_idle "zing-denoiser-${index}" 30000
  done
}

restart_coordinator_rolling() {
  if container_running zing-coordinator && [[ "${ROLLING_RESTART_COORDINATOR}" != "true" ]]; then
    log "keeping existing coordinator during rolling deploy"
    return
  fi

  if container_running zing-coordinator; then
    log "waiting for workers to become idle before coordinator restart"
    wait_all_workers_idle
  fi
  docker rm -f zing-coordinator >/dev/null 2>&1 || true
  start_coordinator
  wait_container_http zing-coordinator http://127.0.0.1:18081/healthz
}

restart_vae_rolling() {
  if container_running zing-vae; then
    drain_worker_container zing-vae 18082
    wait_worker_idle zing-vae 18082
  fi
  docker rm -f zing-vae-heartbeat zing-vae >/dev/null 2>&1 || true
  start_vae
  wait_container_http zing-vae http://127.0.0.1:18082/health
}

restart_denoiser_rolling() {
  local index="$1"
  local name="zing-denoiser-${index}"
  if container_running "${name}"; then
    drain_worker_container "${name}" 30000
    wait_worker_idle "${name}" 30000
  fi
  docker rm -f "${name}-heartbeat" "${name}" >/dev/null 2>&1 || true
  start_denoiser "${index}" "${index}"
  wait_container_http "${name}" http://127.0.0.1:30000/health
}

restart_gateway_rolling() {
  if container_running zing-gateway; then
    log "waiting for workers to become idle before gateway restart"
    wait_all_workers_idle
  fi
  docker rm -f zing-gateway >/dev/null 2>&1 || true
  start_gateway
  wait_container_http zing-gateway http://127.0.0.1:18080/healthz
}

restart_webui_rolling() {
  docker rm -f zing-webui >/dev/null 2>&1 || true
  start_webui
  wait_http "http://127.0.0.1:${PUBLIC_WEB_PORT}/runtime-config.js"
}

remove_retired_bridge_containers() {
  docker rm -f \
    zing-h264-bridge \
    zing-h264ws-bridge \
    zing-webui-h264-bridge \
    torch-cu128-test \
    >/dev/null 2>&1 || true
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

replace_deploy() {
  stop_old_containers
  build_common_mounts
  start_coordinator
  wait_container_http zing-coordinator http://127.0.0.1:18081/healthz
  if [[ "${START_GPU_WORKERS}" == "true" ]]; then
    start_vae
    wait_container_http zing-vae http://127.0.0.1:18082/health
    for index in 1 2 3 4 5 6 7; do
      start_denoiser "${index}" "${index}"
    done
    for index in 1 2 3 4 5 6 7; do
      wait_container_http "zing-denoiser-${index}" http://127.0.0.1:30000/health
    done
  else
    log "skipping VAE and denoiser startup because START_GPU_WORKERS=${START_GPU_WORKERS}"
  fi
  start_gateway
  wait_container_http zing-gateway http://127.0.0.1:18080/healthz
  start_webui
  wait_http "http://127.0.0.1:${PUBLIC_WEB_PORT}/runtime-config.js"
}

rolling_deploy() {
  ensure_docker_network
  remove_retired_bridge_containers
  build_common_mounts
  restart_coordinator_rolling

  if [[ "${START_GPU_WORKERS}" == "true" ]]; then
    if [[ "${ROLLING_RESTART_VAE}" == "true" ]] || ! container_running zing-vae; then
      restart_vae_rolling
    else
      log "keeping existing VAE during rolling deploy"
    fi

    if [[ "${ROLLING_RESTART_DENOISERS}" == "true" ]]; then
      local index
      for index in 1 2 3 4 5 6 7; do
        restart_denoiser_rolling "${index}"
      done
    else
      log "keeping existing denoisers during rolling deploy"
    fi
  else
    log "skipping VAE and denoiser startup because START_GPU_WORKERS=${START_GPU_WORKERS}"
  fi

  if [[ "${ROLLING_RESTART_GATEWAY}" == "true" ]] || ! container_running zing-gateway; then
    restart_gateway_rolling
  else
    log "keeping existing gateway during rolling deploy"
  fi

  if [[ "${ROLLING_RESTART_WEBUI}" == "true" ]] || ! container_running zing-webui; then
    restart_webui_rolling
  else
    log "keeping existing webui during rolling deploy"
  fi
}

validate_deployment_config() {
  case "${DEPLOY_STRATEGY}" in
    replace|rolling) ;;
    *)
      echo "unsupported DEPLOY_STRATEGY=${DEPLOY_STRATEGY}; expected replace or rolling" >&2
      exit 1
      ;;
  esac
  case "${DENOISER_START_MODE}" in
    profile|legacy) ;;
    *)
      echo "unsupported DENOISER_START_MODE=${DENOISER_START_MODE}; expected profile or legacy" >&2
      exit 1
      ;;
  esac
  log "deploy strategy: ${DEPLOY_STRATEGY}"
  log "denoiser startup mode: ${DENOISER_START_MODE}"
  if [[ "${DENOISER_START_MODE}" == "profile" ]]; then
    log "profile launcher: profile=${MINWM_PROFILE}, model=${PROFILE_MODEL_PATH}, taehv=${PROFILE_TAEHV_CHECKPOINT_PATH}"
  fi
}

main() {
  if [[ "${PUBLIC_WEB_HOST}" == "${RELEASED_PUBLIC_WEB_HOST}" ]]; then
    echo "${RELEASED_PUBLIC_WEB_HOST} was released; set PUBLIC_WEB_HOST to the new Aliyun host." >&2
    exit 1
  fi

  validate_deployment_config
  configure_aliyun_cli
  ensure_data_mount
  configure_docker_data_root
  download_code_overlay
  if [[ "${START_GPU_WORKERS}" == "true" ]]; then
    ensure_taehv_checkpoint
    download_model
  else
    log "skipping model download because START_GPU_WORKERS=${START_GPU_WORKERS}"
  fi
  if [[ "${SKIP_IMAGE_PULL:-false}" != "true" ]]; then
    login_acr
    login_ecr
  fi
  pull_images
  ensure_vae_ffmpeg_image
  if [[ "${DEPLOY_STRATEGY}" == "replace" ]]; then
    replace_deploy
  else
    rolling_deploy
  fi
  if ! wait_http "http://127.0.0.1:${PUBLIC_WEB_PORT}/runtime-config.js"; then
    show_failed_logs
    exit 1
  fi
  print_status
  log "webui: http://${PUBLIC_WEB_HOST}/?mode=i2v&playback=smooth_timeline"
  log "gateway: ${PUBLIC_GATEWAY_BASE_URL}"
}

main "$@"
