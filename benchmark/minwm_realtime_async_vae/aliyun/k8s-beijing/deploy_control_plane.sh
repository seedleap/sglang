#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SSH_HOST="${SSH_HOST:?set SSH_HOST=root@<beijing-control-host>}"
REMOTE_DIR="${REMOTE_DIR:-/root/zing-k8s-beijing}"
NAMESPACE="${NAMESPACE:-minwm-realtime}"
PUBLIC_WEB_HOST="${PUBLIC_WEB_HOST:-${SSH_HOST#*@}}"
PUBLIC_GATEWAY_PORT="${PUBLIC_GATEWAY_PORT:-18080}"
PUBLIC_GATEWAY_BASE_URL="${PUBLIC_GATEWAY_BASE_URL:-http://${PUBLIC_WEB_HOST}:${PUBLIC_GATEWAY_PORT}}"
ACR_INSTANCE_ID="${ACR_INSTANCE_ID:-cri-ghpj9pt8jwhxdk0e}"
ACR_REGION="${ACR_REGION:-cn-beijing}"
ACR_REGISTRY="${ACR_REGISTRY:-loopit-registry-bj-registry.cn-beijing.cr.aliyuncs.com}"
ACR_REPOSITORY="${ACR_REPOSITORY:-minwm/sglang-minwm-realtime}"
GPU_RUNTIME_IMAGE="${GPU_RUNTIME_IMAGE:-${RUNTIME_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:minwm-r3-20260820-amd64}}"
CONTROL_IMAGE="${CONTROL_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:gateway-c7ae70a65a2d}"
WEBUI_IMAGE="${WEBUI_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:webui-c7ae70a65a2d}"
NVIDIA_DEVICE_PLUGIN_IMAGE="${NVIDIA_DEVICE_PLUGIN_IMAGE:-nvcr.io/nvidia/k8s-device-plugin:v0.17.1}"
REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS:-24}"
REALTIME_SIZE="${REALTIME_SIZE:-1280x704}"
H264_LIVE_EDGE_TARGET_MS="${H264_LIVE_EDGE_TARGET_MS:-500}"
H264_LIVE_EDGE_SEEK_THRESHOLD_MS="${H264_LIVE_EDGE_SEEK_THRESHOLD_MS:-900}"
UI_CONFIG_JSON="${UI_CONFIG_JSON:-{\"generationModes\":[\"i2v\"],\"defaultGenerationMode\":\"i2v\",\"modelSlots\":[\"minwm\"],\"lockModelSlots\":true,\"size\":\"${REALTIME_SIZE}\",\"targetFps\":${REALTIME_TARGET_FPS},\"sessionMaxLifetimeSeconds\":70,\"playbackAckEnabled\":false,\"h264WebSocketEnabled\":true,\"h264DirectGatewayEnabled\":true,\"h264WebSocketBaseUrl\":\"${PUBLIC_GATEWAY_BASE_URL}\",\"h264CompressedBitrateKbps\":3000,\"h264CompressedCrf\":20,\"h264CompressedPreset\":\"fast\",\"h264CompressedGopSeconds\":2,\"h264CompressedVbvBufferMs\":250,\"h264WebSocketLiveEdgeTargetMs\":${H264_LIVE_EDGE_TARGET_MS},\"h264WebSocketSeekThresholdMs\":${H264_LIVE_EDGE_SEEK_THRESHOLD_MS},\"singleExperience\":true,\"tabScopedUserIds\":true,\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"size\":\"${REALTIME_SIZE}\",\"targetFps\":${REALTIME_TARGET_FPS},\"sinkSize\":8,\"windowFrames\":32,\"continuous\":true,\"h264StartupDropFrames\":0}}}}"
DENOISER_CAPACITY_LIMIT="${DENOISER_CAPACITY_LIMIT:-15}"
VAE_CAPACITY_LIMIT="${VAE_CAPACITY_LIMIT:-160}"
WEBUI_PROMPT_REWRITER_PATH="${WEBUI_PROMPT_REWRITER_PATH:-/data/zing-realtime/secrets/prompt-rewriter-vertex.json}"
HTTPS_PROXY_SECRET_VALUE="${HTTPS_PROXY_SECRET_VALUE:-${https_proxy:-${HTTPS_PROXY:-}}}"
HTTP_PROXY_SECRET_VALUE="${HTTP_PROXY_SECRET_VALUE:-${http_proxy:-${HTTP_PROXY:-${HTTPS_PROXY_SECRET_VALUE}}}}"
K3S_PAUSE_IMAGE="${K3S_PAUSE_IMAGE:-registry.cn-hangzhou.aliyuncs.com/google_containers/pause:3.10}"
K3S_DATA_DIR="${K3S_DATA_DIR:-/data/k3s}"
KUBELET_ROOT_DIR="${KUBELET_ROOT_DIR:-/data/kubelet}"
K3S_DISABLE_LOCAL_STORAGE="${K3S_DISABLE_LOCAL_STORAGE:-true}"
K3S_COREDNS_IMAGE="${K3S_COREDNS_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/mirrored-coredns-coredns:1.14.6}"
K3S_METRICS_SERVER_IMAGE="${K3S_METRICS_SERVER_IMAGE:-registry.cn-hangzhou.aliyuncs.com/google_containers/metrics-server:v0.9.0}"
K3S_LOCAL_PATH_PROVISIONER_IMAGE="${K3S_LOCAL_PATH_PROVISIONER_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/local-path-provisioner:v0.0.36}"
K3S_BUSYBOX_IMAGE="${K3S_BUSYBOX_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/mirrored-library-busybox:1.37.0}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-1800s}"
APPLY_NVIDIA_DEVICE_PLUGIN="${APPLY_NVIDIA_DEVICE_PLUGIN:-false}"
APPLY_GPU_WORKER_TEMPLATES="${APPLY_GPU_WORKER_TEMPLATES:-true}"
APPLY_RUNTIME_SOURCE_PATCH="${APPLY_RUNTIME_SOURCE_PATCH:-true}"
APPLY_WEBUI_STATIC_PATCH="${APPLY_WEBUI_STATIC_PATCH:-true}"
WEBUI_STATIC_PATCH_NAME="${WEBUI_STATIC_PATCH_NAME:-zing-webui-static-patch}"
WEBUI_STATIC_PATCH_VERSION="${WEBUI_STATIC_PATCH_VERSION:-single-zing-v9-20260822}"
WEBUI_STATIC_SOURCE_DIR="${WEBUI_STATIC_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/apps/realtime_webui}"
RUNTIME_SOURCE_PATCH_VERSION="${RUNTIME_SOURCE_PATCH_VERSION:-runtime-v2-h264-20260822}"
RUNTIME_REALTIME_SOURCE_DIR="${RUNTIME_REALTIME_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/realtime}"
RUNTIME_ENTRYPOINT_SOURCE_DIR="${RUNTIME_ENTRYPOINT_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/entrypoints}"
RUNTIME_UTILS_SOURCE_DIR="${RUNTIME_UTILS_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/utils}"
RUNTIME_SHIMS_SOURCE_DIR="${RUNTIME_SHIMS_SOURCE_DIR:-${SCRIPT_DIR}/runtime-shims}"
H264_FFMPEG_BIN="${H264_FFMPEG_BIN:-/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

render_template() {
  local template="$1"
  local output="$2"
  NAMESPACE="${NAMESPACE}" \
  CONTROL_IMAGE="${CONTROL_IMAGE}" \
  GPU_RUNTIME_IMAGE="${GPU_RUNTIME_IMAGE}" \
  WEBUI_IMAGE="${WEBUI_IMAGE}" \
  NVIDIA_DEVICE_PLUGIN_IMAGE="${NVIDIA_DEVICE_PLUGIN_IMAGE}" \
  REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS}" \
  REALTIME_SIZE="${REALTIME_SIZE}" \
  DENOISER_CAPACITY_LIMIT="${DENOISER_CAPACITY_LIMIT}" \
  VAE_CAPACITY_LIMIT="${VAE_CAPACITY_LIMIT}" \
  UI_CONFIG_JSON="${UI_CONFIG_JSON}" \
  H264_FFMPEG_BIN="${H264_FFMPEG_BIN}" \
  RUNTIME_SOURCE_PATCH_VERSION="${RUNTIME_SOURCE_PATCH_VERSION}" \
  envsubst '${NAMESPACE} ${CONTROL_IMAGE} ${GPU_RUNTIME_IMAGE} ${WEBUI_IMAGE} ${NVIDIA_DEVICE_PLUGIN_IMAGE} ${REALTIME_TARGET_FPS} ${REALTIME_SIZE} ${DENOISER_CAPACITY_LIMIT} ${VAE_CAPACITY_LIMIT} ${UI_CONFIG_JSON} ${H264_FFMPEG_BIN} ${RUNTIME_SOURCE_PATCH_VERSION}' \
    <"${template}" >"${output}"
}

install_k3s() {
  log "ensuring k3s is installed on ${SSH_HOST}"
  ssh -o BatchMode=yes "${SSH_HOST}" "K3S_PAUSE_IMAGE='${K3S_PAUSE_IMAGE}' K3S_DATA_DIR='${K3S_DATA_DIR}' KUBELET_ROOT_DIR='${KUBELET_ROOT_DIR}' K3S_DISABLE_LOCAL_STORAGE='${K3S_DISABLE_LOCAL_STORAGE}' bash -s" <<'REMOTE'
set -Eeuo pipefail
mkdir -p "${K3S_DATA_DIR}" "${KUBELET_ROOT_DIR}"
disable_local_storage_arg=""
if [[ "${K3S_DISABLE_LOCAL_STORAGE}" == "true" ]]; then
  disable_local_storage_arg=" --disable=local-storage"
fi
k3s_exec="server --write-kubeconfig-mode 644 --disable=traefik${disable_local_storage_arg} --pause-image ${K3S_PAUSE_IMAGE} --data-dir ${K3S_DATA_DIR} --kubelet-arg root-dir=${KUBELET_ROOT_DIR}"
if ! command -v k3s >/dev/null 2>&1; then
  curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh \
    | INSTALL_K3S_MIRROR=cn \
      INSTALL_K3S_EXEC="${k3s_exec}" \
      sh -
fi
mkdir -p /etc/systemd/system/k3s.service.d
tmp_unit="$(mktemp)"
{
  printf '%s\n' '[Service]'
  printf '%s\n' 'ExecStart='
  printf '%s\n' "ExecStart=/usr/local/bin/k3s ${k3s_exec}"
} >"${tmp_unit}"
unit_changed=false
if ! cmp -s "${tmp_unit}" /etc/systemd/system/k3s.service.d/10-zing-runtime.conf; then
  mv "${tmp_unit}" /etc/systemd/system/k3s.service.d/10-zing-runtime.conf
  unit_changed=true
else
  rm -f "${tmp_unit}"
fi
systemctl daemon-reload
if [[ "${unit_changed}" == "true" ]] || ! systemctl is-active --quiet k3s; then
  systemctl restart k3s
fi
for _ in $(seq 1 60); do
  kubectl get node >/dev/null 2>&1 && exit 0
  sleep 2
done
systemctl status k3s --no-pager
exit 1
REMOTE
}

configure_k3s_system_images() {
  log "ensuring k3s system images use mainland-accessible mirrors"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "K3S_DATA_DIR='${K3S_DATA_DIR}' \
     K3S_DISABLE_LOCAL_STORAGE='${K3S_DISABLE_LOCAL_STORAGE}' \
     K3S_COREDNS_IMAGE='${K3S_COREDNS_IMAGE}' \
     K3S_METRICS_SERVER_IMAGE='${K3S_METRICS_SERVER_IMAGE}' \
     K3S_LOCAL_PATH_PROVISIONER_IMAGE='${K3S_LOCAL_PATH_PROVISIONER_IMAGE}' \
     K3S_BUSYBOX_IMAGE='${K3S_BUSYBOX_IMAGE}' \
     bash -s" <<'REMOTE'
set -Eeuo pipefail
changed="$(python3 - <<'PY'
import os
import re
from pathlib import Path

data_dir = Path(os.environ["K3S_DATA_DIR"])
targets = [
    (
        data_dir / "server/manifests/coredns.yaml",
        r'image:\s*"[^"]*mirrored-coredns-coredns:1\.14\.6"',
        f'image: "{os.environ["K3S_COREDNS_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/local-storage.yaml",
        r'image:\s*"[^"]*local-path-provisioner:v0\.0\.36"',
        f'image: "{os.environ["K3S_LOCAL_PATH_PROVISIONER_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/local-storage.yaml",
        r'image:\s*"[^"]*mirrored-library-busybox:1\.37\.0"',
        f'image: "{os.environ["K3S_BUSYBOX_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/metrics-server/metrics-server-deployment.yaml",
        r'image:\s*"[^"]*mirrored-metrics-server:v0\.9\.0"',
        f'image: "{os.environ["K3S_METRICS_SERVER_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/metrics-server/metrics-server-deployment.yaml",
        r'image:\s*"[^"]*metrics-server:v0\.9\.0"',
        f'image: "{os.environ["K3S_METRICS_SERVER_IMAGE"]}"',
    ),
]
did_change = False
for path, pattern, replacement in targets:
    if not path.exists():
        continue
    text = path.read_text()
    new = re.sub(pattern, replacement, text)
    if new != text:
        path.write_text(new)
        did_change = True
print("true" if did_change else "false")
PY
)"
if [[ "${changed}" == "true" ]]; then
  kubectl apply -f "${K3S_DATA_DIR}/server/manifests/coredns.yaml"
  if [[ "${K3S_DISABLE_LOCAL_STORAGE}" != "true" && -f "${K3S_DATA_DIR}/server/manifests/local-storage.yaml" ]]; then
    kubectl apply -f "${K3S_DATA_DIR}/server/manifests/local-storage.yaml"
  fi
  kubectl apply -f "${K3S_DATA_DIR}/server/manifests/metrics-server/metrics-server-deployment.yaml"
  kubectl -n kube-system rollout restart deploy/coredns deploy/metrics-server
  if [[ "${K3S_DISABLE_LOCAL_STORAGE}" != "true" ]]; then
    kubectl -n kube-system rollout restart deploy/local-path-provisioner || true
  fi
fi
if [[ "${K3S_DISABLE_LOCAL_STORAGE}" == "true" ]]; then
  kubectl -n kube-system delete deploy/local-path-provisioner --ignore-not-found=true
  kubectl delete storageclass/local-path --ignore-not-found=true
fi
kubectl -n kube-system rollout status deploy/coredns --timeout=240s
kubectl -n kube-system rollout status deploy/metrics-server --timeout=120s || true
REMOTE
}

prepare_remote() {
  log "preparing remote k8s assets"
  ssh -o BatchMode=yes "${SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"
}

create_acr_secret() {
  log "creating/updating ACR image pull secret"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' ACR_REGION='${ACR_REGION}' ACR_INSTANCE_ID='${ACR_INSTANCE_ID}' ACR_REGISTRY='${ACR_REGISTRY}' bash -s" <<'REMOTE'
set -Eeuo pipefail
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"
auth_json="$(aliyun cr GetAuthorizationToken --RegionId "${ACR_REGION}" --InstanceId "${ACR_INSTANCE_ID}")"
user="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["TempUsername"])' <<<"${auth_json}")"
token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["AuthorizationToken"])' <<<"${auth_json}")"
kubectl -n "${NAMESPACE}" create secret docker-registry acr-pull \
  --docker-server="${ACR_REGISTRY}" \
  --docker-username="${user}" \
  --docker-password="${token}" \
  --dry-run=client -o yaml | kubectl apply -f -
REMOTE
}

create_optional_webui_secret() {
  log "creating/updating optional webui secret"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' WEBUI_PROMPT_REWRITER_PATH='${WEBUI_PROMPT_REWRITER_PATH}' bash -s" <<'REMOTE'
set -Eeuo pipefail
if [[ -f "${WEBUI_PROMPT_REWRITER_PATH}" ]]; then
  kubectl -n "${NAMESPACE}" create secret generic webui-secrets \
    --from-file=prompt-rewriter-vertex.json="${WEBUI_PROMPT_REWRITER_PATH}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "prompt rewriter secret file not found; leaving optional webui secret absent"
fi
REMOTE
}

create_optional_proxy_secret() {
  if [[ -z "${HTTPS_PROXY_SECRET_VALUE}" && -z "${HTTP_PROXY_SECRET_VALUE}" ]]; then
    log "no proxy secret configured"
    return
  fi
  log "creating/updating optional proxy secret"
  local tmp
  tmp="$(mktemp)"
  {
    [[ -n "${HTTPS_PROXY_SECRET_VALUE}" ]] && printf 'HTTPS_PROXY=%s\nhttps_proxy=%s\n' "${HTTPS_PROXY_SECRET_VALUE}" "${HTTPS_PROXY_SECRET_VALUE}"
    [[ -n "${HTTP_PROXY_SECRET_VALUE}" ]] && printf 'HTTP_PROXY=%s\nhttp_proxy=%s\n' "${HTTP_PROXY_SECRET_VALUE}" "${HTTP_PROXY_SECRET_VALUE}"
  } >"${tmp}"
  scp -q "${tmp}" "${SSH_HOST}:${REMOTE_DIR}/proxy.env"
  rm -f "${tmp}"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_DIR='${REMOTE_DIR}' bash -s" <<'REMOTE'
set -Eeuo pipefail
kubectl -n "${NAMESPACE}" create secret generic zing-proxy-env \
  --from-env-file="${REMOTE_DIR}/proxy.env" \
  --dry-run=client -o yaml | kubectl apply -f -
rm -f "${REMOTE_DIR}/proxy.env"
REMOTE
}

apply_runtime_source_patch() {
  if [[ "${APPLY_RUNTIME_SOURCE_PATCH}" != "true" ]]; then
    log "runtime source patch disabled"
    return
  fi

  log "uploading runtime source patch"
  local remote_patch_dir="${REMOTE_DIR}/runtime-source-patch"
  local entrypoint_files=(
    realtime_gateway_server.py
    realtime_coordinator_server.py
    realtime_vae_server.py
  )
  local utils_files=(
    logging_utils.py
    realtime_video.py
    realtime_trace.py
  )
  local file
  for file in "${entrypoint_files[@]}"; do
    if [[ ! -f "${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime entrypoint file: %s\n' "${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${utils_files[@]}"; do
    if [[ ! -f "${RUNTIME_UTILS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime utils file: %s\n' "${RUNTIME_UTILS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in envs.py prometheus_client.py sitecustomize.py; do
    if [[ ! -f "${RUNTIME_SHIMS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime shim file: %s\n' "${RUNTIME_SHIMS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done

  ssh -o BatchMode=yes "${SSH_HOST}" \
    "rm -rf '${remote_patch_dir}' && mkdir -p '${remote_patch_dir}/realtime' '${remote_patch_dir}/entrypoints' '${remote_patch_dir}/utils' '${remote_patch_dir}/shims'"
  scp -q "${RUNTIME_REALTIME_SOURCE_DIR}"/*.py "${SSH_HOST}:${remote_patch_dir}/realtime/"
  local entrypoint_sources=()
  for file in "${entrypoint_files[@]}"; do
    entrypoint_sources+=("${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}")
  done
  scp -q "${entrypoint_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/entrypoints/"
  local utils_sources=()
  for file in "${utils_files[@]}"; do
    utils_sources+=("${RUNTIME_UTILS_SOURCE_DIR}/${file}")
  done
  scp -q "${utils_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/utils/"
  scp -q \
    "${RUNTIME_SHIMS_SOURCE_DIR}/envs.py" \
    "${RUNTIME_SHIMS_SOURCE_DIR}/prometheus_client.py" \
    "${RUNTIME_SHIMS_SOURCE_DIR}/sitecustomize.py" \
    "${SSH_HOST}:${remote_patch_dir}/shims/"

  ssh -o BatchMode=yes "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_PATCH_DIR='${remote_patch_dir}' bash -s" <<'REMOTE'
set -Eeuo pipefail
: >"${REMOTE_PATCH_DIR}/utils/__init__.py"
kubectl -n "${NAMESPACE}" delete configmap \
  zing-runtime-realtime-patch \
  zing-runtime-entrypoint-patch \
  zing-runtime-utils-patch \
  zing-runtime-envs-patch \
  zing-runtime-dep-shims \
  --ignore-not-found=true
kubectl -n "${NAMESPACE}" create configmap zing-runtime-realtime-patch \
  --from-file="${REMOTE_PATCH_DIR}/realtime"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-entrypoint-patch \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_gateway_server.py" \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_coordinator_server.py" \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_vae_server.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-utils-patch \
  --from-file="${REMOTE_PATCH_DIR}/utils"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-envs-patch \
  --from-file="${REMOTE_PATCH_DIR}/shims/envs.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-dep-shims \
  --from-file="${REMOTE_PATCH_DIR}/shims/prometheus_client.py" \
  --from-file="${REMOTE_PATCH_DIR}/shims/sitecustomize.py"
REMOTE
}

apply_manifests() {
  log "rendering and applying control-plane manifests"
  local rendered
  rendered="$(mktemp)"
  render_template "${SCRIPT_DIR}/manifests/control-plane.yaml.tpl" "${rendered}"
  scp -q "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/control-plane.yaml"
  rm -f "${rendered}"
  ssh -o BatchMode=yes "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/control-plane.yaml'"
  if [[ "${APPLY_NVIDIA_DEVICE_PLUGIN}" == "true" ]]; then
    log "rendering and applying NVIDIA device plugin"
    rendered="$(mktemp)"
    render_template "${SCRIPT_DIR}/manifests/nvidia-device-plugin.yaml.tpl" "${rendered}"
    scp -q "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/nvidia-device-plugin.yaml"
    rm -f "${rendered}"
    ssh -o BatchMode=yes "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/nvidia-device-plugin.yaml'"
  fi
  if [[ "${APPLY_GPU_WORKER_TEMPLATES}" == "true" ]]; then
    log "rendering and applying GPU worker templates"
    rendered="$(mktemp)"
    render_template "${SCRIPT_DIR}/manifests/gpu-workers-5090.yaml.tpl" "${rendered}"
    scp -q "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/gpu-workers-5090.yaml"
    rm -f "${rendered}"
    ssh -o BatchMode=yes "${SSH_HOST}" "kubectl -n '${NAMESPACE}' delete deploy/zing-vae-5090 sts/zing-denoiser-5090 svc/zing-denoiser-5090 --ignore-not-found=true"
    ssh -o BatchMode=yes "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/gpu-workers-5090.yaml'"
  fi
}

apply_webui_static_patch() {
  if [[ "${APPLY_WEBUI_STATIC_PATCH}" != "true" ]]; then
    log "webui static patch disabled"
    return
  fi

  log "uploading single-Zing webui static patch"
  local remote_patch_dir="${REMOTE_DIR}/webui-static-patch"
  local files=(
    index.html
    styles.css
    playback_controller.js
    model_session.js
    dual_model_controller.js
    h264_websocket_session.js
    happy_oyster_session.js
    session_lifetime.js
    prompt_rewrite_controller.js
    world_rules_controller.js
    fullscreen_controller.js
    trace_topology.js
    trace_transport.js
    app.js
  )
  local sources=()
  local file
  for file in "${files[@]}"; do
    if [[ ! -f "${WEBUI_STATIC_SOURCE_DIR}/${file}" ]]; then
      printf 'missing webui static file: %s\n' "${WEBUI_STATIC_SOURCE_DIR}/${file}" >&2
      return 1
    fi
    sources+=("${WEBUI_STATIC_SOURCE_DIR}/${file}")
  done

  ssh -o BatchMode=yes "${SSH_HOST}" "rm -rf '${remote_patch_dir}' && mkdir -p '${remote_patch_dir}'"
  scp -q "${sources[@]}" "${SSH_HOST}:${remote_patch_dir}/"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_PATCH_DIR='${remote_patch_dir}' PATCH_NAME='${WEBUI_STATIC_PATCH_NAME}' bash -s" <<'REMOTE'
set -Eeuo pipefail
printf '%s\n' 'globalThis.HappyOysterSDK = globalThis.HappyOysterSDK || {};' \
  >"${REMOTE_PATCH_DIR}/happy_oyster_sdk.js"
kubectl -n "${NAMESPACE}" delete configmap "${PATCH_NAME}" --ignore-not-found=true
kubectl -n "${NAMESPACE}" create configmap "${PATCH_NAME}" --from-file="${REMOTE_PATCH_DIR}"
REMOTE

  WEBUI_STATIC_PATCH_NAME="${WEBUI_STATIC_PATCH_NAME}" \
  WEBUI_STATIC_PATCH_VERSION="${WEBUI_STATIC_PATCH_VERSION}" \
  WEBUI_IMAGE="${WEBUI_IMAGE}" \
  python3 - <<'PY' | ssh -o BatchMode=yes "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' patch deploy zing-webui --type=strategic --patch-file /dev/stdin"
import json
import os

base = "/opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui"
files = [
    "index.html",
    "styles.css",
    "playback_controller.js",
    "model_session.js",
    "dual_model_controller.js",
    "h264_websocket_session.js",
    "happy_oyster_sdk.js",
    "happy_oyster_session.js",
    "session_lifetime.js",
    "prompt_rewrite_controller.js",
    "world_rules_controller.js",
    "fullscreen_controller.js",
    "trace_topology.js",
    "trace_transport.js",
    "app.js",
]
patch = {
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "seedleap.ai/webui-static-patch": os.environ["WEBUI_STATIC_PATCH_VERSION"]
                }
            },
            "spec": {
                "volumes": [
                    {
                        "name": "webui-static-patch",
                        "configMap": {"name": os.environ["WEBUI_STATIC_PATCH_NAME"]},
                    }
                ],
                "containers": [
                    {
                        "name": "webui",
                        "image": os.environ["WEBUI_IMAGE"],
                        "volumeMounts": [
                            {
                                "name": "webui-static-patch",
                                "mountPath": f"{base}/{name}",
                                "subPath": name,
                                "readOnly": True,
                            }
                            for name in files
                        ],
                    }
                ],
            },
        }
    }
}
print(json.dumps(patch, separators=(",", ":")))
PY
}

wait_rollout() {
  log "waiting for k8s rollouts"
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' rollout status deploy/zing-coordinator --timeout='${ROLLOUT_TIMEOUT}' && \
     kubectl -n '${NAMESPACE}' rollout status deploy/zing-gateway --timeout='${ROLLOUT_TIMEOUT}' && \
     kubectl -n '${NAMESPACE}' rollout status deploy/zing-webui --timeout='${ROLLOUT_TIMEOUT}'"
  if [[ "${APPLY_GPU_WORKER_TEMPLATES}" == "true" ]]; then
    ssh -o BatchMode=yes "${SSH_HOST}" \
      "kubectl -n '${NAMESPACE}' rollout status ds/zing-denoiser-5090-sp2 --timeout='${ROLLOUT_TIMEOUT}' && \
       kubectl -n '${NAMESPACE}' rollout status ds/zing-vae-5090-dual --timeout='${ROLLOUT_TIMEOUT}'"
  fi
}

print_status() {
  ssh -o BatchMode=yes "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' get pods -o wide && \
     kubectl -n '${NAMESPACE}' get svc && \
     curl -fsS http://127.0.0.1/runtime-config.js >/dev/null && \
     curl -fsS http://127.0.0.1:18080/healthz >/dev/null"
  log "webui: http://${PUBLIC_WEB_HOST}/?mode=i2v&playback=smooth_timeline"
  log "gateway: ${PUBLIC_GATEWAY_BASE_URL}"
}

main() {
  install_k3s
  configure_k3s_system_images
  prepare_remote
  create_acr_secret
  create_optional_webui_secret
  create_optional_proxy_secret
  apply_runtime_source_patch
  apply_manifests
  apply_webui_static_patch
  wait_rollout
  print_status
}

main "$@"
