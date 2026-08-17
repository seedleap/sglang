#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

ALIYUN_REGION="${ALIYUN_REGION:-cn-beijing}"
ALIYUN_ZONE="${ALIYUN_ZONE:-cn-beijing-i}"
INSTANCE_ID="${INSTANCE_ID:-i-2zegvp51qv6iuyesw65m}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-2zei6gh3ag2zv34d3jnt}"
OSS_BUCKET="${OSS_BUCKET:-seedleap-sglang-rtx6000-beijing-20260813}"
OSS_ARTIFACT_PREFIX="${OSS_ARTIFACT_PREFIX:-world-model/minwm/serving-artifacts/aliyun-4090-zing-single}"
DATA_DISK_SIZE_GB="${DATA_DISK_SIZE_GB:-200}"
DATA_DISK_NAME="${DATA_DISK_NAME:-zing-realtime-cache-20260817}"
REMOTE_DIR="${REMOTE_DIR:-/root/zing-realtime}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-${REMOTE_DIR}/start_remote.sh}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-7200}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

run_remote() {
  local command="$1"
  local timeout="${2:-600}"
  local encoded invoke_json invoke_id status output
  encoded="$(printf '%s' "${command}" | base64 | tr -d '\n')"
  invoke_json="$(aliyun ecs RunCommand \
    --RegionId "${ALIYUN_REGION}" \
    --Type RunShellScript \
    --InstanceId.1 "${INSTANCE_ID}" \
    --ContentEncoding Base64 \
    --CommandContent "${encoded}" \
    --Timeout "${timeout}")"
  invoke_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["InvokeId"])' <<<"${invoke_json}")"
  while true; do
    sleep 3
    output="$(aliyun ecs DescribeInvocationResults --RegionId "${ALIYUN_REGION}" --InvokeId "${invoke_id}")"
    status="$(python3 -c 'import json,sys; d=json.load(sys.stdin)["Invocation"]["InvocationResults"]["InvocationResult"][0]; print(d["InvocationStatus"])' <<<"${output}")"
    case "${status}" in
      Success|Failed|Stopped|TimedOut|Cancelled)
        python3 -c '
import base64, json, sys
result = json.load(sys.stdin)["Invocation"]["InvocationResults"]["InvocationResult"][0]
payload = result.get("Output") or ""
if payload:
    try:
        print(base64.b64decode(payload).decode("utf-8", "replace"), end="")
    except Exception:
        print(payload)
raise SystemExit(result.get("ExitCode", 0) or 0)
' <<<"${output}"
        [[ "${status}" == "Success" ]] || return 1
        return 0
        ;;
    esac
  done
}

ensure_data_disk() {
  local disks disk_id create_json
  disks="$(aliyun ecs DescribeDisks --RegionId "${ALIYUN_REGION}" --InstanceId "${INSTANCE_ID}")"
  disk_id="$(python3 -c '
import json, sys
data = json.load(sys.stdin)
for disk in data.get("Disks", {}).get("Disk", []):
    if disk.get("Type") == "data" and disk.get("Status") == "In_use":
        print(disk["DiskId"])
        break
' <<<"${disks}")"
  if [[ -n "${disk_id}" ]]; then
    log "reusing attached data disk ${disk_id}"
    return
  fi

  log "creating ${DATA_DISK_SIZE_GB}G ESSD data disk in ${ALIYUN_ZONE}"
  create_json="$(aliyun ecs CreateDisk \
    --RegionId "${ALIYUN_REGION}" \
    --ZoneId "${ALIYUN_ZONE}" \
    --DiskCategory cloud_essd \
    --PerformanceLevel PL0 \
    --Size "${DATA_DISK_SIZE_GB}" \
    --DiskName "${DATA_DISK_NAME}")"
  disk_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["DiskId"])' <<<"${create_json}")"

  log "attaching data disk ${disk_id} to ${INSTANCE_ID}"
  aliyun ecs AttachDisk \
    --RegionId "${ALIYUN_REGION}" \
    --InstanceId "${INSTANCE_ID}" \
    --DiskId "${disk_id}" >/dev/null

  for _ in {1..80}; do
    disks="$(aliyun ecs DescribeDisks --RegionId "${ALIYUN_REGION}" --DiskIds "[\"${disk_id}\"]")"
    if python3 -c '
import json, sys
disk = json.load(sys.stdin)["Disks"]["Disk"][0]
raise SystemExit(0 if disk.get("Status") == "In_use" else 1)
' <<<"${disks}"
    then
      log "data disk is attached"
      return
    fi
    sleep 3
  done
  echo "data disk ${disk_id} did not attach in time" >&2
  exit 1
}

ensure_security_group_port_80() {
  local rules
  rules="$(aliyun ecs DescribeSecurityGroupAttribute --RegionId "${ALIYUN_REGION}" --SecurityGroupId "${SECURITY_GROUP_ID}")"
  if python3 -c '
import json, sys
data = json.load(sys.stdin)
for permission in data.get("Permissions", {}).get("Permission", []):
    if (
        permission.get("IpProtocol") in {"TCP", "tcp", "ALL", "all"}
        and permission.get("PortRange") in {"80/80", "-1/-1"}
        and permission.get("Direction", "ingress").lower() in {"ingress", ""}
        and permission.get("SourceCidrIp") in {"0.0.0.0/0", "::/0"}
    ):
        raise SystemExit(0)
raise SystemExit(1)
' <<<"${rules}"
  then
    log "security group already allows port 80"
    return
  fi
  log "opening inbound TCP/80 on ${SECURITY_GROUP_ID}"
  aliyun ecs AuthorizeSecurityGroup \
    --RegionId "${ALIYUN_REGION}" \
    --SecurityGroupId "${SECURITY_GROUP_ID}" \
    --IpProtocol tcp \
    --PortRange 80/80 \
    --SourceCidrIp 0.0.0.0/0 \
    --Policy accept \
    --Priority 1 >/dev/null
}

upload_code_overlay() {
  local git_sha branch safe_branch archive overlay_uri script_uri
  git_sha="$(git -C "${ROOT}" rev-parse HEAD)"
  branch="$(git -C "${ROOT}" branch --show-current)"
  safe_branch="${branch//\//-}"
  archive="$(mktemp -t sglang-aliyun-overlay.XXXXXX.tar.gz)"
  COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --exclude='._*' \
    --exclude='.DS_Store' \
    -C "${ROOT}" \
    -czf "${archive}" \
    python/sglang/multimodal_gen
  overlay_uri="oss://${OSS_BUCKET}/${OSS_ARTIFACT_PREFIX}/code-overlays/${safe_branch}/${git_sha}.tar.gz"
  script_uri="oss://${OSS_BUCKET}/${OSS_ARTIFACT_PREFIX}/scripts/start_remote-${git_sha}.sh"
  log "uploading code overlay to ${overlay_uri}"
  aliyun oss cp "${archive}" "${overlay_uri}" --region "${ALIYUN_REGION}" -f --jobs 16 --parallel 8 >/dev/null
  log "uploading remote starter to ${script_uri}"
  aliyun oss cp "${SCRIPT_DIR}/start_remote.sh" "${script_uri}" --region "${ALIYUN_REGION}" -f >/dev/null
  rm -f "${archive}"
  printf '%s\n%s\n' "${overlay_uri}" "${script_uri}"
}

start_remote_deployment() {
  local overlay_uri="$1"
  local script_uri="$2"
  log "starting remote deployment through Cloud Assistant"
  run_remote "$(cat <<EOF
set -Eeuo pipefail
mkdir -p '${REMOTE_DIR}'
aliyun oss cp '${script_uri}' '${REMOTE_SCRIPT}' --region '${ALIYUN_REGION}' --endpoint oss-cn-beijing-internal.aliyuncs.com -f >/dev/null
chmod +x '${REMOTE_SCRIPT}'
export ALIYUN_REGION='${ALIYUN_REGION}'
export ALIYUN_ZONE='${ALIYUN_ZONE}'
export CODE_OVERLAY_OSS_URI='${overlay_uri}'
exec '${REMOTE_SCRIPT}'
EOF
)" "${REMOTE_TIMEOUT}"
}

main() {
  local uploaded overlay_uri script_uri
  ensure_data_disk
  ensure_security_group_port_80
  uploaded="$(upload_code_overlay)"
  overlay_uri="$(printf '%s\n' "${uploaded}" | sed -n '1p')"
  script_uri="$(printf '%s\n' "${uploaded}" | sed -n '2p')"
  if [[ -z "${overlay_uri}" || -z "${script_uri}" ]]; then
    echo "failed to resolve uploaded deployment artifacts" >&2
    exit 1
  fi
  start_remote_deployment "${overlay_uri}" "${script_uri}"
  log "WebUI: http://8.147.109.68/?mode=i2v&playback=smooth_timeline"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
