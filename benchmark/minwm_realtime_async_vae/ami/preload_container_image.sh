#!/usr/bin/env bash
set -Eeuo pipefail

: "${AWS_REGION:?AWS_REGION is required}"
: "${IMAGE_REFERENCE:?IMAGE_REFERENCE is required}"
: "${SOURCE_GIT_SHA:?SOURCE_GIT_SHA is required}"

if ! [[ "${IMAGE_REFERENCE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "IMAGE_REFERENCE must be pinned by sha256 digest" >&2
  exit 1
fi
if ! [[ "${SOURCE_GIT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_GIT_SHA must be a full Git commit SHA" >&2
  exit 1
fi

registry="${IMAGE_REFERENCE%%/*}"
case "${registry}" in
  *.dkr.ecr.*.amazonaws.com|*.dkr.ecr.*.amazonaws.com.cn)
    ;;
  *)
    echo "only an Amazon ECR image is accepted: ${registry}" >&2
    exit 1
    ;;
esac

sudo systemctl start containerd
ecr_password="$(aws ecr get-login-password --region "${AWS_REGION}")"
sudo ctr --namespace k8s.io images pull \
  --user "AWS:${ecr_password}" \
  "${IMAGE_REFERENCE}"
unset ecr_password

image_check="$(sudo ctr --namespace k8s.io images check)"
printf '%s\n' "${image_check}"
printf '%s\n' "${image_check}" | awk -v ref="${IMAGE_REFERENCE}" '
  $1 == ref && $4 == "complete" { found = 1 }
  END { exit(found ? 0 : 1) }
'
sudo tee /etc/minwm-baked-image >/dev/null <<EOF
image=${IMAGE_REFERENCE}
source_git_sha=${SOURCE_GIT_SHA}
EOF
sudo chmod 0444 /etc/minwm-baked-image

# Keep the baked containerd content on the persistent AMI root snapshot. The
# EKS setup-local-disks helper still puts kubelet emptyDir data, Pod logs, and
# SOCI state on the instance-store RAID0, but it must not copy the roughly 40
# GiB unpacked image cache on every Spot replacement.
if [[ ! -x /usr/bin/setup-local-disks.eks ]]; then
  sudo mv /usr/bin/setup-local-disks /usr/bin/setup-local-disks.eks
fi
sudo tee /usr/bin/setup-local-disks >/dev/null <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
exec /usr/bin/setup-local-disks.eks "$@" --no-bind-containerd
EOF
sudo chmod 0755 /usr/bin/setup-local-disks

sudo systemctl stop containerd
sudo cloud-init clean --logs
sudo rm -f /etc/ssh/ssh_host_*
sudo truncate -s 0 /etc/machine-id
sudo rm -f /var/lib/dbus/machine-id
sync
