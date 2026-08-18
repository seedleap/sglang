#!/usr/bin/env bash
set -euo pipefail

# Prepare every p6-b200 instance-store device as one RAID0 filesystem before
# EKS starts pulling the large CUDA runtime image. Both containerd and kubelet
# (including emptyDir model-cache volumes) are kept off the small GP3 root disk.

marker=/var/lib/loopit-b200-nvme-ready
if [[ -f "${marker}" ]]; then
  exit 0
fi

# The by-id directory exposes three aliases for every instance-store disk on
# p6-b200.  Feeding those aliases to mdadm repeats the same block device and
# makes RAID creation fail.  Select the eight canonical kernel devices by
# their NVMe model instead.
udevadm settle --timeout=30
mapfile -t instance_store < <(
  lsblk -dn -o NAME,MODEL | awk '$2 == "Amazon" && $3 == "EC2" && $4 == "NVMe" && $5 == "Instance" && $6 == "Storage" { print "/dev/" $1 }' | sort -V
)
if (( ${#instance_store[@]} == 0 )); then
  echo "p6-b200 instance-store NVMe devices were not found" >&2
  exit 1
fi

dnf install -y mdadm xfsprogs rsync
systemctl stop kubelet containerd 2>/dev/null || true

raid=/dev/md/0
mkdir -p /dev/md
if [[ ! -e "${raid}" ]]; then
  mdadm --create "${raid}" --run --level=0 \
    --raid-devices="${#instance_store[@]}" "${instance_store[@]}"
fi

if ! blkid "${raid}" | grep -q 'TYPE="xfs"'; then
  mkfs.xfs -f "${raid}"
fi

mount_root=/mnt/k8s-disks/0
mkdir -p "${mount_root}"
raid_uuid=$(blkid -s UUID -o value "${raid}")
grep -q "UUID=${raid_uuid} " /etc/fstab || \
  printf 'UUID=%s %s xfs defaults,noatime,prjquota,nofail 0 2\n' \
    "${raid_uuid}" "${mount_root}" >>/etc/fstab
mountpoint -q "${mount_root}" || mount "${mount_root}"

for runtime_dir in containerd kubelet; do
  source_dir="/var/lib/${runtime_dir}"
  target_dir="${mount_root}/${runtime_dir}"
  mkdir -p "${source_dir}" "${target_dir}"
  if [[ -n "$(find "${source_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rsync -aHAX "${source_dir}/" "${target_dir}/"
  fi
  mountpoint -q "${source_dir}" || mount --bind "${target_dir}" "${source_dir}"
  grep -qF "${target_dir} ${source_dir} none bind 0 0" /etc/fstab || \
    printf '%s %s none bind 0 0\n' "${target_dir}" "${source_dir}" >>/etc/fstab
done

mkdir -p "$(dirname "${marker}")"
date -u +%FT%TZ >"${marker}"
