# MinWM denoiser GPU AMI

This Packer build starts from an explicitly reviewed EKS 1.35 AL2023 NVIDIA AMI
and preloads the digest-pinned MinWM denoiser image into containerd's `k8s.io`
namespace. Because the final image contains its common GPU/Python base layers,
those content-addressed layers are cached for every container that shares them.

The 200 GiB gp3 root volume is provisioned at 16,000 IOPS and 1,000 MiB/s. The
baked containerd content and unpacked snapshots stay on that root volume, which
avoids copying roughly 40 GiB on every Spot replacement. The bake installs a
small wrapper around the EKS `setup-local-disks` helper that passes
`--no-bind-containerd`; kubelet data (including the model `emptyDir`), Pod logs,
and SOCI state still land on the instance-store RAID0. Snapshot cold-block
initialization can still affect first boot, which is why node-ready time is part
of the canary measurement rather than an assumed saving.

The Packer template has no default source AMI, subnet, security group, or
instance profile. A future bake must pass reviewed `world-model` values
explicitly and is a separate remote infrastructure action; WM-09 did not run
Packer or create an AMI.

The temporary `g6.2xlarge` builder requires an explicitly supplied
least-privilege instance profile with SSM Core and ECR read-only permissions.
No legacy-cluster role or long-lived access key is assumed by the template.

After the new denoiser image exists in ECR:

```bash
packer init benchmark/minwm_realtime_async_vae/ami/minwm-denoiser.pkr.hcl
packer validate \
  -var 'source_ami_id=<reviewed-eks-1.35-nvidia-ami>' \
  -var 'subnet_id=<reviewed-builder-subnet>' \
  -var 'security_group_id=<reviewed-builder-security-group>' \
  -var 'iam_instance_profile=<reviewed-builder-instance-profile>' \
  -var 'image_reference=829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime@sha256:<digest>' \
  -var 'source_git_sha=<40-character-sha>' \
  -var 'ami_name=minwm-denoiser-eks135-<short-sha>-<timestamp>' \
  benchmark/minwm_realtime_async_vae/ami/minwm-denoiser.pkr.hcl
```

`packer build` is a remote write: it creates a temporary EC2 instance, EBS
volume/snapshot, and AMI. Run it only after the reviewed execution plan has been
explicitly approved. The temporary builder is terminated by Packer; the AMI and
its root snapshot remain billable until deregistered/deleted.
