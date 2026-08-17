from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_bake_keeps_cached_containerd_on_the_ami_root_volume():
    script = (ROOT / "ami/preload_container_image.sh").read_text()

    assert "setup-local-disks.eks" in script
    assert "--no-bind-containerd" in script
    assert 'exec /usr/bin/setup-local-disks.eks "$@" --no-bind-containerd' in script


def test_packer_requires_explicit_builder_infrastructure_inputs():
    template = (ROOT / "ami/minwm-denoiser.pkr.hcl").read_text()

    for variable in (
        "source_ami_id",
        "subnet_id",
        "security_group_id",
        "iam_instance_profile",
    ):
        block = template.split(f'variable "{variable}"', 1)[1].split("}", 1)[0]
        assert "default =" not in block
    assert "KarpenterNodeRole-leap-world" not in template
