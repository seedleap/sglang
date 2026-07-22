from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_runner_only_adds_taehv_flag_when_checkpoint_is_set():
    script = (ROOT / "run_capacity_smoke_720p.sh").read_text(encoding="utf-8")

    assert "taehv_args=()" in script
    assert '[[ -n "${taehv_checkpoint_path}" ]]' in script
    assert "--vae-config.taehv-checkpoint-path" in script


def test_runner_image_installs_pinned_taehv():
    dockerfile = (ROOT / "Dockerfile.video-runner").read_text(encoding="utf-8")

    assert (
        "madebyollin/taehv.git@093b918971d59001a0bad6dfd6e0409b5e1752cf" in dockerfile
    )
