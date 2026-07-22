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
    assert "taew2_1.pth" in dockerfile
    assert (
        "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e" in dockerfile
    )


def test_runner_records_startup_and_measurement_boundaries():
    script = (ROOT / "run_capacity_smoke_720p.sh").read_text(encoding="utf-8")

    assert "server_startup_seconds" in script
    assert '"${results_root}/server-startup-seconds"' in script
    assert '"${results_root}/lifecycle.json"' in script
    assert "benchmark_started_epoch" in script


def test_runner_accepts_an_opt_in_fixed_case_limit():
    script = (ROOT / "run_capacity_smoke_720p.sh").read_text(encoding="utf-8")

    assert "SGLANG_VIDEO_CASE_LIMIT" in script
    assert "case_limit_args" in script
    assert "--limit" in script
