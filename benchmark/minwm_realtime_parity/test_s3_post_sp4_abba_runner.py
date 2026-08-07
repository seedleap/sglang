import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_s3_post_sp4_abba_off_only.sh"
MANIFEST = ROOT / "k8s" / "minwm_s3_post_sp4_abba_h200_20260807.yaml"


def _shell_function(name: str) -> str:
    lines = RUNNER.read_text().splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1]) + "\n"
    raise AssertionError(f"unterminated shell function: {name}")


def _run_bash(script: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["bash", "-c", script],
        check=True,
        env={**os.environ, **(env or {})},
        text=True,
    )


def test_wrapper_is_physically_off_only() -> None:
    runner = RUNNER.read_text().lower()
    manifest = MANIFEST.read_text().lower()
    for text in (runner, manifest):
        assert "nsys" not in text
        assert "profiler_on" not in text
    assert "positions=(a1 b1 b2 a2)" in runner
    assert "lanes=(01 00 00 01)" in runner
    assert "backofflimit: 0" in manifest
    assert "kubernetes.io/hostname: i-06888dc1ca88547e1" in manifest
    assert 'value: "900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3"' in manifest
    assert 'value: "29c6ada1a514c137c2ca4cf81b58fdc2065b401a"' in manifest


def test_failed_position_marker_does_not_invalidate_siblings(tmp_path: Path) -> None:
    root = tmp_path / "run"
    failed = root / "measurements" / "A1"
    sibling = root / "measurements" / "B1"
    failed.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (failed / "partial.json").write_text("partial\n")
    (sibling / "valid.json").write_text("valid\n")
    _run_bash(
        "set -euo pipefail\n"
        + _shell_function("record_invalid_attempt")
        + f"record_invalid_attempt 9 {failed}\n"
    )
    assert (failed / "invalid" / "attempt.json").is_file()
    assert not (root / "invalid").exists()
    assert not (sibling / "invalid").exists()


def test_server_and_client_contract_dry_run(tmp_path: Path) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    server_call = tmp_path / "server-call.txt"
    python_calls = tmp_path / "python-calls.txt"
    (mock_bin / "sglang").write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" > "${MOCK_SERVER_CALL}"\n'
        "sleep 30\n"
    )
    (mock_bin / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\n"
        'printf "2026/08/07 00:00:00.000, 0, 0, 1980, P0, 100, 30, 0\\n"\n'
    )
    (mock_bin / "python3").write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${MOCK_PYTHON_CALLS}"\n'
        'if [[ "${1:-}" == "-" ]]; then cat >/dev/null; fi\n'
    )
    for path in mock_bin.iterdir():
        path.chmod(0o755)
    env = {
        "PATH": f"{mock_bin}:{os.environ['PATH']}",
        "MOCK_SERVER_CALL": str(server_call),
        "MOCK_PYTHON_CALLS": str(python_calls),
    }
    server_log = tmp_path / "server.log"
    telemetry = tmp_path / "telemetry.csv"
    output = tmp_path / "result.json"
    shell = (
        "set -euo pipefail\n"
        "MODEL_DIR=/work/model\nSP_DEGREE=4\nserver_pid=\"\"\nmonitor_pid=\"\"\n"
        + _shell_function("start_server")
        + "wait_for_server() { return 0; }\n"
        + f"start_server {server_log} {telemetry}\n"
        + "sleep 0.2\nkill \"${server_pid}\" 2>/dev/null || true\n"
        + "kill \"${monitor_pid}\" 2>/dev/null || true\n"
        + "wait \"${server_pid}\" 2>/dev/null || true\n"
        + "wait \"${monitor_pid}\" 2>/dev/null || true\n"
        + f"SCRIPT_DIR={ROOT}\nCASES={ROOT / 'cases_720p_compile_smoke.json'}\n"
        + "CASE_ID=00_forward_080_pottery_720p\n"
        + "MINWM_RUN_ID=dry-run\nSGLANG_GIT_REF=runner\nMINWM_GIT_REF=minwm\n"
        + "MINWM_CONTAINER_IMAGE=image\nGPU_MODEL='NVIDIA H200'\n"
        + "ALLOCATED_GPU_COUNT=8\nKV_CACHE_NUM_FRAMES=45\n"
        + "WARMUP_CHUNKS=20\nMEASURED_CHUNKS=200\n"
        + _shell_function("run_client")
        + f"run_client A1 01 {output}\n"
    )
    _run_bash(shell, env)
    server_args = server_call.read_text()
    assert "serve --model-path /work/model" in server_args
    assert "--num-gpus 4" in server_args
    assert "--sp-degree 4" in server_args
    assert "--ulysses-degree 4" in server_args
    client_args = python_calls.read_text()
    assert "benchmark_realtime_throughput.py" in client_args
    assert "--measurement-mode profiler_off" in client_args
    assert "--warmup-chunks 20" in client_args
    assert "--measured-chunks 200" in client_args
    assert "--kv-cache-num-frames 45" in client_args
    assert "--gpu-count 4" in client_args
