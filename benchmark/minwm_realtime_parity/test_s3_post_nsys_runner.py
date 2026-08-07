import json
import os
import subprocess
from pathlib import Path

from measurement_tool import build_invalid_marker, load_aggregate_records


ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "run_s3_post_nsys_matrix.sh"
S0_RUNNER = ROOT / "run_s0_measurement.sh"
MEASUREMENT = ROOT / "measurement.py"
NSYS_METRICS = ROOT / "nsys_metrics.py"
MANIFEST = ROOT / "k8s" / "minwm_s3_post_nsys_h200_20260807.yaml"


def _shell_function(path: Path, name: str) -> str:
    lines = path.read_text().splitlines()
    start = lines.index(f"{name}() {{")
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{")
        depth -= lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1]) + "\n"
    raise AssertionError(f"unterminated shell function: {name}")


def _run_bash(
    script: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
        text=True,
    )


def test_wrapper_pins_four_exact_window_lanes_and_order() -> None:
    text = WRAPPER.read_text()
    expected = (
        "export MINWM_S0_PROFILE_PRECONDITION_CHUNKS=20",
        "export MINWM_S0_PROFILE_DISCARD_CHUNKS=1",
        "export MINWM_S0_PROFILE_MEASURED_CHUNKS=10",
        "export MINWM_S0_KV_CACHE_NUM_FRAMES=45",
        'run_variant "${degree}" 00 baseline 0',
        'run_variant "${degree}" 01 candidate 1',
        'compare_degree "${degree}"',
    )
    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)
    assert "for degree in 2 4" in text
    assert "unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR" in text
    assert "--require-complete-stable-nsys" in text


def test_stage_trace_validation_is_lane_scoped_and_explicit() -> None:
    text = WRAPPER.read_text()
    scoped = 'CURRENT_LANE_DIR="${profile_dir}"'
    run = 'bash "${SCRIPT_DIR}/run_s0_measurement.sh"'
    validate = 'validate_stage_trace "${profile_dir}/measurement.json"'
    complete = '"${profile_dir}/S3_LANE_COMPLETE"'
    clear = 'CURRENT_LANE_DIR=""'
    start = text.index(scoped)
    assert start < text.index(run, start)
    assert text.index(run, start) < text.index(validate, start)
    assert text.index(validate, start) < text.index(complete, start)
    assert text.index(complete, start) < text.index(clear, start)
    for required in (
        'on["observed_wall_with_profiler_overhead"][name]',
        'metric["value"]["count"] == 10',
        "list(range(1, 11))",
        'value["collected_target_count"] == 8',
        'len(value["active_pw_gpu_ids"]) == degree',
    ):
        assert required in text


def test_profiler_on_marker_excludes_only_failed_lane(tmp_path: Path) -> None:
    baseline = (
        tmp_path
        / "baseline"
        / "s0-measurement"
        / "sp2"
        / "profiler-on"
        / "measurement.json"
    )
    candidate = (
        tmp_path
        / "candidate"
        / "s0-measurement"
        / "sp2"
        / "profiler-on"
        / "measurement.json"
    )
    for path, label in ((baseline, "baseline"), (candidate, "candidate")):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"lane": label}))

    marker_path = candidate.parent / "invalid-marker-test.json"
    marker = build_invalid_marker(
        candidate.parent,
        reason="stage trace validation failed",
        marker_path=marker_path,
        timestamp_utc="2026-08-07T00:00:00+00:00",
    )
    marker_path.write_text(json.dumps(marker))

    accepted, excluded = load_aggregate_records([baseline, candidate])
    assert accepted == [{"lane": "baseline"}]
    assert excluded == [candidate]
    assert not next(baseline.parent.glob("invalid-marker*.json"), None)


def test_s0_server_and_client_contract_dry_run() -> None:
    script = (
        "set -euo pipefail\n"
        "MODEL_DIR=/work/model\n"
        "CASES=/work/cases.json\nCASE_ID=case\nSGLANG_GIT_REF=runner\n"
        "MINWM_GIT_REF=minwm\nMINWM_CONTAINER_IMAGE=image\nGPU_MODEL='NVIDIA H200'\n"
        "ALLOCATED_GPU_COUNT=8\nKV_CACHE_NUM_FRAMES=45\n"
        + _shell_function(S0_RUNNER, "server_args")
        + _shell_function(S0_RUNNER, "client_common_args")
        + "server_args 4 | tr '\\0' '\\n'\n"
        + "client_common_args 4 /tmp/out.json profile run | tr '\\0' '\\n'\n"
    )
    output = _run_bash(script).stdout
    for expected in (
        "sglang\nserve",
        "--num-gpus\n4",
        "--sp-degree\n4",
        "--ulysses-degree\n4",
        "--gpu-count\n4",
        "--allocated-gpu-count\n8",
        "--kv-cache-num-frames\n45",
        "--require-complete-stage-trace",
    ):
        assert expected in output


def test_canonical_runner_collects_all_targets_and_fails_closed() -> None:
    runner = S0_RUNNER.read_text()
    assert "--gpu-metrics-devices=all" in runner
    assert "--require-complete-stable-nsys" in runner
    assert 'SGLANG_REALTIME_NSYS_WARMUP_CHUNKS="${PROFILE_DISCARD_CHUNKS}"' in runner
    assert 'SGLANG_REALTIME_NSYS_MEASURED_CHUNKS="${PROFILE_MEASURED_CHUNKS}"' in runner
    measurement = MEASUREMENT.read_text()
    assert "API_BOUNDARY_ATTRIBUTION_POLICY" in measurement
    assert "boundary_included_by_start_count" in measurement
    assert "aggregation_mode" in measurement
    metrics = NSYS_METRICS.read_text()
    assert "_discrete_event_start_attribution" in metrics
    assert "boundary_event_examples" in metrics
    assert "streaming selected metricId rows" in metrics


def test_manifest_is_immutable_backoff_zero_and_cannot_expand_pool() -> None:
    text = MANIFEST.read_text()
    assert "backoffLimit: 0" in text
    assert "kubernetes.io/hostname: i-06888dc1ca88547e1" in text
    assert 'value: "58ed4daf7e4208eedde4f8fc8f0a8c1e20e0007d"' in text
    assert 'value: "d5b25227d4487d113e62c86a0fb572a62d6bcc5b"' in text
    assert 'value: "20"' in text
    assert 'value: "1"' in text
    assert 'value: "10"' in text
    assert 'value: "45"' in text
    assert 'nvidia.com/gpu: "8"' in text
    assert "- SYS_ADMIN" in text
    assert "SGLANG_DIFFUSION_TORCH_PROFILER_DIR" not in text
    assert "minwm-s3-a2a-h200-results-20260807-01" in text
