import json
from pathlib import Path

from measurement_tool import build_invalid_marker, load_aggregate_records


ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "run_temb_hoist_nsys_ab.sh"
S0_RUNNER = ROOT / "run_s0_measurement.sh"
MEASUREMENT = ROOT / "measurement.py"
NSYS_METRICS = ROOT / "nsys_metrics.py"


def test_wrapper_pins_exact_sp2_window_and_variant_order() -> None:
    text = WRAPPER.read_text()
    expected = (
        "export MINWM_S0_SP_DEGREES=2",
        "export MINWM_S0_PROFILE_PRECONDITION_CHUNKS=20",
        "export MINWM_S0_PROFILE_DISCARD_CHUNKS=1",
        "export MINWM_S0_PROFILE_MEASURED_CHUNKS=10",
        "export MINWM_S0_KV_CACHE_NUM_FRAMES=45",
        '"${MINWM_S1_PROFILER_OFF_GIT_REF}"',
        'run_variant legacy 0 "${MINWM_S1_LEGACY_OFF_ROOT}"',
        'run_variant candidate 1 "${MINWM_S1_CANDIDATE_OFF_ROOT}"',
    )
    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)
    assert "unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR" in text


def test_wrapper_separates_resumed_off_and_formal_nsys_refs() -> None:
    text = WRAPPER.read_text()
    assert ': "${MINWM_S1_PROFILER_OFF_GIT_REF:' in text
    assert (
        '"${lane}" "${MINWM_S1_PROFILER_OFF_GIT_REF}" "${SCRIPT_DIR}"' in text
    )
    assert '"${variant}" "${SGLANG_GIT_REF}"' in text
    assert "formal Nsight result did not run the pinned product ref" in text
    assert "MINWM_S1_LEGACY_OFF_RUNNER_REF" in text
    assert "MINWM_S1_CANDIDATE_OFF_RUNNER_REF" in text
    assert "_assert_historical_runner_flag_guard(contents, variant)" in text


def test_wrapper_keeps_post_validation_failure_lane_scoped() -> None:
    text = WRAPPER.read_text()
    scoped = 'CURRENT_LANE_DIR="${profile_dir}"'
    run = 'bash "${SCRIPT_DIR}/run_s0_measurement.sh"'
    validate = "--require-complete-stable-nsys"
    clear = 'CURRENT_LANE_DIR=""'
    start = text.index(scoped)
    run_position = text.index(run, start)
    assert start < run_position
    assert 'mkdir -p "${CURRENT_LANE_DIR}"' not in text[start:run_position]
    assert run_position < text.index(validate, start)
    formal_validate = 'validate_profile_result "${variant}"'
    assert text.index(validate, start) < text.index(formal_validate, start)
    assert text.index(formal_validate, start) < text.index(clear, start)
    assert text.index(validate, start) < text.index(clear, start)
    preflight = 'CURRENT_LANE_DIR="${preflight_dir}"'
    assert text.index(preflight) < text.index(scoped)
    assert text.index(preflight) < text.index(
        'mkdir -p "${CURRENT_LANE_DIR}"', text.index(preflight)
    )
    assert 'CURRENT_LANE_DIR="${RESULT_ROOT}/nsys-comparison"' in text
    comparison = text.index('CURRENT_LANE_DIR="${RESULT_ROOT}/nsys-comparison"')
    assert comparison < text.index('mkdir -p "${CURRENT_LANE_DIR}"', comparison)


def test_profiler_on_marker_excludes_only_failed_variant(tmp_path: Path) -> None:
    legacy = (
        tmp_path
        / "run-legacy"
        / "s0-measurement"
        / "sp2"
        / "profiler-on"
        / "measurement.json"
    )
    candidate = (
        tmp_path
        / "run-candidate"
        / "s0-measurement"
        / "sp2"
        / "profiler-on"
        / "measurement.json"
    )
    for path, label in ((legacy, "legacy"), (candidate, "candidate")):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"variant": label}))

    marker_path = candidate.parent / "invalid-marker-test.json"
    marker = build_invalid_marker(
        candidate.parent,
        reason="post-run validation failed",
        marker_path=marker_path,
        timestamp_utc="2026-08-07T00:00:00+00:00",
    )
    marker_path.write_text(json.dumps(marker))

    accepted, excluded = load_aggregate_records([legacy, candidate])
    assert accepted == [{"variant": "legacy"}]
    assert excluded == [candidate]
    assert marker["original_root"] == str(candidate.parent.resolve())
    assert [item["original_path"] for item in marker["files"]] == [
        str(candidate.resolve())
    ]


def test_canonical_runner_collects_all_targets_and_fails_closed() -> None:
    text = S0_RUNNER.read_text()
    assert "--gpu-metrics-devices=all" in text
    assert "--require-complete-stable-nsys" in text
    assert 'SGLANG_REALTIME_NSYS_WARMUP_CHUNKS="${PROFILE_DISCARD_CHUNKS}"' in text
    assert 'SGLANG_REALTIME_NSYS_MEASURED_CHUNKS="${PROFILE_MEASURED_CHUNKS}"' in text
    measurement = MEASUREMENT.read_text()
    assert "API_BOUNDARY_ATTRIBUTION_POLICY" in measurement
    assert "boundary_included_by_start_count" in measurement
    assert "aggregation_mode" in measurement
    metrics = NSYS_METRICS.read_text()
    assert "_discrete_event_start_attribution" in metrics
    assert "boundary_event_examples" in metrics
    assert "streaming selected metricId rows" in metrics


def test_wrapper_asserts_formal_stage_window_and_gpu_target_contract() -> None:
    wrapper = WRAPPER.read_text()
    comparator = (ROOT / "compare_temb_hoist_nsys.py").read_text()
    assert "validate_profile_result" in wrapper
    assert "_assert_contract(record)" in wrapper
    for name in ("dit_wall_ms", "vae_wall_ms", "dit_cuda_ms", "vae_cuda_ms"):
        assert name in comparator
    assert 'value["count"] != 10' in comparator
    assert 'expected_stable_chunk_indices"] != list(range(1, 11))' in comparator
    assert 'value["collected_target_count"] != 8' in comparator
    assert 'value["active_target_count"] != 2' in comparator
