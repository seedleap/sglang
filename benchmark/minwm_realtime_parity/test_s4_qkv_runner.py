from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUALITY_RUNNER = Path(__file__).with_name("run_s4_qkv_quality.sh")
MEASUREMENT_RUNNER = Path(__file__).with_name("run_s0_measurement.sh")
MINWM_MODEL = ROOT / "python/sglang/multimodal_gen/runtime/models/dits/minwm.py"


def test_qkv_quality_runner_uses_emitted_qk_norm_probe_names() -> None:
    runner = QUALITY_RUNNER.read_text()
    model = MINWM_MODEL.read_text()

    for name in ("self_q_norm_000.pt", "self_k_norm_000.pt"):
        assert f'"{name}"' in runner
        stem = name.removesuffix("_000.pt")
        assert f'f"{stem}_{{parity_index:03d}}.pt"' in model

    assert "self_norm_q_output_000.pt" not in runner
    assert "self_norm_k_output_000.pt" not in runner


def test_qkv_quality_runner_scopes_existing_tp2_blocker() -> None:
    runner = QUALITY_RUNNER.read_text()

    for invocation in (
        "run_lane control-compile-reference 0 1 1 1",
        "run_lane qkv-compile-reference 1 1 1 1",
        "run_lane control-compile 0 1 1 1",
        "run_lane qkv-compile 1 1 1 1",
    ):
        assert invocation in runner
    for prefix in (
        "control_compile_reference",
        "qkv_compile_reference",
        "control_compile",
        "qkv_compile",
    ):
        assert f'"{prefix}"' in runner
    assert '"control_eager__control_compile": ["compile_enabled"]' in runner
    assert '"qkv_eager__qkv_compile": ["compile_enabled"]' in runner
    assert '"control_eager__qkv_eager": ["minwm_fused_qkv_projection"]' in runner
    assert '"control_compile__qkv_compile": ["minwm_fused_qkv_projection"]' in runner
    for field in (
        "seed",
        "total_chunks",
        "kv_cache_num_frames",
        "prompt",
        "first_frame_source",
        "sp_degree",
        "precision",
        "minwm_fused_qkv_projection",
    ):
        assert f'"{field}"' in runner
    assert 'for field in ("contract", "request")' in runner
    assert "actual_request_metadata_equal_across_all_lanes" in runner
    assert "first_difference_index" in runner
    assert "left_npy_sha256" in runner
    assert "per_frame_sha256_equal" in runner
    assert "control_eager_vs_control_compile" in runner
    assert "qkv_eager_vs_qkv_compile" in runner
    assert "control_eager_vs_qkv_eager" in runner
    assert "control_compile_vs_qkv_compile" in runner
    assert "existing_whole_model_compile_blocker" in runner
    assert "qkv_compile_additional_regression" in runner
    assert "compile_off_continuation_allowed" in runner
    assert "compile-client-contract-verified.json" in runner
    assert "compile-four-corner-summary.json" in runner
    assert "s4-qkv-quality-summary.json" in runner
    assert "compile gate requires 33 frames" in runner
    assert runner.index('mkdir -p "${SP1_RESULTS}"') < runner.index(
        'python3 - "${COMPILE_CASES}" "${CASE_ID}" "${SP1_RESULTS}"'
    )
    assert runner.index("compile four-corner gate failed") < runner.index(
        "run_lane control 0 1 1 1 baseline"
    )
    assert "run_tp2_existing_blocker tp2-control 0" in runner
    assert "run_tp2_existing_blocker tp2-candidate 1" in runner
    assert "MinWMRMSNorm' object has no attribute 'variance_epsilon" in runner
    assert "candidate[: value.shape[0]]" not in runner
    assert "candidate_tp2.npy" not in runner


def test_qkv_abba_runner_restarts_server_at_every_position() -> None:
    runner = MEASUREMENT_RUNNER.read_text()
    position_start = runner.index("run_profiler_off_position()")
    position_end = runner.index("run_abba_measurements()")
    position = runner[position_start:position_end]

    expected_order = [
        'record_compile_cache_state "${position_dir}/compile-cache-before.json"',
        "start_server \\",
        "run_client \\",
        "stop_server",
        "assert_server_stopped",
        'record_compile_cache_state "${position_dir}/compile-cache-after.json"',
    ]
    offsets = [position.index(statement) for statement in expected_order]
    assert offsets == sorted(offsets)
    assert 'invalid_scope="${position_dir}"' in position
    assert 'local output="${position_dir}/profiler-off.json"' in position

    assert '"${abba_lanes[*]}" != "candidate control control candidate"' in runner
    assert "position-${position}-${lane}" in runner
    assert "server_restart_per_position" in runner
    assert "compile_cache_policy" in runner
    assert runner.index('if [[ -n "${ABBA_SEQUENCE}" ]]') < runner.index(
        'for qkv_lane in "${qkv_lanes[@]}"'
    )
