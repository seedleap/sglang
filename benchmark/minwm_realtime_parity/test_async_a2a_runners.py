from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_RUNNER = Path(__file__).with_name("run_async_a2a_measurement.sh")
QUALITY_RUNNER = Path(__file__).with_name("run_async_a2a_quality.sh")
QUALITY_VALIDATOR = Path(__file__).with_name("validate_async_a2a_quality.py")
MINWM_MODEL = ROOT / "python/sglang/multimodal_gen/runtime/models/dits/minwm.py"


def test_async_a2a_measurement_enforces_alternating_abba_and_sample_floor() -> None:
    runner = MEASUREMENT_RUNNER.read_text()

    assert "candidate baseline baseline candidate" in runner
    assert "baseline candidate candidate baseline" in runner
    assert "MINWM_ASYNC_A2A_MIN_LANE_SAMPLES:-5" in runner
    assert "baseline_count < MIN_LANE_SAMPLES" in runner
    assert "candidate_count < MIN_LANE_SAMPLES" in runner
    assert 'MINWM_ASYNC_A2A="${async_a2a_flag}"' in runner
    assert "MINWM_ASYNC_A2A_OUTPUT=0" in runner
    assert 'MINWM_ASYNC_A2A_BACKEND="${A2A_BACKEND}"' in runner
    assert "SGLANG_REALTIME_TRACE_SYNC_CUDA=0" in runner

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


def test_async_a2a_quality_covers_sp2_sp4_long_run_and_tensor_parity() -> None:
    runner = QUALITY_RUNNER.read_text()
    validator = QUALITY_VALIDATOR.read_text()
    model = MINWM_MODEL.read_text()

    assert "cases_720p_5s.json" in runner
    assert "MINWM_ASYNC_A2A_SP_DEGREES:-2 4" in runner
    assert "MINWM_ASYNC_A2A_STABILITY_REQUESTS:-10" in runner
    assert '--profile bitwise' in runner
    assert 'MINWM_PARITY_DUMP_DIR="${dump_dir}"' in runner
    assert "missing_candidate = baseline_names - candidate_names" in validator
    assert "candidate_names - baseline_names" in validator
    probes = (
        "self_q_norm_000.pt",
        "self_k_norm_000.pt",
        "self_q_roped_000.pt",
        "self_k_roped_000.pt",
        "self_attention_output_000.pt",
        "block0_output_000.pt",
        "output_proj_output_000.pt",
    )
    for probe in probes:
        assert probe in runner
    for probe in probes[:5]:
        assert probe.removesuffix("_000.pt") in model
    assert 'block_name = f"block{block_index}"' in model
    assert '"output_proj": self.proj_out' in model
