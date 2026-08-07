from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_s5_fused_ops", ROOT / "compare_s5_fused_ops.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_has_one_initial_abba_per_sp_and_adaptive_only_followup():
    runner = (ROOT / "run_s5_fused_ops_integration.sh").read_text()
    required_order = (
        'run_headline_lane "${degree}" a1 111 1',
        'run_headline_lane "${degree}" a1 000 1',
        'run_headline_lane "${degree}" a2 000 0',
        'run_headline_lane "${degree}" a2 111 0',
    )
    positions = [runner.index(item) for item in required_order]
    assert positions == sorted(positions)
    assert runner.count('run_headline_lane "${degree}" adaptive "${config}" 0') == 1
    assert '"order": ["111", "000", "000", "111"]' in runner
    assert "MINWM_S0_OFF_REPEAT_COUNT=1" in runner
    assert 'POSITION_DRIFT_PCT="${MINWM_S5_POSITION_DRIFT_PCT:-3.0}"' in runner


def test_runner_covers_primary_and_conditional_pairwise_nsys_matrix():
    runner = (ROOT / "run_s5_fused_ops_integration.sh").read_text()
    assert "primary_order=(000 100 010 001 111)" in runner
    assert "primary_order=(111 001 010 100 000)" in runner
    assert 'if [[ "${pairwise_required}" == "1" ]]' in runner
    assert "for config in 110 101 011" in runner
    for name, value in (
        ("MINWM_S0_PROFILE_PRECONDITION_CHUNKS", 20),
        ("MINWM_S0_PROFILE_DISCARD_CHUNKS", 1),
        ("MINWM_S0_PROFILE_MEASURED_CHUNKS", 10),
    ):
        assert f"export {name}={value}" in runner


def test_runner_records_three_flags_and_strict_correctness_outputs():
    runner = (ROOT / "run_s5_fused_ops_integration.sh").read_text()
    for name in (
        "MINWM_HOIST_TIMESTEP_MODULATION",
        "MINWM_FUSED_POST_A2A_ROPE_CACHE",
        "MINWM_FUSED_QKV_PROJECTION",
    ):
        assert name in runner
    assert "np.array_equal(baseline, candidate)" in runner
    assert "torch.equal(base_tensor, candidate_tensor)" in runner
    assert "chunk_*_latents.pt" in runner
    assert "compatible three-projection fallback" in runner


def test_s0_runner_modes_keep_profiler_off_and_nsys_isolated():
    runner = (ROOT / "run_s0_measurement.sh").read_text()
    assert 'PROFILER_OFF_ONLY="${MINWM_S0_PROFILER_OFF_ONLY:-0}"' in runner
    assert 'NSYS_ONLY="${MINWM_S0_NSYS_ONLY:-0}"' in runner
    assert 'if [[ "${NSYS_ONLY}" == "1" && "${PROFILER_OFF_ONLY}" == "1" ]]' in runner
    assert 'if [[ "${PROFILER_OFF_ONLY}" != "1" ]]; then\n  install_nsys\nfi' in runner
    assert runner.count("assert_no_nsys_processes") >= 4
    assert '"${lane_dir}/correctness-server.log"' in runner
    assert "MINWM_S0_PARITY_DUMP_DIR= \\\n" in runner
    assert '"${lane_dir}/profiler-off-server.log"' in runner


def test_triple_interaction_residual_uses_all_three_singletons():
    module = _load_compare_module()
    summaries = {
        config: {metric: 100.0 for metric in module.SCALAR_METRICS}
        for config in ("000", "100", "010", "001", "111")
    }
    for metric in module.SCALAR_METRICS:
        summaries["100"][metric] = 99.0
        summaries["010"][metric] = 98.0
        summaries["001"][metric] = 97.0
        summaries["111"][metric] = 93.5
    residual = module._interaction_residual(summaries, "111", ("100", "010", "001"))
    for metric in module.SCALAR_METRICS:
        assert residual[metric]["combined_delta"] == -6.5
        assert residual[metric]["component_delta_sum"] == -6.0
        assert residual[metric]["absolute"] == -0.5
        assert residual[metric]["percentage_points_of_000"] == -0.5
