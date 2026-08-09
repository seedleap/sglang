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
        'run_headline_lane "${degree}" a1 111',
        'run_headline_lane "${degree}" a1 000',
        'run_headline_lane "${degree}" a2 000',
        'run_headline_lane "${degree}" a2 111',
    )
    positions = [runner.index(item) for item in required_order]
    assert positions == sorted(positions)
    assert runner.count('run_headline_lane "${degree}" adaptive "${config}"') == 1
    assert '"order": ["111", "000", "000", "111"]' in runner
    assert "MINWM_S0_OFF_REPEAT_COUNT=1" in runner
    assert 'POSITION_DRIFT_PCT="${MINWM_S5_POSITION_DRIFT_PCT:-3.0}"' in runner
    assert 'assert_headline_cv_gate "${degree}"' in runner
    assert "recorded_triggers.update(triggers)" in runner


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
    assert '"candidate_stable_fallback_launch_slots"' in runner


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
    assert 'BITWISE_ONLY="${MINWM_S0_BITWISE_ONLY:-0}"' in runner
    assert 'if [[ "${BITWISE_ONLY}" == "1" ]]; then\n      return' in runner


def test_correctness_servers_do_not_precondition_headline_abba():
    runner = (ROOT / "run_s5_fused_ops_integration.sh").read_text()
    assert "run_correctness_lane()" in runner
    assert "export MINWM_S0_BITWISE_ONLY=1" in runner
    assert "export MINWM_S0_BITWISE_ONLY=0" in runner
    last_headline = runner.index('run_headline_lane "${degree}" a2 111')
    first_correctness = runner.index('run_correctness_lane "${degree}" 111')
    assert last_headline < first_correctness


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


def test_nsys_requires_full_post_a2a_launch_coverage():
    comparator = (ROOT / "compare_s5_fused_ops.py").read_text()
    assert 'degree * 10 * 150 if config[1] == "1" else 0' in comparator
    assert '"explicit_fallback_launch_slots"' in comparator
    assert '"expected_fused_post_launches_per_active_gpu_per_chunk": 150' in comparator


def test_measurement_trace_relay_does_not_restore_removed_client_trace_state():
    source = (
        ROOT.parents[1]
        / "python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime"
        / "realtime_video_api.py"
    ).read_text()
    assert "_install_realtime_trace_sink(session, trace_sink)" in source
    assert "_send_realtime_trace_events(ws, trace_queue)" in source
    assert "session.client_trace" not in source


def test_attempt06_pins_the_relay_fix_and_current_product_tree():
    manifest = (
        ROOT
        / "k8s/minwm_s5_fusedops_h200_20260809_attempt06.yaml"
    ).read_text()
    runner_ref = "2adb6e1437fd4d06127dc938786354b4a7b1f63c"
    product_ref = "dc4c865a6e41dd26f5feaeb8f9236facd5725082"
    assert "minwm-s5-fusedops-h200-20260809-05" not in manifest
    assert manifest.count("minwm-s5-fusedops-h200-20260809-06") == 6
    assert manifest.count(runner_ref) == 3
    assert manifest.count(product_ref) == 3
    assert "backoffLimit: 0" in manifest
    assert 'nvidia.com/gpu: "8"' in manifest


def test_attempt07_pins_heartbeat_runner_and_current_product_tree():
    manifest = (
        ROOT
        / "k8s/minwm_s5_fusedops_h200_20260809_attempt07.yaml"
    ).read_text()
    runner_ref = "ed255b3c6b2af96f81b08f55393a5a8bb32f4644"
    product_ref = "dc4c865a6e41dd26f5feaeb8f9236facd5725082"
    assert "minwm-s5-fusedops-h200-20260809-06" not in manifest
    assert manifest.count("minwm-s5-fusedops-h200-20260809-07") == 6
    assert manifest.count(runner_ref) == 3
    assert manifest.count(product_ref) == 3
    assert "backoffLimit: 0" in manifest
    assert 'nvidia.com/gpu: "8"' in manifest


def test_attempt08_pins_chunk_trace_runner_and_current_product_tree():
    manifest = (
        ROOT
        / "k8s/minwm_s5_fusedops_h200_20260809_attempt08.yaml"
    ).read_text()
    runner_ref = "b2c3227d1d31ae95d18bf97a41337f4311f1cca2"
    product_ref = "dc4c865a6e41dd26f5feaeb8f9236facd5725082"
    assert "minwm-s5-fusedops-h200-20260809-07" not in manifest
    assert manifest.count("minwm-s5-fusedops-h200-20260809-08") == 6
    assert manifest.count(runner_ref) == 3
    assert manifest.count(product_ref) == 3
    assert "backoffLimit: 0" in manifest
    assert 'nvidia.com/gpu: "8"' in manifest
