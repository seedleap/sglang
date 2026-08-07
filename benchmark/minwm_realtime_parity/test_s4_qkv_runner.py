from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUALITY_RUNNER = Path(__file__).with_name("run_s4_qkv_quality.sh")
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

    assert "run_lane candidate-compile-reference 1 1 1 1" in runner
    assert "candidate_compile_reference.npy" in runner
    assert '"allowed_execution_difference": ["compile_enabled"]' in runner
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
    assert 'reference_record["contract"] != compiled_record["contract"]' in runner
    assert 'reference_record["request"] != compiled_record["request"]' in runner
    assert "reference_frame_sha256" in runner
    assert "per_frame_sha256_equal" in runner
    assert "compile-client-contract-verified.json" in runner
    assert "s4-qkv-quality-summary.json" in runner
    assert "reference.shape != compiled.shape" in runner
    assert "run_tp2_existing_blocker tp2-control 0" in runner
    assert "run_tp2_existing_blocker tp2-candidate 1" in runner
    assert "MinWMRMSNorm' object has no attribute 'variance_epsilon" in runner
    assert "candidate[: value.shape[0]]" not in runner
    assert "candidate_tp2.npy" not in runner
