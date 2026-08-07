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
