from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_postprocess_fusions import (  # noqa: E402
    _bitquant_self_post,
    _minwm_adaln,
    _minwm_layer_norm,
    _round_fp32_to_bf16_fp32,
    difference_metrics,
)


def test_explicit_bf16_rne_matches_materialized_cast_bits() -> None:
    bit_patterns = torch.tensor(
        [
            0x00000000,
            0x80000000,
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
            0x3F818000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    values = bit_patterns.view(torch.float32)

    actual = _round_fp32_to_bf16_fp32(values)
    expected = values.to(torch.bfloat16).float()

    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_explicit_bf16_rne_canonicalizes_nan_before_reduction() -> None:
    nan_patterns = torch.tensor(
        [0x7F800001, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF], dtype=torch.int64
    ).to(torch.int32)

    actual = _round_fp32_to_bf16_fp32(nan_patterns.view(torch.float32))

    assert torch.equal(
        actual.view(torch.int32),
        torch.full((4,), 0x7FC00000, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("gate_shape", "noncontiguous", "autocast"),
    [
        ("vector", False, False),
        ("row", False, True),
        ("batch", True, False),
        ("batch", True, True),
    ],
)
def test_bitquant_golden_matches_materialized_boundary_on_cpu(
    gate_shape: str, noncontiguous: bool, autocast: bool
) -> None:
    generator = torch.Generator().manual_seed(20260807)

    def activation() -> torch.Tensor:
        if not noncontiguous:
            return torch.randn(2, 7, 3072, generator=generator, dtype=torch.bfloat16)
        storage = torch.randn(2, 7, 3072 * 2, generator=generator, dtype=torch.bfloat16)
        return storage[..., ::2]

    hidden_states = activation()
    update = activation()
    timestep_storage = torch.randn(
        2, 7, 6, 3072, generator=generator, dtype=torch.bfloat16
    )
    weight = torch.randn(3072, generator=generator, dtype=torch.bfloat16)
    bias = torch.randn(3072, generator=generator, dtype=torch.bfloat16)
    if gate_shape == "vector":
        model_gate = torch.randn(3072, generator=generator, dtype=torch.bfloat16)
    elif gate_shape == "row":
        model_gate = torch.randn(1, 3072, generator=generator, dtype=torch.bfloat16)
    else:
        model_gate = torch.randn(2, 1, 3072, generator=generator, dtype=torch.bfloat16)
    timestep_gate = timestep_storage.select(2, 2)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        residual = _minwm_adaln(
            hidden_states,
            y=update,
            m_gate=model_gate,
            e_gate=timestep_gate,
        )
        expected = (
            residual,
            _minwm_layer_norm(
                residual,
                eps=1e-6,
                weight=weight,
                bias=bias,
            ),
        )
        actual = _bitquant_self_post(
            hidden_states,
            update,
            model_gate,
            timestep_gate,
            weight,
            bias,
            1e-6,
        )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_difference_metrics_reports_bf16_ulp_and_first_location() -> None:
    actual = torch.tensor([1.0, 1.0078125], dtype=torch.bfloat16)
    expected = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)

    metrics = difference_metrics(actual, expected)

    assert metrics["changed_count"] == 1
    assert metrics["changed_fraction"] == 0.5
    assert metrics["max_abs"] == 0.0078125
    assert metrics["max_ulp"] == 1
    assert metrics["first_difference"]["index"] == [1]
    assert metrics["first_difference"]["ulp"] == 1
