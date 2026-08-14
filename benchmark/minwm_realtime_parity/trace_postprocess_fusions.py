#!/usr/bin/env python3
"""Trace one MinWM postprocessing segment without defining benchmark metrics."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.models.dits.minwm import (
    _minwm_adaln,
    _minwm_adaln_op,
    _minwm_layer_norm,
    _minwm_layer_norm_op,
    _MinWMSegmentCompile,
)
from sglang.multimodal_gen.runtime.models.dits import minwm as minwm_module


def _round_fp32_to_bf16_fp32(values: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to BF16 while keeping an FP32 tensor.

    A normal ``values.to(torch.bfloat16).float()`` round-trip can be folded away
    when its producer and consumer are fused by Inductor. Expressing BF16 RNE in
    integer bits makes the rounding data-dependent and therefore observable to
    the LayerNorm reduction without requiring a store/load boundary.
    """
    bits = values.view(torch.int32)
    lower = bits & 0xFFFF
    upper = bits & -0x10000
    upper_lsb = (upper >> 16) & 1
    round_up = (lower > 0x8000) | ((lower == 0x8000) & (upper_lsb == 1))
    rounded_bits = upper + (round_up.to(torch.int32) << 16)
    is_nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    rounded_bits = torch.where(
        is_nan,
        torch.full_like(rounded_bits, 0x7FC00000),
        rounded_bits,
    )
    return rounded_bits.view(torch.float32)


def _proposed_self_post_op(
    hidden_states: torch.Tensor,
    attn_output: torch.Tensor,
    model_gate: torch.Tensor,
    timestep_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the rejected S2 proposal without changing the product path."""
    residual = (
        hidden_states.float()
        + attn_output.float() * (model_gate.float() + timestep_gate.float())
    ).type_as(hidden_states)
    normalized = torch.nn.functional.layer_norm(
        residual.float(),
        (residual.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).type_as(hidden_states)
    return residual, normalized


def _proposed_self_post(
    hidden_states: torch.Tensor,
    attn_output: torch.Tensor,
    model_gate: torch.Tensor,
    timestep_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.to(hidden_states.dtype) if weight is not None else None
    bias = bias.to(hidden_states.dtype) if bias is not None else None
    return _MinWMSegmentCompile.get(_proposed_self_post_op, hidden_states.is_cuda)(
        hidden_states,
        attn_output,
        model_gate,
        timestep_gate,
        weight,
        bias,
        eps,
    )


def _bitquant_self_post_op(
    hidden_states: torch.Tensor,
    attn_output: torch.Tensor,
    model_gate: torch.Tensor,
    timestep_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test a single graph with an explicit, non-foldable BF16 RNE boundary."""
    residual_fp32 = hidden_states.float() + attn_output.float() * (
        model_gate.float() + timestep_gate.float()
    )
    quantized_residual_fp32 = _round_fp32_to_bf16_fp32(residual_fp32)
    residual = quantized_residual_fp32.type_as(hidden_states)
    normalized = torch.nn.functional.layer_norm(
        quantized_residual_fp32,
        (residual.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).type_as(hidden_states)
    return residual, normalized


def _bitquant_self_post(
    hidden_states: torch.Tensor,
    attn_output: torch.Tensor,
    model_gate: torch.Tensor,
    timestep_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.to(hidden_states.dtype) if weight is not None else None
    bias = bias.to(hidden_states.dtype) if bias is not None else None
    return _MinWMSegmentCompile.get(_bitquant_self_post_op, hidden_states.is_cuda)(
        hidden_states,
        attn_output,
        model_gate,
        timestep_gate,
        weight,
        bias,
        eps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        choices=(
            "self-baseline",
            "self-proposed",
            "self-bitquant",
            "cross",
            "ffn",
        ),
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument(
        "--gate-shape", choices=("vector", "row", "batch"), default="row"
    )
    parser.add_argument(
        "--input-layout",
        choices=("contiguous", "noncontiguous"),
        default="contiguous",
    )
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--disable-segment-compile", action="store_true")
    parser.add_argument("--detailed-correctness", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-kernels", action="store_true")
    return parser.parse_args()


def _coordinates(flat_index: int, shape: torch.Size) -> list[int]:
    coordinates = []
    for dimension in reversed(shape):
        coordinates.append(flat_index % dimension)
        flat_index //= dimension
    return list(reversed(coordinates))


def _ordered_bf16_bits(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    return torch.where(negative, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)


def difference_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "bitwise_exact": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    changed = actual != expected
    changed_count = torch.count_nonzero(changed).item()
    metrics = {
        "bitwise_exact": changed_count == 0,
        "changed_count": changed_count,
        "changed_fraction": changed_count / actual.numel(),
        "max_abs": (actual.float() - expected.float()).abs().max().item(),
    }
    if actual.dtype == torch.bfloat16:
        ulp_distance = (_ordered_bf16_bits(actual) - _ordered_bf16_bits(expected)).abs()
        metrics["max_ulp"] = ulp_distance.max().item()
    if changed_count:
        flat_index = torch.nonzero(changed.flatten(), as_tuple=False)[0].item()
        metrics["first_difference"] = {
            "flat_index": flat_index,
            "index": _coordinates(flat_index, actual.shape),
            "actual": actual.flatten()[flat_index].float().item(),
            "expected": expected.flatten()[flat_index].float().item(),
        }
        if actual.dtype == torch.bfloat16:
            metrics["first_difference"]["ulp"] = int(
                (
                    _ordered_bf16_bits(actual.flatten()[flat_index : flat_index + 1])
                    - _ordered_bf16_bits(
                        expected.flatten()[flat_index : flat_index + 1]
                    )
                )
                .abs()
                .item()
            )
    return metrics


def compare_bitwise(
    actual: torch.Tensor | tuple[torch.Tensor, ...],
    expected: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[bool, float]:
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_values) != len(expected_values):
        raise AssertionError("candidate and reference return different arity")
    exact = True
    max_abs = 0.0
    for index, (actual_value, expected_value) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        metrics = difference_metrics(actual_value, expected_value)
        print(f"comparison={json.dumps({'output': index, **metrics}, sort_keys=True)}")
        if not metrics["bitwise_exact"]:
            exact = False
            difference = metrics["max_abs"]
            max_abs = max(max_abs, difference)
    return exact, max_abs


def build_candidate(
    candidate: str,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    gate_shape: str,
    input_layout: str,
) -> tuple[
    Callable[[], object],
    Callable[[], object],
    Callable[[object], object] | None,
]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260807)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

    def activation() -> torch.Tensor:
        if input_layout == "contiguous":
            return randn(batch_size, sequence_length, hidden_size)
        storage = randn(batch_size, sequence_length, hidden_size * 2)
        value = storage[..., ::2]
        if value.is_contiguous():
            raise AssertionError("failed to construct a non-contiguous activation")
        return value

    hidden_states = activation()
    update = activation()
    model_values = randn(1, 6, hidden_size)
    # This mirrors temb[:, frame_index].select(-2, index): the selected view
    # has a 6*D token stride instead of a contiguous D token stride.
    timestep_storage = randn(batch_size, sequence_length, 6, hidden_size)
    weight = randn(hidden_size)
    bias = randn(hidden_size)
    eps = 1e-6

    if gate_shape == "vector":
        model_gate = randn(hidden_size)
    elif gate_shape == "row":
        model_gate = randn(1, hidden_size)
    else:
        model_gate = randn(batch_size, 1, hidden_size)

    if candidate in {"self-baseline", "self-proposed", "self-bitquant"}:

        def reference():
            residual = _minwm_adaln_op(
                hidden_states,
                y=update,
                m_gate=model_gate,
                e_gate=timestep_storage.select(2, 2),
            )
            normalized = _minwm_layer_norm_op(residual, weight, bias, eps)
            return residual, normalized

        if candidate == "self-baseline":

            def operation():
                residual = _minwm_adaln(
                    hidden_states,
                    y=update,
                    m_gate=model_gate,
                    e_gate=timestep_storage.select(2, 2),
                )
                normalized = _minwm_layer_norm(
                    residual,
                    eps=eps,
                    weight=weight,
                    bias=bias,
                )
                return residual, normalized

        elif candidate == "self-proposed":

            def operation():
                return _proposed_self_post(
                    hidden_states,
                    update,
                    model_gate,
                    timestep_storage.select(2, 2),
                    weight,
                    bias,
                    eps,
                )

            # Compare the proposed one-segment graph with the actual two-segment
            # product path. This is the numerical contract that rejected it.
            def reference():
                residual = _minwm_adaln(
                    hidden_states,
                    y=update,
                    m_gate=model_gate,
                    e_gate=timestep_storage.select(2, 2),
                )
                normalized = _minwm_layer_norm(
                    residual,
                    eps=eps,
                    weight=weight,
                    bias=bias,
                )
                return residual, normalized

        else:

            def operation():
                return _bitquant_self_post(
                    hidden_states,
                    update,
                    model_gate,
                    timestep_storage.select(2, 2),
                    weight,
                    bias,
                    eps,
                )

            def reference():
                residual = _minwm_adaln(
                    hidden_states,
                    y=update,
                    m_gate=model_gate,
                    e_gate=timestep_storage.select(2, 2),
                )
                normalized = _minwm_layer_norm(
                    residual,
                    eps=eps,
                    weight=weight,
                    bias=bias,
                )
                return residual, normalized

        def materialized_norm(actual):
            residual, _ = actual
            return _minwm_layer_norm(
                residual,
                eps=eps,
                weight=weight,
                bias=bias,
            )

        return operation, reference, materialized_norm

    if candidate == "cross":

        def operation():
            return _minwm_adaln(
                hidden_states,
                model_values[:, 3],
                model_values[:, 4],
                timestep_storage.select(2, 3),
                timestep_storage.select(2, 4),
                eps,
                r=update,
            )

        def reference():
            return _minwm_adaln_op(
                hidden_states,
                model_values[:, 3],
                model_values[:, 4],
                timestep_storage.select(2, 3),
                timestep_storage.select(2, 4),
                eps,
                r=update,
            )

        return operation, reference, None

    def operation():
        return _minwm_adaln(
            hidden_states,
            y=update,
            m_gate=model_values[:, 5],
            e_gate=timestep_storage.select(2, 5),
        )

    def reference():
        return _minwm_adaln_op(
            hidden_states,
            y=update,
            m_gate=model_values[:, 5],
            e_gate=timestep_storage.select(2, 5),
        )

    return operation, reference, None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    minwm_module._MINWM_SEGMENT_COMPILE = not args.disable_segment_compile
    minwm_module._MinWMSegmentCompile._compiled.clear()
    operation, reference, materialized_reference = build_candidate(
        args.candidate,
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        args.gate_shape,
        args.input_layout,
    )
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=args.autocast
    )
    with autocast:
        expected = reference()
        for _ in range(args.warmup):
            actual = operation()
    torch.cuda.synchronize()
    bitwise_exact, max_abs = compare_bitwise(actual, expected)
    if args.detailed_correctness and materialized_reference is not None:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=args.autocast
        ):
            materialized_norm = materialized_reference(actual)
        print(
            "comparison="
            + json.dumps(
                {
                    "output": "candidate_norm_vs_layer_norm_of_returned_residual",
                    **difference_metrics(actual[1], materialized_norm),
                },
                sort_keys=True,
            )
        )
        print(
            "comparison="
            + json.dumps(
                {
                    "output": "baseline_norm_vs_layer_norm_of_returned_residual",
                    **difference_metrics(expected[1], materialized_norm),
                },
                sort_keys=True,
            )
        )

    if args.profile_kernels:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.autocast
            ):
                profiled = operation()
            torch.cuda.synchronize()
        profiled_exact, profiled_max_abs = compare_bitwise(profiled, expected)
        bitwise_exact = bitwise_exact and profiled_exact
        max_abs = max(max_abs, profiled_max_abs)
        print(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        )

    start = time.perf_counter()
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=args.autocast
    ):
        for _ in range(args.iterations):
            actual = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / args.iterations
    final_exact, final_max_abs = compare_bitwise(actual, expected)
    bitwise_exact = bitwise_exact and final_exact
    max_abs = max(max_abs, final_max_abs)
    print(
        f"candidate={args.candidate} batch_size={args.batch_size} "
        f"sequence_length={args.sequence_length} "
        f"hidden_size={args.hidden_size} mean_ms={elapsed_ms:.6f} "
        f"gate_shape={args.gate_shape} input_layout={args.input_layout} "
        f"autocast={str(args.autocast).lower()} "
        f"segment_compile={str(not args.disable_segment_compile).lower()} "
        f"bitwise_exact={str(bitwise_exact).lower()} max_abs={max_abs}"
    )


if __name__ == "__main__":
    main()
