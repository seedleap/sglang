#!/usr/bin/env python3
"""Trace one MinWM postprocessing segment without defining benchmark metrics."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.models.dits.minwm import (
    _minwm_adaln,
    _minwm_adaln_op,
    _minwm_layer_norm,
    _minwm_layer_norm_op,
    _minwm_self_attn_post,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("self-baseline", "self-fused", "cross", "ffn"),
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile-kernels", action="store_true")
    return parser.parse_args()


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
        if not torch.equal(actual_value, expected_value):
            exact = False
            difference = (
                (actual_value.float() - expected_value.float()).abs().max().item()
            )
            max_abs = max(max_abs, difference)
            print(f"output={index} bitwise_exact=false max_abs={difference}")
    return exact, max_abs


def build_candidate(
    candidate: str, sequence_length: int, hidden_size: int
) -> tuple[Callable[[], object], Callable[[], object]]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260807)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

    hidden_states = randn(1, sequence_length, hidden_size)
    update = randn(1, sequence_length, hidden_size)
    model_values = randn(1, 6, hidden_size)
    # This mirrors temb[:, frame_index].select(-2, index): the selected view
    # has a 6*D token stride instead of a contiguous D token stride.
    timestep_storage = randn(1, sequence_length, 6, hidden_size)
    weight = randn(hidden_size)
    bias = randn(hidden_size)
    eps = 1e-6

    if candidate in {"self-baseline", "self-fused"}:

        def reference():
            residual = _minwm_adaln_op(
                hidden_states,
                y=update,
                m_gate=model_values[:, 2],
                e_gate=timestep_storage.select(2, 2),
            )
            normalized = _minwm_layer_norm_op(residual, weight, bias, eps)
            return residual, normalized

        if candidate == "self-baseline":

            def operation():
                residual = _minwm_adaln(
                    hidden_states,
                    y=update,
                    m_gate=model_values[:, 2],
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
                return _minwm_self_attn_post(
                    hidden_states,
                    update,
                    model_values[:, 2],
                    timestep_storage.select(2, 2),
                    weight=weight,
                    bias=bias,
                    eps=eps,
                )

        return operation, reference

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

        return operation, reference

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

    return operation, reference


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    operation, reference = build_candidate(
        args.candidate, args.sequence_length, args.hidden_size
    )
    expected = reference()
    for _ in range(args.warmup):
        actual = operation()
    torch.cuda.synchronize()
    bitwise_exact, max_abs = compare_bitwise(actual, expected)

    if args.profile_kernels:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            profiled = operation()
            torch.cuda.synchronize()
        profiled_exact, profiled_max_abs = compare_bitwise(profiled, expected)
        bitwise_exact = bitwise_exact and profiled_exact
        max_abs = max(max_abs, profiled_max_abs)
        print(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        )

    start = time.perf_counter()
    for _ in range(args.iterations):
        actual = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / args.iterations
    final_exact, final_max_abs = compare_bitwise(actual, expected)
    bitwise_exact = bitwise_exact and final_exact
    max_abs = max(max_abs, final_max_abs)
    print(
        f"candidate={args.candidate} sequence_length={args.sequence_length} "
        f"hidden_size={args.hidden_size} mean_ms={elapsed_ms:.6f} "
        f"bitwise_exact={str(bitwise_exact).lower()} max_abs={max_abs}"
    )


if __name__ == "__main__":
    main()
