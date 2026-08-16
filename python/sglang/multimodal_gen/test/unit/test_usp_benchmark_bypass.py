import torch

from sglang.multimodal_gen.runtime.layers.usp import (
    _benchmark_reshape_input_without_a2a,
    _benchmark_reshape_output_without_a2a,
)


def test_benchmark_bypass_round_trips_shapes_head_dim_2():
    x = torch.empty(1, 585, 40, 384)

    attention_shape = _benchmark_reshape_input_without_a2a(x, 8, head_dim=2)
    restored_shape = _benchmark_reshape_output_without_a2a(
        attention_shape, 8, head_dim=2
    )

    assert attention_shape.shape == (1, 4680, 5, 384)
    assert restored_shape.shape == x.shape
    assert attention_shape.numel() == x.numel()


def test_benchmark_bypass_round_trips_shapes_head_dim_1():
    x = torch.empty(1, 40, 585, 128)

    attention_shape = _benchmark_reshape_input_without_a2a(x, 8, head_dim=1)
    restored_shape = _benchmark_reshape_output_without_a2a(
        attention_shape, 8, head_dim=1
    )

    assert attention_shape.shape == (1, 5, 4680, 128)
    assert restored_shape.shape == x.shape
    assert attention_shape.numel() == x.numel()
