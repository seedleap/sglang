from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).with_name("benchmark_attention_kernels.py")
SPEC = importlib.util.spec_from_file_location(
    "benchmark_attention_kernels", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("preset", "tokens_per_frame", "query_length", "key_length"),
    [
        ("480p", 1560, 6240, 31200),
        ("704p", 3432, 13728, 68640),
    ],
)
def test_production_shape_geometry(
    preset: str, tokens_per_frame: int, query_length: int, key_length: int
) -> None:
    shape = MODULE.PRESETS[preset]
    shape.validate()
    assert shape.sink_frames == 4
    assert shape.window_frames == 20
    assert shape.num_heads == 24
    assert shape.head_dim == 128
    assert shape.tokens_per_frame == tokens_per_frame
    assert shape.query_length == query_length
    assert shape.key_length == key_length


def test_sink_must_fit_inside_window() -> None:
    shape = MODULE.MinWMAttentionShape(
        name="invalid", width=832, height=480, sink_frames=20, window_frames=20
    )
    with pytest.raises(ValueError, match="sink_frames"):
        shape.validate()
