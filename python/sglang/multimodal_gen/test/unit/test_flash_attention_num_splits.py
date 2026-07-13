from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn


def _make_impl(config, *, num_splits=None):
    args = SimpleNamespace(attention_backend_config=config)
    with patch(
        "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
        return_value=args,
    ):
        return flash_attn.FlashAttentionImpl(
            num_heads=2,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            num_splits=num_splits,
        )


def test_flash_attention_forwards_explicit_num_splits():
    impl = _make_impl({"fa_num_splits": 2})
    query = torch.empty(1, 4, 2, 128, dtype=torch.bfloat16)
    key = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    sentinel = torch.empty_like(query)

    with (
        patch.object(flash_attn, "fa_ver", 3),
        patch.object(flash_attn, "flash_attn_varlen_func", return_value=sentinel) as op,
    ):
        assert impl.forward(query, key, value) is sentinel

    assert op.call_args.kwargs["num_splits"] == 2


def test_flash_attention_num_splits_defaults_to_heuristic():
    assert _make_impl({}).num_splits == 0


def test_flash_attention_explicit_num_splits_overrides_global_config():
    assert _make_impl({"fa_num_splits": 4}, num_splits=2).num_splits == 2


def test_flash_attention_rejects_negative_num_splits():
    with pytest.raises(ValueError, match="fa_num_splits must be non-negative"):
        _make_impl({"fa_num_splits": -1})
