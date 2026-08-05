# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from types import ModuleType

import pytest
import torch


def test_sage_attention3_does_not_mutate_key(monkeypatch):
    fake_sageattn3 = ModuleType("sageattn3")
    observed = {}

    def mutating_sage_attention(query, key, value, *, is_causal):
        del value, is_causal
        observed["key_data_ptr"] = key.data_ptr()
        key.zero_()
        return query.clone()

    fake_sageattn3.sageattn3_blackwell = mutating_sage_attention
    monkeypatch.setitem(sys.modules, "sageattn3", fake_sageattn3)

    module_name = "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3"
    previous_module = sys.modules.pop(module_name, None)
    try:
        backend_module = importlib.import_module(module_name)
        impl = backend_module.SageAttention3Impl(
            num_heads=2,
            head_size=4,
            causal=False,
            softmax_scale=0.5,
        )
        query = torch.randn(1, 3, 2, 4)
        key = torch.randn(1, 5, 2, 4)
        value = torch.randn_like(key)
        original_key = key.clone()

        output = impl.forward(query, key, value, attn_metadata=None)

        torch.testing.assert_close(output, query)
        torch.testing.assert_close(key, original_key)
        assert observed["key_data_ptr"] != key.data_ptr()
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("sageattn3") is None,
    reason="requires CUDA and the SageAttention3 extension",
)
def test_sage_attention3_real_kernel_does_not_mutate_key():
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (12, 0), (12, 1)}:
        pytest.skip(f"SageAttention3 does not support SM{capability[0]}{capability[1]}")

    from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3 import (
        SageAttention3Impl,
    )

    impl = SageAttention3Impl(
        num_heads=8,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
    )
    query = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    original_key = key.clone()

    output = impl.forward(query, key, value, attn_metadata=None)
    torch.cuda.synchronize()

    torch.testing.assert_close(key, original_key, rtol=0, atol=0)
    assert torch.isfinite(output).all()
