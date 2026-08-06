import unittest
import sys
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
)

SDPA_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)


class FakeCudaPlatform(CudaPlatformBase):
    is_sm120_device = False
    is_blackwell_device = False
    supports_flash_attention = True
    device_capability = (9, 0)

    @classmethod
    def is_sm120(cls):
        return cls.is_sm120_device

    @classmethod
    def is_blackwell(cls):
        return cls.is_blackwell_device

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls.supports_flash_attention

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        del device_id
        return DeviceCapability(*cls.device_capability)


class TestCudaAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        FakeCudaPlatform.is_sm120_device = False
        FakeCudaPlatform.is_blackwell_device = False
        FakeCudaPlatform.supports_flash_attention = True
        FakeCudaPlatform.device_capability = (9, 0)

    def resolve(
        self,
        selected_backend: AttentionBackendEnum | None,
        dtype: torch.dtype = torch.float16,
    ) -> str:
        return FakeCudaPlatform.get_attn_backend_cls_str(
            selected_backend=selected_backend,
            head_size=128,
            dtype=dtype,
        )

    def test_direct_torch_sdpa_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.TORCH_SDPA), SDPA_BACKEND_CLS_STR
        )

    def test_direct_aiter_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.AITER),
            "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend",
        )

    def test_default_backend_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_requested_flash_attention_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(AttentionBackendEnum.FA), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_for_non_flash_attention_dtype(self):
        self.assertEqual(self.resolve(None, torch.float32), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_without_flash_attention_capability(self):
        FakeCudaPlatform.supports_flash_attention = False

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_blackwell_falls_back_when_flash_attention_prepare_fails(self):
        FakeCudaPlatform.is_blackwell_device = True

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=False,
        ) as prepare_flash_attention:
            self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

        prepare_flash_attention.assert_called_once_with()

    def test_invalid_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid attention backend"):
            self.resolve(AttentionBackendEnum.AITER_SAGE)

    def test_requested_sage_attention_fails_without_dependency(self):
        FakeCudaPlatform.device_capability = (12, 0)
        with patch.dict(sys.modules, {"sageattention": None}):
            with self.assertRaisesRegex(ImportError, "requested but is not installed"):
                self.resolve(AttentionBackendEnum.SAGE_ATTN)

    def test_requested_sage_attention_rejects_b200(self):
        FakeCudaPlatform.device_capability = (10, 0)
        with self.assertRaisesRegex(
            RuntimeError, "does not support CUDA capability 10.0"
        ):
            self.resolve(AttentionBackendEnum.SAGE_ATTN)

    def test_requested_sage_attention3_rejects_datacenter_blackwell(self):
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                FakeCudaPlatform.device_capability = capability
                version = f"{capability[0]}.{capability[1]}"
                with self.assertRaisesRegex(
                    RuntimeError, f"does not support CUDA capability {version}"
                ):
                    self.resolve(AttentionBackendEnum.SAGE_ATTN_3)


if __name__ == "__main__":
    unittest.main()
