from __future__ import annotations

import asyncio
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import msgspec.msgpack
from PIL import Image


MODULE_PATH = (
    Path(__file__).parents[2]
    / "python/sglang/multimodal_gen/runtime/realtime/startup_warmup.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("realtime_startup_warmup", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeSocket:
    def __init__(self, messages: list[bytes]) -> None:
        self.messages = iter(messages)

    async def recv(self) -> bytes:
        return next(self.messages)


class RealtimeStartupWarmupTest(unittest.TestCase):
    def test_lingbot2_request_matches_production_720p_shape(self) -> None:
        module = _load_module()
        request = module.build_warmup_request(
            model="lingbot-world-v2-14b-causal-fast-diffusers",
            first_frame=b"jpeg-bytes",
            trace_id="startup-warmup",
        )

        self.assertEqual(request["type"], "init")
        self.assertEqual(request["generation_mode"], "i2v")
        self.assertEqual(request["size"], "1280x720")
        self.assertEqual(request["first_frame"], b"jpeg-bytes")
        self.assertEqual(request["max_chunks"], 1)
        self.assertEqual(request["realtime_causal_sink_size"], 3)
        self.assertEqual(request["realtime_causal_kv_cache_num_frames"], 12)

    def test_minwm_request_matches_production_shape_and_cache(self) -> None:
        module = _load_module()
        request = module.build_warmup_request(
            model="minwm-production",
            first_frame=b"jpeg-bytes",
            trace_id="startup-warmup",
            size="1280x704",
            guidance_scale=0.0,
            sink_size=8,
            kv_cache_num_frames=32,
        )

        self.assertEqual(request["size"], "1280x704")
        self.assertEqual(request["guidance_scale"], 0.0)
        self.assertEqual(request["realtime_causal_sink_size"], 8)
        self.assertEqual(request["realtime_causal_kv_cache_num_frames"], 32)
        with Image.open(BytesIO(module.create_reference_frame("1280x704"))) as image:
            self.assertEqual(image.size, (1280, 704))

    def test_rejects_invalid_size_and_cache_window(self) -> None:
        module = _load_module()
        with self.assertRaisesRegex(ValueError, "WIDTHxHEIGHT"):
            module.create_reference_frame("720p")
        with self.assertRaisesRegex(ValueError, "larger than sink"):
            module.build_warmup_request(
                model="minwm",
                first_frame=b"jpeg-bytes",
                trace_id="startup-warmup",
                sink_size=8,
                kv_cache_num_frames=8,
            )

    def test_accepts_only_normal_generation_complete_close(self) -> None:
        module = _load_module()
        complete = SimpleNamespace(
            rcvd=SimpleNamespace(code=1000, reason="generation complete"), sent=None
        )
        other = SimpleNamespace(
            rcvd=SimpleNamespace(code=1000, reason="other"), sent=None
        )

        self.assertTrue(module._is_generation_complete_close(complete))
        self.assertFalse(module._is_generation_complete_close(other))

    def test_waits_for_split_frame_payload(self) -> None:
        module = _load_module()
        socket = _FakeSocket(
            [
                msgspec.msgpack.encode(
                    {
                        "type": "frame_batch_header",
                        "chunk_index": 0,
                        "num_frames": 1,
                        "total_size": 4,
                    }
                ),
                b"webp",
            ]
        )

        result = asyncio.run(module.wait_for_first_frame(socket, timeout_s=1))

        self.assertEqual(result["chunk_index"], 0)

    def test_rejects_server_error_before_readiness(self) -> None:
        module = _load_module()
        socket = _FakeSocket(
            [msgspec.msgpack.encode({"type": "error", "content": "compile failed"})]
        )

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            asyncio.run(module.wait_for_first_frame(socket, timeout_s=1))


if __name__ == "__main__":
    unittest.main()
