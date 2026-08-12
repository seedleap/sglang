#!/usr/bin/env python3

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from world_creator import WorldCreator


class FakeGeminiModels:
    def __init__(self):
        self.calls = []
        self.validation_results = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["config"].response_json_schema
        if "passed" in schema.get("properties", {}):
            passed = self.validation_results.pop(0) if self.validation_results else True
            return SimpleNamespace(
                parsed={
                    "camera_mode_matches": passed,
                    "avatar_orientation_valid": passed,
                    "camera_distance_and_scale_valid": passed,
                    "continuous_walkable_route": passed,
                    "explorable_space": passed,
                    "not_poster_or_portrait": passed,
                    "passed": passed,
                    "feedback": "Show the subject from behind on a walkable road."
                    if not passed
                    else "",
                }
            )
        return SimpleNamespace(
            parsed={
                "camera_mode": "third_person",
                "world_description": "A complete explorable fantasy game world.",
            }
        )


class FakeImageClient:
    def __init__(self):
        self.calls = []
        image = Image.new("RGB", (1536, 1024), "navy")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.encoded = base64.b64encode(buffer.getvalue()).decode()
        self.images = self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=self.encoded)])


class WorldCreatorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "image.json"
        self.config.write_text(
            json.dumps(
                {
                    "model_name": "gpt-image-2",
                    "api_key": "unused-test-key",
                    "api_version": "test",
                    "client_args": {"base_url": "https://example.invalid"},
                }
            )
        )
        self.models = FakeGeminiModels()
        self.gemini = SimpleNamespace(aio=SimpleNamespace(models=self.models))
        self.images = FakeImageClient()
        self.creator = WorldCreator(
            gemini_client_provider=lambda: self._client(),
            image_client=self.images,
            image_config_path=self.config,
            generated_root=self.root / "generated",
        )

    async def _client(self):
        return self.gemini

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_text_creates_description_and_16_by_9_first_frame(self):
        result = await self.creator.complete("龙飞过山谷")
        self.assertTrue(result.image_generated)
        self.assertEqual(len(self.models.calls), 2)
        self.assertEqual(len(self.images.calls), 1)
        with Image.open(io.BytesIO(result.image_bytes)) as image:
            self.assertEqual(image.size, (1280, 720))
        image_prompt = self.images.calls[0]["prompt"]
        self.assertIn("trailing follow-camera view 4-7 meters", image_prompt)
        self.assertIn("subject's back facing the camera", image_prompt)
        self.assertIn("broad unobstructed walkable route", image_prompt)
        self.assertIn("not key art", image_prompt)

    async def test_first_person_uses_gameplay_camera_and_walkable_route(self):
        self.models.generate_content = self._first_person_description
        await self.creator.complete("第一人称走进森林遗迹")
        image_prompt = self.images.calls[0]["prompt"]
        self.assertIn("eye-level forward in-engine gameplay view", image_prompt)
        self.assertIn("Do not show a third-person avatar", image_prompt)
        self.assertIn("broad unobstructed walkable route", image_prompt)

    async def _first_person_description(self, **kwargs):
        self.models.calls.append(kwargs)
        schema = kwargs["config"].response_json_schema
        if "passed" in schema.get("properties", {}):
            return SimpleNamespace(
                parsed={
                    "camera_mode_matches": True,
                    "avatar_orientation_valid": True,
                    "camera_distance_and_scale_valid": True,
                    "continuous_walkable_route": True,
                    "explorable_space": True,
                    "not_poster_or_portrait": True,
                    "passed": True,
                    "feedback": "",
                }
            )
        return SimpleNamespace(
            parsed={
                "camera_mode": "first_person",
                "world_description": "A first-person forest path leads into ruins.",
            }
        )

    async def test_image_is_preserved_and_only_description_is_generated(self):
        result = await self.creator.complete(
            "", b"uploaded-image", "image/jpeg"
        )
        self.assertFalse(result.image_generated)
        self.assertIsNone(result.image_bytes)
        self.assertEqual(len(self.images.calls), 0)
        self.assertEqual(len(self.models.calls[0]["contents"]), 2)

    async def test_failed_visual_gate_regenerates_with_review_feedback(self):
        self.models.validation_results = [False, True]
        result = await self.creator.complete("孙悟空在仙境")
        self.assertTrue(result.image_generated)
        self.assertEqual(len(self.images.calls), 2)
        self.assertIn(
            "Show the subject from behind on a walkable road.",
            self.images.calls[1]["prompt"],
        )

    async def test_text_and_image_are_combined_without_redrawing(self):
        result = await self.creator.complete(
            "夜晚下雪", b"uploaded-image", "image/png"
        )
        self.assertFalse(result.image_generated)
        self.assertIn("夜晚下雪", self.models.calls[0]["contents"][0])
        self.assertEqual(len(self.images.calls), 0)

    async def test_neither_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "required"):
            await self.creator.complete("")


if __name__ == "__main__":
    unittest.main()
