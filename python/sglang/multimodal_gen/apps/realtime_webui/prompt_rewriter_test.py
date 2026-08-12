#!/usr/bin/env python3

import asyncio
import unittest
from types import SimpleNamespace

from prompt_rewriter import PromptRewriter, build_user_message


class FakeModels:
    def __init__(self):
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed={
                "prompt": "A stable cinematic camera follows the dragon through falling snow.",
                "change_type": "persistent",
            }
        )


class PromptRewriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_rewrite_returns_prompt_and_lifetime(self):
        models = FakeModels()
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        rewriter = PromptRewriter(client=client, request_timeout_seconds=1)
        result = await rewriter.rewrite("让世界下雪", "A dragon flies over a valley.")
        self.assertEqual(result.change_type.value, "persistent")
        self.assertIn("falling snow", result.prompt)
        self.assertIn("让世界下雪", models.calls[0]["contents"][0])
        self.assertIn("A dragon flies over a valley.", models.calls[0]["contents"][0])
        self.assertEqual(models.calls[0]["model"], "gemini-3.1-flash-lite")

    def test_build_message_rejects_missing_state(self):
        with self.assertRaisesRegex(ValueError, "previous_prompt"):
            build_user_message("", "下雪")


if __name__ == "__main__":
    unittest.main()
