#!/usr/bin/env python3

import asyncio
import unittest
from types import SimpleNamespace

from prompt_rewriter import PromptRewriter, build_user_message


class FakeModels:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(
            responses
            or [
                {
                    "prompt": "A stable cinematic camera follows the dragon through falling snow.",
                    "change_type": "persistent",
                }
            ]
        )

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(parsed=response)


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
        self.assertEqual(len(models.calls), 1)

    async def test_forbidden_process_words_are_accepted_without_retry(self):
        models = FakeModels(
            [
                {
                    "prompt": "The original dragon retains its previous position.",
                    "change_type": "persistent",
                }
            ]
        )
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        result = await PromptRewriter(client=client).rewrite(
            "保持位置", "A dragon flies over a valley."
        )
        self.assertIn("original", result.prompt)
        self.assertIn("previous", result.prompt)
        self.assertEqual(len(models.calls), 1)

    async def test_invalid_schema_gets_one_structure_retry(self):
        models = FakeModels(
            [
                {"prompt": "Missing its lifetime label."},
                {"prompt": "A dragon jumps once.", "change_type": "one_time"},
            ]
        )
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        result = await PromptRewriter(client=client).rewrite(
            "跳一下", "A dragon flies over a valley."
        )
        self.assertEqual(result.change_type.value, "one_time")
        self.assertEqual(len(models.calls), 2)
        self.assertIn("STRUCTURE CORRECTION", models.calls[1]["contents"][0])

    async def test_api_failure_is_not_retried(self):
        models = FakeModels([RuntimeError("vertex unavailable")])
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        with self.assertRaisesRegex(RuntimeError, "vertex unavailable"):
            await PromptRewriter(client=client).rewrite(
                "跳一下", "A dragon flies over a valley."
            )
        self.assertEqual(len(models.calls), 1)

    def test_build_message_rejects_missing_state(self):
        with self.assertRaisesRegex(ValueError, "previous_prompt"):
            build_user_message("", "下雪")


if __name__ == "__main__":
    unittest.main()
