#!/usr/bin/env python3

import asyncio
import io
import json
import unittest
from unittest import mock

import server
from PIL import Image


class FakeResponse:
    def __init__(self, payload):
        self.status = 200
        self.payload = payload

    async def json(self, content_type=None):
        return self.payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/openapi/v1/worlds/get-travel-credential"):
            return FakeResponse({"code": 0, "data": {"ticket": "ticket-test"}})
        if url.endswith("/api/v1/tokens"):
            return FakeResponse({"output": {"token": "temporary-token-test"}})
        raise AssertionError(f"unexpected upstream URL {url}")


class FakeRequest:
    def __init__(self, app):
        self.app = app

    async def json(self):
        return {"encryptedWorldId": "world-test"}


class HappyOysterBffTest(unittest.IsolatedAsyncioTestCase):
    def test_first_frame_is_normalized_to_bounded_16_by_9_jpeg(self):
        source = io.BytesIO()
        Image.new("RGB", (1393, 1129), (20, 80, 140)).save(source, format="PNG")

        normalized = server._normalize_happyoyster_first_frame(source.getvalue())

        self.assertLess(len(normalized), len(source.getvalue()))
        with Image.open(io.BytesIO(normalized)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (1280, 720))

    async def test_prepare_uses_origin_for_temporary_token(self):
        session = FakeSession()
        app = {server.SESSION: session}
        with mock.patch.object(server, "HAPPYOYSTER_API_KEY", "main-key-test"):
            response = await server._happyoyster_prepare(FakeRequest(app))
        payload = json.loads(response.text)
        self.assertEqual(payload["token"], "temporary-token-test")
        self.assertEqual(payload["ticket"], "ticket-test")
        urls = [url for _method, url, _kwargs in session.calls]
        self.assertIn(
            "https://llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com/api/v1/tokens",
            urls,
        )
        self.assertIn(
            "https://llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com/"
            "api/v2/apps/happyoyster-1.0/openapi/v1/worlds/get-travel-credential",
            urls,
        )
        for _method, _url, kwargs in session.calls:
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer main-key-test")


if __name__ == "__main__":
    unittest.main()
