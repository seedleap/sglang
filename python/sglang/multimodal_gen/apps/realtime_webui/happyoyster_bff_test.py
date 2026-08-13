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
    def __init__(self, app, body=None):
        self.app = app
        self.body = body or {"encryptedWorldId": "world-test"}

    async def json(self):
        return self.body


class HappyOysterBffTest(unittest.IsolatedAsyncioTestCase):
    def test_prebuilt_manifest_accepts_only_safe_nonempty_entries(self):
        payload = {
            "worlds": {
                "dragon-ride": {"encryptedWorldId": "world-dragon"},
                "Unsafe Key": {"encryptedWorldId": "world-unsafe"},
                "missing": {},
            }
        }
        with mock.patch.object(server, "HAPPYOYSTER_PREBUILT_WORLDS_PATH") as path:
            path.read_text.return_value = json.dumps(payload)
            self.assertEqual(
                server._load_happyoyster_prebuilt_worlds(),
                {"dragon-ride": "world-dragon"},
            )

    async def test_resolve_ready_prebuilt_world(self):
        request = FakeRequest({
            server.HAPPYOYSTER_WORLD_CACHE: {"dragon-ride": "world-dragon"},
        }, {"presetKey": "dragon-ride"})
        with mock.patch.object(
            server,
            "_happyoyster_request",
            mock.AsyncMock(return_value={"status": "ready"}),
        ):
            response = await server._happyoyster_resolve_world(request)
        self.assertEqual(
            json.loads(response.text),
            {
                "status": "ready",
                "source": "prebuilt",
                "encryptedWorldId": "world-dragon",
            },
        )

    async def test_resolve_expired_world_evicts_runtime_cache(self):
        cache = {"dragon-ride": "world-expired"}
        request = FakeRequest(
            {server.HAPPYOYSTER_WORLD_CACHE: cache},
            {"presetKey": "dragon-ride"},
        )
        with mock.patch.object(
            server,
            "_happyoyster_request",
            mock.AsyncMock(side_effect=server.web.HTTPBadGateway()),
        ):
            response = await server._happyoyster_resolve_world(request)
        self.assertEqual(json.loads(response.text)["source"], "expired")
        self.assertNotIn("dragon-ride", cache)

    async def test_create_reuses_cached_world_under_per_preset_lock(self):
        cache = {"dragon-ride": "world-existing"}
        app = {
            server.HAPPYOYSTER_WORLD_CACHE: cache,
            server.HAPPYOYSTER_WORLD_BUILD_LOCKS: {},
        }
        request = FakeRequest(
            app,
            {
                "prompt": "Dragon flight",
                "firstFrameUrl": "https://example.test/dragon.jpg",
                "presetKey": "dragon-ride",
            },
        )
        upstream = mock.AsyncMock(return_value={"status": "building"})
        with mock.patch.object(server, "_happyoyster_request", upstream):
            first, second = await asyncio.gather(
                server._happyoyster_create_world(request),
                server._happyoyster_create_world(request),
            )
        self.assertEqual(json.loads(first.text)["encryptedWorldId"], "world-existing")
        self.assertEqual(json.loads(second.text)["encryptedWorldId"], "world-existing")
        self.assertTrue(
            all(call.args[1] == "GET" for call in upstream.await_args_list),
            "a cached preset must not create another upstream World",
        )

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
