# SPDX-License-Identifier: Apache-2.0
"""Executable H.264 WebSocket bridge smoke test using real ffmpeg.

Run this file in the WebUI image.  It stands up a fake raw-RGB realtime
backend, connects both production backend routes, and verifies that each route
produces fragmented MP4/H.264 plus forwards interactive controls upstream.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from typing import Any

import msgspec
from aiohttp import ClientSession, WSMsgType, web
from h264_websocket_bridge import install_h264_websocket_bridge

WIDTH = 96
HEIGHT = 64
FPS = 24


def _rgb_frame(index: int) -> bytes:
    red = (index * 19) % 256
    green = (64 + index * 7) % 256
    blue = (192 - index * 5) % 256
    return bytes((red, green, blue)) * (WIDTH * HEIGHT)


async def _fake_realtime(request: web.Request) -> web.WebSocketResponse:
    websocket = web.WebSocketResponse(max_msg_size=0)
    await websocket.prepare(request)
    init_message = await websocket.receive()
    assert init_message.type == WSMsgType.BINARY
    init = msgspec.msgpack.decode(bytes(init_message.data))
    assert init["realtime_output_format"] == "raw"
    request.app["connected_backends"].add(request.match_info["backend"])

    async def produce() -> None:
        for index in range(36):
            header: dict[str, Any] = {
                "type": "frame_batch",
                "content_type": "application/x-raw-rgb",
                "width": WIDTH,
                "height": HEIGHT,
                "channels": 3,
                "num_frames": 1,
                "chunk_index": index // 6,
                "event_id": 1,
                "frame_batch_index": index % 6,
                "num_frame_batches": 6,
                "is_final_frame_batch": index % 6 == 5,
                "payload": _rgb_frame(index),
            }
            await websocket.send_bytes(msgspec.msgpack.encode(header))
            await asyncio.sleep(1 / FPS)
        await asyncio.sleep(0.25)
        await websocket.close()

    producer = asyncio.create_task(produce())
    try:
        async for message in websocket:
            if message.type != WSMsgType.BINARY:
                continue
            event = msgspec.msgpack.decode(bytes(message.data))
            if event.get("kind") == "prompt":
                request.app["prompt_backends"].add(request.match_info["backend"])
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
    return websocket


async def _start_app(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    return runner, int(sockets[0].getsockname()[1])


async def _exercise_backend(port: int, backend: str) -> None:
    init_frame = base64.b64encode(b"not-decoded-by-the-fake-backend").decode()
    media = bytearray()
    saw_connected = False
    saw_media_batch = False
    saw_encode_timing = False
    saw_payload_timing = False
    first_media_chunk = None
    async with ClientSession() as client:
        async with client.ws_connect(
            f"http://127.0.0.1:{port}/api/h264ws/{backend}",
            max_msg_size=0,
        ) as websocket:
            await websocket.send_json(
                {
                    "type": "init",
                    "trace_id": f"e2e-{backend}",
                    "fps": FPS,
                    "h264_bitrate_kbps": 3000,
                    "h264_crf": 20,
                    "h264_preset": "fast",
                    "h264_profile": "main",
                    "h264_gop_seconds": 2,
                    "h264_vbv_buffer_ms": 250,
                    "first_frame": f"data:application/octet-stream;base64,{init_frame}",
                }
            )
            deadline = asyncio.get_running_loop().time() + 8
            while asyncio.get_running_loop().time() < deadline:
                message = await asyncio.wait_for(websocket.receive(), timeout=2)
                if message.type == WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event.get("type") == "error":
                        raise AssertionError(event["message"])
                    if (
                        event.get("type") == "status"
                        and event.get("state") == "connected"
                    ):
                        assert event["bitrate_kbps"] == 3000
                        assert event["crf"] == 20
                        assert event["preset"] == "fast"
                        assert event["profile"] == "main"
                        assert event["gop_seconds"] == 2
                        assert event["vbv_buffer_ms"] == 250
                        assert event["startup_drop_frames"] == (
                            8 if backend == "lingbot2" else 0
                        )
                        saw_connected = True
                        await websocket.send_json(
                            {
                                "type": "event",
                                "kind": "prompt",
                                "event_id": 1,
                                "payload": {"prompt": "turn left"},
                            }
                        )
                    if event.get("type") == "media_batch":
                        saw_media_batch = True
                        if first_media_chunk is None:
                            first_media_chunk = int(event.get("chunk_index") or 0)
                    if event.get("type") == "media_encode_timing":
                        saw_encode_timing = True
                        assert float(event.get("bridge_encoder_feed_ms") or 0) >= 0
                    if event.get("type") == "media_payload":
                        saw_payload_timing = True
                        assert int(event.get("num_bytes") or 0) > 0
                        assert float(event.get("server_sent_epoch_ms") or 0) > 0
                elif message.type == WSMsgType.BINARY:
                    media.extend(message.data)
                    if b"ftyp" in media and b"moof" in media and b"mdat" in media:
                        break
                elif message.type in {
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                }:
                    break
    assert saw_connected, f"{backend}: bridge never connected"
    assert saw_media_batch, f"{backend}: frame metadata was not emitted"
    assert saw_encode_timing, f"{backend}: encoder timing was not emitted"
    assert saw_payload_timing, f"{backend}: payload timing was not emitted"
    assert first_media_chunk == (1 if backend == "lingbot2" else 0)
    assert b"ftyp" in media, f"{backend}: MP4 init segment was not emitted"
    assert b"moof" in media and b"mdat" in media, f"{backend}: fMP4 media missing"


async def main() -> None:
    upstream_app = web.Application()
    upstream_app["connected_backends"] = set()
    upstream_app["prompt_backends"] = set()
    upstream_app.router.add_get("/{backend}/v1/realtime_video/generate", _fake_realtime)
    upstream_runner, upstream_port = await _start_app(upstream_app)

    bridge_app = web.Application()
    session_key = web.AppKey("e2e_client_session", object)
    bridge_app[session_key] = ClientSession()
    install_h264_websocket_bridge(
        bridge_app,
        upstream_session_key=session_key,
        upstream_resolver=lambda backend: (
            f"ws://127.0.0.1:{upstream_port}/{backend}/v1/realtime_video/generate"
        ),
    )
    bridge_runner, bridge_port = await _start_app(bridge_app)
    try:
        for backend in ("minwm", "lingbot2"):
            await _exercise_backend(bridge_port, backend)
        assert upstream_app["connected_backends"] == {"minwm", "lingbot2"}
        assert upstream_app["prompt_backends"] == {"minwm", "lingbot2"}
    finally:
        await bridge_app[session_key].close()
        await bridge_runner.cleanup()
        await upstream_runner.cleanup()
    print("H.264 WebSocket E2E passed for minwm and lingbot2")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
