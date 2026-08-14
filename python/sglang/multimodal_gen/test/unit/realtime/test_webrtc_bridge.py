# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import msgspec.msgpack

from sglang.multimodal_gen.apps.realtime_webui.webrtc_bridge import (
    RAW_RGB_CONTENT_TYPE,
    WebRTCBridgeSession,
)


class _FakeUpstream:
    closed = False

    def __init__(self) -> None:
        self.messages: list[bytes] = []

    async def send_bytes(self, payload: bytes) -> None:
        self.messages.append(payload)


def _session() -> WebRTCBridgeSession:
    manager = SimpleNamespace(media_http_base="http://media", h264_preset="superfast")
    return WebRTCBridgeSession(
        manager=manager,
        session_id="test",
        init={"fps": 24},
        codec="h264",
        bitrate_kbps=3500,
    )


def test_bridge_control_event_cuts_over_stale_media_before_encoding():
    async def run():
        session = _session()
        upstream = _FakeUpstream()
        session.upstream = upstream
        envelope = {
            "type": "event",
            "kind": "camera_actions",
            "event_id": 7,
            "payload": {"mode": "state", "transitions": []},
        }
        await session.send_control(envelope)

        assert session.minimum_event_id == 7
        assert msgspec.msgpack.decode(upstream.messages[0]) == envelope

        await session._handle_frame_payload(
            {
                "type": "frame_batch",
                "chunk_index": 12,
                "event_id": 6,
                "content_type": RAW_RGB_CONTENT_TYPE,
                "width": 2,
                "height": 2,
                "channels": 3,
                "num_frames": 1,
            },
            b"x" * 12,
        )

        assert session.ffmpeg is None
        assert session.frames == 0
        assert session.dropped_batches == 1
        assert session.dropped_frames == 1
        assert session.dropped_source_bytes == 12

    asyncio.run(run())
