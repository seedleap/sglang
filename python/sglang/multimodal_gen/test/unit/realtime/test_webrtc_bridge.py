# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import msgspec.msgpack

from sglang.multimodal_gen.apps.realtime_webui.webrtc_bridge import (
    RAW_RGB_CONTENT_TYPE,
    WebRTCBridgeSession,
    _playback_ack_enabled,
    _raw_channel_filter,
)


class _FakeUpstream:
    closed = False

    def __init__(self) -> None:
        self.messages: list[bytes] = []

    async def send_bytes(self, payload: bytes) -> None:
        self.messages.append(payload)


def _session() -> WebRTCBridgeSession:
    manager = SimpleNamespace(
        media_http_base="http://media",
        h264_preset="superfast",
        bridge_max_queued_frames=4,
    )
    return WebRTCBridgeSession(
        manager=manager,
        session_id="test",
        init={"fps": 24},
        codec="h264",
        bitrate_kbps=3500,
    )


def test_bridge_control_event_waits_for_new_media_before_cutover():
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

        assert session.minimum_event_id == 0
        assert session.pending_cutover_event_id == 7
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

        assert session.frame_queue.qsize() == 1
        assert session.dropped_batches == 0

        await session._handle_frame_payload(
            {
                "type": "frame_batch",
                "chunk_index": 13,
                "event_id": 7,
                "content_type": RAW_RGB_CONTENT_TYPE,
                "width": 2,
                "height": 2,
                "channels": 3,
                "num_frames": 1,
            },
            b"y" * 12,
        )

        assert session.minimum_event_id == 7
        assert session.pending_cutover_event_id == 0
        assert session.transition_cutovers == 1
        assert session.control_dropped_frames == 1
        assert session.dropped_frames == 1
        assert session.dropped_source_bytes == 12
        assert session.frame_queue.qsize() == 1
        queued = session.frame_queue.get_nowait()
        session.frame_queue.task_done()
        assert queued.event_id == 7

    asyncio.run(run())


def test_raw_channel_filter_normalizes_lab_gbr_transport():
    assert _raw_channel_filter("rgb") == ""
    assert _raw_channel_filter("invalid") == ""
    assert _raw_channel_filter("gbr") == (
        "colorchannelmixer="
        "rr=0:rg=0:rb=1:"
        "gr=1:gg=0:gb=0:"
        "br=0:bg=1:bb=0,"
    )


def test_bridge_control_event_discards_stale_queue_when_new_media_arrives():
    async def run():
        session = _session()
        upstream = _FakeUpstream()
        session.upstream = upstream
        await session._handle_frame_payload(
            {
                "type": "frame_batch",
                "chunk_index": 12,
                "event_id": 6,
                "content_type": RAW_RGB_CONTENT_TYPE,
                "width": 2,
                "height": 2,
                "channels": 3,
                "num_frames": 2,
            },
            b"x" * 24,
        )
        assert session.frame_queue.qsize() == 2

        await session.send_control(
            {
                "type": "event",
                "kind": "camera_actions",
                "event_id": 7,
                "payload": {"mode": "state", "transitions": []},
            }
        )

        assert session.frame_queue.qsize() == 2
        assert session.pending_cutover_event_id == 7
        assert session.control_dropped_frames == 0

        await session._handle_frame_payload(
            {
                "type": "frame_batch",
                "chunk_index": 13,
                "event_id": 7,
                "content_type": RAW_RGB_CONTENT_TYPE,
                "width": 2,
                "height": 2,
                "channels": 3,
                "num_frames": 1,
            },
            b"y" * 12,
        )

        assert session.frame_queue.qsize() == 1
        assert session.control_dropped_frames == 2
        assert session.dropped_frames == 2

    asyncio.run(run())


def test_bridge_queue_is_bounded_and_keeps_newest_frames():
    async def run():
        session = _session()
        for event_id in range(1, 6):
            await session._handle_frame_payload(
                {
                    "type": "frame_batch",
                    "chunk_index": event_id,
                    "event_id": event_id,
                    "content_type": RAW_RGB_CONTENT_TYPE,
                    "width": 1,
                    "height": 1,
                    "channels": 3,
                    "num_frames": 1,
                },
                bytes([event_id]) * 3,
            )

        assert session.frame_queue.qsize() == 4
        assert session.queue_overflow_dropped_frames == 1
        assert session.dropped_frames == 1
        queued_event_ids = []
        while not session.frame_queue.empty():
            frame = session.frame_queue.get_nowait()
            session.frame_queue.task_done()
            queued_event_ids.append(frame.event_id)
        assert queued_event_ids == [2, 3, 4, 5]

    asyncio.run(run())


def test_source_only_bridge_fans_out_without_local_encoder_queue():
    async def run():
        session = _session()
        session.source_only = True
        subscriber = asyncio.Queue(maxsize=2)
        session.comparison_frame_subscribers.add(subscriber)

        await session._handle_frame_payload(
            {
                "type": "frame_batch",
                "chunk_index": 4,
                "event_id": 3,
                "content_type": RAW_RGB_CONTENT_TYPE,
                "width": 2,
                "height": 2,
                "channels": 3,
                "num_frames": 1,
            },
            b"z" * 12,
        )

        assert session.frames == 1
        assert session.state == "streaming"
        assert session.frame_queue.empty()
        shared_frame = subscriber.get_nowait()
        subscriber.task_done()
        assert shared_frame.event_id == 3
        assert shared_frame.rgb == b"z" * 12

    asyncio.run(run())


def test_native_webrtc_session_preserves_playback_ack_opt_in():
    assert _playback_ack_enabled({"playback_ack_enabled": True}) is True
    assert _playback_ack_enabled({"playback_ack_enabled": False}) is False
    assert _playback_ack_enabled({}) is False
