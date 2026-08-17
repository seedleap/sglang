# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from aiohttp import web

from h264_websocket_bridge import (
    H264WebSocketBridgeManager,
    H264WebSocketSession,
    _raw_channel_filter,
    _split_payload,
)


def test_split_raw_rgb_payload_into_frames():
    header = {
        "content_type": "application/x-raw-rgb",
        "width": 2,
        "height": 1,
        "channels": 3,
    }
    assert _split_payload(header, b"abcdefABCDEF") == [b"abcdef", b"ABCDEF"]


def test_backend_resolver_and_channel_order(monkeypatch):
    monkeypatch.setenv("H264_WS_RAW_CHANNEL_ORDER", "rgb")
    monkeypatch.setenv("MINWM_H264_WS_RAW_CHANNEL_ORDER", "gbr")
    app = web.Application()
    session_key = web.AppKey("session", object)
    manager = H264WebSocketBridgeManager(
        app,
        session_key,
        lambda backend: f"ws://gateway/backends/{backend}/v1/realtime_video/generate",
    )
    assert manager.upstream_url("minwm").endswith(
        "/backends/minwm/v1/realtime_video/generate"
    )
    assert manager.upstream_url("lingbot2").endswith(
        "/backends/lingbot2/v1/realtime_video/generate"
    )
    assert manager.raw_channel_order("minwm") == "gbr"
    assert manager.raw_channel_order("lingbot2") == "rgb"
    assert "colorchannelmixer" in _raw_channel_filter("gbr")
    assert _raw_channel_filter("rgb") == ""


def test_manager_session_limit_is_bounded(monkeypatch):
    monkeypatch.setenv("H264_WS_MAX_SESSIONS", "999")
    app = web.Application()
    manager = H264WebSocketBridgeManager(
        app,
        web.AppKey("session", object),
        lambda backend: f"ws://gateway/{backend}",
    )
    assert manager.max_sessions == 64


def test_session_accepts_bounded_encoder_overrides(monkeypatch):
    monkeypatch.setenv("H264_WS_PRESET", "veryfast")
    monkeypatch.setenv("H264_WS_CRF", "16")
    monkeypatch.setenv("H264_WS_BITRATE_KBPS", "6000")
    app = web.Application()
    manager = H264WebSocketBridgeManager(
        app,
        web.AppKey("session", object),
        lambda backend: f"ws://gateway/{backend}",
    )
    session = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="minwm",
        init={
            "h264_preset": "fast",
            "h264_profile": "main",
            "h264_crf": 20,
            "h264_bitrate_kbps": 3000,
            "h264_vbv_buffer_ms": 250,
            "h264_gop_seconds": 2,
            "h264_startup_drop_frames": 8,
        },
    )
    assert session.encoder_preset == "fast"
    assert session.encoder_profile == "main"
    assert session.encoder_crf == 20
    assert session.encoder_bitrate_kbps == 3000
    assert session.encoder_vbv_buffer_ms == 250
    assert session.encoder_gop_seconds == 2
    assert session.startup_drop_frames == 0


def test_lingbot_drops_transition_frames_before_encoder_start(monkeypatch):
    monkeypatch.delenv("LINGBOT2_H264_WS_STARTUP_DROP_FRAMES", raising=False)
    app = web.Application()
    manager = H264WebSocketBridgeManager(
        app,
        web.AppKey("session", object),
        lambda backend: f"ws://gateway/{backend}",
    )
    session = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="lingbot2",
        init={},
    )
    frames = [bytes([index]) * 6 for index in range(10)]
    header = {
        "content_type": "application/x-raw-rgb",
        "width": 2,
        "height": 1,
        "channels": 3,
        "chunk_index": 0,
        "num_frames": len(frames),
    }

    asyncio.run(session._handle_frame_payload(header, b"".join(frames)))

    retained = [session.frame_queue.get_nowait() for _ in range(2)]
    assert [frame.rgb for frame in retained] == frames[8:]
    assert all(frame.chunk_index == 0 for frame in retained)
    assert session.startup_drop_remaining == 0
    assert session.startup_dropped_frames == 8
    assert session.dropped_frames == 8


def test_lingbot_does_not_drop_stable_chunk_if_chunk_zero_was_shed(monkeypatch):
    monkeypatch.delenv("LINGBOT2_H264_WS_STARTUP_DROP_FRAMES", raising=False)
    app = web.Application()
    manager = H264WebSocketBridgeManager(
        app,
        web.AppKey("session", object),
        lambda backend: f"ws://gateway/{backend}",
    )
    session = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="lingbot2",
        init={},
    )
    frame = b"stable"
    header = {
        "content_type": "application/x-raw-rgb",
        "width": 2,
        "height": 1,
        "channels": 3,
        "chunk_index": 1,
        "num_frames": 1,
    }

    asyncio.run(session._handle_frame_payload(header, frame))

    assert session.frame_queue.get_nowait().rgb == frame
    assert session.startup_drop_remaining == 0
    assert session.startup_dropped_frames == 0
