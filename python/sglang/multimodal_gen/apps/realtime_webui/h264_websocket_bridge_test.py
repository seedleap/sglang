# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time

import msgspec
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


def test_encoder_drains_frames_immediately_without_server_side_pacing():
    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    class FakeStdin:
        def __init__(self):
            self.writes = []
            self.two_frames_written = asyncio.Event()

        def write(self, payload):
            self.writes.append(payload)
            if len(self.writes) == 2:
                self.two_frames_written.set()

        async def drain(self):
            await asyncio.sleep(0)

    class FakeFFmpeg:
        def __init__(self):
            self.stdin = FakeStdin()

    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        websocket = FakeWebSocket()
        session = H264WebSocketSession(
            manager=manager,
            websocket=websocket,
            backend="minwm",
            # One FPS used to make the old server-side pacing fail this test.
            # It must now affect only H.264 timestamps, never frame delivery.
            init={"fps": 1},
        )
        session.width = 2
        session.height = 1
        session.ffmpeg = FakeFFmpeg()
        frames = [b"abcdef", b"ABCDEF"]
        await session._handle_frame_payload(
            {
                "content_type": "application/x-raw-rgb",
                "width": 2,
                "height": 1,
                "channels": 3,
                "chunk_index": 0,
                "num_frames": 2,
            },
            b"".join(frames),
        )

        encoder_task = asyncio.create_task(session._encode_frames())
        try:
            await asyncio.wait_for(
                session.ffmpeg.stdin.two_frames_written.wait(), timeout=0.1
            )
            await asyncio.sleep(0.02)
            assert session.ffmpeg.stdin.writes == frames
            media_batches = [
                message
                for message in websocket.messages
                if message["type"] == "media_batch"
            ]
            encode_timings = [
                message
                for message in websocket.messages
                if message["type"] == "media_encode_timing"
            ]
            assert len(media_batches) == 2
            assert len(encode_timings) == 2
            assert all(message["repeated_frame"] is False for message in media_batches)
            assert all(
                message["bridge_encoder_feed_ms"] >= 0 for message in encode_timings
            )
        finally:
            encoder_task.cancel()
            await asyncio.gather(encoder_task, return_exceptions=True)

    asyncio.run(exercise())


def test_finite_session_backpressures_instead_of_dropping_tail():
    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        manager.max_queued_frames = 1
        session = H264WebSocketSession(
            manager=manager,
            websocket=object(),
            backend="minwm",
            init={"max_chunks": 1},
        )
        header = {
            "content_type": "application/x-raw-rgb",
            "width": 2,
            "height": 1,
            "channels": 3,
            "chunk_index": 0,
            "num_frames": 1,
        }
        await session._handle_frame_payload(header, b"first!")
        blocked_put = asyncio.create_task(
            session._handle_frame_payload(header, b"second")
        )
        await asyncio.sleep(0)

        assert not blocked_put.done()
        assert session.dropped_frames == 0
        first = session.frame_queue.get_nowait()
        session.frame_queue.task_done()
        await blocked_put
        second = session.frame_queue.get_nowait()
        session.frame_queue.task_done()
        assert first.rgb == b"first!"
        assert second.rgb == b"second"
        assert session.dropped_frames == 0

    asyncio.run(exercise())


def test_continuous_session_keeps_existing_live_edge_drop_policy():
    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        manager.max_queued_frames = 1
        session = H264WebSocketSession(
            manager=manager,
            websocket=object(),
            backend="minwm",
            init={},
        )
        header = {
            "content_type": "application/x-raw-rgb",
            "width": 2,
            "height": 1,
            "channels": 3,
            "chunk_index": 0,
            "num_frames": 1,
        }
        await session._handle_frame_payload(header, b"first!")
        await session._handle_frame_payload(header, b"second")

        retained = session.frame_queue.get_nowait()
        session.frame_queue.task_done()
        assert retained.rgb == b"second"
        assert session.dropped_frames == 1

    asyncio.run(exercise())


def test_i2v_num_frames_does_not_turn_default_continuous_stream_finite():
    app = web.Application()
    manager = H264WebSocketBridgeManager(
        app,
        web.AppKey("session", object),
        lambda backend: f"ws://gateway/{backend}",
    )

    continuous_i2v = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="minwm",
        init={
            "generation_mode": "i2v",
            "first_frame": "data:image/png;base64,AA==",
            "num_frames": 17,
        },
    )
    finite_i2v = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="minwm",
        init={
            "generation_mode": "i2v",
            "first_frame": "data:image/png;base64,AA==",
            "num_frames": 17,
            "max_chunks": 1,
        },
    )
    finite_t2v = H264WebSocketSession(
        manager=manager,
        websocket=object(),
        backend="minwm",
        init={"generation_mode": "t2v", "num_frames": 121},
    )

    assert continuous_i2v.finite_request is False
    assert finite_i2v.finite_request is True
    assert finite_t2v.finite_request is True


def test_final_completion_is_published_only_after_queue_and_output_flush():
    class FakeWebSocket:
        closed = False

        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        websocket = FakeWebSocket()
        session = H264WebSocketSession(
            manager=manager,
            websocket=websocket,
            backend="minwm",
            init={"max_chunks": 1},
        )
        await session._receive_binary(
            msgspec.msgpack.encode(
                {
                    "type": "media_chunk_complete",
                    "chunk_index": 0,
                    "event_id": 7,
                    "num_frames": 2,
                    "is_final_chunk": True,
                }
            )
        )
        assert websocket.messages == []

        await session.frame_queue.put(b"frame-0")
        await session.frame_queue.put(b"frame-1")
        processed = []
        flushed_after = []

        async def encode():
            while True:
                frame = await session.frame_queue.get()
                try:
                    await asyncio.sleep(0)
                    processed.append(frame)
                    session.frames += 1
                finally:
                    session.frame_queue.task_done()

        async def stop_ffmpeg(*, flush_output=False):
            assert flush_output is True
            flushed_after.append(tuple(processed))

        session.encoder_task = asyncio.create_task(encode())
        session._stop_ffmpeg = stop_ffmpeg
        await session._finish_graceful_stream()

        assert processed == [b"frame-0", b"frame-1"]
        assert flushed_after == [(b"frame-0", b"frame-1")]
        assert websocket.messages == [
            {
                "type": "stream_complete",
                "chunk_index": 0,
                "event_id": 7,
                "num_frames": 2,
                "encoded_frames": 2,
                "dropped_frames": 0,
                "media_bytes": 0,
            }
        ]

    asyncio.run(exercise())


def test_ffmpeg_timeout_cancels_blocked_output_before_bounded_kill_wait():
    shutdown_events = []

    class BlockingWebSocket:
        def __init__(self):
            self.send_started = asyncio.Event()
            self.send_cancelled = asyncio.Event()

        async def send_json(self, _payload):
            return None

        async def send_bytes(self, _payload):
            self.send_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                shutdown_events.append("output_cancelled")
                self.send_cancelled.set()

    class FakeStdin:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class OneFragmentStdout:
        def __init__(self):
            self.sent = False

        async def read(self, _size):
            if self.sent:
                return b""
            self.sent = True
            return b"fragment"

    class HungProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = OneFragmentStdout()
            self.stderr = None
            self.returncode = None
            self.killed = False
            self.exited = asyncio.Event()
            self.wait_calls = 0

        async def wait(self):
            self.wait_calls += 1
            await self.exited.wait()
            return self.returncode

        def kill(self):
            self.killed = True
            shutdown_events.append("process_killed")
            # Deliberately never resolve wait(): the production code must put
            # a deadline around the post-kill reap as well.

    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        manager.drain_timeout_ms = 20
        websocket = BlockingWebSocket()
        session = H264WebSocketSession(
            manager=manager,
            websocket=websocket,
            backend="minwm",
            init={"max_chunks": 1},
        )
        process = HungProcess()
        session.ffmpeg = process
        session.stdout_task = asyncio.create_task(session._pump_stdout())
        await asyncio.wait_for(websocket.send_started.wait(), timeout=0.1)

        started = asyncio.get_running_loop().time()
        try:
            await session._stop_ffmpeg(flush_output=True)
        except TimeoutError:
            pass
        else:
            raise AssertionError("strict flush must report its initial process timeout")
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.5
        assert process.stdin.closed is True
        assert websocket.send_cancelled.is_set()
        assert process.killed is True
        assert process.wait_calls == 2
        assert shutdown_events[:2] == ["output_cancelled", "process_killed"]
        assert session.stdout_task is None

    asyncio.run(exercise())


def test_graceful_ffmpeg_flush_keeps_successful_encode_metric(monkeypatch):
    observations = []

    def observe(stage, _duration, **labels):
        observations.append((stage, labels.get("result")))

    monkeypatch.setattr("h264_websocket_bridge.observe_stage_seconds", observe)

    class FakeWebSocket:
        async def send_json(self, _payload):
            return None

        async def send_bytes(self, _payload):
            return None

    class FakeStdin:
        def close(self):
            return None

    class OneFragmentStdout:
        def __init__(self):
            self.sent = False

        async def read(self, _size):
            if self.sent:
                return b""
            self.sent = True
            await asyncio.sleep(0)
            return b"fragment"

    class CleanProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = OneFragmentStdout()
            self.stderr = None
            self.returncode = 0

        async def wait(self):
            await asyncio.sleep(0)
            return self.returncode

    async def exercise():
        app = web.Application()
        manager = H264WebSocketBridgeManager(
            app,
            web.AppKey("session", object),
            lambda backend: f"ws://gateway/{backend}",
        )
        session = H264WebSocketSession(
            manager=manager,
            websocket=FakeWebSocket(),
            backend="minwm",
            init={"max_chunks": 1},
        )
        process = CleanProcess()
        session.ffmpeg = process
        session.encoder_start_times_s.append(time.perf_counter() - 0.01)
        session.stdout_task = asyncio.create_task(session._pump_stdout(process))

        await session._stop_ffmpeg(flush_output=True)

        frame_results = [
            result for stage, result in observations if stage == "frame_encode"
        ]
        assert frame_results == ["success"]
        assert not session.encoder_start_times_s

    asyncio.run(exercise())
