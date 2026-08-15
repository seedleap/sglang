# SPDX-License-Identifier: Apache-2.0

import asyncio

import msgspec.msgpack
import pytest

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import encode_message
from sglang.multimodal_gen.runtime.realtime.gateway import (
    AdmissionQueueFull,
    BoundedAdmissionWaiterGate,
    BrowserPlaybackAckWindow,
    GatewayOutputRegistry,
    OutputProtocolError,
    build_denoiser_url,
    worker_message_allowed,
)


def test_gateway_admission_waiter_gate_rejects_overflow_without_blocking():
    async def run():
        gate = BoundedAdmissionWaiterGate(max_waiters=1)
        async with gate.waiter():
            assert gate.waiters == 1
            with pytest.raises(AdmissionQueueFull, match="ADMISSION_QUEUE_FULL"):
                async with gate.waiter():
                    raise AssertionError("overflow waiter must not enter")
        assert gate.waiters == 0

    asyncio.run(run())


def _frame(
    chunk: int,
    batch: int = 0,
    *,
    generation: str = "g",
    num_frames: int = 1,
    event_id: int | None = None,
) -> bytes:
    return encode_message(
        "frame_batch",
        session_id="s",
        generation_id=generation,
        request_id=f"r{chunk}",
        chunk_index=chunk,
        frame_batch_index=batch,
        payload_lengths=[1] * num_frames,
        payload=b"x" * num_frames,
        content_type="image/webp",
        width=8,
        height=8,
        num_frames=num_frames,
        event_id=event_id,
    )


def _media_complete(chunk: int, *, generation: str = "g") -> bytes:
    return encode_message(
        "media_chunk_complete",
        session_id="s",
        generation_id=generation,
        request_id=f"r{chunk}",
        chunk_index=chunk,
        num_frames=1,
    )


def test_gateway_playback_ack_window_bounds_chunk_lead_and_sheds_stale_media():
    async def run():
        window = BrowserPlaybackAckWindow(
            max_unacked_chunks=2,
            wait_timeout_s=0.01,
        )
        assert await window.allow_output(_frame(8))

        await window.observe_browser_message(
            encode_message("init", playback_ack_enabled=True)
        )
        assert await window.allow_output(_frame(0))
        assert await window.allow_output(_frame(0, 1))
        assert await window.allow_output(_frame(1))
        assert not await window.allow_output(_frame(2))
        assert await window.allow_output(_media_complete(2))

        await window.observe_browser_message(
            encode_message(
                "event",
                kind="playback_ack",
                payload={
                    "last_received_chunk": 0,
                    "last_rendered_chunk": 0,
                },
            )
        )
        assert not await window.allow_output(_frame(2))
        assert await window.allow_output(_frame(3))

    asyncio.run(run())


def test_gateway_playback_ack_window_prioritizes_interactive_event_cutover():
    async def run():
        window = BrowserPlaybackAckWindow(
            max_unacked_chunks=2,
            wait_timeout_s=1,
        )
        await window.observe_browser_message(
            encode_message("init", playback_ack_enabled=True)
        )
        assert await window.allow_output(_frame(10, event_id=3))
        assert await window.allow_output(_frame(11, event_id=3))

        await window.observe_browser_message(
            encode_message(
                "event",
                kind="camera_actions",
                event_id=7,
                payload={"mode": "state", "transitions": []},
            )
        )

        started = asyncio.get_running_loop().time()
        assert not await window.allow_output(_frame(12, event_id=6))
        assert asyncio.get_running_loop().time() - started < 0.1
        assert await window.allow_output(_frame(13, event_id=7))
        assert window.minimum_event_id == 7

    asyncio.run(run())


def test_gateway_output_route_is_fenced_ordered_and_bounded():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=2, enqueue_timeout_s=0.01)
        route = await registry.register("s", "g", token="secret")
        with pytest.raises(OutputProtocolError, match="token"):
            await registry.bind("s", "g", token="wrong")
        assert await registry.bind("s", "g", token="secret") is route
        with pytest.raises(OutputProtocolError, match="already bound"):
            await registry.bind("s", "g", token="secret")
        await registry.unbind("s", "g", token="secret")
        assert await registry.bind("s", "g", token="secret") is route

        await route.put(_frame(0, 0))
        await route.put(_frame(0, 1))
        await route.put(_frame(1, 0))
        assert route.dropped_messages == 1

        assert await route.get() == _frame(0, 1)
        route.task_done()
        with pytest.raises(OutputProtocolError, match="stale generation"):
            await route.put(_frame(1, 0, generation="old"))
        with pytest.raises(OutputProtocolError, match="stale chunk"):
            await route.put(_frame(0, 1))

        assert await route.get() == _frame(1, 0)
        route.task_done()
        await route.join()

        await registry.unregister("s", "g", token="secret")
        assert route.closed

    asyncio.run(run())


def test_gateway_output_route_never_waits_for_browser_capacity():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1, enqueue_timeout_s=0.1)
        route = await registry.register("s", "g", token="secret")

        await route.put(_frame(0, 0))
        await asyncio.wait_for(route.put(_frame(0, 1)), timeout=0.01)
        assert route.dropped_messages == 1
        assert route.dropped_frames == 1
        assert await route.get() == _frame(0, 1)
        route.task_done()

    asyncio.run(run())


def test_gateway_output_route_uses_media_completion_as_authoritative_marker():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=3)
        route = await registry.register("s", "g", token="secret")

        await route.put(_frame(0, 0))
        completion_waiter = asyncio.create_task(
            route.wait_until_chunk_completed(0)
        )
        await asyncio.sleep(0)
        assert not completion_waiter.done()

        completion = _media_complete(0)
        await route.put(completion)
        await completion_waiter
        assert await route.get() == _frame(0, 0)
        route.task_done()
        assert await route.get() == completion
        route.task_done()
        await route.join()

        with pytest.raises(OutputProtocolError, match="duplicate completion"):
            await route.put(completion)

    asyncio.run(run())


def test_gateway_output_route_prioritizes_media_completion_under_backpressure():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1, enqueue_timeout_s=0.5)
        route = await registry.register("s", "g", token="secret")

        await route.put(_frame(0, 0))
        completion = _media_complete(0)
        await asyncio.wait_for(route.put(completion), timeout=0.05)

        assert route.dropped_messages == 0
        assert await route.get() == _frame(0, 0)
        route.task_done()
        assert await route.get() == completion
        route.task_done()
        await route.wait_until_chunk_completed(0)

    asyncio.run(run())


def test_gateway_output_route_keeps_controls_when_shedding_old_media():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1)
        route = await registry.register("s", "g", token="secret")

        await route.put(_frame(0, 0))
        completion = _media_complete(0)
        await route.put(completion)
        await route.put(_frame(1, 0))

        assert route.dropped_messages == 1
        assert route.dropped_frames == 1
        assert await route.get() == completion
        route.task_done()
        assert await route.get() == _frame(1, 0)
        route.task_done()

    asyncio.run(run())


def test_gateway_output_route_bounds_actual_frames_and_reports_queue_metrics():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=4)
        route = await registry.register(
            "s", "g", token="secret", trace_id="trace-a"
        )

        await route.put(_frame(0, 0, num_frames=3))
        await route.put(_frame(0, 1, num_frames=3))

        metrics = route.queue_metrics()
        assert route.trace_id == "trace-a"
        assert metrics["gateway_queue_depth"] == 3
        assert metrics["gateway_queue_capacity_frames"] == 4
        assert metrics["gateway_dropped_frames"] == 3
        assert metrics["gateway_queue_bytes"] > 0
        assert metrics["gateway_oldest_frame_age_ms"] >= 0

        assert await route.get() == _frame(0, 1, num_frames=3)
        route.task_done()
        await route.join()

    asyncio.run(run())


def test_gateway_rejects_a_second_live_registration_for_same_session():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1)
        await registry.register("s", "g", token="one")
        with pytest.raises(OutputProtocolError, match="already registered"):
            await registry.register("s", "g2", token="two")

    asyncio.run(run())


def test_gateway_builds_a_fenced_worker_route_and_rejects_media_or_trace():
    url = build_denoiser_url(
        "ws://denoiser-a:30000/v1/realtime_video/generate",
        session_id="session a",
        generation_id="generation-a",
        coordinator_token="lease-secret",
        worker_epoch="denoiser-epoch",
        vae_url="ws://vae-a:18081/v1/realtime_vae/decode",
        vae_worker_epoch="vae-epoch",
        output_url="ws://10.0.0.4:18080/v1/internal/realtime_output",
        output_token="output-secret",
        trace_id="trace-a",
    )
    assert "gateway_managed=1" in url
    assert "session_id=session+a" in url
    assert "realtime_vae_worker_url=ws%3A%2F%2Fvae-a" in url
    assert "worker_epoch=denoiser-epoch" in url
    assert "realtime_vae_worker_epoch=vae-epoch" in url
    assert "trace_id=trace-a" in url
    assert not worker_message_allowed(encode_message("chunk_stats"))
    assert worker_message_allowed(encode_message("error"))
    assert worker_message_allowed(
        encode_message("chunk_telemetry", chunk_index=4, event_id=9)
    )
    assert not worker_message_allowed(msgspec.msgpack.encode({"type": "chunk_stats"}))
    assert not worker_message_allowed(_frame(0))
    assert not worker_message_allowed(encode_message("trace_events", traces=[]))
