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
    OutputBackpressureError,
    OutputIncompleteError,
    OutputProtocolError,
    OutputRouteClosed,
    build_denoiser_url,
    build_gateway_output_url,
    init_requests_finite_output,
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
    is_final: bool = False,
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
        is_final_frame_batch=is_final,
    )


def _media_complete(
    chunk: int,
    *,
    generation: str = "g",
    is_final: bool = False,
) -> bytes:
    return encode_message(
        "media_chunk_complete",
        session_id="s",
        generation_id=generation,
        request_id=f"r{chunk}",
        chunk_index=chunk,
        num_frames=1,
        is_final_chunk=is_final,
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
        assert window.metrics()["gateway_playback_ack_shed_messages"] == 1
        assert window.metrics()["gateway_playback_ack_rejected_messages"] == 0

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


@pytest.mark.parametrize(
    ("init", "expected"),
    (
        ({"type": "init", "max_chunks": 1}, True),
        ({"type": "init", "generation_mode": "i2v", "max_chunks": 2}, True),
        (
            {"type": "init", "generation_mode": "t2v", "num_frames": 49},
            True,
        ),
        ({"type": "init", "mode": "t2v", "num_frames": 49}, True),
        ({"type": "init", "num_frames": 49}, True),
        ({"type": "init", "num_frames": 49, "first_frame": b"png"}, False),
        (
            {"type": "init", "generation_mode": "i2v", "num_frames": 17},
            False,
        ),
        ({"type": "init", "generation_mode": "t2v", "num_frames": 0}, False),
        ({"type": "event", "max_chunks": 1}, False),
    ),
)
def test_gateway_classifies_only_terminal_init_boundaries_as_finite(init, expected):
    assert init_requests_finite_output(init) is expected


def test_gateway_finite_playback_ack_window_waits_for_ack_without_shedding():
    async def run():
        window = BrowserPlaybackAckWindow(
            max_unacked_chunks=1,
            wait_timeout_s=0.1,
        )
        await window.observe_browser_message(
            encode_message(
                "init",
                generation_mode="t2v",
                num_frames=17,
                playback_ack_enabled=True,
            )
        )
        assert window.finite_request
        assert await window.allow_output(_frame(0))

        pending = asyncio.create_task(window.allow_output(_frame(1)))
        await asyncio.sleep(0.01)
        assert not pending.done()
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
        assert await pending
        metrics = window.metrics()
        assert metrics["gateway_playback_ack_shed_messages"] == 0
        assert metrics["gateway_playback_ack_rejected_messages"] == 0

    asyncio.run(run())


def test_gateway_finite_playback_ack_timeout_fails_and_blocks_completion():
    async def run():
        window = BrowserPlaybackAckWindow(
            max_unacked_chunks=1,
            wait_timeout_s=0.01,
        )
        await window.observe_browser_message(
            encode_message(
                "init",
                max_chunks=2,
                playback_ack_enabled=True,
            )
        )
        assert await window.allow_output(_frame(0))
        with pytest.raises(OutputBackpressureError, match="ACK window"):
            await window.allow_output(_frame(1))
        with pytest.raises(OutputBackpressureError, match="ACK window"):
            await window.allow_output(_media_complete(1))
        metrics = window.metrics()
        assert metrics["gateway_playback_ack_shed_messages"] == 0
        assert metrics["gateway_playback_ack_rejected_messages"] == 1
        assert metrics["gateway_playback_ack_failed"] is True

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
        route.configure_request_mode(finite_request=False)

        await route.put(_frame(0, 0))
        await asyncio.wait_for(route.put(_frame(0, 1)), timeout=0.01)
        assert route.dropped_messages == 1
        assert route.dropped_frames == 1
        assert await route.get() == _frame(0, 1)
        route.task_done()

    asyncio.run(run())


def test_gateway_continuous_route_does_not_end_on_transient_output_unbind():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=False)
        route.bind_output()
        route.unbind_output()

        pending = asyncio.create_task(route.get())
        await asyncio.sleep(0.01)
        assert not pending.done()
        await route.close()
        with pytest.raises(OutputRouteClosed, match="output route is closed"):
            await pending

    asyncio.run(run())


def test_gateway_finite_output_route_backpressures_without_shedding():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1, enqueue_timeout_s=0.1)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)

        first = _frame(0, 0)
        second = _frame(0, 1)
        await route.put(first)
        pending = asyncio.create_task(route.put(second))
        await asyncio.sleep(0.01)
        assert not pending.done()
        assert route.dropped_messages == 0

        assert await route.get() == first
        route.task_done()
        await pending
        assert await route.get() == second
        route.task_done()
        await route.join()
        metrics = route.queue_metrics()
        assert metrics["gateway_dropped_messages"] == 0
        assert metrics["gateway_rejected_messages"] == 0
        assert metrics["gateway_output_finite"] is True

    asyncio.run(run())


def test_gateway_finite_completion_waits_until_prior_frames_are_forwarded():
    async def run():
        registry = GatewayOutputRegistry(
            queue_depth=2,
            enqueue_timeout_s=0.1,
            completion_timeout_s=0.1,
        )
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True, expected_final_chunk=0)

        frame = _frame(0, 0)
        completion = _media_complete(0, is_final=True)
        await route.put(frame)
        completion_put = asyncio.create_task(route.put(completion))
        await asyncio.sleep(0.01)
        assert not completion_put.done()
        assert route.accepted_completions == 0

        assert await route.get() == frame
        route.mark_output_forwarded(frame)
        route.task_done()
        await completion_put
        assert route.accepted_completions == 1
        assert await route.get() == completion
        route.mark_output_forwarded(completion)
        route.task_done()
        await route.wait_until_chunk_forwarded(0)
        await route.wait_until_final_completion_forwarded()
        assert route.final_completion_forwarded is True

    asyncio.run(run())


def test_gateway_finite_completion_is_rejected_after_downstream_send_failure():
    async def run():
        registry = GatewayOutputRegistry(
            queue_depth=2,
            enqueue_timeout_s=0.1,
            completion_timeout_s=0.1,
        )
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)

        frame = _frame(0, 0)
        await route.put(frame)
        completion_put = asyncio.create_task(route.put(_media_complete(0)))
        assert await route.get() == frame
        route.fail_output("finite browser send failed: RuntimeError")
        route.task_done()

        with pytest.raises(OutputBackpressureError, match="browser send failed"):
            await completion_put
        assert route.accepted_completions == 0
        assert route.rejected_completions == 1
        assert route.failure_reason is not None
        await route.close()

    asyncio.run(run())


def test_gateway_failure_wakes_chunk_and_final_forward_waiters():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=2)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)
        route.bind_output()

        chunk_waiter = asyncio.create_task(route.wait_until_chunk_forwarded(0))
        final_waiter = asyncio.create_task(
            route.wait_until_final_completion_forwarded()
        )
        await asyncio.sleep(0)
        route.fail_output("downstream send failed")

        for waiter in (chunk_waiter, final_waiter):
            with pytest.raises(OutputBackpressureError, match="downstream send failed"):
                await asyncio.wait_for(waiter, timeout=0.05)

    asyncio.run(run())


def test_gateway_finite_waiter_rejects_output_close_without_final_marker():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=2)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)
        route.bind_output()
        waiter = asyncio.create_task(route.wait_until_final_completion_forwarded())
        await asyncio.sleep(0)
        route.unbind_output()

        with pytest.raises(OutputIncompleteError, match="output closed"):
            await asyncio.wait_for(waiter, timeout=0.05)

    asyncio.run(run())


def test_gateway_finite_expected_chunk_requires_final_marker():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=2)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True, expected_final_chunk=0)
        frame = _frame(0)
        await route.put(frame)
        assert await route.get() == frame
        route.task_done()

        with pytest.raises(OutputProtocolError, match="lacks final media marker"):
            await route.put(_media_complete(0))

    asyncio.run(run())


def test_gateway_finite_output_route_timeout_fails_before_completion():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1, enqueue_timeout_s=0.01)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)

        await route.put(_frame(0, 0))
        with pytest.raises(OutputBackpressureError, match="queue remained full"):
            await route.put(_frame(0, 1))
        with pytest.raises(OutputBackpressureError, match="queue remained full"):
            await route.put(_media_complete(0))

        metrics = route.queue_metrics()
        assert metrics["gateway_dropped_messages"] == 0
        assert metrics["gateway_rejected_messages"] == 1
        assert metrics["gateway_rejected_frames"] == 1
        assert metrics["gateway_accepted_completions"] == 0
        assert metrics["gateway_rejected_completions"] == 1
        assert metrics["gateway_output_failed"] is True
        await route.close()

    asyncio.run(run())


def test_gateway_finite_zero_enqueue_timeout_rejects_immediately_without_shedding():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=1, enqueue_timeout_s=0)
        route = await registry.register("s", "g", token="secret")
        route.configure_request_mode(finite_request=True)
        await route.put(_frame(0, 0))

        started = asyncio.get_running_loop().time()
        with pytest.raises(OutputBackpressureError, match="0.000s"):
            await route.put(_frame(0, 1))
        assert asyncio.get_running_loop().time() - started < 0.05
        assert route.dropped_messages == 0
        assert route.rejected_messages == 1

    asyncio.run(run())


def test_gateway_output_route_uses_media_completion_as_authoritative_marker():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=3)
        route = await registry.register("s", "g", token="secret")

        await route.put(_frame(0, 0))
        completion_waiter = asyncio.create_task(route.wait_until_chunk_completed(0))
        await asyncio.sleep(0)
        assert not completion_waiter.done()

        completion = _media_complete(0)
        await route.put(completion)
        await completion_waiter
        forwarded_waiter = asyncio.create_task(route.wait_until_chunk_forwarded(0))
        await asyncio.sleep(0)
        assert not forwarded_waiter.done()
        assert await route.get() == _frame(0, 0)
        route.mark_output_forwarded(_frame(0, 0))
        route.task_done()
        assert await route.get() == completion
        route.mark_output_forwarded(completion)
        route.task_done()
        await forwarded_waiter
        await route.join()

        metrics = route.queue_metrics()
        assert metrics["gateway_accepted_completions"] == 1
        assert metrics["gateway_forwarded_completions"] == 1
        assert metrics["gateway_forwarded_frames"] == 1

        with pytest.raises(OutputProtocolError, match="duplicate completion"):
            await route.put(completion)

    asyncio.run(run())


def test_gateway_final_frame_batch_does_not_replace_media_completion():
    async def run():
        registry = GatewayOutputRegistry(queue_depth=2)
        route = await registry.register("s", "g", token="secret")
        await route.put(_frame(0, 0, is_final=True))

        completion_waiter = asyncio.create_task(route.wait_until_chunk_completed(0))
        await asyncio.sleep(0)
        assert not completion_waiter.done()
        await route.put(_media_complete(0))
        await completion_waiter
        await route.close()

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
        route = await registry.register("s", "g", token="secret", trace_id="trace-a")

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
    assert not worker_message_allowed(msgspec.msgpack.encode({"type": "chunk_stats"}))
    assert not worker_message_allowed(_frame(0))
    assert not worker_message_allowed(encode_message("trace_events", traces=[]))


def test_gateway_output_url_carries_completion_ack_budget_and_preserves_query():
    url = build_gateway_output_url(
        "ws://gateway/v1/internal/realtime_output?existing=1",
        completion_ack_timeout_s=95,
    )

    assert url == (
        "ws://gateway/v1/internal/realtime_output"
        "?existing=1&completion_ack_timeout_s=95.000"
    )
