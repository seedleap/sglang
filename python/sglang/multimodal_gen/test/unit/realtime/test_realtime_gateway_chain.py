# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import uvicorn
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    SessionAssignment,
    WorkerSlot,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Coordinator:
    def __init__(self, denoiser_endpoint: str):
        self.denoiser_endpoint = denoiser_endpoint
        self.admitted = []
        self.released = []
        self.released_at = None

    async def health(self):
        return {"status": "ready"}

    async def admit(self, **request):
        self.admitted.append(request)
        return SessionAssignment(
            user_id=request["user_id"],
            session_id=request["session_id"],
            generation_id=request["generation_id"],
            token="lease-token",
            expires_at=time.monotonic() + 60,
            denoiser=WorkerSlot(
                worker_id="denoiser-1",
                role="denoiser",
                endpoint=self.denoiser_endpoint,
                az="test-a",
                slot_index=0,
                model_revision=request["model_revision"],
                vae_fingerprint=request["vae_fingerprint"],
            ),
            vae=WorkerSlot(
                worker_id="vae-1",
                role="vae",
                endpoint="ws://vae-1:18081/v1/realtime_vae/decode",
                az="test-a",
                slot_index=0,
                model_revision="",
                vae_fingerprint=request["vae_fingerprint"],
            ),
        )

    async def renew(self, assignment):
        return assignment

    async def release(self, assignment):
        self.released_at = time.monotonic()
        self.released.append(assignment)


class _TraceQuery:
    async def query(self, trace_id, **_kwargs):
        return {
            "trace_id": trace_id,
            "events": [{"event": "gateway.ws_accepted", "trace_seq": 1}],
            "next_cursor": 1,
        }


async def _run_gateway(app, port: int):
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    return server, task


def test_gateway_routes_control_and_direct_vae_media_and_queries_trace_over_http():
    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        denoiser_close_started_at = None

        async def denoiser(connection):
            nonlocal denoiser_close_started_at
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]
            assert parse_qs(urlsplit(output_url).query)["completion_ack_timeout_s"] == [
                "95.000"
            ]

            await connection.send(encode_message("session_ready"))
            init = decode_message(await connection.recv())
            assert init["type"] == "init"
            assert init["max_chunks"] == 1
            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                accepted = decode_message(await output.recv())
                assert accepted["type"] == "session_output_accepted"

                async def send_frame(index: int):
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id="request-0",
                            chunk_index=0,
                            frame_batch_index=index,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                            is_final_frame_batch=False,
                        )
                    )

                await send_frame(0)
                denoiser_close_started_at = time.monotonic()
                await connection.close(code=1000, reason="generation complete")
                # The VAE media route is independent and can still be flushing
                # accepted frames after the Denoiser control route closes.
                await asyncio.sleep(0.05)
                await send_frame(1)
                await send_frame(2)
                await send_frame(3)
                await output.send(
                    encode_message(
                        "media_chunk_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-0",
                        chunk_index=0,
                        num_frames=4,
                        is_final_chunk=True,
                    )
                )
                completion_accepted = decode_message(await output.recv())
                assert completion_accepted["type"] == ("media_chunk_complete_accepted")
                assert completion_accepted["chunk_index"] == 0

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            output_drain_timeout_s=90,
            output_completion_timeout_s=90,
            output_ack_network_margin_s=5,
            trace_query=_TraceQuery(),
            release_grace_s=0.05,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-a"
                )
                async with connect(url, max_size=None, compression=None) as browser:

                    async def send_actions_until_closed():
                        await browser.send(
                            encode_message(
                                "init",
                                max_chunks=1,
                            )
                        )
                        await browser.send(
                            encode_message(
                                "camera_actions",
                                event_id=1,
                                actions=["w"],
                            )
                        )
                        await browser.wait_closed()

                    action_task = asyncio.create_task(send_actions_until_closed())
                    messages = []
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), 4)
                                )
                            )
                    except ConnectionClosedOK:
                        pass
                    await action_task
                assert {message["type"] for message in messages} == {
                    "session_ready",
                    "frame_batch",
                    "media_chunk_complete",
                }
                frame_batches = [
                    message for message in messages if message["type"] == "frame_batch"
                ]
                assert [message["frame_batch_index"] for message in frame_batches] == [
                    0,
                    1,
                    2,
                    3,
                ]
                assert frame_batches[-1]["is_final_frame_batch"] is False
                media_complete = next(
                    message
                    for message in messages
                    if message["type"] == "media_chunk_complete"
                )
                assert media_complete["chunk_index"] == 0
                assert all("trace" not in message["type"] for message in messages)

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{gateway_port}"
                        "/v1/realtime_video/traces/trace-a"
                    )
                assert response.status_code == 200
                assert response.json()["events"][0]["event"] == ("gateway.ws_accepted")
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.admitted) == 1
        assert coordinator.admitted[0]["wait_for_capacity"] is True
        assert len(coordinator.released) == 1
        assert denoiser_close_started_at is not None
        assert coordinator.released_at - denoiser_close_started_at >= 0.045

    asyncio.run(run())


def test_gateway_finite_final_marker_completes_while_upstream_remains_open():
    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        observed = {}

        async def denoiser(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]
            assert decode_message(await connection.recv())["max_chunks"] == 1

            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                assert decode_message(await output.recv())["type"] == (
                    "session_output_accepted"
                )
                await output.send(
                    encode_message(
                        "frame_batch",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-0",
                        chunk_index=0,
                        frame_batch_index=0,
                        payload_lengths=[4],
                        payload=b"webp",
                        content_type="image/webp",
                        width=8,
                        height=8,
                        num_frames=1,
                        is_final_frame_batch=True,
                    )
                )
                await output.send(
                    encode_message(
                        "media_chunk_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-0",
                        chunk_index=0,
                        num_frames=1,
                        is_final_chunk=True,
                    )
                )
                assert decode_message(await output.recv())["type"] == (
                    "media_chunk_complete_accepted"
                )
                # Do not close the Denoiser control route. The browser-visible
                # final marker itself must wake the Gateway session.
                await connection.wait_closed()
                observed["upstream_closed_by_gateway"] = True

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                messages = []
                async with connect(url, max_size=None, compression=None) as browser:
                    await browser.send(encode_message("init", max_chunks=1))
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), 2)
                                )
                            )
                    except ConnectionClosedOK:
                        pass
                    assert browser.close_code == 1000
            finally:
                server.should_exit = True
                await server_task

        assert observed["upstream_closed_by_gateway"] is True
        assert [message["type"] for message in messages] == [
            "frame_batch",
            "media_chunk_complete",
        ]
        assert messages[-1]["is_final_chunk"] is True

    asyncio.run(run())


def test_gateway_marks_t2v_finite_route_and_ack_window_before_forwarding_init():
    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        observed = {}
        app = None

        async def denoiser(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]

            init = decode_message(await connection.recv())
            route = app.state.output_registry._routes[session_id]
            observed.update(
                init=init,
                route_finite=route.finite_request,
                route_mode_configured=route.request_mode_configured,
            )
            await connection.send(encode_message("session_ready"))
            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                assert decode_message(await output.recv())["type"] == (
                    "session_output_accepted"
                )
                for chunk_index in range(3):
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=f"request-{chunk_index}",
                            chunk_index=chunk_index,
                            frame_batch_index=0,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                            is_final_frame_batch=True,
                        )
                    )
                await output.send(
                    encode_message(
                        "media_chunk_complete",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-2",
                        chunk_index=2,
                        num_frames=1,
                    )
                )
                # The producer may submit completion before the ACK timeout is
                # observed.  The finite completion barrier must reject it: the
                # browser never receives the marker and the VAE receives 1013.
                try:
                    await output.recv()
                except ConnectionClosed as exc:
                    observed["output_close_code"] = (
                        exc.rcvd.code if exc.rcvd is not None else None
                    )
            await connection.wait_closed()

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            output_queue_depth=8,
            output_enqueue_timeout_s=0.5,
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-finite"
                )
                messages = []
                async with connect(url, max_size=None, compression=None) as browser:
                    await browser.send(
                        encode_message(
                            "init",
                            num_frames=3,
                            playback_ack_enabled=True,
                        )
                    )
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), 3)
                                )
                            )
                    except ConnectionClosed as exc:
                        observed["browser_close_code"] = (
                            exc.rcvd.code if exc.rcvd is not None else None
                        )
            finally:
                server.should_exit = True
                await server_task

        assert observed["init"].get("generation_mode") is None
        assert observed["route_finite"] is True
        assert observed["route_mode_configured"] is True
        assert observed["output_close_code"] == 1013
        assert observed["browser_close_code"] == 1013
        assert [
            message["chunk_index"]
            for message in messages
            if message["type"] == "frame_batch"
        ] == [0, 1]
        assert not any(
            message["type"] == "media_chunk_complete" for message in messages
        )
        error = next(message for message in messages if message["type"] == "error")
        assert "ACK window" in error["content"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure_mode",
    ("output_closed", "output_timeout", "missing_final_marker"),
)
def test_gateway_finite_session_rejects_missing_final_completion(
    failure_mode, monkeypatch
):
    traces = []
    monkeypatch.setattr(
        realtime_gateway_server,
        "_log_gateway_trace",
        lambda trace_id, event, **fields: traces.append(
            {"trace_id": trace_id, "event": event, **fields}
        ),
    )

    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        observed = {}

        async def denoiser(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]
            assert decode_message(await connection.recv())["max_chunks"] == 1

            if failure_mode != "output_timeout":
                async with connect(
                    output_url, max_size=None, compression=None
                ) as output:
                    await output.send(
                        encode_message(
                            "session_output_open",
                            session_id=session_id,
                            generation_id=generation_id,
                            token=output_token,
                        )
                    )
                    assert decode_message(await output.recv())["type"] == (
                        "session_output_accepted"
                    )
                    await output.send(
                        encode_message(
                            "frame_batch",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id="request-0",
                            chunk_index=0,
                            frame_batch_index=0,
                            payload_lengths=[4],
                            payload=b"webp",
                            content_type="image/webp",
                            width=8,
                            height=8,
                            num_frames=1,
                            is_final_frame_batch=True,
                        )
                    )
                    if failure_mode == "missing_final_marker":
                        await output.send(
                            encode_message(
                                "media_chunk_complete",
                                session_id=session_id,
                                generation_id=generation_id,
                                request_id="request-0",
                                chunk_index=0,
                                num_frames=1,
                            )
                        )
                        try:
                            await output.recv()
                        except ConnectionClosed as exc:
                            observed["output_close_code"] = (
                                exc.rcvd.code if exc.rcvd is not None else None
                            )
            await connection.close(code=1000, reason="producer stopped")

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            output_drain_timeout_s=0.05,
            output_completion_timeout_s=0.05,
            output_ack_network_margin_s=0.05,
            release_grace_s=0,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                    f"?trace_id=trace-{failure_mode}"
                )
                messages = []
                async with connect(url, max_size=None, compression=None) as browser:
                    await browser.send(encode_message("init", max_chunks=1))
                    try:
                        while True:
                            messages.append(
                                decode_message(
                                    await asyncio.wait_for(browser.recv(), 2)
                                )
                            )
                    except ConnectionClosed as exc:
                        observed["browser_close_code"] = (
                            exc.rcvd.code if exc.rcvd is not None else None
                        )
            finally:
                server.should_exit = True
                await server_task

        assert observed["browser_close_code"] == 1011
        assert not any(
            message["type"] == "media_chunk_complete" for message in messages
        )
        error = next(message for message in messages if message["type"] == "error")
        if failure_mode == "missing_final_marker":
            assert "lacks final media marker" in error["content"]
            assert observed["output_close_code"] == 1008
            rejected = next(
                trace for trace in traces if trace["event"] == "gateway.output_rejected"
            )
            assert rejected["reject_kind"] == "protocol"
            assert rejected["reject_reason"] == (
                "expected final chunk lacks final media marker"
            )
            assert rejected["gateway_rejected_completions"] == 1
        else:
            assert "final media completion" in error["content"]
        incomplete = next(
            trace for trace in traces if trace["event"] == "gateway.output_incomplete"
        )
        assert incomplete["gateway_final_completion_forwarded"] is False
        closed = next(
            trace for trace in traces if trace["event"] == "gateway.session_closed"
        )
        assert closed["session_succeeded"] is False
        assert closed["session_outcome"] == "output_incomplete"
        assert closed["close_code"] == 1011

    asyncio.run(run())


def test_gateway_routes_named_lingbot2_websocket_through_coordinator():
    async def run():
        gateway_port = _free_port()
        lingbot2_port = _free_port()
        received = []

        async def lingbot2_denoiser(connection):
            query = parse_qs(urlsplit(connection.request.path).query)
            await connection.send(encode_message("session_ready"))
            received.append(
                {
                    "query": query,
                    "payload": await connection.recv(),
                }
            )
            await connection.close(code=1000)

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{lingbot2_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            lingbot2_vae_fingerprint="taew2_1",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            lingbot2_model_revision="lingbot2-r1",
            release_grace_s=0,
        )

        async with serve(lingbot2_denoiser, "127.0.0.1", lingbot2_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}"
                    "/backends/lingbot2/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-a"
                )
                payload = encode_message("camera_actions", event_id=7, actions=["w"])
                async with connect(url, max_size=None, compression=None) as browser:
                    ready = decode_message(await browser.recv())
                    assert ready["type"] == "session_ready", ready
                    await browser.send(payload)
                    await browser.wait_closed()
                assert coordinator.admitted[0]["model_revision"] == "lingbot2-r1"
                assert coordinator.admitted[0]["vae_fingerprint"] == "taew2_1"
                assert received[0]["query"]["trace_id"] == ["trace-a"]
                assert received[0]["query"]["realtime_vae_worker_url"] == [
                    "ws://vae-1:18081/v1/realtime_vae/decode"
                ]
                assert decode_message(received[0]["payload"])["event_id"] == 7
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.released) == 1

    asyncio.run(run())
