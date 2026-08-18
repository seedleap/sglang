# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.async_vae_client import (
    GatewayOutputClient,
    RemoteVAEError,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import encode_message


def test_gateway_output_client_uses_completion_ack_budget_from_route_url():
    client = GatewayOutputClient(
        "ws://gateway/v1/internal/realtime_output?completion_ack_timeout_s=95.000",
        session_id="s",
        generation_id="g",
        token="secret",
        timeout_s=5,
    )

    assert client.timeout_s == 5
    assert client.completion_ack_timeout_s == 95


def test_gateway_output_client_timeout_discards_late_ack_connection():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()
            self.closed = False

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            self.closed = True

    def completion(chunk_index):
        return encode_message(
            "media_chunk_complete",
            session_id="s",
            generation_id="g",
            request_id=f"r{chunk_index}",
            chunk_index=chunk_index,
            num_frames=1,
        )

    def accepted(chunk_index):
        return encode_message(
            "media_chunk_complete_accepted",
            session_id="s",
            generation_id="g",
            request_id=f"r{chunk_index}",
            chunk_index=chunk_index,
        )

    async def scenario():
        first = FakeSocket()
        second = FakeSocket()
        sockets = iter((first, second))

        async def connect_factory(*_args, **_kwargs):
            return next(sockets)

        await first.received.put(
            encode_message(
                "session_output_accepted",
                session_id="s",
                generation_id="g",
            )
        )
        client = GatewayOutputClient(
            "ws://gateway/v1/internal/realtime_output?completion_ack_timeout_s=0.010",
            session_id="s",
            generation_id="g",
            token="secret",
            connect_factory=connect_factory,
        )
        await client.open()
        await first.sent.get()

        with pytest.raises(RemoteVAEError, match="timed out after 0.010s"):
            await client.send(completion(0))
        assert first.closed is True
        assert client._ws is None
        await first.received.put(accepted(0))

        await second.received.put(
            encode_message(
                "session_output_accepted",
                session_id="s",
                generation_id="g",
            )
        )
        await client.open()
        await second.sent.get()
        await second.received.put(accepted(1))
        await client.send(completion(1))

        assert first.received.qsize() == 1
        assert second.received.empty()

    asyncio.run(scenario())


def test_gateway_output_client_cancellation_discards_late_ack_connection():
    class FakeSocket:
        def __init__(self):
            self.sent = asyncio.Queue()
            self.received = asyncio.Queue()
            self.closed = False

        async def send(self, payload):
            await self.sent.put(payload)

        async def recv(self):
            return await self.received.get()

        async def close(self):
            self.closed = True

    def completion(chunk_index):
        return encode_message(
            "media_chunk_complete",
            session_id="s",
            generation_id="g",
            request_id=f"r{chunk_index}",
            chunk_index=chunk_index,
            num_frames=1,
        )

    def accepted(chunk_index):
        return encode_message(
            "media_chunk_complete_accepted",
            session_id="s",
            generation_id="g",
            request_id=f"r{chunk_index}",
            chunk_index=chunk_index,
        )

    async def scenario():
        first = FakeSocket()
        second = FakeSocket()
        sockets = iter((first, second))

        async def connect_factory(*_args, **_kwargs):
            return next(sockets)

        await first.received.put(
            encode_message(
                "session_output_accepted",
                session_id="s",
                generation_id="g",
            )
        )
        client = GatewayOutputClient(
            "ws://gateway/v1/internal/realtime_output?completion_ack_timeout_s=30",
            session_id="s",
            generation_id="g",
            token="secret",
            connect_factory=connect_factory,
        )
        await client.open()
        await first.sent.get()

        pending = asyncio.create_task(client.send(completion(0)))
        assert await first.sent.get() == completion(0)
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert first.closed is True
        assert client._ws is None
        await first.received.put(accepted(0))

        await second.received.put(
            encode_message(
                "session_output_accepted",
                session_id="s",
                generation_id="g",
            )
        )
        await client.open()
        await second.sent.get()
        await second.received.put(accepted(1))
        await client.send(completion(1))

        assert first.received.qsize() == 1
        assert second.received.empty()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ("not-a-number", "0", "3601"))
def test_gateway_output_client_rejects_invalid_completion_ack_budget(value):
    with pytest.raises(ValueError, match="completion_ack_timeout_s"):
        GatewayOutputClient(
            "ws://gateway/v1/internal/realtime_output"
            f"?completion_ack_timeout_s={value}",
            session_id="s",
            generation_id="g",
            token="secret",
        )
