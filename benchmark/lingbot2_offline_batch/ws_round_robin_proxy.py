#!/usr/bin/env python3
"""Round-robin SGLang WebSockets and split oversized raw frame messages."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import logging
from collections.abc import Iterator

import msgspec
from aiohttp import ClientSession, WSMsgType, web


LOGGER = logging.getLogger("ws_round_robin_proxy")
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"


def parse_ports(value: str) -> tuple[int, ...]:
    try:
        ports = tuple(int(port) for port in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid target ports: {value!r}") from error
    if not ports or any(not 1 <= port <= 65535 for port in ports):
        raise argparse.ArgumentTypeError(f"invalid target ports: {value!r}")
    return ports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=31443)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-ports", type=parse_ports, required=True)
    parser.add_argument("--upstream-path", default="/v1/realtime_video/generate")
    parser.add_argument("--max-raw-frames-per-message", type=int, default=4)
    return parser.parse_args()


def split_raw_frame_message(
    header: dict,
    payload: bytes,
    *,
    max_frames: int,
) -> list[tuple[bytes, bytes]]:
    num_frames = int(header.get("num_frames", 0))
    bytes_per_frame = int(header.get("bytes_per_frame", 0))
    if (
        header.get("type") != "frame_batch_header"
        or header.get("content_type") != RAW_RGB_CONTENT_TYPE
        or num_frames <= max_frames
        or bytes_per_frame <= 0
        or len(payload) != num_frames * bytes_per_frame
    ):
        return [(msgspec.msgpack.encode(header), payload)]

    frame_ranges = [
        (start, min(start + max_frames, num_frames))
        for start in range(0, num_frames, max_frames)
    ]
    original_batch_index = int(header.get("frame_batch_index", 0))
    original_batch_count = int(header.get("num_frame_batches", 1))
    original_is_final = bool(header.get("is_final_frame_batch", True))
    part_count = len(frame_ranges)
    messages: list[tuple[bytes, bytes]] = []
    for part_index, (start, end) in enumerate(frame_ranges):
        part_payload = payload[start * bytes_per_frame : end * bytes_per_frame]
        part_header = dict(header)
        part_header.update(
            {
                "num_frames": end - start,
                "total_size": len(part_payload),
                "frame_batch_index": original_batch_index * part_count + part_index,
                "num_frame_batches": original_batch_count * part_count,
                "is_final_frame_batch": original_is_final
                and part_index == part_count - 1,
            }
        )
        messages.append((msgspec.msgpack.encode(part_header), part_payload))
    return messages


async def forward_client_to_upstream(client_ws, upstream_ws) -> None:
    async for message in client_ws:
        if message.type == WSMsgType.BINARY:
            await upstream_ws.send_bytes(message.data)
        elif message.type == WSMsgType.TEXT:
            await upstream_ws.send_str(message.data)
        elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
            break


async def forward_upstream_to_client(
    upstream_ws,
    client_ws,
    *,
    max_raw_frames: int,
) -> None:
    async for message in upstream_ws:
        if message.type == WSMsgType.TEXT:
            await client_ws.send_str(message.data)
            continue
        if message.type != WSMsgType.BINARY:
            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break
            continue

        raw_message = message.data
        try:
            decoded = msgspec.msgpack.decode(raw_message)
        except msgspec.DecodeError:
            await client_ws.send_bytes(raw_message)
            continue
        if not isinstance(decoded, dict) or decoded.get("type") != "frame_batch_header":
            await client_ws.send_bytes(raw_message)
            continue

        payload_message = await upstream_ws.receive()
        if payload_message.type != WSMsgType.BINARY:
            raise RuntimeError("frame_batch_header was not followed by binary payload")
        for header_payload, frame_payload in split_raw_frame_message(
            decoded,
            payload_message.data,
            max_frames=max_raw_frames,
        ):
            await client_ws.send_bytes(header_payload)
            await client_ws.send_bytes(frame_payload)


def create_app(args: argparse.Namespace) -> web.Application:
    target_ports: Iterator[int] = itertools.cycle(args.target_ports)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def proxy(request: web.Request) -> web.WebSocketResponse:
        client_ws = web.WebSocketResponse(max_msg_size=0, autoping=True)
        await client_ws.prepare(request)
        target_port = next(target_ports)
        upstream_url = (
            f"ws://{args.target_host}:{target_port}{args.upstream_path}"
        )
        peer = request.transport.get_extra_info("peername") if request.transport else None
        LOGGER.info("proxying %s to %s", peer, upstream_url)
        try:
            async with ClientSession() as session:
                async with session.ws_connect(
                    upstream_url,
                    max_msg_size=0,
                    autoping=True,
                ) as upstream_ws:
                    tasks = {
                        asyncio.create_task(
                            forward_client_to_upstream(client_ws, upstream_ws)
                        ),
                        asyncio.create_task(
                            forward_upstream_to_client(
                                upstream_ws,
                                client_ws,
                                max_raw_frames=args.max_raw_frames_per_message,
                            )
                        ),
                    }
                    _done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for task in tasks:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
        except Exception:
            LOGGER.exception("proxy connection failed for %s", peer)
        finally:
            await client_ws.close()
        return client_ws

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get(args.upstream_path, proxy)
    return app


def main() -> None:
    args = parse_args()
    if args.max_raw_frames_per_message < 1:
        raise SystemExit("--max-raw-frames-per-message must be positive")
    web.run_app(
        create_app(args),
        host=args.listen_host,
        port=args.listen_port,
        print=None,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
