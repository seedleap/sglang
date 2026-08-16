#!/usr/bin/env python3
"""Expose loopback-only SGLang WebSocket listeners through TCP relays."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging


LOGGER = logging.getLogger("ws_tcp_relay")


def parse_mapping(value: str) -> tuple[int, int]:
    try:
        listen_port_text, target_port_text = value.split(":", 1)
        listen_port = int(listen_port_text)
        target_port = int(target_port_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"mapping must be LISTEN_PORT:TARGET_PORT, got {value!r}"
        ) from error
    for name, port in (("listen", listen_port), ("target", target_port)):
        if not 1 <= port <= 65535:
            raise argparse.ArgumentTypeError(f"{name} port is out of range: {port}")
    return listen_port, target_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument(
        "--mapping",
        action="append",
        type=parse_mapping,
        required=True,
        help="TCP mapping in LISTEN_PORT:TARGET_PORT form; repeat as needed",
    )
    return parser.parse_args()


async def copy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while data := await reader.read(1024 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        if writer.can_write_eof():
            with contextlib.suppress(ConnectionError):
                writer.write_eof()
                await writer.drain()
        else:
            writer.close()


async def relay_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        target_reader, target_writer = await asyncio.open_connection(
            target_host, target_port
        )
    except OSError as error:
        LOGGER.warning("failed to connect %s to %s:%d: %s", peer, target_host, target_port, error)
        client_writer.close()
        with contextlib.suppress(ConnectionError):
            await client_writer.wait_closed()
        return

    LOGGER.info("relaying %s to %s:%d", peer, target_host, target_port)
    upstream = asyncio.create_task(copy_stream(client_reader, target_writer))
    downstream = asyncio.create_task(copy_stream(target_reader, client_writer))
    await asyncio.gather(upstream, downstream, return_exceptions=True)
    target_writer.close()
    client_writer.close()
    with contextlib.suppress(ConnectionError):
        await target_writer.wait_closed()
    with contextlib.suppress(ConnectionError):
        await client_writer.wait_closed()


async def main() -> None:
    args = parse_args()
    servers: list[asyncio.Server] = []
    for listen_port, target_port in args.mapping:
        server = await asyncio.start_server(
            lambda reader, writer, port=target_port: relay_connection(
                reader, writer, args.target_host, port
            ),
            args.listen_host,
            listen_port,
        )
        servers.append(server)
        LOGGER.info(
            "listening on %s:%d -> %s:%d",
            args.listen_host,
            listen_port,
            args.target_host,
            target_port,
        )
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(main())
