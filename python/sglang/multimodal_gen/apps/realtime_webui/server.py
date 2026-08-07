# SPDX-License-Identifier: Apache-2.0

"""Serve the realtime UI and proxy its HTTP/WebSocket API on one port."""

import asyncio
import json
import os
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web


ROOT = Path(__file__).resolve().parent
UPSTREAM_HTTP = os.environ.get("REALTIME_UPSTREAM_HTTP", "http://127.0.0.1:30000")
UPSTREAM_WS = os.environ.get("REALTIME_UPSTREAM_WS", "ws://127.0.0.1:30000")
BACKEND_ENV_PREFIXES = {
    "minwm": "MINWM",
    "lingbot2": "LINGBOT2",
}
SESSION = web.AppKey("upstream_session", ClientSession)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _forward_headers(headers):
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }


async def _index(_request):
    return web.FileResponse(ROOT / "index.html")


async def _runtime_config(_request):
    raw_config = os.environ.get("REALTIME_UI_CONFIG_JSON", "{}")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as error:
        raise web.HTTPInternalServerError(
            text=f"invalid REALTIME_UI_CONFIG_JSON: {error.msg}"
        ) from error
    if not isinstance(config, dict):
        raise web.HTTPInternalServerError(
            text="REALTIME_UI_CONFIG_JSON must contain a JSON object"
        )
    return web.Response(
        text=f"globalThis.SGLANG_REALTIME_UI_CONFIG = {json.dumps(config)};\n",
        content_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


async def _proxy_http(request):
    upstream_url = f"{UPSTREAM_HTTP}{request.rel_url}"
    async with request.app[SESSION].request(
        request.method,
        upstream_url,
        headers=_forward_headers(request.headers),
        data=request.content,
        allow_redirects=False,
    ) as response:
        return web.Response(
            status=response.status,
            headers=_forward_headers(response.headers),
            body=await response.read(),
        )


def _backend_upstream(backend, transport, path, query=""):
    prefix = BACKEND_ENV_PREFIXES[backend]
    default = UPSTREAM_WS if transport == "ws" else UPSTREAM_HTTP
    base = os.environ.get(f"{prefix}_UPSTREAM_{transport.upper()}", default)
    return f"{base.rstrip('/')}/{path.lstrip('/')}{query}"


def _named_backend_url(request, transport):
    backend = request.match_info["backend"]
    path = f"/v1/{request.match_info.get('path', '')}".rstrip("/")
    if request.path.endswith("/generate"):
        path = "/v1/realtime_video/generate"
    query = f"?{request.query_string}" if request.query_string else ""
    try:
        return _backend_upstream(backend, transport, path, query)
    except KeyError as error:
        raise web.HTTPNotFound(text=f"unknown realtime backend: {backend}") from error


async def _proxy_backend_http(request):
    upstream_url = _named_backend_url(request, "http")
    async with request.app[SESSION].request(
        request.method,
        upstream_url,
        headers=_forward_headers(request.headers),
        data=request.content,
        allow_redirects=False,
    ) as response:
        return web.Response(
            status=response.status,
            headers=_forward_headers(response.headers),
            body=await response.read(),
        )


async def _relay_websocket(source, destination):
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await destination.send_str(message.data)
        elif message.type == WSMsgType.BINARY:
            await destination.send_bytes(message.data)
        elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
            break


async def _proxy_websocket(request):
    downstream = web.WebSocketResponse(max_msg_size=0)
    await downstream.prepare(request)
    upstream_url = f"{UPSTREAM_WS}{request.rel_url}"

    try:
        async with request.app[SESSION].ws_connect(
            upstream_url,
            max_msg_size=0,
        ) as upstream:
            relays = {
                asyncio.create_task(_relay_websocket(downstream, upstream)),
                asyncio.create_task(_relay_websocket(upstream, downstream)),
            }
            done, pending = await asyncio.wait(
                relays,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except Exception:
        await downstream.close(
            code=1011,
            message=b"upstream websocket unavailable",
        )
    finally:
        if not downstream.closed:
            await downstream.close()

    return downstream


async def _proxy_backend_websocket(request):
    upstream_url = _named_backend_url(request, "ws")
    downstream = web.WebSocketResponse(max_msg_size=0)
    await downstream.prepare(request)

    try:
        async with request.app[SESSION].ws_connect(
            upstream_url,
            max_msg_size=0,
        ) as upstream:
            relays = {
                asyncio.create_task(_relay_websocket(downstream, upstream)),
                asyncio.create_task(_relay_websocket(upstream, downstream)),
            }
            done, pending = await asyncio.wait(
                relays,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except Exception:
        await downstream.close(
            code=1011,
            message=b"upstream websocket unavailable",
        )
    finally:
        if not downstream.closed:
            await downstream.close()

    return downstream


async def _session_context(app):
    app[SESSION] = ClientSession(timeout=ClientTimeout(total=None))
    yield
    await app[SESSION].close()


def create_app():
    app = web.Application(client_max_size=1024**3)
    app.cleanup_ctx.append(_session_context)
    app.router.add_get("/", _index)
    app.router.add_get("/runtime-config.js", _runtime_config)
    app.router.add_get(
        "/backends/{backend}/v1/realtime_video/generate",
        _proxy_backend_websocket,
    )
    app.router.add_route(
        "*", "/backends/{backend}/v1/{path:.*}", _proxy_backend_http
    )
    app.router.add_get("/v1/realtime_video/generate", _proxy_websocket)
    app.router.add_route("*", "/v1/{path:.*}", _proxy_http)
    app.router.add_static("/", ROOT)
    return app


if __name__ == "__main__":
    web.run_app(
        create_app(),
        host=os.environ.get("WEBUI_HOST", "0.0.0.0"),
        port=int(os.environ.get("WEBUI_PORT", "18080")),
    )
