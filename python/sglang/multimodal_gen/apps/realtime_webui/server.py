# SPDX-License-Identifier: Apache-2.0

"""Serve the realtime UI and proxy its HTTP/WebSocket API on one port."""

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from prompt_rewriter import PromptRewriter
from world_creator import WorldCreator


ROOT = Path(__file__).resolve().parent
UPSTREAM_HTTP = os.environ.get("REALTIME_UPSTREAM_HTTP", "http://127.0.0.1:30000")
UPSTREAM_WS = os.environ.get("REALTIME_UPSTREAM_WS", "ws://127.0.0.1:30000")
BACKEND_ENV_PREFIXES = {
    "minwm": "MINWM",
    "lingbot2": "LINGBOT2",
}
SESSION = web.AppKey("upstream_session", ClientSession)
PROMPT_REWRITER = web.AppKey("prompt_rewriter", PromptRewriter)
WORLD_CREATOR = web.AppKey("world_creator", WorldCreator)
MAX_WORLD_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_WORLD_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
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


async def _rewrite_prompt(request):
    """Rewrite one Live Direction without exposing Vertex credentials."""

    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "request body must be valid JSON"}),
            content_type="application/json",
        ) from error
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "request body must be a JSON object"}),
            content_type="application/json",
        )
    instruction = str(body.get("instruction", "")).strip()
    previous_prompt = str(body.get("previous_prompt", "")).strip()
    if not instruction or not previous_prompt:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"error": "instruction and previous_prompt are required"}
            ),
            content_type="application/json",
        )
    if len(instruction) > 2000 or len(previous_prompt) > 20000:
        raise web.HTTPRequestEntityTooLarge(
            max_size=22000,
            actual_size=len(instruction) + len(previous_prompt),
        )

    rewriter = request.app[PROMPT_REWRITER]
    if not rewriter.configured:
        raise web.HTTPServiceUnavailable(
            text=json.dumps({"error": "prompt rewriter is not configured"}),
            content_type="application/json",
        )
    try:
        result = await rewriter.rewrite(instruction, previous_prompt)
    except Exception as error:
        logging.exception("Live Direction prompt rewrite failed")
        raise web.HTTPBadGateway(
            text=json.dumps({"error": "prompt rewrite failed; please try again"}),
            content_type="application/json",
        ) from error
    return web.json_response(result.model_dump(mode="json"))


async def _complete_world(request):
    """Complete a world from text, an uploaded first frame, or both."""

    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "world completion requires multipart form data"}),
            content_type="application/json",
        )
    world_description = ""
    image_bytes = bytearray()
    image_mime_type = ""
    try:
        reader = await request.multipart()
        async for part in reader:
            if part.name == "world_description":
                world_description = (await part.text()).strip()
                if len(world_description) > 20000:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=20000, actual_size=len(world_description)
                    )
            elif part.name == "first_frame":
                image_mime_type = (part.headers.get("Content-Type") or "").lower()
                if image_mime_type not in ALLOWED_WORLD_IMAGE_TYPES:
                    raise web.HTTPUnsupportedMediaType(
                        text=json.dumps(
                            {"error": "first frame must be PNG, JPEG, or WebP"}
                        ),
                        content_type="application/json",
                    )
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    image_bytes.extend(chunk)
                    if len(image_bytes) > MAX_WORLD_IMAGE_BYTES:
                        raise web.HTTPRequestEntityTooLarge(
                            max_size=MAX_WORLD_IMAGE_BYTES,
                            actual_size=len(image_bytes),
                        )
    except web.HTTPException:
        raise
    except Exception as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "unable to read world completion input"}),
            content_type="application/json",
        ) from error

    if not world_description and not image_bytes:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"error": "write a world description or upload a first frame"}
            ),
            content_type="application/json",
        )
    creator = request.app[WORLD_CREATOR]
    if not creator.description_configured:
        raise web.HTTPServiceUnavailable(
            text=json.dumps({"error": "world completion is not configured"}),
            content_type="application/json",
        )
    if not image_bytes and not creator.image_configured:
        raise web.HTTPServiceUnavailable(
            text=json.dumps({"error": "world image generation is not configured"}),
            content_type="application/json",
        )
    try:
        completed = await creator.complete(
            world_description,
            bytes(image_bytes) if image_bytes else None,
            image_mime_type or "image/png",
        )
    except Exception as error:
        logging.exception("World completion failed")
        raise web.HTTPBadGateway(
            text=json.dumps({"error": "world completion failed; please try again"}),
            content_type="application/json",
        ) from error

    image_url = None
    if completed.image_bytes:
        image_id = creator.save_generated_image(completed.image_bytes)
        image_url = f"/api/world/images/{image_id}"
    return web.json_response(
        {
            "world_description": completed.world_description,
            "image_generated": completed.image_generated,
            "image_url": image_url,
        }
    )


async def _generated_world_image(request):
    path = request.app[WORLD_CREATOR].generated_image_path(
        request.match_info["image_id"]
    )
    if path is None:
        raise web.HTTPNotFound()
    return web.FileResponse(
        path,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
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
    if app[PROMPT_REWRITER].configured:
        try:
            await app[PROMPT_REWRITER]._get_client()
        except Exception:
            logging.exception("Unable to initialize the Live Direction rewriter")
    yield
    await app[SESSION].close()


def create_app():
    app = web.Application(client_max_size=1024**3)
    prompt_rewriter = PromptRewriter()
    app[PROMPT_REWRITER] = prompt_rewriter
    app[WORLD_CREATOR] = WorldCreator(
        gemini_client_provider=prompt_rewriter._get_client
    )
    app.cleanup_ctx.append(_session_context)
    app.router.add_get("/", _index)
    app.router.add_get("/runtime-config.js", _runtime_config)
    app.router.add_post("/api/prompt/rewrite", _rewrite_prompt)
    app.router.add_post("/api/world/complete", _complete_world)
    app.router.add_get(
        "/api/world/images/{image_id}", _generated_world_image
    )
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
