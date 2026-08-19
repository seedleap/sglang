# SPDX-License-Identifier: Apache-2.0

"""Serve the realtime UI and proxy its HTTP/WebSocket API on one port."""

import asyncio
import io
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
    WSMsgType,
    web,
)
from h264_websocket_bridge import install_h264_websocket_bridge
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
HAPPYOYSTER_WORLD_CACHE = web.AppKey("happyoyster_world_cache", dict)
HAPPYOYSTER_WORLD_BUILD_LOCKS = web.AppKey("happyoyster_world_build_locks", dict)
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
HAPPYOYSTER_API_KEY = os.environ.get("HAPPYOYSTER_API_KEY", "").strip()
HAPPYOYSTER_API_BASE_URL = os.environ.get(
    "HAPPYOYSTER_API_BASE_URL",
    "https://llm-0jcmcer24vyvd7rr.cn-beijing.maas.aliyuncs.com/api/v2/apps/happyoyster-1.0/",
).rstrip("/")
HAPPYOYSTER_TOKEN_BASE_URL = os.environ.get("HAPPYOYSTER_TOKEN_BASE_URL", "").rstrip(
    "/"
)
HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL = os.environ.get(
    "HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL", ""
).rstrip("/")
HAPPYOYSTER_TIMEOUT_SECONDS = float(
    os.environ.get("HAPPYOYSTER_TIMEOUT_SECONDS", "130")
)
HAPPYOYSTER_FIRST_FRAME_SIZE = (1280, 720)
HAPPYOYSTER_PREBUILT_WORLDS_PATH = Path(
    os.environ.get(
        "HAPPYOYSTER_PREBUILT_WORLDS_PATH",
        ROOT / "happyoyster_prebuilt_worlds.json",
    )
).expanduser()
HAPPYOYSTER_PRESET_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


def _forward_headers(headers):
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }


async def _index(_request):
    return web.FileResponse(
        ROOT / "index.html",
        headers={"Cache-Control": "no-store"},
    )


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


def _direct_h264_gateway_enabled():
    try:
        config = json.loads(os.environ.get("REALTIME_UI_CONFIG_JSON", "{}"))
    except json.JSONDecodeError:
        return False
    return isinstance(config, dict) and config.get("h264DirectGatewayEnabled") is True


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
            text=json.dumps({"error": "instruction and previous_prompt are required"}),
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


async def _complete_world_rule(request):
    """Complete a skill/goal label and prompt without exposing credentials."""

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
    rule_input = str(body.get("input", "")).strip()
    previous_prompt = str(body.get("previous_prompt", "")).strip()
    kind = str(body.get("kind", "")).strip().lower()
    if not rule_input or not previous_prompt or kind not in {"skill", "goal"}:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {"error": ("input, previous_prompt, and kind=skill|goal are required")}
            ),
            content_type="application/json",
        )
    if len(rule_input) > 2000 or len(previous_prompt) > 20000:
        raise web.HTTPRequestEntityTooLarge(
            max_size=22000,
            actual_size=len(rule_input) + len(previous_prompt),
        )

    rewriter = request.app[PROMPT_REWRITER]
    if not rewriter.configured:
        raise web.HTTPServiceUnavailable(
            text=json.dumps({"error": "prompt rewriter is not configured"}),
            content_type="application/json",
        )
    try:
        result = await rewriter.complete_world_rule(rule_input, previous_prompt, kind)
    except Exception as error:
        logging.exception("World rule completion failed")
        raise web.HTTPBadGateway(
            text=json.dumps({"error": "rule completion failed; please try again"}),
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


def _happyoyster_configured():
    return bool(HAPPYOYSTER_API_KEY and HAPPYOYSTER_API_BASE_URL)


def _normalize_happyoyster_preset_key(value):
    preset_key = str(value or "").strip().lower()
    return preset_key if HAPPYOYSTER_PRESET_KEY_RE.fullmatch(preset_key) else ""


def _load_happyoyster_prebuilt_worlds():
    raw_config = os.environ.get("HAPPYOYSTER_PREBUILT_WORLDS_JSON", "").strip()
    try:
        payload = (
            json.loads(raw_config)
            if raw_config
            else json.loads(HAPPYOYSTER_PREBUILT_WORLDS_PATH.read_text())
        )
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        logging.exception("Unable to load HappyOyster prebuilt World manifest")
        return {}
    worlds = payload.get("worlds", payload) if isinstance(payload, dict) else {}
    if not isinstance(worlds, dict):
        return {}
    result = {}
    for raw_key, raw_value in worlds.items():
        key = _normalize_happyoyster_preset_key(raw_key)
        if isinstance(raw_value, dict):
            world_id = str(raw_value.get("encryptedWorldId", "")).strip()
        else:
            world_id = str(raw_value or "").strip()
        if key and world_id:
            result[key] = world_id
    return result


def _normalize_happyoyster_first_frame(image_bytes):
    """Return a bounded 16:9 JPEG accepted by the HappyOyster first-frame API."""

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        fitted = ImageOps.fit(
            rgb,
            HAPPYOYSTER_FIRST_FRAME_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = io.BytesIO()
        fitted.save(
            output,
            format="JPEG",
            quality=88,
            optimize=True,
            progressive=True,
        )
        return output.getvalue()


def _happyoyster_origin():
    parsed = urlsplit(HAPPYOYSTER_API_BASE_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


def _happyoyster_token_base_url():
    return HAPPYOYSTER_TOKEN_BASE_URL or _happyoyster_origin()


def _unwrap_happyoyster_payload(payload):
    if isinstance(payload, dict) and isinstance(payload.get("output"), dict):
        payload = payload["output"]
    if isinstance(payload, dict) and payload.get("code") == 0:
        return payload.get("data")
    if isinstance(payload, dict) and payload.get("code") not in (None, 0):
        raise RuntimeError(
            str(payload.get("message") or f"HappyOyster error {payload['code']}")
        )
    return payload


async def _happyoyster_request(
    app,
    method,
    path,
    *,
    params=None,
    json_body=None,
    base_url=None,
):
    if not _happyoyster_configured():
        raise web.HTTPServiceUnavailable(
            text=json.dumps({"error": "HappyOyster is not configured"}),
            content_type="application/json",
        )
    request_base_url = (base_url or HAPPYOYSTER_API_BASE_URL).rstrip("/")
    url = f"{request_base_url}/{path.lstrip('/')}"
    try:
        async with app[SESSION].request(
            method,
            url,
            params=params,
            json=json_body,
            headers={
                "Authorization": f"Bearer {HAPPYOYSTER_API_KEY}",
                "Accept": "application/json",
                "User-Agent": "sglang-world-studio/1.0",
            },
            timeout=ClientTimeout(total=HAPPYOYSTER_TIMEOUT_SECONDS),
        ) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(
                    str(
                        payload.get("message") if isinstance(payload, dict) else payload
                    )
                )
            return _unwrap_happyoyster_payload(payload)
    except web.HTTPException:
        raise
    except Exception as error:
        logging.exception("HappyOyster request failed: %s %s", method, url)
        raise web.HTTPBadGateway(
            text=json.dumps({"error": str(error) or "HappyOyster request failed"}),
            content_type="application/json",
        ) from error


async def _happyoyster_config(_request):
    return web.json_response(
        {
            "enabled": _happyoyster_configured(),
            "apiBaseUrl": HAPPYOYSTER_API_BASE_URL,
            "sessionMaxLifetimeSeconds": 60,
        },
        headers={"Cache-Control": "no-store"},
    )


async def _happyoyster_share_image(request):
    if not HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL:
        raise web.HTTPServiceUnavailable(
            text=json.dumps(
                {"error": "HappyOyster public image URL is not configured"}
            ),
            content_type="application/json",
        )
    image_bytes = await request.read()
    if not image_bytes:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "first frame is required"}),
            content_type="application/json",
        )
    if len(image_bytes) > MAX_WORLD_IMAGE_BYTES:
        raise web.HTTPRequestEntityTooLarge(
            max_size=MAX_WORLD_IMAGE_BYTES, actual_size=len(image_bytes)
        )
    mime_type = request.content_type.lower()
    if mime_type not in ALLOWED_WORLD_IMAGE_TYPES:
        raise web.HTTPUnsupportedMediaType(
            text=json.dumps({"error": "first frame must be PNG, JPEG, or WebP"}),
            content_type="application/json",
        )
    try:
        normalized = await asyncio.to_thread(
            _normalize_happyoyster_first_frame, image_bytes
        )
    except Exception as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "first frame is not a valid image"}),
            content_type="application/json",
        ) from error
    image_name = request.app[WORLD_CREATOR].save_shared_image(normalized, ".jpg")
    return web.json_response(
        {
            "url": f"{HAPPYOYSTER_PUBLIC_IMAGE_BASE_URL}/api/world/images/{image_name}",
            "contentType": "image/jpeg",
            "width": HAPPYOYSTER_FIRST_FRAME_SIZE[0],
            "height": HAPPYOYSTER_FIRST_FRAME_SIZE[1],
        }
    )


async def _happyoyster_create_world(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "request body must be valid JSON"}),
            content_type="application/json",
        ) from error
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "prompt is required"}),
            content_type="application/json",
        )
    upstream_body = {
        "mode": 1,
        "async": True,
        "creationModel": "simple",
        "prompt": prompt[:2000],
        "perspective": body.get("perspective", "third_person"),
        "uploadMode": "first_frame",
        "resolution": "720p",
        "eventStyle": "normal",
    }
    first_frame_url = str(body.get("firstFrameUrl", "")).strip()
    if first_frame_url:
        upstream_body["firstFrameImage"] = {
            "url": first_frame_url,
            "referenceType": "default",
        }
    preset_key = _normalize_happyoyster_preset_key(body.get("presetKey"))

    async def create_or_reuse():
        if preset_key:
            cached_world_id = request.app[HAPPYOYSTER_WORLD_CACHE].get(preset_key)
            if cached_world_id:
                try:
                    status_data = await _happyoyster_request(
                        request.app,
                        "GET",
                        "/openapi/v1/worlds/build-status",
                        params={"encryptedWorldId": cached_world_id},
                    )
                except web.HTTPBadGateway:
                    request.app[HAPPYOYSTER_WORLD_CACHE].pop(preset_key, None)
                else:
                    status = str(
                        status_data.get("status", "")
                        if isinstance(status_data, dict)
                        else ""
                    ).lower()
                    if status not in {"failed", "expired", "deleted", "not_found"}:
                        return {
                            "encryptedWorldId": cached_world_id,
                            "status": status or "building",
                            "source": "runtime-cache",
                        }
                    request.app[HAPPYOYSTER_WORLD_CACHE].pop(preset_key, None)
        data = await _happyoyster_request(
            request.app, "POST", "/openapi/v1/worlds", json_body=upstream_body
        )
        world_id = data.get("encryptedWorldId") if isinstance(data, dict) else None
        if preset_key and world_id:
            request.app[HAPPYOYSTER_WORLD_CACHE][preset_key] = str(world_id)
        return data

    if not preset_key:
        return web.json_response(await create_or_reuse())
    lock = request.app[HAPPYOYSTER_WORLD_BUILD_LOCKS].setdefault(
        preset_key, asyncio.Lock()
    )
    async with lock:
        return web.json_response(await create_or_reuse())


async def _happyoyster_resolve_world(request):
    """Resolve a reusable preset World, discarding expired or failed IDs."""

    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "request body must be valid JSON"}),
            content_type="application/json",
        ) from error
    preset_key = _normalize_happyoyster_preset_key(
        body.get("presetKey") if isinstance(body, dict) else ""
    )
    if not preset_key:
        return web.json_response({"status": "missing", "source": "none"})
    world_id = request.app[HAPPYOYSTER_WORLD_CACHE].get(preset_key)
    if not world_id:
        return web.json_response({"status": "missing", "source": "none"})
    try:
        data = await _happyoyster_request(
            request.app,
            "GET",
            "/openapi/v1/worlds/build-status",
            params={"encryptedWorldId": world_id},
        )
    except web.HTTPBadGateway:
        logging.warning(
            "HappyOyster cached World is unavailable; regenerating preset=%s",
            preset_key,
        )
        request.app[HAPPYOYSTER_WORLD_CACHE].pop(preset_key, None)
        return web.json_response(
            {"status": "missing", "source": "expired", "expiredWorldId": world_id}
        )
    status = str(data.get("status", "") if isinstance(data, dict) else "").lower()
    if status in {"failed", "expired", "deleted", "not_found"}:
        request.app[HAPPYOYSTER_WORLD_CACHE].pop(preset_key, None)
        return web.json_response(
            {"status": "missing", "source": "expired", "expiredWorldId": world_id}
        )
    return web.json_response(
        {
            "status": status or "building",
            "source": "prebuilt",
            "encryptedWorldId": world_id,
        },
        headers={"Cache-Control": "no-store"},
    )


async def _happyoyster_world_status(request):
    world_id = str(request.query.get("encryptedWorldId", "")).strip()
    if not world_id:
        raise web.HTTPBadRequest(text="encryptedWorldId is required")
    data = await _happyoyster_request(
        request.app,
        "GET",
        "/openapi/v1/worlds/build-status",
        params={"encryptedWorldId": world_id},
    )
    return web.json_response(data)


async def _happyoyster_prepare(request):
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
    world_id = str(body.get("encryptedWorldId", "")).strip()
    if not world_id:
        raise web.HTTPBadRequest(text="encryptedWorldId is required")
    credential, token_payload = await asyncio.gather(
        _happyoyster_request(
            request.app,
            "POST",
            "/openapi/v1/worlds/get-travel-credential",
            json_body={"encryptedWorldId": world_id},
        ),
        _happyoyster_request(
            request.app,
            "POST",
            "/api/v1/tokens",
            params={"expire_in_seconds": 1800},
            base_url=_happyoyster_token_base_url(),
        ),
    )
    token = token_payload.get("token") if isinstance(token_payload, dict) else None
    ticket = credential.get("ticket") if isinstance(credential, dict) else None
    if not token or not ticket:
        raise web.HTTPBadGateway(
            text=json.dumps({"error": "HappyOyster credentials are incomplete"}),
            content_type="application/json",
        )
    return web.json_response(
        {
            "token": token,
            "ticket": ticket,
            "apiBaseUrl": HAPPYOYSTER_API_BASE_URL,
            "apiHost": urlsplit(HAPPYOYSTER_API_BASE_URL).netloc,
            "encryptedWorldId": world_id,
        },
        headers={"Cache-Control": "no-store"},
    )


async def _proxy_http(request):
    upstream_url = f"{UPSTREAM_HTTP}{request.rel_url}"
    try:
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
    except ClientError as error:
        logging.warning("Upstream HTTP unavailable: %s", upstream_url, exc_info=error)
        raise web.HTTPBadGateway(text="upstream HTTP unavailable") from error


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
    try:
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
    except ClientError as error:
        logging.warning("Backend HTTP unavailable: %s", upstream_url, exc_info=error)
        raise web.HTTPBadGateway(text="backend HTTP unavailable") from error


async def _relay_websocket(source, destination):
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await destination.send_str(message.data)
        elif message.type == WSMsgType.BINARY:
            await destination.send_bytes(message.data)
        elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
            break


async def _proxy_websocket(request):
    upstream_url = f"{UPSTREAM_WS}{request.rel_url}"
    downstream = None

    try:
        async with request.app[SESSION].ws_connect(
            upstream_url,
            max_msg_size=0,
        ) as upstream:
            downstream = web.WebSocketResponse(max_msg_size=0)
            await downstream.prepare(request)
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
    except ClientError as error:
        logging.warning(
            "Upstream websocket unavailable: %s", upstream_url, exc_info=error
        )
        raise web.HTTPBadGateway(text="upstream websocket unavailable") from error
    except Exception:
        if downstream is None:
            raise
        await downstream.close(code=1011, message=b"upstream websocket unavailable")
    finally:
        if downstream is not None and not downstream.closed:
            await downstream.close()

    return downstream


async def _proxy_backend_websocket(request):
    upstream_url = _named_backend_url(request, "ws")
    downstream = None

    try:
        async with request.app[SESSION].ws_connect(
            upstream_url,
            max_msg_size=0,
        ) as upstream:
            downstream = web.WebSocketResponse(max_msg_size=0)
            await downstream.prepare(request)
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
    except ClientError as error:
        logging.warning(
            "Backend websocket unavailable: %s", upstream_url, exc_info=error
        )
        raise web.HTTPBadGateway(text="backend websocket unavailable") from error
    except Exception:
        if downstream is None:
            raise
        await downstream.close(code=1011, message=b"upstream websocket unavailable")
    finally:
        if downstream is not None and not downstream.closed:
            await downstream.close()

    return downstream


async def _session_context(app):
    # The production cluster occasionally experiences short CoreDNS/VPC DNS
    # stalls. Re-resolving the stable Gateway Service every ten seconds (the
    # aiohttp default) turned those stalls into HTTP/WebSocket 502s. Cache the
    # successful Service resolution across several realtime sessions while
    # keeping a bounded lifetime for normal Kubernetes Service changes.
    app[SESSION] = ClientSession(
        timeout=ClientTimeout(total=None),
        connector=TCPConnector(ttl_dns_cache=300),
    )
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
    app[HAPPYOYSTER_WORLD_CACHE] = _load_happyoyster_prebuilt_worlds()
    app[HAPPYOYSTER_WORLD_BUILD_LOCKS] = {}
    app.cleanup_ctx.append(_session_context)
    app.router.add_get("/", _index)
    app.router.add_get("/runtime-config.js", _runtime_config)
    app.router.add_post("/api/prompt/rewrite", _rewrite_prompt)
    app.router.add_post("/api/world-rule/complete", _complete_world_rule)
    app.router.add_post("/api/world/complete", _complete_world)
    app.router.add_get("/api/world/images/{image_id}", _generated_world_image)
    app.router.add_get("/api/happyoyster/config", _happyoyster_config)
    app.router.add_post("/api/happyoyster/share-image", _happyoyster_share_image)
    app.router.add_post("/api/happyoyster/worlds/resolve", _happyoyster_resolve_world)
    app.router.add_post("/api/happyoyster/worlds", _happyoyster_create_world)
    app.router.add_get(
        "/api/happyoyster/worlds/build-status", _happyoyster_world_status
    )
    app.router.add_post("/api/happyoyster/prepare", _happyoyster_prepare)
    app.router.add_get(
        "/backends/{backend}/v1/realtime_video/generate",
        _proxy_backend_websocket,
    )
    app.router.add_route("*", "/backends/{backend}/v1/{path:.*}", _proxy_backend_http)
    app.router.add_get("/v1/realtime_video/generate", _proxy_websocket)
    app.router.add_route("*", "/v1/{path:.*}", _proxy_http)
    if not _direct_h264_gateway_enabled():
        install_h264_websocket_bridge(
            app,
            upstream_session_key=SESSION,
            upstream_resolver=lambda backend: _backend_upstream(
                backend,
                "ws",
                "/v1/realtime_video/generate",
            ),
        )
    app.router.add_static("/", ROOT)
    return app


if __name__ == "__main__":
    web.run_app(
        create_app(),
        host=os.environ.get("WEBUI_HOST", "0.0.0.0"),
        port=int(os.environ.get("WEBUI_PORT", "18080")),
    )
