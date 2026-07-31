"""One-slot synchronous HTTP adapter for one SGLang LingBot instance."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException, Response

from .contracts import ValidationError, parse_video_request
from .realtime_client import generate_mp4


API_KEY = os.environ.get("API_KEY", "")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "lingbot-batch-api")
SGLANG_WS_URL = os.environ.get(
    "SGLANG_WS_URL", "ws://127.0.0.1:30000/v1/realtime_video/generate"
)
RETRY_AFTER_SECONDS = int(os.environ.get("RETRY_AFTER_SECONDS", "5"))

app = FastAPI(title="LingBot Batch API", version="1.0.0")
generation_lock = asyncio.Lock()
request_count = 0
success_count = 0
failure_count = 0
_s3 = boto3.client("s3")


def _authorize(api_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if api_key is None or not hmac.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


def _output_key(request_id: str) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"{OUTPUT_PREFIX.strip('/')}/{digest[:2]}/{digest}.mp4"


def _request_fingerprint(request: Any) -> str:
    normalized = {
        "request_id": request.request_id,
        "source_id": request.source_id,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "first_frame": request.first_frame,
        "movement_key": request.action_pair.movement_key,
        "camera_key": request.action_pair.camera_key,
        "video_seed": request.video_seed,
    }
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_frame_url(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "https":
        return uri
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")},
        ExpiresIn=3600,
    )


def _existing_output(key: str, fingerprint: str) -> dict[str, Any] | None:
    try:
        response = _s3.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    if response.get("Metadata", {}).get("request-fingerprint") != fingerprint:
        raise ValueError("request_id already exists with a different payload")
    return {
        "output_s3_uri": f"s3://{OUTPUT_BUCKET}/{key}",
        "bytes": response["ContentLength"],
        "idempotent_replay": True,
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    # Readiness must not flap for every 20-30 second request: with ALB IP
    # targets, frequent endpoint deregistration is slower than returning 429.
    # The caller observes available_slots and retries another target via ALB.
    return {"ready": True, "available_slots": int(not generation_lock.locked())}


@app.get("/v1/capacity")
async def capacity(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _authorize(x_api_key)
    return {
        "worker_concurrency": 1,
        "inflight": int(generation_lock.locked()),
        "available_slots": int(not generation_lock.locked()),
        "server_queue_depth": 0,
    }


@app.get("/metrics", response_class=Response)
async def metrics() -> str:
    return (
        "# TYPE lingbot_inflight_requests gauge\n"
        f"lingbot_inflight_requests {int(generation_lock.locked())}\n"
        "# TYPE lingbot_requests_total counter\n"
        f"lingbot_requests_total {request_count}\n"
        "# TYPE lingbot_success_total counter\n"
        f"lingbot_success_total {success_count}\n"
        "# TYPE lingbot_failure_total counter\n"
        f"lingbot_failure_total {failure_count}\n"
    )


@app.post("/v1/videos/generate")
async def generate(
    payload: dict[str, Any],
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    global request_count, success_count, failure_count
    _authorize(x_api_key)
    try:
        request = parse_video_request(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not OUTPUT_BUCKET:
        raise HTTPException(status_code=503, detail="OUTPUT_BUCKET is not configured")

    key = _output_key(request.request_id)
    fingerprint = _request_fingerprint(request)
    try:
        existing = await asyncio.to_thread(_existing_output, key, fingerprint)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if existing is not None:
        return {"request_id": request.request_id, "status": "succeeded", **existing}
    if generation_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="all capacity on this worker is busy",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )

    request_count += 1
    started = time.perf_counter()
    async with generation_lock:
        try:
            with tempfile.TemporaryDirectory(prefix="lingbot-api-") as directory:
                output = Path(directory) / "output.mp4"
                media = await generate_mp4(
                    ws_url=SGLANG_WS_URL,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    first_frame=await asyncio.to_thread(
                        _first_frame_url, request.first_frame
                    ),
                    pair=request.action_pair,
                    seed=request.video_seed,
                    output=output,
                )
                await asyncio.to_thread(
                    _s3.upload_file,
                    str(output),
                    OUTPUT_BUCKET,
                    key,
                    ExtraArgs={
                        "ContentType": "video/mp4",
                        "Metadata": {
                            "request-id-sha256": hashlib.sha256(
                                request.request_id.encode("utf-8")
                            ).hexdigest(),
                            "request-fingerprint": fingerprint,
                            "source-id-sha256": hashlib.sha256(
                                request.source_id.encode("utf-8")
                            ).hexdigest(),
                            "movement-key": request.action_pair.movement_key,
                            "camera-key": request.action_pair.camera_key,
                        },
                    },
                )
            success_count += 1
            response.headers["X-LingBot-Processing-Seconds"] = (
                f"{time.perf_counter() - started:.3f}"
            )
            response.headers["X-LingBot-Worker-Concurrency"] = "1"
            return {
                "request_id": request.request_id,
                "source_id": request.source_id,
                "status": "succeeded",
                "movement_key": request.action_pair.movement_key,
                "camera_key": request.action_pair.camera_key,
                "output_s3_uri": f"s3://{OUTPUT_BUCKET}/{key}",
                "media": media,
                "idempotent_replay": False,
            }
        except HTTPException:
            raise
        except Exception as exc:
            failure_count += 1
            raise HTTPException(
                status_code=502, detail=f"generation failed: {exc}"
            ) from exc
