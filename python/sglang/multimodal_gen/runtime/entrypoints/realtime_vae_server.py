# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from sglang.multimodal_gen.runtime.realtime.async_vae_client import GatewayOutputClient
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    PAYLOAD_TRANSPORT_SHARED_MEMORY,
    PAYLOAD_TRANSPORT_WEBSOCKET,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    discard_shared_memory_payload,
    encode_message,
    latent_header_from_message,
    materialize_payload_from_shared_memory,
    store_payload_in_shared_memory,
    validate_payload,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
    TAEHVEngine,
)
from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
    WorkerReservationRegistry,
    install_worker_reservation_routes,
    resolve_worker_epoch,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    log_realtime_trace,
    normalize_trace_id,
)

logger = logging.getLogger(__name__)


def _tensor_from_payload(header, payload: bytes) -> torch.Tensor:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[header.dtype]
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(header.shape)


async def _bind_socket_session(
    worker: AsyncVAEWorker,
    current_identity: tuple[str, str] | None,
    opened: SessionOpen,
) -> tuple[str, str]:
    if current_identity is not None:
        raise ProtocolViolation("WebSocket already owns a VAE session")
    await worker.open(opened)
    return opened.session_id, opened.generation_id


def create_app(
    worker: AsyncVAEWorker,
    *,
    max_message_bytes: int,
    reservation_registry: WorkerReservationRegistry | None = None,
    shared_memory_dir: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await worker.close_all()

    app = FastAPI(lifespan=lifespan)
    if reservation_registry is not None:
        reservation_registry.set_load_provider(worker.runtime_state)
        install_worker_reservation_routes(app, reservation_registry)

    @app.get("/health")
    async def health():
        return JSONResponse(
            {
                "status": "ok",
                "decoder_backend": worker.decoder_backend,
                "decoder_fidelity": (
                    "approximate" if worker.decoder_backend == "taehv" else "exact"
                ),
                "active_sessions": worker.active_sessions,
                "max_sessions": worker.max_sessions,
            }
        )

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/v1/realtime_vae/decode")
    async def decode_socket(ws: WebSocket):
        await ws.accept()
        identity = None
        trace_session = None
        output_client = None
        reservation_token = None
        reservation_owner = uuid4().hex
        reservation_consumed = False
        send_lock = asyncio.Lock()
        finish_tasks: set[asyncio.Task] = set()

        async def send(message: bytes):
            async with send_lock:
                await ws.send_bytes(message)

        try:
            while True:
                wire = await ws.receive_bytes()
                message = decode_message(wire, max_message_bytes=max_message_bytes)
                message_type = message["type"]
                if message_type == "session_open":
                    opened = SessionOpen(
                        session_id=str(message["session_id"]),
                        generation_id=str(message["generation_id"]),
                        decoder_backend=str(message.get("decoder_backend") or ""),
                        response_transport=str(
                            message.get("response_transport")
                            or PAYLOAD_TRANSPORT_WEBSOCKET
                        ),
                        trace_id=normalize_trace_id(
                            message.get("trace_id"),
                            fallback=str(message["session_id"]),
                        ),
                        output_format=str(message.get("output_format") or "webp"),
                        quality=int(message.get("quality") or 90),
                        preview_max_width=message.get("preview_max_width"),
                        output_url=message.get("output_url"),
                        output_token=message.get("output_token"),
                    )
                    if reservation_registry is not None:
                        reservation_token = str(message.get("coordinator_token") or "")
                        await reservation_registry.consume(
                            reservation_token,
                            session_id=opened.session_id,
                            generation_id=opened.generation_id,
                            worker_epoch=str(message.get("worker_epoch") or ""),
                            owner_id=reservation_owner,
                        )
                        reservation_consumed = True
                    if bool(opened.output_url) != bool(opened.output_token):
                        raise ProtocolViolation(
                            "output_url and output_token must be provided together"
                        )
                    if (
                        opened.output_url is not None
                        and opened.response_transport == PAYLOAD_TRANSPORT_SHARED_MEMORY
                    ):
                        raise ProtocolViolation(
                            "shared-memory responses cannot use a Gateway output route"
                        )
                    if opened.output_url is not None:
                        output_client = GatewayOutputClient(
                            opened.output_url,
                            session_id=opened.session_id,
                            generation_id=opened.generation_id,
                            token=opened.output_token or "",
                            max_message_bytes=max_message_bytes,
                        )
                        await output_client.open()
                    identity = await _bind_socket_session(worker, identity, opened)
                    trace_session = SimpleNamespace(
                        id=opened.session_id,
                        generation_id=opened.generation_id,
                        trace_id=opened.trace_id,
                        trace_started_at=time.perf_counter(),
                    )
                    log_realtime_trace(
                        logger,
                        trace_session,
                        "server.vae_session_open",
                        output_direct=output_client is not None,
                    )
                    await send(
                        encode_message(
                            "session_accepted",
                            session_id=opened.session_id,
                            generation_id=opened.generation_id,
                            decoder_backend=worker.decoder_backend,
                            decoder_fidelity=(
                                "approximate"
                                if worker.decoder_backend == "taehv"
                                else "exact"
                            ),
                            credit_chunk_index=0,
                        )
                    )
                    continue
                if message_type == "abort":
                    break
                if message_type != "latent_chunk":
                    raise ProtocolViolation(f"unsupported message type: {message_type}")
                if identity is None:
                    raise ProtocolViolation("session_open must precede latent chunks")

                header = latent_header_from_message(message)
                if (header.session_id, header.generation_id) != identity:
                    raise ProtocolViolation("latent identity does not match WebSocket")
                payload_transport = message.get(
                    "payload_transport", PAYLOAD_TRANSPORT_WEBSOCKET
                )
                payload = message.get("payload")
                if payload_transport == PAYLOAD_TRANSPORT_SHARED_MEMORY:
                    reference = message.get("payload_reference")
                    if not isinstance(reference, dict):
                        raise ProtocolViolation(
                            "latent shared-memory reference is required"
                        )
                    payload = materialize_payload_from_shared_memory(
                        reference,
                        root=shared_memory_dir,
                        max_bytes=max_message_bytes,
                    )
                elif payload_transport != PAYLOAD_TRANSPORT_WEBSOCKET:
                    raise ProtocolViolation("unsupported latent payload transport")
                if not isinstance(payload, bytes):
                    raise ProtocolViolation("latent payload is required")
                validate_payload(header, payload)
                latents = _tensor_from_payload(header, payload)
                log_realtime_trace(
                    logger,
                    trace_session,
                    "server.vae_latent_received",
                    request_id=header.request_id,
                    chunk_index=header.chunk_index,
                    latent_bytes=len(payload),
                )

                async def on_frame_batch(frame_batch, *, header=header):
                    payload_lengths = [len(value) for value in frame_batch.payloads]
                    source_width = getattr(
                        frame_batch, "source_width", frame_batch.width
                    )
                    source_height = getattr(
                        frame_batch, "source_height", frame_batch.height
                    )
                    preview_width = getattr(
                        frame_batch, "preview_width", frame_batch.width
                    )
                    preview_height = getattr(
                        frame_batch, "preview_height", frame_batch.height
                    )
                    frame_payload = b"".join(frame_batch.payloads)
                    frame_fields = dict(
                        session_id=header.session_id,
                        generation_id=header.generation_id,
                        request_id=header.request_id,
                        chunk_index=header.chunk_index,
                        event_id=header.event_id,
                        content_type=frame_batch.content_type,
                        width=frame_batch.width,
                        height=frame_batch.height,
                        source_width=source_width,
                        source_height=source_height,
                        preview_width=preview_width,
                        preview_height=preview_height,
                        payload_lengths=payload_lengths,
                        num_frames=frame_batch.num_frames,
                        frame_batch_index=frame_batch.frame_batch_index,
                        is_final_frame_batch=frame_batch.is_final,
                        is_final_chunk=header.is_final_chunk,
                        encode_ms=frame_batch.encode_ms,
                    )
                    shared_reference = None
                    if (
                        output_client is None
                        and opened.response_transport == PAYLOAD_TRANSPORT_SHARED_MEMORY
                    ):
                        shared_reference = store_payload_in_shared_memory(
                            frame_payload,
                            root=shared_memory_dir,
                        )
                        wire = encode_message(
                            "frame_batch",
                            payload_transport=PAYLOAD_TRANSPORT_SHARED_MEMORY,
                            payload_reference=shared_reference,
                            payload_checksum=checksum_payload(frame_payload),
                            **frame_fields,
                        )
                    else:
                        wire = encode_message(
                            "frame_batch",
                            payload_transport=PAYLOAD_TRANSPORT_WEBSOCKET,
                            payload=frame_payload,
                            **frame_fields,
                        )
                    send_started = time.perf_counter()
                    try:
                        if output_client is not None:
                            await output_client.send(wire)
                        else:
                            await send(wire)
                    except Exception:
                        discard_shared_memory_payload(
                            shared_reference,
                            root=shared_memory_dir,
                        )
                        raise
                    log_realtime_trace(
                        logger,
                        trace_session,
                        "server.vae_frame_batch_sent",
                        request_id=header.request_id,
                        chunk_index=header.chunk_index,
                        frame_batch_index=frame_batch.frame_batch_index,
                        num_frames=frame_batch.num_frames,
                        duration_ms=round(
                            (time.perf_counter() - send_started) * 1000, 3
                        ),
                        output_direct=output_client is not None,
                    )

                async def on_decode_started(*, header=header):
                    await send(
                        encode_message(
                            "latent_accepted",
                            session_id=header.session_id,
                            generation_id=header.generation_id,
                            request_id=header.request_id,
                            chunk_index=header.chunk_index,
                            next_credit_chunk_index=header.chunk_index + 1,
                        )
                    )

                future = await worker.submit(
                    header,
                    latents,
                    on_frame_batch=on_frame_batch,
                    on_decode_started=on_decode_started,
                )

                async def finish_chunk(future=future, header=header):
                    try:
                        result = await future
                        common_trace = {
                            "request_id": header.request_id,
                            "chunk_index": header.chunk_index,
                        }
                        log_realtime_trace(
                            logger,
                            trace_session,
                            "server.vae_queue_wait_complete",
                            duration_ms=round(result.queue_wait_ms, 3),
                            **common_trace,
                        )
                        log_realtime_trace(
                            logger,
                            trace_session,
                            "server.vae_decode_complete",
                            duration_ms=round(result.decode_ms, 3),
                            **common_trace,
                        )
                        log_realtime_trace(
                            logger,
                            trace_session,
                            "server.vae_encode_complete",
                            duration_ms=round(result.encode_ms, 3),
                            **common_trace,
                        )
                        if output_client is not None:
                            completion_started = time.perf_counter()
                            await output_client.send(
                                encode_message(
                                    "media_chunk_complete",
                                    session_id=header.session_id,
                                    generation_id=header.generation_id,
                                    request_id=header.request_id,
                                    chunk_index=header.chunk_index,
                                    event_id=header.event_id,
                                    num_frames=result.num_frames,
                                    is_final_chunk=header.is_final_chunk,
                                )
                            )
                            log_realtime_trace(
                                logger,
                                trace_session,
                                "server.vae_media_completion_accepted",
                                duration_ms=round(
                                    (time.perf_counter() - completion_started) * 1000,
                                    3,
                                ),
                                **common_trace,
                            )
                        await send(
                            encode_message(
                                "chunk_complete",
                                session_id=header.session_id,
                                generation_id=header.generation_id,
                                request_id=header.request_id,
                                chunk_index=header.chunk_index,
                                num_frames=result.num_frames,
                                is_final_chunk=header.is_final_chunk,
                                queue_wait_ms=result.queue_wait_ms,
                                decode_ms=result.decode_ms,
                                encode_ms=result.encode_ms,
                            )
                        )
                    except Exception as exc:
                        logger.exception(
                            "VAE chunk completion failed: session_id=%s "
                            "generation_id=%s request_id=%s chunk_index=%s",
                            header.session_id,
                            header.generation_id,
                            header.request_id,
                            header.chunk_index,
                        )
                        await send(
                            encode_message(
                                "error",
                                request_id=header.request_id,
                                chunk_index=header.chunk_index,
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )

                task = asyncio.create_task(finish_chunk())
                finish_tasks.add(task)
                task.add_done_callback(finish_tasks.discard)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await send(
                    encode_message(
                        "error",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            except Exception:
                pass
        finally:
            if (
                reservation_registry is not None
                and reservation_token is not None
                and reservation_consumed
            ):
                await reservation_registry.release(
                    reservation_token,
                    owner_id=reservation_owner,
                )
            for task in finish_tasks:
                task.cancel()
            if finish_tasks:
                await asyncio.gather(*finish_tasks, return_exceptions=True)
            if identity is not None:
                await worker.close(*identity)
            if output_client is not None:
                await output_client.close()

    return app


def _add_worker_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decoder-backend", choices=("exact", "taehv"), required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--vae-path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--queue-depth-per-session", type=int, default=1)
    parser.add_argument("--encoded-frames-per-batch", type=int, default=1)
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument("--shared-memory-dir")
    parser.add_argument("--worker-epoch")


def _parse_worker_args(argv: list[str] | None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    _add_worker_cli_args(pre_parser)
    known, remaining = pre_parser.parse_known_args(argv)
    if known.decoder_backend == "taehv":
        if remaining:
            pre_parser.error("unrecognized arguments: " + " ".join(remaining))
        if not known.checkpoint_path:
            pre_parser.error("--checkpoint-path is required for taehv backend")
        if known.vae_path:
            pre_parser.error("--vae-path is only valid for exact backend")
        return known, None

    if not known.vae_path:
        pre_parser.error("--vae-path is required for exact backend")
    if known.checkpoint_path:
        pre_parser.error("--checkpoint-path is only valid for taehv backend")

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        conflict_handler="resolve",
    )
    ServerArgs.add_cli_args(parser)
    raw_args, unknown = parser.parse_known_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args, unknown)
    return raw_args, server_args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args, exact_server_args = _parse_worker_args(argv)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if args.decoder_backend == "taehv":
        engine = TAEHVEngine(args.checkpoint_path, device=args.device, dtype=dtype)
        max_sessions = args.max_sessions or 8
    else:
        if args.max_sessions not in (None, 1):
            raise ValueError("exact backend requires --max-sessions=1")
        from sglang.multimodal_gen.runtime.realtime.exact_vae_backend import (
            ExactCausalVAEEngine,
        )
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        set_global_server_args(exact_server_args)
        engine = ExactCausalVAEEngine(exact_server_args, args.vae_path)
        max_sessions = 1
    warmup_started = time.perf_counter()
    engine.warmup()
    logger.info(
        "%s startup warmup completed in %.1f ms",
        args.decoder_backend,
        (time.perf_counter() - warmup_started) * 1000,
    )
    worker = AsyncVAEWorker(
        engine,
        max_sessions=max_sessions,
        queue_depth_per_session=args.queue_depth_per_session,
        encoded_frames_per_batch=args.encoded_frames_per_batch,
    )
    reservations = WorkerReservationRegistry(
        worker_epoch=resolve_worker_epoch(args.worker_epoch),
        capacity=max_sessions,
    )
    app = create_app(
        worker,
        max_message_bytes=args.max_message_mb * 1024 * 1024,
        reservation_registry=reservations,
        shared_memory_dir=args.shared_memory_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
