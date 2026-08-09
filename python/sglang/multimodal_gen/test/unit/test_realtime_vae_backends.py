# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import realtime_vae_server
from sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server import create_app
from sglang.multimodal_gen.runtime.realtime.async_vae_client import RealtimeVAEClient
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    PAYLOAD_TRANSPORT_SHARED_MEMORY,
    ChunkSequenceTracker,
    LatentChunkHeader,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    encode_message,
    materialize_payload_from_shared_memory,
    store_payload_in_shared_memory,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
)
from sglang.multimodal_gen.runtime.realtime_vae_config import (
    uses_remote_vae,
    worker_decoder_backend,
)


class _Decoder:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _ExactLikeEngine:
    backend = "exact"
    max_sessions = 1
    rgb_quantization = "truncate"

    def __init__(self) -> None:
        self.decoders: list[_Decoder] = []

    def create_decoder(self, _identity) -> _Decoder:
        decoder = _Decoder()
        self.decoders.append(decoder)
        return decoder

    def decode(self, decoder, latents, *, first_chunk):
        if first_chunk:
            decoder.reset()
        return latents.clamp(0, 1)


def test_realtime_vae_deployment_backends_are_explicit_and_mutually_exclusive():
    assert not uses_remote_vae("local")
    assert worker_decoder_backend("exact_remote") == "exact"
    assert worker_decoder_backend("taehv_remote") == "taehv"
    with pytest.raises(ValueError, match="must be one of"):
        worker_decoder_backend("remote")


def test_loopback_transport_defaults_to_shared_memory_and_rejects_remote_force():
    client = RealtimeVAEClient(
        "ws://127.0.0.1:18081/v1/realtime_vae/decode",
        session_id="session",
        generation_id="generation",
    )
    assert client.payload_transport == PAYLOAD_TRANSPORT_SHARED_MEMORY

    with pytest.raises(ValueError, match="requires a loopback"):
        RealtimeVAEClient(
            "ws://vae.internal:18081/v1/realtime_vae/decode",
            session_id="session",
            generation_id="generation",
            transport="shared_memory",
        )


def test_chunk_sequence_rejects_work_after_explicit_final_chunk():
    tracker = ChunkSequenceTracker("session", "generation")
    final = LatentChunkHeader(
        session_id="session",
        generation_id="generation",
        request_id="request-0",
        chunk_index=0,
        dtype="float32",
        shape=(1, 3, 1, 2, 2),
        byte_length=48,
        checksum="checksum",
        is_final_chunk=True,
    )
    tracker.accept(final)

    with pytest.raises(ProtocolViolation, match="after final chunk"):
        tracker.accept(
            replace(
                final,
                request_id="request-1",
                chunk_index=1,
                is_final_chunk=False,
            )
        )


def test_exact_backend_rejects_taehv_checkpoint_before_loading_weights():
    from sglang.multimodal_gen.runtime.realtime.exact_vae_backend import (
        ExactCausalVAEEngine,
    )

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(taehv_checkpoint_path="/tmp/taehv.pth")
        )
    )

    with pytest.raises(ValueError, match="cannot use taehv_checkpoint_path"):
        ExactCausalVAEEngine(server_args, "/tmp/native-vae")


def test_unified_worker_cli_selects_one_exact_backend(monkeypatch):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parsed_server_args = SimpleNamespace(pipeline_class_name="MinWMCausalDMDPipeline")
    monkeypatch.setattr(
        ServerArgs,
        "from_cli_args",
        classmethod(lambda _cls, _args, _unknown: parsed_server_args),
    )

    worker_args, server_args = realtime_vae_server._parse_worker_args(
        [
            "--decoder-backend",
            "exact",
            "--vae-path",
            "/models/minwm/vae",
            "--model-path",
            "/models/minwm",
            "--pipeline-class-name",
            "MinWMCausalDMDPipeline",
            "--max-sessions",
            "1",
        ]
    )

    assert worker_args.decoder_backend == "exact"
    assert worker_args.vae_path == "/models/minwm/vae"
    assert worker_args.max_sessions == 1
    assert server_args is parsed_server_args


def test_exact_worker_rejects_multi_session_capacity():
    with pytest.raises(ValueError, match="at most 1 active session"):
        AsyncVAEWorker(_ExactLikeEngine(), max_sessions=2)


def test_worker_rejects_decoder_backend_mismatch():
    async def run_test():
        worker = AsyncVAEWorker(_ExactLikeEngine(), max_sessions=1)
        with pytest.raises(ProtocolViolation, match="does not match"):
            await worker.open(
                SessionOpen(
                    session_id="session",
                    generation_id="generation",
                    decoder_backend="taehv",
                )
            )

    asyncio.run(run_test())


def test_exact_worker_preserves_native_rgb24_truncation():
    worker = AsyncVAEWorker(_ExactLikeEngine(), max_sessions=1)
    opened = SimpleNamespace(
        output_format="raw",
        quality=90,
        preview_max_width=None,
    )
    frames = torch.full((1, 3, 1, 1, 1), 0.5, dtype=torch.float32)

    encoded = worker._encode_frames(frames, opened)

    assert encoded[0].payloads == (bytes([127, 127, 127]),)


def test_exact_worker_releases_model_global_cache_after_decode_failure():
    async def run_test():
        engine = _ExactLikeEngine()

        def fail_decode(_decoder, _latents, *, first_chunk):
            del first_chunk
            raise RuntimeError("decode failed")

        engine.decode = fail_decode
        worker = AsyncVAEWorker(engine, max_sessions=1)
        await worker.open(
            SessionOpen(
                session_id="session",
                generation_id="generation",
                decoder_backend="exact",
            )
        )
        latents = torch.ones((1, 3, 1, 2, 2), dtype=torch.float32)
        payload = latents.view(torch.uint8).numpy().tobytes()
        header = LatentChunkHeader(
            session_id="session",
            generation_id="generation",
            request_id="request-0",
            chunk_index=0,
            dtype="float32",
            shape=tuple(latents.shape),
            byte_length=len(payload),
            checksum=checksum_payload(payload),
        )

        with pytest.raises(RuntimeError, match="decode failed"):
            await worker.decode(header, latents)
        await asyncio.sleep(0)

        assert worker.active_sessions == 0
        assert engine.decoders[0].reset_calls >= 1

    asyncio.run(run_test())


def test_shared_memory_payload_rejects_escape_and_cleans_success(tmp_path):
    root = tmp_path / "shm"
    reference = store_payload_in_shared_memory(b"latent", root=root)

    assert materialize_payload_from_shared_memory(reference, root=root) == b"latent"
    assert not (root / reference["path"].split("/")[-1]).exists()

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    with pytest.raises(ProtocolViolation, match="escapes configured root"):
        materialize_payload_from_shared_memory(
            {"path": str(outside), "num_bytes": 6},
            root=root,
        )
    assert outside.exists()

    oversized = store_payload_in_shared_memory(b"too large", root=root)
    with pytest.raises(ProtocolViolation, match="exceeds protocol limit"):
        materialize_payload_from_shared_memory(
            oversized,
            root=root,
            max_bytes=2,
        )
    assert not (root / oversized["path"].split("/")[-1]).exists()


def test_unified_exact_worker_shared_memory_final_chunk_round_trip(tmp_path):
    engine = _ExactLikeEngine()
    worker = AsyncVAEWorker(engine, max_sessions=1, encoded_frames_per_batch=1)
    app = create_app(
        worker,
        max_message_bytes=1024 * 1024,
        shared_memory_dir=tmp_path,
    )
    latents = torch.ones((1, 3, 1, 2, 2), dtype=torch.float32)
    payload = latents.view(torch.uint8).numpy().tobytes()
    header = LatentChunkHeader(
        session_id="session",
        generation_id="generation",
        request_id="request-0",
        chunk_index=0,
        dtype="float32",
        shape=tuple(latents.shape),
        byte_length=len(payload),
        checksum=checksum_payload(payload),
        event_id=17,
        is_final_chunk=True,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_vae/decode") as socket:
            socket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="session",
                    generation_id="generation",
                    decoder_backend="exact",
                    response_transport=PAYLOAD_TRANSPORT_SHARED_MEMORY,
                    output_format="raw",
                )
            )
            accepted = decode_message(socket.receive_bytes())
            assert accepted["decoder_backend"] == "exact"
            assert accepted["decoder_fidelity"] == "exact"

            reference = store_payload_in_shared_memory(payload, root=tmp_path)
            socket.send_bytes(
                encode_message(
                    "latent_chunk",
                    header=header,
                    payload_transport=PAYLOAD_TRANSPORT_SHARED_MEMORY,
                    payload_reference=reference,
                )
            )

            messages = {}
            for _ in range(3):
                message = decode_message(socket.receive_bytes())
                messages[message["type"]] = message

            assert messages["latent_accepted"]["chunk_index"] == 0
            frame = messages["frame_batch"]
            assert frame["event_id"] == 17
            assert frame["is_final_chunk"] is True
            frame_payload = materialize_payload_from_shared_memory(
                frame["payload_reference"],
                root=tmp_path,
            )
            assert checksum_payload(frame_payload) == frame["payload_checksum"]
            assert frame_payload == bytes([255]) * 12
            assert messages["chunk_complete"]["is_final_chunk"] is True
            assert worker.active_sessions == 0

    assert engine.decoders[0].reset_calls >= 2
