# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib

import pytest
import torch
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server import (
    _parse_worker_args,
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_client import (
    RealtimeVAEClient,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    PAYLOAD_TRANSPORT_WEBSOCKET,
    LatentChunkHeader,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
)
from sglang.multimodal_gen.runtime.realtime.media_profile import (
    MediaProfileAcceptance,
    RealtimeMediaProfile,
    parse_media_profile,
    resolve_remote_media_profile,
)
from sglang.multimodal_gen.runtime.realtime.rife_media_processor import (
    RIFE2xMediaProcessor,
    validate_rife_weights,
)


class _FrameEngine:
    backend = "taehv"
    rgb_quantization = "round"

    def __init__(self, outputs: list[torch.Tensor]) -> None:
        self.outputs = list(outputs)

    def create_decoder(self, identity):
        return identity

    async def decode(self, decoder, latents, *, first_chunk):
        del decoder, latents, first_chunk
        return self.outputs.pop(0).clone()


class _AveragingRIFE:
    weights_sha256 = "a" * 64
    ready = True

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.ranges: list[tuple[float, float]] = []

    def interpolate_midpoints(self, source_frames: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(source_frames.shape))
        self.ranges.append((float(source_frames.min()), float(source_frames.max())))
        return (
            source_frames[:, :3, :-1].float() + source_frames[:, :3, 1:].float()
        ) / 2


class _NegotiationSocket:
    def __init__(self, response: bytes) -> None:
        self.received: asyncio.Queue[bytes] = asyncio.Queue()
        self.received.put_nowait(response)
        self.sent: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    async def send(self, payload: bytes) -> None:
        await self.sent.put(payload)

    async def recv(self) -> bytes:
        return await self.received.get()

    async def close(self) -> None:
        self.closed = True


def _frames(*values: float) -> torch.Tensor:
    channels = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1, 1)
    return channels.expand(1, 3, -1, 2, 2).contiguous()


def _header(
    chunk_index: int,
    *,
    event_id: int | None,
    prompt_version: int,
) -> LatentChunkHeader:
    return LatentChunkHeader(
        session_id="session",
        generation_id="generation",
        request_id=f"request-{chunk_index}",
        chunk_index=chunk_index,
        dtype="bfloat16",
        shape=(1, 4, 1, 1, 1),
        byte_length=8,
        checksum="test",
        event_id=event_id,
        prompt_version=prompt_version,
        has_reference=True,
    )


def _red_values(result) -> list[int]:
    return [payload[0] for batch in result.frame_batches for payload in batch.payloads]


def test_media_profile_parser_is_strict_and_typed():
    assert parse_media_profile(None) is RealtimeMediaProfile.NATIVE_V1
    assert (
        parse_media_profile(RealtimeMediaProfile.RIFE2X_V1)
        is RealtimeMediaProfile.RIFE2X_V1
    )
    with pytest.raises(ProtocolViolation, match="unsupported realtime media profile"):
        parse_media_profile("rife-latest")


def test_remote_legacy_2x_request_maps_to_negotiated_rife():
    assert (
        resolve_remote_media_profile(
            "native_v1",
            legacy_enabled=True,
            legacy_exp=1,
            legacy_scale=1.0,
            legacy_model_path=None,
        )
        is RealtimeMediaProfile.RIFE2X_V1
    )
    assert (
        resolve_remote_media_profile(
            "native_v1",
            legacy_enabled=False,
            legacy_exp=4,
            legacy_scale=0.25,
            legacy_model_path="dormant-client-default",
        )
        is RealtimeMediaProfile.NATIVE_V1
    )


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"legacy_exp": 2}, "exp=1"),
        ({"legacy_exp": 1.5}, "exp=1"),
        ({"legacy_scale": 0.5}, "scale=1.0"),
        ({"legacy_scale": float("nan")}, "scale=1.0"),
        ({"legacy_model_path": "/client/weights"}, "configured only"),
    ],
)
def test_remote_legacy_mapping_rejects_unsupported_or_client_owned_weights(
    overrides,
    error,
):
    kwargs = {
        "legacy_enabled": True,
        "legacy_exp": 1,
        "legacy_scale": 1.0,
        "legacy_model_path": None,
        **overrides,
    }
    with pytest.raises(ProtocolViolation, match=error):
        resolve_remote_media_profile("native_v1", **kwargs)


def test_explicit_rife_rejects_client_weight_path_without_legacy_flag():
    with pytest.raises(ProtocolViolation, match="configured only"):
        resolve_remote_media_profile(
            "rife2x_v1",
            legacy_enabled=False,
            legacy_model_path="/client/weights",
        )


def test_rife_weights_require_absolute_local_path_and_exact_digest(tmp_path):
    weight_file = tmp_path / "flownet.pkl"
    weight_file.write_bytes(b"controlled-rife-weights")
    digest = hashlib.sha256(weight_file.read_bytes()).hexdigest()

    resolved, actual = validate_rife_weights(tmp_path, digest)
    assert resolved == weight_file.resolve()
    assert actual == digest
    with pytest.raises(ValueError, match="absolute local path"):
        validate_rife_weights("elfgum/RIFE-4.22.lite", digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_rife_weights(tmp_path, "0" * 64)


def test_worker_cli_requires_paired_server_owned_rife_configuration():
    with pytest.raises(SystemExit):
        _parse_worker_args(
            [
                "--decoder-backend",
                "taehv",
                "--checkpoint-path",
                "/checkpoint",
                "--rife-model-path",
                "/weights",
            ]
        )
    with pytest.raises(SystemExit):
        _parse_worker_args(
            [
                "--decoder-backend",
                "exact",
                "--vae-path",
                "/vae",
                "--rife-model-path",
                "/weights",
                "--rife-model-sha256",
                "a" * 64,
            ]
        )
    args, exact_args = _parse_worker_args(
        [
            "--decoder-backend",
            "taehv",
            "--checkpoint-path",
            "/checkpoint",
            "--rife-model-path",
            "/weights",
            "--rife-model-sha256",
            "a" * 64,
        ]
    )
    assert args.rife_model_path == "/weights"
    assert exact_args is None


def test_strict_local_rife_processor_batches_midpoints(tmp_path):
    weight_file = tmp_path / "flownet.pkl"

    class _Flownet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _Model:
        def __init__(self):
            self.flownet = _Flownet()

        def eval(self):
            return self

        def inference(self, first, second, scale=1.0):
            assert scale == 1.0
            return (first + second) / 2

    torch.save({"module.weight": torch.ones(1)}, weight_file)
    digest = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    processor = RIFE2xMediaProcessor(
        tmp_path,
        digest,
        device="cpu",
        max_batch_pairs=2,
        model_factory=_Model,
    )
    assert processor.ready is False
    processor.warmup()
    assert processor.ready is True
    result = processor.interpolate_midpoints(_frames(0.0, 0.2, 0.4, 0.8))
    assert result.shape == (1, 3, 3, 2, 2)
    torch.testing.assert_close(
        result[0, 0, :, 0, 0],
        torch.tensor([0.1, 0.3, 0.6]),
    )


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"module.not_a_real_parameter": torch.ones(1)},
        {"module.weight": torch.ones(1), "other": torch.ones(1)},
    ],
)
def test_strict_local_rife_processor_rejects_incomplete_or_mixed_state(
    tmp_path,
    state,
):
    class _Model:
        def __init__(self):
            self.flownet = torch.nn.Linear(1, 1, bias=False)

        def eval(self):
            return self

    weight_file = tmp_path / "flownet.pkl"
    torch.save(state, weight_file)
    digest = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="state dict"):
        RIFE2xMediaProcessor(
            tmp_path,
            digest,
            device="cpu",
            model_factory=_Model,
        )


def test_worker_does_not_advertise_rife_before_processor_warmup():
    class _NotReadyRIFE(_AveragingRIFE):
        ready = False

    worker = AsyncVAEWorker(
        _FrameEngine([]),
        max_sessions=1,
        rife_processor=_NotReadyRIFE(),
    )
    assert worker.supported_media_profiles == (RealtimeMediaProfile.NATIVE_V1,)
    assert worker.media_capability_fingerprint is None
    health = (
        TestClient(create_app(worker, max_message_bytes=1024 * 1024))
        .get("/health")
        .json()
    )
    assert "supported_media_profiles" not in health
    asyncio.run(worker.close_all())


def test_worker_native_profile_is_byte_and_count_compatible():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine([_frames(0.0, 0.4, 0.8)]),
            max_sessions=1,
            encoded_frames_per_batch=16,
            rife_processor=processor,
        )
        acceptance = await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.NATIVE_V1,
                source_timeline_fps=12,
            )
        )
        result = await worker.decode(
            _header(0, event_id=1, prompt_version=0),
            torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16),
        )
        assert acceptance.effective is RealtimeMediaProfile.NATIVE_V1
        assert acceptance.source_timeline_fps == acceptance.output_timeline_fps == 12
        assert result.source_num_frames == result.output_num_frames == 3
        assert result.num_frames == 3
        assert _red_values(result) == [0, 102, 204]
        assert processor.calls == []
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rife2x_preserves_seam_and_resets_on_event_or_prompt_cutover():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine(
                [
                    _frames(0.0, 0.4, 0.8),
                    _frames(0.9, 1.0),
                    _frames(0.2, 0.4),
                    _frames(0.6, 0.8),
                ]
            ),
            max_sessions=1,
            encoded_frames_per_batch=16,
            rife_processor=processor,
        )
        acceptance = await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
                source_timeline_fps=12,
            )
        )
        assert acceptance.output_timeline_fps == 24
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)

        first = await worker.decode(_header(0, event_id=7, prompt_version=0), latent)
        second = await worker.decode(_header(1, event_id=7, prompt_version=0), latent)
        event_cutover = await worker.decode(
            _header(2, event_id=8, prompt_version=0), latent
        )
        prompt_cutover = await worker.decode(
            _header(3, event_id=8, prompt_version=1), latent
        )

        assert (first.source_num_frames, first.output_num_frames) == (3, 5)
        assert first.rife_interpolation_ms >= 0
        assert _red_values(first) == [0, 51, 102, 153, 204]
        # Same cutover: emit midpoint(previous chunk tail, current head) first,
        # then each real source frame.  No endpoint is duplicated.
        assert (second.source_num_frames, second.output_num_frames) == (2, 4)
        assert _red_values(second) == [217, 230, 242, 255]
        # An event or prompt boundary must never synthesize a blended seam.
        assert (event_cutover.source_num_frames, event_cutover.output_num_frames) == (
            2,
            3,
        )
        assert _red_values(event_cutover)[0] == 51
        assert (prompt_cutover.source_num_frames, prompt_cutover.output_num_frames) == (
            2,
            3,
        )
        assert _red_values(prompt_cutover)[0] == 153
        assert [shape[2] for shape in processor.calls] == [3, 3, 2, 2]
        metrics = generate_latest()
        assert b'stage="rife_interpolation"' in metrics
        assert b'stage="frame_interpolation"' in metrics
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_fails_closed_when_rife_capability_is_missing():
    async def scenario():
        worker = AsyncVAEWorker(_FrameEngine([]), max_sessions=1)
        with pytest.raises(ProtocolViolation, match="profile is unavailable"):
            await worker.open(
                SessionOpen(
                    "session",
                    "generation",
                    media_profile=RealtimeMediaProfile.RIFE2X_V1,
                )
            )
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_clamps_rife_inputs_to_normalized_rgb_range():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine([_frames(-0.25, 1.25)]),
            max_sessions=1,
            rife_processor=processor,
        )
        await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
            )
        )
        result = await worker.decode(
            _header(0, event_id=1, prompt_version=0),
            torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16),
        )
        assert processor.ranges == [(0.0, 1.0)]
        assert _red_values(result) == [0, 128, 255]
        await worker.close_all()

    asyncio.run(scenario())


def test_client_uses_only_server_accepted_rife_profile_and_timing():
    async def scenario():
        digest = "b" * 64
        socket = _NegotiationSocket(
            encode_message(
                "session_accepted",
                session_id="session",
                generation_id="generation",
                decoder_backend="taehv",
                decoder_fidelity="approximate",
                requested_media_profile="rife2x_v1",
                effective_media_profile="rife2x_v1",
                source_timeline_fps=12,
                output_timeline_fps=24,
                media_weights_sha256=digest,
            )
        )

        async def connect_factory(*args, **kwargs):
            del args, kwargs
            return socket

        client = RealtimeVAEClient(
            "ws://vae",
            session_id="session",
            generation_id="generation",
            transport="websocket",
            connect_factory=connect_factory,
        )
        acceptance = await client.open(
            decoder_backend="taehv",
            output_format="webp",
            quality=80,
            preview_max_width=832,
            media_profile=RealtimeMediaProfile.RIFE2X_V1,
            source_timeline_fps=12,
        )
        session_open = decode_message(await socket.sent.get())
        assert session_open["media_profile"] == "rife2x_v1"
        assert session_open["source_timeline_fps"] == 12
        assert acceptance.effective is RealtimeMediaProfile.RIFE2X_V1
        assert acceptance.output_timeline_fps == 24
        assert acceptance.weights_sha256 == digest

        async def ignore_frame(_batch):
            return None

        submit_task = asyncio.create_task(
            client.submit(
                torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16),
                {
                    "session_id": "session",
                    "generation_id": "generation",
                    "request_id": "request",
                    "chunk_index": 0,
                },
                on_frame_batch=ignore_frame,
            )
        )
        latent_message = decode_message(await socket.sent.get())
        assert latent_message["type"] == "latent_chunk"
        await socket.received.put(
            encode_message(
                "latent_accepted",
                session_id="session",
                generation_id="generation",
                request_id="request",
                chunk_index=0,
            )
        )
        handle = await submit_task
        await socket.received.put(
            encode_message(
                "chunk_complete",
                session_id="session",
                generation_id="generation",
                request_id="request",
                chunk_index=0,
                num_frames=3,
                source_num_frames=2,
                output_num_frames=3,
                media_profile="rife2x_v1",
                source_timeline_fps=12,
                output_timeline_fps=24,
                rife_interpolation_ms=4.5,
                post_decode_ms=3.25,
            )
        )
        result = await handle.wait()
        assert (result.source_num_frames, result.output_num_frames) == (2, 3)
        assert result.post_decode_ms == 3.25
        assert result.rife_interpolation_ms == 4.5
        await client.close()
        assert socket.closed

    asyncio.run(scenario())


def test_client_native_handshake_adds_no_wire_fields_or_messages():
    async def scenario():
        socket = _NegotiationSocket(
            encode_message(
                "session_accepted",
                session_id="session",
                generation_id="generation",
                decoder_backend="taehv",
                decoder_fidelity="approximate",
                credit_chunk_index=0,
            )
        )

        async def connect_factory(*args, **kwargs):
            del args, kwargs
            return socket

        client = RealtimeVAEClient(
            "ws://vae",
            session_id="session",
            generation_id="generation",
            transport="websocket",
            connect_factory=connect_factory,
        )
        acceptance = await client.open(
            decoder_backend="taehv",
            output_format="webp",
            quality=80,
            preview_max_width=832,
            media_profile=RealtimeMediaProfile.NATIVE_V1,
            source_timeline_fps=24,
        )
        session_open = decode_message(await socket.sent.get())
        assert "media_profile" not in session_open
        assert "source_timeline_fps" not in session_open
        assert acceptance.effective is RealtimeMediaProfile.NATIVE_V1
        assert acceptance.source_timeline_fps == acceptance.output_timeline_fps == 24
        assert socket.sent.empty()
        await client.close()

    asyncio.run(scenario())


def test_client_fails_closed_when_rife_chunk_omits_or_forges_media_fields():
    client = RealtimeVAEClient(
        "ws://vae",
        session_id="session",
        generation_id="generation",
        transport="websocket",
    )
    client.media_profile_acceptance = MediaProfileAcceptance(
        requested=RealtimeMediaProfile.RIFE2X_V1,
        effective=RealtimeMediaProfile.RIFE2X_V1,
        source_timeline_fps=12,
        output_timeline_fps=24,
        weights_sha256="b" * 64,
    )
    with pytest.raises(ProtocolViolation, match="omitted RIFE fields"):
        client._validated_media_fields({}, response_kind="frame")
    with pytest.raises(ProtocolViolation, match="timing is invalid"):
        client._validated_media_fields(
            {
                "media_profile": "rife2x_v1",
                "source_timeline_fps": float("nan"),
                "output_timeline_fps": 24,
            },
            response_kind="frame",
        )


@pytest.mark.parametrize(
    "acceptance_fields, error",
    [
        ({}, "did not negotiate"),
        (
            {
                "requested_media_profile": "rife2x_v1",
                "effective_media_profile": "native_v1",
                "source_timeline_fps": 12,
                "output_timeline_fps": 12,
            },
            "silently changed",
        ),
        (
            {
                "requested_media_profile": "rife2x_v1",
                "effective_media_profile": "rife2x_v1",
                "source_timeline_fps": 12,
                "output_timeline_fps": 24,
            },
            "weights digest is missing",
        ),
    ],
)
def test_client_fails_closed_without_exact_rife_acceptance(
    acceptance_fields,
    error,
):
    async def scenario():
        socket = _NegotiationSocket(
            encode_message(
                "session_accepted",
                session_id="session",
                generation_id="generation",
                decoder_backend="taehv",
                decoder_fidelity="approximate",
                **acceptance_fields,
            )
        )

        async def connect_factory(*args, **kwargs):
            del args, kwargs
            return socket

        client = RealtimeVAEClient(
            "ws://vae",
            session_id="session",
            generation_id="generation",
            transport="websocket",
            connect_factory=connect_factory,
        )
        with pytest.raises(ProtocolViolation, match=error):
            await client.open(
                decoder_backend="taehv",
                output_format="webp",
                quality=80,
                preview_max_width=832,
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
                source_timeline_fps=12,
            )
        await client.close()

    asyncio.run(scenario())


def test_vae_server_reports_authoritative_profile_capability_and_acceptance():
    processor = _AveragingRIFE()
    worker = AsyncVAEWorker(
        _FrameEngine([_frames(0.0, 1.0)]),
        max_sessions=1,
        rife_processor=processor,
    )
    app = create_app(worker, max_message_bytes=1024 * 1024)

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["supported_media_profiles"] == ["native_v1", "rife2x_v1"]
        assert health["rife_weights_sha256"] == processor.weights_sha256
        assert health["media_capability_fingerprint"] == (
            f"rife2x_v1:{processor.weights_sha256}"
        )
        assert health["media_pool_rollout"] == "homogeneous_only"
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="session",
                    generation_id="generation",
                    decoder_backend="taehv",
                    output_format="webp",
                    quality=80,
                    media_profile="rife2x_v1",
                    source_timeline_fps=12,
                )
            )
            accepted = decode_message(websocket.receive_bytes())
            assert accepted["type"] == "session_accepted"
            assert accepted["requested_media_profile"] == "rife2x_v1"
            assert accepted["effective_media_profile"] == "rife2x_v1"
            assert accepted["source_timeline_fps"] == 12
            assert accepted["output_timeline_fps"] == 24
            assert accepted["media_weights_sha256"] == processor.weights_sha256
            latents = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
            payload = latents.view(torch.uint8).numpy().tobytes()
            websocket.send_bytes(
                encode_message(
                    "latent_chunk",
                    header=LatentChunkHeader(
                        session_id="session",
                        generation_id="generation",
                        request_id="request",
                        chunk_index=0,
                        dtype="bfloat16",
                        shape=tuple(latents.shape),
                        byte_length=len(payload),
                        checksum=checksum_payload(payload),
                        event_id=1,
                        prompt_version=0,
                        has_reference=True,
                    ),
                    payload_transport=PAYLOAD_TRANSPORT_WEBSOCKET,
                    payload=payload,
                )
            )
            messages = []
            while not any(message["type"] == "chunk_complete" for message in messages):
                messages.append(decode_message(websocket.receive_bytes()))
            frame_messages = [
                message for message in messages if message["type"] == "frame_batch"
            ]
            assert len(frame_messages) == 3
            assert all(
                message["media_profile"] == "rife2x_v1"
                and message["source_timeline_fps"] == 12
                and message["output_timeline_fps"] == 24
                for message in frame_messages
            )
            complete = next(
                message for message in messages if message["type"] == "chunk_complete"
            )
            assert (complete["source_num_frames"], complete["output_num_frames"]) == (
                2,
                3,
            )
            assert complete["rife_interpolation_ms"] >= 0
            websocket.send_bytes(
                encode_message(
                    "abort",
                    session_id="session",
                    generation_id="generation",
                )
            )


def test_rife_capable_server_keeps_native_session_wire_exactly_legacy():
    worker = AsyncVAEWorker(
        _FrameEngine([_frames(0.0, 0.5)]),
        max_sessions=1,
        rife_processor=_AveragingRIFE(),
    )
    with TestClient(create_app(worker, max_message_bytes=1024 * 1024)) as client:
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="native-session",
                    generation_id="native-generation",
                    decoder_backend="taehv",
                    output_format="webp",
                    quality=80,
                )
            )
            assert decode_message(websocket.receive_bytes()) == {
                "version": 2,
                "type": "session_accepted",
                "session_id": "native-session",
                "generation_id": "native-generation",
                "decoder_backend": "taehv",
                "decoder_fidelity": "approximate",
                "credit_chunk_index": 0,
            }
            latents = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
            payload = latents.view(torch.uint8).numpy().tobytes()
            websocket.send_bytes(
                encode_message(
                    "latent_chunk",
                    header=LatentChunkHeader(
                        session_id="native-session",
                        generation_id="native-generation",
                        request_id="native-request",
                        chunk_index=0,
                        dtype="bfloat16",
                        shape=tuple(latents.shape),
                        byte_length=len(payload),
                        checksum=checksum_payload(payload),
                        event_id=1,
                        prompt_version=0,
                        has_reference=True,
                    ),
                    payload_transport=PAYLOAD_TRANSPORT_WEBSOCKET,
                    payload=payload,
                )
            )
            native_messages = []
            while not any(
                message["type"] == "chunk_complete" for message in native_messages
            ):
                native_messages.append(decode_message(websocket.receive_bytes()))
            assert sorted(message["type"] for message in native_messages) == [
                "chunk_complete",
                "frame_batch",
                "frame_batch",
                "latent_accepted",
            ]
            for message in native_messages:
                assert "media_profile" not in message
                assert "source_num_frames" not in message
                assert "output_num_frames" not in message
                assert "source_timeline_fps" not in message
                assert "output_timeline_fps" not in message
                assert "rife_interpolation_ms" not in message
            websocket.send_bytes(
                encode_message(
                    "abort",
                    session_id="native-session",
                    generation_id="native-generation",
                )
            )


def test_vae_server_rejects_rife_when_worker_has_no_capability():
    app = create_app(
        AsyncVAEWorker(_FrameEngine([_frames(0.0, 0.5)]), max_sessions=1),
        max_message_bytes=1024 * 1024,
    )
    with TestClient(app) as client:
        assert "supported_media_profiles" not in client.get("/health").json()
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="native-session",
                    generation_id="native-generation",
                    decoder_backend="taehv",
                    output_format="webp",
                    quality=80,
                )
            )
            native_accepted = decode_message(websocket.receive_bytes())
            assert native_accepted == {
                "version": 2,
                "type": "session_accepted",
                "session_id": "native-session",
                "generation_id": "native-generation",
                "decoder_backend": "taehv",
                "decoder_fidelity": "approximate",
                "credit_chunk_index": 0,
            }
            latents = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
            payload = latents.view(torch.uint8).numpy().tobytes()
            header = LatentChunkHeader(
                session_id="native-session",
                generation_id="native-generation",
                request_id="native-request",
                chunk_index=0,
                dtype="bfloat16",
                shape=tuple(latents.shape),
                byte_length=len(payload),
                checksum=checksum_payload(payload),
                event_id=1,
                prompt_version=0,
                has_reference=True,
            )
            websocket.send_bytes(
                encode_message(
                    "latent_chunk",
                    header=header,
                    payload_transport=PAYLOAD_TRANSPORT_WEBSOCKET,
                    payload=payload,
                )
            )
            native_messages = []
            while not any(
                message["type"] == "chunk_complete" for message in native_messages
            ):
                native_messages.append(decode_message(websocket.receive_bytes()))
            assert sorted(message["type"] for message in native_messages) == [
                "chunk_complete",
                "frame_batch",
                "frame_batch",
                "latent_accepted",
            ]
            for message in native_messages:
                assert "media_profile" not in message
                assert "source_timeline_fps" not in message
                assert "output_timeline_fps" not in message
                assert "rife_interpolation_ms" not in message
            websocket.send_bytes(
                encode_message(
                    "abort",
                    session_id="native-session",
                    generation_id="native-generation",
                )
            )
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="session",
                    generation_id="generation",
                    decoder_backend="taehv",
                    output_format="webp",
                    quality=80,
                    media_profile="rife2x_v1",
                    source_timeline_fps=12,
                )
            )
            error = decode_message(websocket.receive_bytes())
            assert error["type"] == "error"
            assert error["error_type"] == "ProtocolViolation"
            assert "profile is unavailable" in error["message"]
