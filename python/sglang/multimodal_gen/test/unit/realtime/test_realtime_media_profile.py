# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import msgspec.msgpack
import pytest
import torch
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
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
    profiles = ("rife2x_v1", "rife3x_v1")

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.ranges: list[tuple[float, float]] = []
        self.intermediate_multipliers: list[int] = []

    def interpolate_midpoints(self, source_frames: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(source_frames.shape))
        self.ranges.append((float(source_frames.min()), float(source_frames.max())))
        return (
            source_frames[:, :3, :-1].float() + source_frames[:, :3, 1:].float()
        ) / 2

    def interpolate_intermediates(
        self,
        source_frames: torch.Tensor,
        *,
        multiplier: int,
    ) -> torch.Tensor:
        self.calls.append(tuple(source_frames.shape))
        self.ranges.append((float(source_frames.min()), float(source_frames.max())))
        self.intermediate_multipliers.append(multiplier)
        left = source_frames[:, :3, :-1].float()
        right = source_frames[:, :3, 1:].float()
        return torch.stack(
            [
                left + (right - left) * (offset / multiplier)
                for offset in range(1, multiplier)
            ],
            dim=3,
        )


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


class _InitSocket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.sent: list[bytes] = []

    async def receive_bytes(self) -> bytes:
        if self.payload is not None:
            payload = self.payload
            self.payload = None
            return payload
        raise WebSocketDisconnect()

    async def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)


def _frames(*values: float) -> torch.Tensor:
    channels = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1, 1)
    return channels.expand(1, 3, -1, 2, 2).contiguous()


def _header(
    chunk_index: int,
    *,
    event_id: int | None,
    action_version: int = 0,
    prompt_version: int,
    session_id: str = "session",
    generation_id: str = "generation",
) -> LatentChunkHeader:
    return LatentChunkHeader(
        session_id=session_id,
        generation_id=generation_id,
        request_id=f"{session_id}-request-{chunk_index}",
        chunk_index=chunk_index,
        dtype="bfloat16",
        shape=(1, 4, 1, 1, 1),
        byte_length=8,
        checksum="test",
        event_id=event_id,
        action_version=action_version,
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
    assert parse_media_profile("rife3x_v1") is RealtimeMediaProfile.RIFE3X_V1
    assert RealtimeMediaProfile.RIFE3X_V1.output_timeline_fps_multiplier == 3
    with pytest.raises(ProtocolViolation, match="unsupported realtime media profile"):
        parse_media_profile("rife-latest")


def test_remote_legacy_interpolation_fails_closed_without_changing_native_defaults():
    with pytest.raises(ProtocolViolation, match="upgrade the client"):
        resolve_remote_media_profile(
            "native_v1",
            legacy_enabled=True,
            legacy_exp=1,
            legacy_scale=1.0,
            legacy_model_path=None,
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
def test_explicit_rife_rejects_conflicting_legacy_options(
    overrides,
    error,
):
    kwargs = {
        "legacy_enabled": False,
        "legacy_exp": 1,
        "legacy_scale": 1.0,
        "legacy_model_path": None,
        **overrides,
    }
    with pytest.raises(ProtocolViolation, match=error):
        resolve_remote_media_profile("rife2x_v1", **kwargs)


def test_explicit_rife_accepts_null_dormant_legacy_numbers():
    assert (
        resolve_remote_media_profile(
            "rife2x_v1",
            legacy_enabled=False,
            legacy_exp=None,
            legacy_scale=None,
        )
        is RealtimeMediaProfile.RIFE2X_V1
    )


def test_explicit_rife_rejects_client_weight_path_without_legacy_flag():
    with pytest.raises(ProtocolViolation, match="configured only"):
        resolve_remote_media_profile(
            "rife2x_v1",
            legacy_enabled=False,
            legacy_model_path="/client/weights",
        )


@pytest.mark.parametrize(
    "transport_fields",
    [
        {"realtime_output_format": "webp"},
        {
            "realtime_output_format": "raw",
            "h264_bitrate_kbps": 3000,
            "h264_gop_seconds": 2,
        },
    ],
    ids=("webp", "h264-bridge"),
)
def test_remote_legacy_init_is_rejected_before_session_or_media_messages(
    monkeypatch,
    transport_fields,
):
    async def scenario():
        monkeypatch.setattr(
            realtime_video_api,
            "get_global_server_args",
            lambda: SimpleNamespace(realtime_vae_backend="taehv_remote"),
        )
        socket = _InitSocket(
            encode_message(
                "init",
                prompt="legacy interpolation request",
                enable_frame_interpolation=True,
                **transport_fields,
            )
        )
        session = GenerateSession()
        with pytest.raises(WebSocketDisconnect):
            await realtime_video_api._listen_generate_request(socket, session)

        assert session.request is None
        assert session.vae_client is None
        messages = [msgspec.msgpack.decode(payload) for payload in socket.sent]
        assert len(messages) == 1
        assert messages[0]["type"] == "error"
        assert "explicit realtime_media_profile" in messages[0]["content"]
        assert not any(
            message.get("type") in {"session_ready", "frame_batch"}
            for message in messages
        )

    asyncio.run(scenario())


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
            self.timesteps = []

        def eval(self):
            return self

        def inference(self, first, second, scale=1.0, timestep=0.5):
            assert scale == 1.0
            self.timesteps.append(timestep)
            if torch.is_tensor(timestep):
                return first + (second - first) * timestep
            return first + (second - first) * float(timestep)

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
    exact_3x = processor.interpolate_intermediates(
        _frames(0.0, 0.3, 0.6, 0.9),
        multiplier=3,
    )
    assert exact_3x.shape == (1, 3, 3, 2, 2, 2)
    torch.testing.assert_close(
        exact_3x[0, 0, :, :, 0, 0],
        torch.tensor(
            [
                [0.1, 0.2],
                [0.4, 0.5],
                [0.7, 0.8],
            ]
        ),
    )
    arbitrary_timesteps = [
        value.flatten().tolist()
        for value in processor.model.timesteps
        if torch.is_tensor(value)
    ]
    assert arbitrary_timesteps
    assert all(
        values == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
        for values in arbitrary_timesteps
    )


def test_rife_processor_loads_the_exact_weight_bytes_that_were_hashed(tmp_path):
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

    torch.save({"module.weight": torch.ones(1)}, weight_file)
    digest = hashlib.sha256(weight_file.read_bytes()).hexdigest()

    def replacing_factory():
        # Simulate a writable mount or symlink target changing after digest
        # validation but before state loading.
        torch.save({"module.weight": torch.zeros(1)}, weight_file)
        return _Model()

    processor = RIFE2xMediaProcessor(
        tmp_path,
        digest,
        device="cpu",
        model_factory=replacing_factory,
    )
    torch.testing.assert_close(processor.model.flownet.weight, torch.ones(1))


class _SingleWeightRIFEModel:
    def __init__(self):
        self.flownet = torch.nn.Linear(1, 1, bias=False)

    def eval(self):
        return self


def _create_single_weight_rife_processor(tmp_path, state):
    weight_file = tmp_path / "flownet.pkl"
    torch.save(state, weight_file)
    digest = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    return RIFE2xMediaProcessor(
        tmp_path,
        digest,
        device="cpu",
        model_factory=_SingleWeightRIFEModel,
    )


def test_strict_local_rife_processor_accepts_official_training_auxiliaries(tmp_path):
    expected_weight = torch.full((1, 1), 2.0)
    processor = _create_single_weight_rife_processor(
        tmp_path,
        {
            "module.weight": expected_weight,
            "module.teacher.block0.conv0.0.1.weight": torch.ones(1),
            "module.caltime.0.weight": torch.ones(1),
        },
    )
    torch.testing.assert_close(processor.model.flownet.weight, expected_weight)


def test_strict_local_rife_processor_rejects_unknown_extra_key(tmp_path):
    with pytest.raises(ValueError, match="exactly match IFNet"):
        _create_single_weight_rife_processor(
            tmp_path,
            {
                "module.weight": torch.ones((1, 1)),
                "module.unrecognized.weight": torch.ones(1),
            },
        )


def test_strict_local_rife_processor_rejects_missing_inference_key_after_strip(
    tmp_path,
):
    with pytest.raises(ValueError, match="exactly match IFNet"):
        _create_single_weight_rife_processor(
            tmp_path,
            {
                "module.teacher.block0.conv0.0.1.weight": torch.ones(1),
                "module.caltime.0.weight": torch.ones(1),
            },
        )


def test_strict_local_rife_processor_rejects_mixed_module_prefixes(tmp_path):
    with pytest.raises(ValueError, match="cannot mix prefixed and unprefixed"):
        _create_single_weight_rife_processor(
            tmp_path,
            {
                "module.weight": torch.ones((1, 1)),
                "teacher.block0.conv0.0.1.weight": torch.ones(1),
            },
        )


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"module.weight": torch.ones(1)},
    ],
)
def test_strict_local_rife_processor_rejects_empty_or_misshaped_state(
    tmp_path,
    state,
):
    with pytest.raises(ValueError, match="state dict"):
        _create_single_weight_rife_processor(tmp_path, state)


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


def test_worker_keeps_legacy_midpoint_only_processor_at_rife2x_capability():
    class _LegacyMidpointRIFE:
        ready = True
        weights_sha256 = "d" * 64

        def interpolate_midpoints(self, source_frames):
            return (
                source_frames[:, :3, :-1].float() + source_frames[:, :3, 1:].float()
            ) / 2

    processor = _LegacyMidpointRIFE()
    worker = AsyncVAEWorker(
        _FrameEngine([]),
        max_sessions=1,
        rife_processor=processor,
    )
    assert worker.supported_media_profiles == (
        RealtimeMediaProfile.NATIVE_V1,
        RealtimeMediaProfile.RIFE2X_V1,
    )
    assert worker.media_capability_fingerprint == (
        f"rife2x_v1:{processor.weights_sha256}"
    )
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
        # It holds the new first frame in that slot to preserve 2x cadence.
        assert (event_cutover.source_num_frames, event_cutover.output_num_frames) == (
            2,
            4,
        )
        assert _red_values(event_cutover)[:2] == [51, 51]
        assert (prompt_cutover.source_num_frames, prompt_cutover.output_num_frames) == (
            2,
            4,
        )
        assert _red_values(prompt_cutover)[:2] == [153, 153]
        assert (
            sum(
                result.output_num_frames
                for result in (first, second, event_cutover, prompt_cutover)
            )
            == 2
            * sum(
                result.source_num_frames
                for result in (first, second, event_cutover, prompt_cutover)
            )
            - 1
        )
        assert [shape[2] for shape in processor.calls] == [3, 3, 2, 2]
        metrics = generate_latest()
        assert b'stage="actor_wait"' in metrics
        assert b'stage="rife_interpolation"' in metrics
        assert b'stage="vae_actor_wait"' in metrics
        assert b'stage="frame_interpolation"' in metrics
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rife3x_preserves_exact_cadence_and_holds_cutover_slots():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine(
                [
                    _frames(0.0, 0.6),
                    _frames(0.0, 0.6),
                    _frames(0.4, 1.0),
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
                media_profile=RealtimeMediaProfile.RIFE3X_V1,
                source_timeline_fps=24,
            )
        )
        assert acceptance.output_timeline_fps == 72
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)

        first = await worker.decode(_header(0, event_id=7, prompt_version=0), latent)
        second = await worker.decode(_header(1, event_id=7, prompt_version=0), latent)
        cutover = await worker.decode(_header(2, event_id=8, prompt_version=0), latent)

        # The session begins with 3N-2 because no preceding source endpoint
        # exists. Every later chunk emits exactly 3N frames.
        assert (first.source_num_frames, first.output_num_frames) == (2, 4)
        assert _red_values(first) == [0, 51, 102, 153]
        assert (second.source_num_frames, second.output_num_frames) == (2, 6)
        assert _red_values(second) == [102, 51, 0, 51, 102, 153]
        # A control cutover holds both missing 1/3 and 2/3 seam slots at the
        # new first endpoint. It never interpolates the old state into it.
        assert (cutover.source_num_frames, cutover.output_num_frames) == (2, 6)
        assert _red_values(cutover) == [102, 102, 102, 153, 204, 255]
        assert (
            sum(result.output_num_frames for result in (first, second, cutover))
            == 3 * sum(result.source_num_frames for result in (first, second, cutover))
            - 2
        )
        assert [shape[2] for shape in processor.calls] == [2, 3, 2]
        assert processor.intermediate_multipliers == [3, 3, 3]
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rife3x_empty_single_and_shape_change_keep_global_cadence():
    async def scenario():
        shape_changed = torch.full(
            (1, 3, 1, 3, 2),
            0.2,
            dtype=torch.float32,
        )
        worker = AsyncVAEWorker(
            _FrameEngine(
                [
                    _frames(0.0),
                    torch.empty((1, 3, 0, 2, 2), dtype=torch.float32),
                    _frames(0.6),
                    shape_changed,
                ]
            ),
            max_sessions=1,
            encoded_frames_per_batch=16,
            rife_processor=_AveragingRIFE(),
        )
        await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.RIFE3X_V1,
            )
        )
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
        first = await worker.decode(
            _header(0, event_id=None, action_version=0, prompt_version=0),
            latent,
        )
        empty_cutover = await worker.decode(
            _header(1, event_id=None, action_version=1, prompt_version=0),
            latent,
        )
        held_single = await worker.decode(
            _header(2, event_id=None, action_version=1, prompt_version=0),
            latent,
        )
        changed_shape = await worker.decode(
            _header(3, event_id=None, action_version=1, prompt_version=0),
            latent,
        )

        assert (first.source_num_frames, first.output_num_frames) == (1, 1)
        assert (empty_cutover.source_num_frames, empty_cutover.output_num_frames) == (
            0,
            0,
        )
        assert (held_single.source_num_frames, held_single.output_num_frames) == (
            1,
            3,
        )
        assert _red_values(held_single) == [153, 153, 153]
        assert (changed_shape.source_num_frames, changed_shape.output_num_frames) == (
            1,
            3,
        )
        assert _red_values(changed_shape) == [51, 51, 51]
        results = (first, empty_cutover, held_single, changed_shape)
        assert (
            sum(result.output_num_frames for result in results)
            == 3 * sum(result.source_num_frames for result in results) - 2
        )
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rife3x_shape_failure_does_not_commit_previous_frame():
    class _ShapeFailingRIFE(_AveragingRIFE):
        fail = True

        def interpolate_intermediates(
            self,
            source_frames: torch.Tensor,
            *,
            multiplier: int,
        ) -> torch.Tensor:
            result = super().interpolate_intermediates(
                source_frames,
                multiplier=multiplier,
            )
            return result[:, :, :, :1] if self.fail else result

    async def scenario():
        processor = _ShapeFailingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine([]),
            max_sessions=1,
            rife_processor=processor,
        )
        previous = _frames(0.9)
        state = SimpleNamespace(previous_source_frame=previous.clone())
        current = _frames(0.0, 0.6)
        with pytest.raises(RuntimeError, match="intermediate shape mismatch"):
            await worker._interpolate_rife(
                state,
                current,
                multiplier=3,
                hold_new_boundary_frame=True,
                reset_previous=True,
            )
        torch.testing.assert_close(state.previous_source_frame, previous)

        processor.fail = False
        output = await worker._interpolate_rife(
            state,
            current,
            multiplier=3,
            hold_new_boundary_frame=True,
            reset_previous=True,
        )
        torch.testing.assert_close(
            output[0, 0, :, 0, 0],
            torch.tensor([0.0, 0.0, 0.0, 0.2, 0.4, 0.6]),
        )
        torch.testing.assert_close(state.previous_source_frame, _frames(0.6))
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_rife_resets_seam_on_action_change_without_event_id():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine(
                [
                    _frames(0.0, 0.4),
                    _frames(0.6, 0.8),
                    _frames(0.9, 1.0),
                    _frames(0.0, 0.4),
                ]
            ),
            max_sessions=1,
            encoded_frames_per_batch=16,
            rife_processor=processor,
        )
        await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
            )
        )
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
        first = await worker.decode(
            _header(
                0,
                event_id=None,
                action_version=0,
                prompt_version=0,
            ),
            latent,
        )
        action_cutover = await worker.decode(
            _header(
                1,
                event_id=None,
                action_version=1,
                prompt_version=0,
            ),
            latent,
        )
        same_action = await worker.decode(
            _header(
                2,
                event_id=None,
                action_version=1,
                prompt_version=0,
            ),
            latent,
        )
        second_action_cutover = await worker.decode(
            _header(
                3,
                event_id=None,
                action_version=2,
                prompt_version=0,
            ),
            latent,
        )

        assert (first.source_num_frames, first.output_num_frames) == (2, 3)
        # event_id=None is valid.  action_version remains an authoritative
        # control boundary and must suppress the old-tail/new-head midpoint.
        assert (action_cutover.source_num_frames, action_cutover.output_num_frames) == (
            2,
            4,
        )
        assert _red_values(action_cutover) == [153, 153, 179, 204]
        # Once the action version is stable again, the next chunk restores the
        # cross-chunk seam and therefore emits 2N frames.
        assert (same_action.source_num_frames, same_action.output_num_frames) == (2, 4)
        assert _red_values(same_action) == [217, 230, 242, 255]
        assert (
            second_action_cutover.source_num_frames,
            second_action_cutover.output_num_frames,
        ) == (2, 4)
        assert _red_values(second_action_cutover) == [0, 0, 51, 102]
        results = (first, action_cutover, same_action, second_action_cutover)
        assert (
            sum(result.output_num_frames for result in results)
            == 2 * sum(result.source_num_frames for result in results) - 1
        )
        await worker.close_all()

    asyncio.run(scenario())


def test_worker_preserves_cutover_hold_across_an_empty_source_chunk():
    async def scenario():
        processor = _AveragingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine(
                [
                    _frames(0.0, 0.4),
                    torch.empty((1, 3, 0, 2, 2), dtype=torch.float32),
                    _frames(0.6, 0.8),
                ]
            ),
            max_sessions=1,
            encoded_frames_per_batch=16,
            rife_processor=processor,
        )
        await worker.open(
            SessionOpen(
                "session",
                "generation",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
            )
        )
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
        first = await worker.decode(
            _header(0, event_id=None, action_version=0, prompt_version=0),
            latent,
        )
        empty_cutover = await worker.decode(
            _header(1, event_id=None, action_version=1, prompt_version=0),
            latent,
        )
        first_nonempty_after_cutover = await worker.decode(
            _header(2, event_id=None, action_version=1, prompt_version=0),
            latent,
        )

        assert (first.source_num_frames, first.output_num_frames) == (2, 3)
        assert (empty_cutover.source_num_frames, empty_cutover.output_num_frames) == (
            0,
            0,
        )
        assert (
            first_nonempty_after_cutover.source_num_frames,
            first_nonempty_after_cutover.output_num_frames,
        ) == (2, 4)
        assert _red_values(first_nonempty_after_cutover) == [153, 153, 179, 204]
        results = (first, empty_cutover, first_nonempty_after_cutover)
        assert (
            sum(result.output_num_frames for result in results)
            == 2 * sum(result.source_num_frames for result in results) - 1
        )
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


def test_rife_cancellation_waits_for_native_call_before_next_session_enters():
    class _BlockingRIFE(_AveragingRIFE):
        def __init__(self) -> None:
            super().__init__()
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self._lock = threading.Lock()
            self._active = 0
            self.max_active = 0

        def interpolate_midpoints(self, source_frames: torch.Tensor) -> torch.Tensor:
            with self._lock:
                call_index = len(self.calls) + 1
                self.calls.append(tuple(source_frames.shape))
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                if call_index == 1:
                    self.first_started.set()
                    if not self.release_first.wait(timeout=5):
                        raise TimeoutError("test RIFE call was not released")
                else:
                    self.second_started.set()
                return (
                    source_frames[:, :3, :-1].float() + source_frames[:, :3, 1:].float()
                ) / 2
            finally:
                with self._lock:
                    self._active -= 1

    async def scenario():
        processor = _BlockingRIFE()
        worker = AsyncVAEWorker(
            _FrameEngine([_frames(0.0, 0.5), _frames(0.2, 0.8)]),
            max_sessions=2,
            encoded_frames_per_batch=16,
            rife_processor=processor,
        )
        await worker.open(
            SessionOpen(
                "session-1",
                "generation-1",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
            )
        )
        await worker.open(
            SessionOpen(
                "session-2",
                "generation-2",
                media_profile=RealtimeMediaProfile.RIFE2X_V1,
            )
        )
        first_state = worker._sessions[("session-1", "generation-1")]
        second_state = worker._sessions[("session-2", "generation-2")]
        emitted_first: list[int] = []

        async def on_first_frame(batch):
            emitted_first.append(batch.num_frames)

        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
        first_decode = asyncio.create_task(
            worker.decode(
                _header(
                    0,
                    event_id=1,
                    prompt_version=0,
                    session_id="session-1",
                    generation_id="generation-1",
                ),
                latent,
                on_frame_batch=on_first_frame,
            )
        )
        close_first = None
        second_decode = None
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(processor.first_started.wait, 1),
                timeout=2,
            )
            second_decode = asyncio.create_task(
                worker.decode(
                    _header(
                        0,
                        event_id=1,
                        prompt_version=0,
                        session_id="session-2",
                        generation_id="generation-2",
                    ),
                    latent,
                )
            )
            for _ in range(100):
                if second_state.processing:
                    break
                await asyncio.sleep(0.005)
            assert second_state.processing

            close_first = asyncio.create_task(worker.close("session-1", "generation-1"))
            await asyncio.sleep(0.05)
            assert not close_first.done()
            assert not processor.second_started.is_set()
            assert processor.max_active == 1
            assert first_state.previous_source_frame is None
            assert emitted_first == []

            # A repeated abort/close signal must not punch through the drain
            # barrier while the first native thread is still using the actor.
            assert first_state.runner is not None
            first_state.runner.cancel()
            await asyncio.sleep(0.05)
            assert not close_first.done()
            assert not processor.second_started.is_set()
            assert processor.max_active == 1
            assert first_state.previous_source_frame is None
            assert emitted_first == []

            processor.release_first.set()
            await asyncio.wait_for(close_first, timeout=2)
            with pytest.raises(asyncio.CancelledError):
                await first_decode
            second = await asyncio.wait_for(second_decode, timeout=2)
            assert (second.source_num_frames, second.output_num_frames) == (2, 3)
            assert second.actor_wait_ms >= 40
            assert second.decode_ms < second.actor_wait_ms
            assert processor.second_started.is_set()
            assert processor.max_active == 1
            assert first_state.previous_source_frame is None
            assert emitted_first == []
        finally:
            processor.release_first.set()
            pending = [
                task
                for task in (close_first, first_decode, second_decode)
                if task is not None and not task.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await worker.close_all()

    asyncio.run(scenario())


def test_sync_iterator_cancellation_keeps_shared_actor_serialized():
    class _BlockingIteratorEngine:
        backend = "taehv"
        rgb_quantization = "round"

        def __init__(self) -> None:
            self.outputs = [_frames(0.0, 0.5), _frames(0.2, 0.8)]
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self._lock = threading.Lock()
            self._active = 0
            self.frame_calls = 0
            self.max_active = 0

        def create_decoder(self, identity):
            return identity

        def iter_decode(self, decoder, latents, *, first_chunk):
            del decoder, latents, first_chunk
            engine = self
            output = self.outputs.pop(0)

            class _OneFrameIterator:
                def __init__(self) -> None:
                    self.yielded = False

                def __iter__(self):
                    return self

                def __next__(self):
                    if self.yielded:
                        raise StopIteration
                    self.yielded = True
                    with engine._lock:
                        engine.frame_calls += 1
                        call_index = engine.frame_calls
                        engine._active += 1
                        engine.max_active = max(engine.max_active, engine._active)
                    try:
                        if call_index == 1:
                            engine.first_started.set()
                            if not engine.release_first.wait(timeout=5):
                                raise TimeoutError("test iterator was not released")
                        else:
                            engine.second_started.set()
                        return output.clone()
                    finally:
                        with engine._lock:
                            engine._active -= 1

            return _OneFrameIterator()

    async def scenario():
        engine = _BlockingIteratorEngine()
        worker = AsyncVAEWorker(
            engine,
            max_sessions=2,
            encoded_frames_per_batch=16,
        )
        await worker.open(SessionOpen("session-1", "generation-1"))
        await worker.open(SessionOpen("session-2", "generation-2"))
        second_state = worker._sessions[("session-2", "generation-2")]
        latent = torch.zeros((1, 4, 1, 1, 1), dtype=torch.bfloat16)
        first_decode = asyncio.create_task(
            worker.decode(
                _header(
                    0,
                    event_id=1,
                    prompt_version=0,
                    session_id="session-1",
                    generation_id="generation-1",
                ),
                latent,
            )
        )
        close_first = None
        second_decode = None
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(engine.first_started.wait, 1),
                timeout=2,
            )
            second_decode = asyncio.create_task(
                worker.decode(
                    _header(
                        0,
                        event_id=1,
                        prompt_version=0,
                        session_id="session-2",
                        generation_id="generation-2",
                    ),
                    latent,
                )
            )
            for _ in range(100):
                if second_state.processing:
                    break
                await asyncio.sleep(0.005)
            assert second_state.processing

            close_first = asyncio.create_task(worker.close("session-1", "generation-1"))
            await asyncio.sleep(0.05)
            assert not close_first.done()
            assert not engine.second_started.is_set()
            assert engine.max_active == 1

            engine.release_first.set()
            await asyncio.wait_for(close_first, timeout=2)
            with pytest.raises(asyncio.CancelledError):
                await first_decode
            second = await asyncio.wait_for(second_decode, timeout=2)
            assert (second.source_num_frames, second.output_num_frames) == (2, 2)
            assert second.actor_wait_ms >= 40
            assert second.decode_ms >= second.actor_wait_ms
            assert engine.second_started.is_set()
            assert engine.max_active == 1
        finally:
            engine.release_first.set()
            pending = [
                task
                for task in (close_first, first_decode, second_decode)
                if task is not None and not task.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
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
                actor_wait_ms=2.25,
                rife_interpolation_ms=4.5,
                post_decode_ms=3.25,
            )
        )
        result = await handle.wait()
        assert (result.source_num_frames, result.output_num_frames) == (2, 3)
        assert result.post_decode_ms == 3.25
        assert result.actor_wait_ms == 2.25
        assert result.rife_interpolation_ms == 4.5
        await client.close()
        assert socket.closed

    asyncio.run(scenario())


def test_client_accepts_explicit_rife3x_timeline_72_profile():
    async def scenario():
        digest = "c" * 64
        socket = _NegotiationSocket(
            encode_message(
                "session_accepted",
                session_id="session",
                generation_id="generation",
                decoder_backend="taehv",
                decoder_fidelity="approximate",
                requested_media_profile="rife3x_v1",
                effective_media_profile="rife3x_v1",
                source_timeline_fps=24,
                output_timeline_fps=72,
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
            media_profile=RealtimeMediaProfile.RIFE3X_V1,
            source_timeline_fps=24,
        )
        session_open = decode_message(await socket.sent.get())
        assert session_open["media_profile"] == "rife3x_v1"
        assert session_open["source_timeline_fps"] == 24
        assert acceptance.effective is RealtimeMediaProfile.RIFE3X_V1
        assert acceptance.output_timeline_fps == 72
        assert acceptance.weights_sha256 == digest
        await client.close()

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


def test_client_rejects_silently_downgraded_rife_frame_counts():
    client = RealtimeVAEClient(
        "ws://vae",
        session_id="session",
        generation_id="generation",
        transport="websocket",
    )
    with pytest.raises(ProtocolViolation, match="expected 3, got 2"):
        client._validate_rife_frame_counts(
            2,
            2,
            RealtimeMediaProfile.RIFE2X_V1,
        )
    # An empty source chunk neither starts nor advances the media cadence.
    client._validate_rife_frame_counts(0, 0, RealtimeMediaProfile.RIFE2X_V1)
    client._validate_rife_frame_counts(2, 3, RealtimeMediaProfile.RIFE2X_V1)
    with pytest.raises(ProtocolViolation, match="expected 4, got 3"):
        client._validate_rife_frame_counts(2, 3, RealtimeMediaProfile.RIFE2X_V1)
    client._validate_rife_frame_counts(2, 4, RealtimeMediaProfile.RIFE2X_V1)

    rife3_client = RealtimeVAEClient(
        "ws://vae",
        session_id="session",
        generation_id="generation",
        transport="websocket",
    )
    rife3_client._validate_rife_frame_counts(
        0,
        0,
        RealtimeMediaProfile.RIFE3X_V1,
    )
    with pytest.raises(ProtocolViolation, match="expected 4, got 3"):
        rife3_client._validate_rife_frame_counts(
            2,
            3,
            RealtimeMediaProfile.RIFE3X_V1,
        )
    rife3_client._validate_rife_frame_counts(
        2,
        4,
        RealtimeMediaProfile.RIFE3X_V1,
    )
    with pytest.raises(ProtocolViolation, match="expected 6, got 5"):
        rife3_client._validate_rife_frame_counts(
            2,
            5,
            RealtimeMediaProfile.RIFE3X_V1,
        )
    rife3_client._validate_rife_frame_counts(
        2,
        6,
        RealtimeMediaProfile.RIFE3X_V1,
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


def test_vae_server_reports_rife3x_timeline_72_capability_and_acceptance():
    processor = _AveragingRIFE()
    worker = AsyncVAEWorker(
        _FrameEngine([_frames(0.0, 1.0)]),
        max_sessions=1,
        rife_processor=processor,
    )
    app = create_app(worker, max_message_bytes=1024 * 1024)

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["supported_media_profiles"] == [
            "native_v1",
            "rife2x_v1",
            "rife3x_v1",
        ]
        assert health["rife_weights_sha256"] == processor.weights_sha256
        assert health["media_capability_fingerprint"] == (
            f"rife2x_v1+rife3x_v1:{processor.weights_sha256}"
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
                    media_profile="rife3x_v1",
                    source_timeline_fps=24,
                )
            )
            accepted = decode_message(websocket.receive_bytes())
            assert accepted["type"] == "session_accepted"
            assert accepted["requested_media_profile"] == "rife3x_v1"
            assert accepted["effective_media_profile"] == "rife3x_v1"
            assert accepted["source_timeline_fps"] == 24
            assert accepted["output_timeline_fps"] == 72
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
            assert len(frame_messages) == 4
            assert all(
                message["media_profile"] == "rife3x_v1"
                and message["source_timeline_fps"] == 24
                and message["output_timeline_fps"] == 72
                for message in frame_messages
            )
            complete = next(
                message for message in messages if message["type"] == "chunk_complete"
            )
            assert (complete["source_num_frames"], complete["output_num_frames"]) == (
                2,
                4,
            )
            assert complete["rife_interpolation_ms"] >= 0
            assert complete["actor_wait_ms"] >= 0
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
