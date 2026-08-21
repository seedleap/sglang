# SPDX-License-Identifier: Apache-2.0

import asyncio

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
    decode_message,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    EncodedFrameBatch,
)
from sglang.multimodal_gen.runtime.realtime.h264_media_pipeline import (
    H264MediaPipeline,
    H264PipelineConfig,
)


class _FakeStdin:
    def __init__(self):
        self.frames = []

    def write(self, value):
        self.frames.append(value)

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        pass


class _FakeStdout:
    def __init__(self):
        self.calls = 0

    async def read(self, _size):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0)
            return b"fmp4-payload"
        await asyncio.Future()


class _FakeStderr:
    async def readline(self):
        await asyncio.Future()


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()

    async def wait(self):
        return 0

    def kill(self):
        pass


def _header() -> LatentChunkHeader:
    return LatentChunkHeader(
        session_id="session",
        generation_id="generation",
        request_id="request-0",
        chunk_index=0,
        dtype="float16",
        shape=(1, 1, 1, 1, 1),
        byte_length=2,
        checksum="checksum",
    )


def _batch(*frames: bytes) -> EncodedFrameBatch:
    return EncodedFrameBatch(
        payloads=frames,
        content_type="application/x-raw-rgb",
        width=2,
        height=2,
        frame_batch_index=0,
        is_final=True,
        encode_ms=0.0,
        source_width=2,
        source_height=2,
        preview_width=2,
        preview_height=2,
    )


def test_h264_pipeline_bounds_rgb_before_encoding_and_emits_fmp4(monkeypatch):
    async def run():
        process = _FakeProcess()

        async def create_subprocess(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                max_queued_frames=1,
                live_edge_frames=0,
            ),
        )
        first = bytes([1, 2, 3]) * 4
        latest = bytes([4, 5, 6]) * 4
        pipeline.enqueue(_header(), _batch(first, latest))
        pipeline.enqueue_completion(_header(), num_frames=2)

        for _ in range(100):
            messages = [decode_message(wire) for wire in wires]
            if any(message["type"] == "media_payload" for message in messages):
                break
            await asyncio.sleep(0.001)

        messages = [decode_message(wire) for wire in wires]
        assert pipeline.dropped_frames == 1
        assert process.stdin.frames == [latest]
        assert [message["type"] for message in messages] == [
            "media_init",
            "media_batch",
            "media_encode_timing",
            "media_chunk_complete",
            "media_payload",
        ]
        media = messages[-1]
        assert media["codec"] == "h264"
        assert media["container"] == "fmp4"
        assert media["payload"] == b"fmp4-payload"
        assert all(
            message.get("content_type") != "application/x-raw-rgb"
            for message in messages
        )
        await pipeline.close()

    asyncio.run(run())


def test_h264_pipeline_drains_final_chunk_before_close(monkeypatch):
    """收尾必须把最后一块的帧和 media_chunk_complete 发出去。

    回归的是线上真事故：VAE 收到 is_final_chunk 后立刻拆会话 → close() 直接
    cancel+clear → 最后一块的完成标记被丢 → 网关等 chunk_index==max_chunks-1
    的标记等不到，空转到 --output-drain-timeout-s（线上 90s）才收摊，一局白占
    90 秒座位，还把正常跑完的生成误标成 "maximum session lifetime reached"。
    """

    async def run():
        process = _FakeProcess()

        async def create_subprocess(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                # 队列放得下整块，避免掉帧策略干扰本用例
                max_queued_frames=8,
                live_edge_frames=8,
            ),
        )
        frame = bytes([1, 2, 3]) * 4
        # 入队后不给事件循环任何喘息就 close()：模拟"解码完最后一块立刻收摊"
        pipeline.enqueue(_header(), _batch(frame, frame))
        pipeline.enqueue_completion(_header(), num_frames=2)
        await pipeline.close()

        messages = [decode_message(wire) for wire in wires]
        types = [message["type"] for message in messages]
        assert types.count("media_batch") == 2, f"尾帧被丢了: {types}"
        completions = [m for m in messages if m["type"] == "media_chunk_complete"]
        assert len(completions) == 1, f"最后一块的完成标记被丢了: {types}"
        assert completions[0]["chunk_index"] == 0
        assert completions[0]["media_transport"] == "h264"
        assert process.stdin.frames == [frame, frame]

    asyncio.run(run())
