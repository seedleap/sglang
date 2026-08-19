# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    SHARED_MEMORY_DIR_ENV,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)


def test_critical_path_observer_failure_cancels_unconsumed_async_output(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(SHARED_MEMORY_DIR_ENV, str(tmp_path))
    reference = reserve_async_shared_memory_payload(4, root=tmp_path)
    result = OutputBatch(raw_frame_shared_memory_ref=reference)
    terminal = []

    def wait_for_cancel():
        terminal.append(
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
        )

    producer = threading.Thread(target=wait_for_cancel)
    producer.start()

    class _Adapter:
        @staticmethod
        async def wait_for_next_chunk(_session):
            return None

        @staticmethod
        def prepare_next_request(_session, _server_args, chunk):
            return SimpleNamespace(
                block_idx=chunk.index,
                condition_inputs={},
                realtime_event_id=None,
                request_id=chunk.request_id,
            )

        @staticmethod
        def on_chunk_complete(_session, _result):
            raise AssertionError("observer failure must precede chunk completion")

    async def process_generation_batch(_client, _batch):
        return None, result

    def fail_observer(_result, *, server_args):
        del server_args
        raise RuntimeError("critical-path observer failed")

    async def ignore_error_message(_message, _websocket):
        return None

    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(model_id="test-model"),
    )
    monkeypatch.setattr(
        realtime_video_api,
        "process_generation_batch",
        process_generation_batch,
    )
    monkeypatch.setattr(
        realtime_video_api,
        "_observe_realtime_result_stage_metrics",
        fail_observer,
    )
    monkeypatch.setattr(realtime_video_api, "observe_stage_ms", lambda *_a, **_k: True)
    monkeypatch.setattr(
        realtime_video_api, "log_realtime_trace", lambda *_a, **_k: None
    )
    monkeypatch.setattr(realtime_video_api, "write_error_msg", ignore_error_message)

    session = GenerateSession()
    session.adapter = _Adapter()
    session.request = SimpleNamespace(
        max_chunks=1,
        realtime_interactive_event_grace_ms=0,
    )

    asyncio.run(realtime_video_api._generate_loop_local(SimpleNamespace(), session))
    producer.join(timeout=1)

    assert not producer.is_alive()
    assert terminal == ["cancel"]
    assert result.raw_frame_shared_memory_ref is None
    assert list(tmp_path.iterdir()) == []
