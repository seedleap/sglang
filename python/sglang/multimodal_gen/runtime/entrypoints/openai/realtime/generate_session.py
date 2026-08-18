# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.utils.realtime_trace import normalize_trace_id

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )


def _non_negative_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


@dataclass(frozen=True, slots=True)
class RealtimeChunkContext:
    session_id: str
    generation_id: str
    index: int
    request_id: str
    action_version: int = 0
    prompt_version: int = 0


class GenerateSession:
    """A realtime generation session"""

    def __init__(
        self,
        *,
        max_inflight_chunks: int = 1,
        session_id: str | None = None,
        generation_id: str | None = None,
    ):
        if max_inflight_chunks < 1:
            raise ValueError("max_inflight_chunks must be positive")
        self.id = session_id or uuid4().hex
        self.generation_id = generation_id or uuid4().hex
        self.trace_id = self.id
        self.trace_started_at = time.perf_counter()
        self.trace_started_epoch_ms = int(time.time() * 1000)
        self.created_at = time.monotonic()
        self.playable_at: float | None = None
        self.playback_ack_enabled = False
        self.last_received_chunk = -1
        self.last_rendered_chunk = -1
        self.last_rendered_event_id = 0
        self.last_client_activity_at = self.created_at
        self.client_activity_version = 0
        self.interactive_event_version = 0
        self._interactive_event = asyncio.Event()
        self.action_version = 0
        self.prompt_version = 0
        self.denoise_intervals: dict[int, tuple[float, float]] = {}
        self.vae_intervals: dict[int, tuple[float, float]] = {}
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.next_chunk_index = 0
        self.max_inflight_chunks = max_inflight_chunks
        self.active_chunks: dict[int, RealtimeChunkContext] = {}
        self.active_batches: dict[int, Any] = {}
        self._completed_chunks: set[int] = set()
        self.realtime_session = RealtimeSession()
        self.adapter: BaseRealtimeModelAdapter | None = None
        self.adapter_state: Any = None
        self.output_pace_next_send_at: float | None = None
        self.output_pace_last_event_id: int | None = None
        self.vae_client: Any = None
        self.vae_decoder_backend: str | None = None
        self.media_profile_acceptance: Any = None
        self.vae_worker_url: str | None = None
        self.vae_worker_epoch: str | None = None
        self.coordinator_token: str | None = None
        self.gateway_output_url: str | None = None
        self.gateway_output_token: str | None = None
        self.pending_control_refresh: tuple[str, int | None] | None = None
        self.control_refresh_task: Any = None
        self.input_event_timings: dict[int, tuple[float, float]] = {}

    def set_adapter(self, adapter: BaseRealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def bind_trace(self, request: RealtimeVideoGenerationsRequest):
        self.trace_id = normalize_trace_id(request.trace_id, fallback=self.trace_id)

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.bind_trace(request)
        self.request = request
        self.playback_ack_enabled = bool(request.playback_ack_enabled)

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.request = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.next_chunk_index = 0
        self.active_chunks.clear()
        self.active_batches.clear()
        self._completed_chunks.clear()
        self.adapter = None
        self.adapter_state = None
        self.output_pace_next_send_at = None
        self.output_pace_last_event_id = None
        self.vae_client = None
        self.vae_decoder_backend = None
        self.media_profile_acceptance = None
        self.pending_control_refresh = None
        self.control_refresh_task = None
        self.input_event_timings.clear()
        self.playable_at = None
        self.playback_ack_enabled = False
        self.last_received_chunk = -1
        self.last_rendered_chunk = -1
        self.last_rendered_event_id = 0
        self.denoise_intervals.clear()
        self.vae_intervals.clear()
        self.realtime_session.dispose()

    def record_input_event(
        self,
        event_id: int | None,
        client_sent_epoch_ms: float | None,
        server_received_epoch_ms: float,
    ) -> None:
        if event_id is None or event_id <= 0 or not client_sent_epoch_ms:
            return
        self.input_event_timings[int(event_id)] = (
            float(client_sent_epoch_ms),
            float(server_received_epoch_ms),
        )
        while len(self.input_event_timings) > 128:
            self.input_event_timings.pop(next(iter(self.input_event_timings)))

    def consume_input_timing(self, event_id: int | None) -> dict[str, float] | None:
        """Consume the first user input included in a generated event burst."""
        if event_id is None or event_id <= 0:
            return None
        consumed = [
            candidate
            for candidate in self.input_event_timings
            if candidate <= int(event_id)
        ]
        if not consumed:
            return None
        first_event_id = min(consumed)
        client_sent, server_received = self.input_event_timings[first_event_id]
        for candidate in consumed:
            self.input_event_timings.pop(candidate, None)
        return {
            "input_event_id": float(first_event_id),
            "client_sent_epoch_ms": client_sent,
            "server_received_epoch_ms": server_received,
            "input_uplink_ms": max(0.0, server_received - client_sent),
        }

    def mark_playable(self) -> None:
        if self.playable_at is None:
            self.playable_at = time.monotonic()

    def apply_playback_ack(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.last_received_chunk = max(
            self.last_received_chunk,
            _non_negative_int(payload.get("last_received_chunk"), -1),
        )
        self.last_rendered_chunk = max(
            self.last_rendered_chunk,
            _non_negative_int(payload.get("last_rendered_chunk"), -1),
        )
        self.last_rendered_event_id = max(
            self.last_rendered_event_id,
            _non_negative_int(payload.get("last_rendered_event_id"), 0),
        )
        if payload.get("playable") is True:
            self.mark_playable()

    def mark_client_activity(self) -> None:
        self.last_client_activity_at = time.monotonic()
        self.client_activity_version += 1

    def mark_event_version(self, kind: str) -> None:
        if kind in {"camera_actions", "action_labels", "action_weights"}:
            self.action_version += 1
        elif kind in {"prompt", "scene_cut"}:
            self.prompt_version += 1
        self.interactive_event_version += 1
        self._interactive_event.set()

    async def wait_for_interactive_event_after(
        self, version: int, timeout_s: float
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self.interactive_event_version <= version:
            self._interactive_event.clear()
            if self.interactive_event_version > version:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(
                    self._interactive_event.wait(), timeout=remaining
                )
            except TimeoutError:
                return False
        return True

    @property
    def current_chunk(self) -> RealtimeChunkContext | None:
        if not self.active_chunks:
            return None
        return self.active_chunks[min(self.active_chunks)]

    @property
    def latest_active_chunk(self) -> RealtimeChunkContext | None:
        if not self.active_chunks:
            return None
        return self.active_chunks[max(self.active_chunks)]

    def can_schedule_chunk(self) -> bool:
        if len(self.active_chunks) >= self.max_inflight_chunks:
            return False
        if self.request is None or self.request.max_chunks is None:
            return True
        return self.next_chunk_index < self.request.max_chunks

    def new_chunk(
        self,
        *,
        action_version: int | None = None,
        prompt_version: int | None = None,
    ) -> RealtimeChunkContext:
        if len(self.active_chunks) >= self.max_inflight_chunks:
            if self.max_inflight_chunks == 1:
                raise RuntimeError("previous realtime chunk is still active")
            raise RuntimeError("realtime chunk in-flight limit reached")
        if not self.can_schedule_chunk():
            raise RuntimeError("realtime session reached max chunks")
        chunk = RealtimeChunkContext(
            session_id=self.id,
            generation_id=self.generation_id,
            index=self.next_chunk_index,
            request_id=f"{self.id}_{uuid4().hex}",
            action_version=(
                self.action_version if action_version is None else action_version
            ),
            prompt_version=(
                self.prompt_version if prompt_version is None else prompt_version
            ),
        )
        self.next_chunk_index += 1
        self.active_chunks[chunk.index] = chunk
        return chunk

    def bind_chunk_request(self, chunk: RealtimeChunkContext, batch: Any) -> None:
        if self.active_chunks.get(chunk.index) != chunk:
            raise RuntimeError(f"realtime chunk {chunk.index} is not active")
        self.active_batches[chunk.index] = batch

    def generate_chunk_completed(
        self, chunk: RealtimeChunkContext | None = None
    ) -> None:
        if chunk is None:
            if not self.active_chunks:
                raise RuntimeError("no active realtime chunk to complete")
            chunk = self.active_chunks[min(self.active_chunks)]
        active = self.active_chunks.pop(chunk.index, None)
        if active != chunk:
            raise RuntimeError(f"realtime chunk {chunk.index} is not active")
        self.active_batches.pop(chunk.index, None)
        self._completed_chunks.add(chunk.index)
        while self.generate_chunk_cnt in self._completed_chunks:
            self._completed_chunks.remove(self.generate_chunk_cnt)
            self.generate_chunk_cnt += 1

    def reached_max_chunks(self) -> bool:
        return (
            self.request is not None
            and self.request.max_chunks is not None
            and self.generate_chunk_cnt >= self.request.max_chunks
        )
