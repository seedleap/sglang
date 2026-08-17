# SPDX-License-Identifier: Apache-2.0

"""Session-fenced media routing primitives for the realtime Gateway."""

from __future__ import annotations

import asyncio
import hmac
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import msgspec.msgpack

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
)
from sglang.multimodal_gen.runtime.realtime.request_mode import (
    init_requests_finite_output,
)


class OutputProtocolError(ProtocolViolation):
    pass


class OutputBackpressureError(RuntimeError):
    pass


class OutputIncompleteError(RuntimeError):
    pass


class OutputRouteClosed(RuntimeError):
    pass


class AdmissionQueueFull(RuntimeError):
    reason = "ADMISSION_QUEUE_FULL"

    def __init__(self) -> None:
        super().__init__(self.reason)


def _positive_int(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


class BrowserPlaybackAckWindow:
    """Bound media lead for ACK-aware browsers without blocking the VAE.

    The browser opts in via the init message. Once enabled, at most
    ``max_unacked_chunks`` chunks are sent beyond the latest received ACK.
    If the browser cannot catch up within the short wait window, continuous
    sessions retain the historical live-edge shedding policy.  Finite
    sessions fail explicitly instead: an exact timeline must never silently
    lose a batch and then report completion.
    """

    def __init__(
        self,
        *,
        max_unacked_chunks: int = 2,
        wait_timeout_s: float = 0.25,
    ) -> None:
        if max_unacked_chunks < 1:
            raise ValueError("max_unacked_chunks must be positive")
        if wait_timeout_s < 0:
            raise ValueError("wait_timeout_s must be non-negative")
        self.max_unacked_chunks = max_unacked_chunks
        self.wait_timeout_s = wait_timeout_s
        self.enabled = False
        self.finite_request = False
        self.request_mode_configured = False
        self.last_received_chunk = -1
        self.last_rendered_chunk = -1
        self.minimum_event_id = 0
        self._sent_chunks: set[int] = set()
        self._shed_chunks: set[int] = set()
        self._changed = asyncio.Condition()
        self.shed_messages = 0
        self.shed_frames = 0
        self.rejected_messages = 0
        self.rejected_frames = 0
        self.failure_reason: str | None = None

    def configure_request_mode(self, *, finite_request: bool) -> None:
        if self.request_mode_configured and self.finite_request != finite_request:
            raise OutputProtocolError("request output mode cannot change after init")
        self.request_mode_configured = True
        self.finite_request = finite_request

    async def observe_browser_message(self, wire: bytes) -> None:
        try:
            message = decode_message(wire)
        except ProtocolViolation:
            return
        if message.get("type") == "init":
            self.configure_request_mode(
                finite_request=init_requests_finite_output(message)
            )
            self.enabled = message.get("playback_ack_enabled") is True
            return
        if message.get("type") != "event":
            return
        if message.get("kind") in {"camera_actions", "prompt", "scene_cut"}:
            event_id = self._non_negative_int(message.get("event_id"))
            if event_id is None:
                return
            async with self._changed:
                if event_id > self.minimum_event_id:
                    self.minimum_event_id = event_id
                    self._sent_chunks.clear()
                    self._shed_chunks.clear()
                self._changed.notify_all()
            return
        if message.get("kind") != "playback_ack":
            return
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return
        received = self._non_negative_int(payload.get("last_received_chunk"))
        rendered = self._non_negative_int(payload.get("last_rendered_chunk"))
        async with self._changed:
            if received is not None:
                self.last_received_chunk = max(self.last_received_chunk, received)
                self._sent_chunks = {
                    chunk for chunk in self._sent_chunks if chunk > received
                }
                self._shed_chunks = {
                    chunk for chunk in self._shed_chunks if chunk > received
                }
            if rendered is not None:
                self.last_rendered_chunk = max(self.last_rendered_chunk, rendered)
            self._changed.notify_all()

    async def allow_output(self, wire: bytes) -> bool:
        if self.failure_reason is not None:
            raise OutputBackpressureError(self.failure_reason)
        if not self.enabled:
            return True
        try:
            message = decode_message(wire)
        except ProtocolViolation:
            return True
        if message.get("type") != "frame_batch":
            return True
        chunk_index = self._non_negative_int(message.get("chunk_index"))
        if chunk_index is None:
            return True
        event_id = self._non_negative_int(message.get("event_id"))
        if self.minimum_event_id and (
            event_id is None or event_id < self.minimum_event_id
        ):
            return self._reject_or_shed(
                chunk_index,
                message,
                reason="frame batch predates the active browser event",
            )
        if chunk_index in self._shed_chunks:
            self._record_shed(message)
            return False
        if chunk_index in self._sent_chunks:
            return True
        deadline = time.monotonic() + self.wait_timeout_s
        async with self._changed:
            while len(self._sent_chunks) >= self.max_unacked_chunks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._reject_or_shed(
                        chunk_index,
                        message,
                        reason=(
                            "browser playback ACK window remained full for "
                            f"{self.wait_timeout_s:.3f}s"
                        ),
                    )
                try:
                    await asyncio.wait_for(self._changed.wait(), remaining)
                except TimeoutError:
                    return self._reject_or_shed(
                        chunk_index,
                        message,
                        reason=(
                            "browser playback ACK window remained full for "
                            f"{self.wait_timeout_s:.3f}s"
                        ),
                    )
            self._sent_chunks.add(chunk_index)
        return True

    def _reject_or_shed(self, chunk_index: int, message: dict, *, reason: str) -> bool:
        frame_count = max(1, _positive_int(message.get("num_frames")))
        if self.finite_request:
            self.rejected_messages += 1
            self.rejected_frames += frame_count
            self.failure_reason = f"finite output rejected: {reason}"
            raise OutputBackpressureError(self.failure_reason)
        self._shed_chunks.add(chunk_index)
        self._record_shed(message)
        return False

    def _record_shed(self, message: dict) -> None:
        frame_count = max(1, _positive_int(message.get("num_frames")))
        self.shed_messages += 1
        self.shed_frames += frame_count

    def metrics(self) -> dict[str, int | bool | str]:
        return {
            "gateway_playback_ack_enabled": self.enabled,
            "gateway_playback_ack_finite": self.finite_request,
            "gateway_playback_ack_shed_messages": self.shed_messages,
            "gateway_playback_ack_shed_frames": self.shed_frames,
            "gateway_playback_ack_rejected_messages": self.rejected_messages,
            "gateway_playback_ack_rejected_frames": self.rejected_frames,
            "gateway_playback_ack_failed": self.failure_reason is not None,
            "gateway_playback_ack_failure_reason": self.failure_reason or "",
        }

    @staticmethod
    def _non_negative_int(value) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None


class BoundedAdmissionWaiterGate:
    def __init__(self, *, max_waiters: int = 64) -> None:
        if max_waiters < 1:
            raise ValueError("max_waiters must be positive")
        self.max_waiters = max_waiters
        self.waiters = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def waiter(self):
        async with self._lock:
            if self.waiters >= self.max_waiters:
                raise AdmissionQueueFull()
            self.waiters += 1
        try:
            yield
        finally:
            async with self._lock:
                self.waiters -= 1


_WORKER_CONTROL_MESSAGES = {
    "error",
    "session_ready",
    "control_ack",
    "heartbeat",
    "chunk_telemetry",
}


@dataclass(frozen=True, slots=True)
class _QueuedOutput:
    wire: bytes
    message_type: str
    frame_count: int
    enqueued_at: float


def worker_message_allowed(wire: bytes) -> bool:
    """Only forward business control data from Denoiser to the browser."""
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError:
        return False
    return isinstance(message, dict) and message.get("type") in _WORKER_CONTROL_MESSAGES


def worker_message_type(wire: bytes) -> str:
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError as exc:
        raise ProtocolViolation("invalid Denoiser control message") from exc
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolViolation("Denoiser control message type is required")
    return message["type"]


def build_denoiser_url(
    endpoint: str,
    *,
    session_id: str,
    generation_id: str,
    coordinator_token: str,
    vae_url: str,
    output_url: str,
    output_token: str,
    trace_id: str,
    worker_epoch: str = "",
    vae_worker_epoch: str = "",
) -> str:
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(
        gateway_managed="1",
        session_id=session_id,
        generation_id=generation_id,
        coordinator_token=coordinator_token,
        worker_epoch=worker_epoch,
        realtime_vae_worker_url=vae_url,
        realtime_vae_worker_epoch=vae_worker_epoch,
        gateway_output_url=output_url,
        gateway_output_token=output_token,
        trace_id=trace_id,
    )
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def build_gateway_output_url(
    endpoint: str,
    *,
    completion_ack_timeout_s: float,
) -> str:
    """Carry the Gateway completion budget to the direct VAE output client."""

    if not 0 < completion_ack_timeout_s <= 3600:
        raise ValueError("completion_ack_timeout_s must be in (0, 3600]")
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["completion_ack_timeout_s"] = f"{completion_ack_timeout_s:.3f}"
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


@dataclass(slots=True)
class GatewayOutputRoute:
    session_id: str
    generation_id: str
    token: str
    queue_depth: int
    enqueue_timeout_s: float
    completion_timeout_s: float
    trace_id: str = ""
    _queue: deque[_QueuedOutput | None] = field(init=False)
    _queue_ready: asyncio.Event = field(init=False)
    _queue_drained: asyncio.Event = field(init=False)
    _unfinished_tasks: int = field(default=0, init=False)
    _queued_media_frames: int = field(default=0, init=False)
    _queued_bytes: int = field(default=0, init=False)
    _last_chunk_index: int = field(default=-1, init=False)
    _last_frame_batch_index: int = field(default=-1, init=False)
    _seen_chunks: set[int] = field(default_factory=set, init=False)
    _output_closed: asyncio.Event = field(init=False)
    _chunk_completed: dict[int, asyncio.Event] = field(default_factory=dict, init=False)
    _chunk_forwarded: dict[int, asyncio.Event] = field(default_factory=dict, init=False)
    _capacity_changed: asyncio.Event = field(init=False)
    _output_failed: asyncio.Event = field(init=False)
    _final_completion_forwarded: asyncio.Event = field(init=False)
    _output_state_changed: asyncio.Event = field(init=False)
    dropped_messages: int = field(default=0, init=False)
    dropped_frames: int = field(default=0, init=False)
    rejected_messages: int = field(default=0, init=False)
    rejected_frames: int = field(default=0, init=False)
    forwarded_messages: int = field(default=0, init=False)
    forwarded_frames: int = field(default=0, init=False)
    accepted_completions: int = field(default=0, init=False)
    forwarded_completions: int = field(default=0, init=False)
    rejected_completions: int = field(default=0, init=False)
    finite_request: bool = field(default=False, init=False)
    request_mode_configured: bool = field(default=False, init=False)
    expected_final_chunk: int | None = field(default=None, init=False)
    final_completion_received: bool = field(default=False, init=False)
    final_completion_forwarded: bool = field(default=False, init=False)
    final_completion_chunk: int | None = field(default=None, init=False)
    failure_reason: str | None = field(default=None, init=False)
    failure_kind: str = field(default="", init=False)
    bound: bool = field(default=False, init=False)
    ever_bound: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._queue = deque()
        self._queue_ready = asyncio.Event()
        self._queue_drained = asyncio.Event()
        self._queue_drained.set()
        self._output_closed = asyncio.Event()
        self._output_closed.set()
        self._capacity_changed = asyncio.Event()
        self._output_failed = asyncio.Event()
        self._final_completion_forwarded = asyncio.Event()
        self._output_state_changed = asyncio.Event()

    def configure_request_mode(
        self,
        *,
        finite_request: bool,
        expected_final_chunk: int | None = None,
    ) -> None:
        if self.request_mode_configured and self.finite_request != finite_request:
            raise OutputProtocolError("request output mode cannot change after init")
        if self._seen_chunks and self.finite_request != finite_request:
            raise OutputProtocolError("request output mode arrived after media")
        self.request_mode_configured = True
        self.finite_request = finite_request
        self.expected_final_chunk = expected_final_chunk if finite_request else None

    def fail_output(self, reason: str, *, kind: str = "backpressure") -> None:
        if self.failure_reason is None:
            self.failure_reason = reason
            self.failure_kind = kind
            self._output_failed.set()
        self._output_state_changed.set()
        self._queue_ready.set()
        self._capacity_changed.set()

    def _raise_if_failed(self) -> None:
        if self.failure_reason is not None:
            if self.failure_kind == "protocol":
                raise OutputProtocolError(self.failure_reason)
            raise OutputBackpressureError(self.failure_reason)

    def bind_output(self) -> None:
        self._raise_if_failed()
        self.bound = True
        self.ever_bound = True
        self._output_closed.clear()
        self._output_state_changed.set()

    def unbind_output(self) -> None:
        self.bound = False
        self._output_closed.set()
        self._queue_ready.set()
        self._capacity_changed.set()
        self._output_state_changed.set()

    async def wait_until_output_closed(self) -> None:
        await self._output_closed.wait()

    async def wait_until_chunk_completed(self, chunk_index: int) -> None:
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        event = self._chunk_completed.setdefault(chunk_index, asyncio.Event())
        await event.wait()

    async def wait_until_chunk_forwarded(self, chunk_index: int) -> None:
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        event = self._chunk_forwarded.setdefault(chunk_index, asyncio.Event())
        await self._wait_until_forwarded(
            event,
            description=f"media completion for chunk {chunk_index}",
        )

    async def wait_until_final_completion_forwarded(self) -> None:
        await self._wait_until_forwarded(
            self._final_completion_forwarded,
            description="final media completion",
        )

    async def _wait_until_forwarded(
        self,
        event: asyncio.Event,
        *,
        description: str,
    ) -> None:
        while True:
            if event.is_set():
                return
            self._raise_if_failed()
            if self.finite_request and self.closed:
                raise OutputIncompleteError(
                    f"Gateway output route closed before {description} was forwarded"
                )
            if self.finite_request and self.ever_bound and not self.bound:
                raise OutputIncompleteError(
                    f"Gateway output closed before {description} was forwarded"
                )
            self._output_state_changed.clear()
            if event.is_set():
                return
            self._raise_if_failed()
            if self.finite_request and self.closed:
                continue
            if self.finite_request and self.ever_bound and not self.bound:
                continue
            await self._output_state_changed.wait()

    def token_matches(self, token: str) -> bool:
        return hmac.compare_digest(self.token, token)

    async def put(self, wire: bytes) -> None:
        if self.closed:
            raise OutputRouteClosed("output route is closed")
        message = decode_message(wire)
        message_type = message.get("type")
        if message_type not in {"frame_batch", "media_chunk_complete"}:
            raise OutputProtocolError(
                "Gateway output accepts frame_batch or media_chunk_complete only"
            )
        if message.get("session_id") != self.session_id:
            raise OutputProtocolError("wrong session")
        if message.get("generation_id") != self.generation_id:
            raise OutputProtocolError("stale generation")
        if self.final_completion_received:
            raise OutputProtocolError("output received after final media completion")
        chunk_index = int(message.get("chunk_index", -1))
        if chunk_index < 0:
            raise OutputProtocolError("invalid chunk sequence")
        if self.failure_reason is not None:
            if message_type == "media_chunk_complete":
                self.rejected_completions += 1
            self._raise_if_failed()
        if message_type == "media_chunk_complete":
            if chunk_index not in self._seen_chunks:
                raise OutputProtocolError("completion before frame batch")
            completed = self._chunk_completed.setdefault(chunk_index, asyncio.Event())
            if completed.is_set():
                raise OutputProtocolError("duplicate completion")
            is_final_chunk = message.get("is_final_chunk") is True
            if is_final_chunk and self.final_completion_received:
                raise OutputProtocolError("duplicate final media completion")
            if self.finite_request and self.expected_final_chunk is not None:
                if is_final_chunk and chunk_index != self.expected_final_chunk:
                    self._reject_completion_protocol(
                        "final media completion has unexpected chunk index"
                    )
                if chunk_index == self.expected_final_chunk and not is_final_chunk:
                    self._reject_completion_protocol(
                        "expected final chunk lacks final media marker"
                    )
            if self.finite_request and (
                self.dropped_messages or self.rejected_messages
            ):
                self.rejected_completions += 1
                reason = "finite output cannot complete after media loss"
                self.fail_output(reason)
                raise OutputBackpressureError(reason)
            if self.finite_request:
                # Do not acknowledge a finite completion to the VAE until all
                # earlier media has actually crossed both the Gateway queue and
                # the browser ACK window.  If either path fails, fail_output()
                # wakes this barrier and completion is rejected.
                try:
                    await self._wait_for_prior_output()
                except OutputBackpressureError:
                    self.rejected_completions += 1
                    raise
                except TimeoutError:
                    self.rejected_completions += 1
                    reason = (
                        "finite Gateway completion barrier timed out after "
                        f"{self.completion_timeout_s:.3f}s"
                    )
                    self.fail_output(reason)
                    raise OutputBackpressureError(reason) from None
                self._raise_if_failed()
            # Control markers are tiny and are not counted against the media
            # frame capacity. They must survive media shedding because the
            # upstream VAE uses them to release decode credit.
            self._append_output(
                _QueuedOutput(
                    wire=wire,
                    message_type=message_type,
                    frame_count=0,
                    enqueued_at=time.monotonic(),
                )
            )
            self.accepted_completions += 1
            if is_final_chunk:
                self.final_completion_received = True
                self.final_completion_chunk = chunk_index
            completed.set()
            return

        frame_batch_index = int(message.get("frame_batch_index", -1))
        if frame_batch_index < 0:
            raise OutputProtocolError("invalid frame sequence")
        if chunk_index < self._last_chunk_index:
            raise OutputProtocolError("stale chunk")
        if chunk_index > self._last_chunk_index + 1:
            raise OutputProtocolError("out-of-order chunk")
        if chunk_index == self._last_chunk_index:
            if frame_batch_index <= self._last_frame_batch_index:
                raise OutputProtocolError("duplicate frame batch")
        elif frame_batch_index != 0:
            raise OutputProtocolError("new chunk must start at frame batch zero")
        frame_count = max(1, int(message.get("num_frames") or 1))
        output = _QueuedOutput(
            wire=wire,
            message_type=message_type,
            frame_count=frame_count,
            enqueued_at=time.monotonic(),
        )
        if self.finite_request:
            await self._put_media_reliable(output)
        else:
            self._put_media_latest(output)
        self._last_chunk_index = chunk_index
        self._last_frame_batch_index = frame_batch_index
        self._seen_chunks.add(chunk_index)

    async def _wait_for_prior_output(self) -> None:
        join_task = asyncio.create_task(self.join())
        failure_task = asyncio.create_task(self._output_failed.wait())
        tasks = {join_task, failure_task}
        try:
            done, _ = await asyncio.wait(
                tasks,
                timeout=self.completion_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError
            self._raise_if_failed()
            await join_task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _put_media_reliable(self, output: _QueuedOutput) -> None:
        deadline = time.monotonic() + self.enqueue_timeout_s
        while not self._media_fits(output):
            self._raise_if_failed()
            if self.closed:
                raise OutputRouteClosed("output route is closed")
            self._capacity_changed.clear()
            if self._media_fits(output):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_reliable_enqueue(output)
            try:
                await asyncio.wait_for(self._capacity_changed.wait(), remaining)
            except TimeoutError:
                self._fail_reliable_enqueue(output)
        self._raise_if_failed()
        self._append_output(output)

    def _media_fits(self, output: _QueuedOutput) -> bool:
        # Keep a frame batch indivisible.  A single oversized batch is allowed
        # only when no other media is queued, matching the continuous route.
        return self._queued_media_frames == 0 or (
            self._queued_media_frames + output.frame_count <= self.queue_depth
        )

    def _fail_reliable_enqueue(self, output: _QueuedOutput) -> None:
        self.rejected_messages += 1
        self.rejected_frames += output.frame_count
        reason = (
            "finite Gateway output queue remained full for "
            f"{self.enqueue_timeout_s:.3f}s"
        )
        self.fail_output(reason)
        raise OutputBackpressureError(reason)

    def _put_media_latest(self, output: _QueuedOutput) -> None:
        # Never propagate browser/network backpressure into the VAE worker.
        # Drop whole encoded frame batches from the oldest edge until the new
        # batch fits. A single oversized batch remains indivisible and may
        # temporarily exceed the configured frame capacity.
        while (
            self._queued_media_frames
            and self._queued_media_frames + output.frame_count > self.queue_depth
        ):
            if not self._drop_oldest_media():
                break
        self._append_output(output)

    def _append_output(self, output: _QueuedOutput) -> None:
        if self.closed:
            raise OutputRouteClosed("output route is closed")
        self._queue.append(output)
        self._unfinished_tasks += 1
        self._queue_drained.clear()
        self._queued_media_frames += output.frame_count
        self._queued_bytes += len(output.wire)
        self._queue_ready.set()

    def _drop_oldest_media(self) -> bool:
        for index, output in enumerate(self._queue):
            if output is None or output.message_type != "frame_batch":
                continue
            del self._queue[index]
            self._queued_media_frames -= output.frame_count
            self._queued_bytes -= len(output.wire)
            self._finish_task()
            self.dropped_messages += 1
            self.dropped_frames += output.frame_count
            self._capacity_changed.set()
            return True
        return False

    def queue_metrics(self) -> dict[str, int | float | bool | str]:
        oldest_frame_at = next(
            (
                output.enqueued_at
                for output in self._queue
                if output is not None and output.message_type == "frame_batch"
            ),
            None,
        )
        oldest_frame_age_ms = (
            max(0.0, (time.monotonic() - oldest_frame_at) * 1000.0)
            if oldest_frame_at is not None
            else 0.0
        )
        return {
            "gateway_queue_depth": self._queued_media_frames,
            "gateway_queue_messages": sum(
                1 for output in self._queue if output is not None
            ),
            "gateway_queue_bytes": self._queued_bytes,
            "gateway_oldest_frame_age_ms": round(oldest_frame_age_ms, 3),
            "gateway_dropped_frames": self.dropped_frames,
            "gateway_dropped_messages": self.dropped_messages,
            "gateway_rejected_frames": self.rejected_frames,
            "gateway_rejected_messages": self.rejected_messages,
            "gateway_forwarded_frames": self.forwarded_frames,
            "gateway_forwarded_messages": self.forwarded_messages,
            "gateway_accepted_completions": self.accepted_completions,
            "gateway_forwarded_completions": self.forwarded_completions,
            "gateway_rejected_completions": self.rejected_completions,
            "gateway_queue_capacity_frames": self.queue_depth,
            "gateway_output_finite": self.finite_request,
            "gateway_output_mode_configured": self.request_mode_configured,
            "gateway_expected_final_chunk": (
                self.expected_final_chunk
                if self.expected_final_chunk is not None
                else -1
            ),
            "gateway_final_completion_received": self.final_completion_received,
            "gateway_final_completion_forwarded": self.final_completion_forwarded,
            "gateway_final_completion_chunk": (
                self.final_completion_chunk
                if self.final_completion_chunk is not None
                else -1
            ),
            "gateway_output_failed": self.failure_reason is not None,
            "gateway_output_failure_kind": self.failure_kind,
            "gateway_output_failure_reason": self.failure_reason or "",
        }

    def _reject_completion_protocol(self, reason: str) -> None:
        self.rejected_completions += 1
        self.fail_output(reason, kind="protocol")
        raise OutputProtocolError(reason)

    async def get_output(self) -> _QueuedOutput:
        while True:
            self._raise_if_failed()
            if self._queue:
                output = self._queue.popleft()
                if not self._queue:
                    self._queue_ready.clear()
                if output is None:
                    raise OutputRouteClosed("output route is closed")
                self._queued_media_frames -= output.frame_count
                self._queued_bytes -= len(output.wire)
                self._capacity_changed.set()
                return output
            if self.closed:
                raise OutputRouteClosed("output route is closed")
            if self.finite_request and self.ever_bound and not self.bound:
                if not self.final_completion_forwarded:
                    raise OutputIncompleteError(
                        "Gateway output closed before final media completion"
                    )
                raise OutputRouteClosed("output route is closed")
            self._queue_ready.clear()
            if self._queue:
                continue
            await self._queue_ready.wait()

    async def get(self) -> bytes:
        return (await self.get_output()).wire

    def mark_output_forwarded(self, wire: bytes) -> None:
        message = decode_message(wire)
        message_type = message.get("type")
        if message_type == "frame_batch":
            self.forwarded_messages += 1
            self.forwarded_frames += max(1, _positive_int(message.get("num_frames")))
            return
        if message_type == "media_chunk_complete":
            chunk_index = int(message.get("chunk_index", -1))
            if chunk_index < 0:
                raise OutputProtocolError("invalid chunk sequence")
            self.forwarded_completions += 1
            self._chunk_forwarded.setdefault(chunk_index, asyncio.Event()).set()
            self._output_state_changed.set()
            if message.get("is_final_chunk") is True:
                self.final_completion_forwarded = True
                self.final_completion_chunk = chunk_index
                self._final_completion_forwarded.set()

    def task_done(self) -> None:
        self._finish_task()

    def _finish_task(self) -> None:
        if self._unfinished_tasks <= 0:
            raise ValueError("task_done() called too many times")
        self._unfinished_tasks -= 1
        if self._unfinished_tasks == 0:
            self._queue_drained.set()

    async def join(self) -> None:
        await self._queue_drained.wait()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.unbind_output()
        for event in self._chunk_completed.values():
            event.set()
        while self._queue:
            output = self._queue.popleft()
            if output is not None:
                self._queued_media_frames -= output.frame_count
                self._queued_bytes -= len(output.wire)
                self._finish_task()
        self._queue.append(None)
        self._queue_ready.set()
        self._capacity_changed.set()


class GatewayOutputRegistry:
    def __init__(
        self,
        *,
        queue_depth: int = 64,
        enqueue_timeout_s: float = 0.0,
        completion_timeout_s: float = 5.0,
    ) -> None:
        if queue_depth < 1:
            raise ValueError("queue_depth must be positive")
        if enqueue_timeout_s < 0:
            raise ValueError("enqueue_timeout_s must be non-negative")
        if completion_timeout_s <= 0:
            raise ValueError("completion_timeout_s must be positive")
        self.queue_depth = queue_depth
        self.enqueue_timeout_s = enqueue_timeout_s
        self.completion_timeout_s = completion_timeout_s
        self._routes: dict[str, GatewayOutputRoute] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
        trace_id: str = "",
    ) -> GatewayOutputRoute:
        if not session_id or not generation_id or not token:
            raise OutputProtocolError("output route identity is required")
        async with self._lock:
            current = self._routes.get(session_id)
            if current is not None and not current.closed:
                raise OutputProtocolError("session output route is already registered")
            route = GatewayOutputRoute(
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                queue_depth=self.queue_depth,
                enqueue_timeout_s=self.enqueue_timeout_s,
                completion_timeout_s=self.completion_timeout_s,
                trace_id=trace_id,
            )
            self._routes[session_id] = route
            return route

    async def bind(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> GatewayOutputRoute:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None or route.closed:
                raise OutputProtocolError("unknown output route")
            if route.generation_id != generation_id:
                raise OutputProtocolError("stale generation")
            if not route.token_matches(token):
                raise OutputProtocolError("invalid output token")
            if route.bound:
                raise OutputProtocolError("output route is already bound")
            route.bind_output()
            return route

    async def unbind(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> None:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None:
                return
            if route.generation_id != generation_id or not route.token_matches(token):
                return
            route.unbind_output()

    async def unregister(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> None:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None:
                return
            if route.generation_id != generation_id or not route.token_matches(token):
                return
            self._routes.pop(session_id, None)
        await route.close()
