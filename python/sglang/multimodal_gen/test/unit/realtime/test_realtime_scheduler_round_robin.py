# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    ReplaceQueuedRealtimeReq,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _chunk(session_id: str, chunk_index: int) -> Req:
    request = Req.__new__(Req)
    request.request_id = f"{session_id}-{chunk_index}"
    request.realtime_session_id = session_id
    request.realtime_generation_id = f"generation-{session_id}"
    request.block_idx = chunk_index
    request.realtime_action_version = 0
    request.realtime_prompt_version = 0
    return request


def _scheduler(policy: str) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(model_id="minwm", model_path=None)
    scheduler.waiting_queue = deque()
    scheduler._realtime_scheduling_policy = policy
    scheduler._realtime_rr_sessions = []
    scheduler._realtime_rr_last_session = None
    scheduler._pending_realtime_replacements = {}
    scheduler._dispatched_realtime_requests = {}
    scheduler._batch_metrics_enabled = False
    scheduler._dynamic_batching_enabled = lambda: False
    scheduler._observe_realtime_scheduler_queue = lambda *_args, **_kwargs: None
    return scheduler


def _enqueue(scheduler: Scheduler, *requests) -> None:
    for request in requests:
        identity = getattr(request, "request_id", type(request).__name__).encode()
        scheduler.waiting_queue.append((identity, request, 1.0))


def _next(scheduler: Scheduler):
    items = scheduler.get_next_batch_to_run()
    assert items is not None and len(items) == 1
    return items[0][1]


def test_realtime_scheduling_cli_defaults_to_fifo_and_accepts_round_robin():
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)

    defaults, unknown = parser.parse_known_args(["--model-path", "/fake"])
    assert unknown == []
    assert defaults.realtime_scheduling_policy == "fifo"

    enabled, unknown = parser.parse_known_args(
        [
            "--model-path",
            "/fake",
            "--realtime-scheduling-policy",
            "session_round_robin",
        ]
    )
    assert unknown == []
    assert enabled.realtime_scheduling_policy == "session_round_robin"

    with pytest.raises(SystemExit):
        parser.parse_known_args(
            [
                "--model-path",
                "/fake",
                "--realtime-scheduling-policy",
                "not-a-policy",
            ]
        )


def test_default_fifo_policy_preserves_global_arrival_order():
    scheduler = _scheduler("fifo")
    requests = [_chunk("A", 0), _chunk("A", 1), _chunk("B", 0), _chunk("B", 1)]
    _enqueue(scheduler, *requests)

    assert [_next(scheduler).request_id for _ in requests] == [
        "A-0",
        "A-1",
        "B-0",
        "B-1",
    ]


def test_round_robin_alternates_two_sessions_one_whole_chunk_at_a_time():
    scheduler = _scheduler("session_round_robin")
    requests = [_chunk("A", 0), _chunk("A", 1), _chunk("B", 0), _chunk("B", 1)]
    _enqueue(scheduler, *requests)

    assert [_next(scheduler).request_id for _ in requests] == [
        "A-0",
        "B-0",
        "A-1",
        "B-1",
    ]


def test_round_robin_has_no_starvation_across_three_backlogged_sessions():
    scheduler = _scheduler("session_round_robin")
    requests = [
        *(_chunk("A", index) for index in range(3)),
        *(_chunk("B", index) for index in range(3)),
        *(_chunk("C", index) for index in range(3)),
    ]
    _enqueue(scheduler, *requests)

    assert [_next(scheduler).request_id for _ in requests] == [
        "A-0",
        "B-0",
        "C-0",
        "A-1",
        "B-1",
        "C-1",
        "A-2",
        "B-2",
        "C-2",
    ]


def test_round_robin_late_join_and_idle_sessions_are_work_conserving():
    scheduler = _scheduler("session_round_robin")
    _enqueue(scheduler, _chunk("A", 0))
    assert _next(scheduler).request_id == "A-0"

    # B joins after A's turn and is selected before A receives another turn.
    _enqueue(scheduler, _chunk("A", 1), _chunk("B", 0))
    assert _next(scheduler).request_id == "B-0"

    # A is idle while C joins; neither idle A nor temporarily idle B blocks C.
    _enqueue(scheduler, _chunk("C", 0))
    assert _next(scheduler).request_id == "C-0"

    # The scheduler immediately uses whichever Session becomes runnable again.
    assert _next(scheduler).request_id == "A-1"


def test_round_robin_never_reorders_across_control_or_non_realtime_barrier():
    scheduler = _scheduler("session_round_robin")
    scheduler._dynamic_batching_enabled = lambda: True
    scheduler._realtime_rr_sessions = ["A", "B"]
    scheduler._realtime_rr_last_session = "A"
    release = ReleaseRealtimeSessionReq(session_id="A")
    non_realtime = Req.__new__(Req)
    non_realtime.request_id = "ordinary"
    non_realtime.realtime_session_id = None
    _enqueue(
        scheduler,
        _chunk("A", 0),
        _chunk("A", 1),
        release,
        _chunk("B", 0),
        non_realtime,
        _chunk("B", 1),
    )

    assert _next(scheduler).request_id == "A-0"
    assert _next(scheduler).request_id == "A-1"
    assert _next(scheduler) is release
    assert _next(scheduler).request_id == "B-0"
    assert _next(scheduler) is non_realtime
    assert _next(scheduler).request_id == "B-1"


def test_release_removes_session_and_preserves_round_robin_cursor():
    class Worker:
        def release_realtime_session(self, session_id):
            assert session_id == "B"
            return OutputBatch()

    scheduler = _scheduler("session_round_robin")
    scheduler.worker = Worker()
    scheduler._realtime_rr_sessions = ["A", "B", "C"]
    scheduler._realtime_rr_last_session = "B"

    scheduler._handle_release_realtime_session(
        [ReleaseRealtimeSessionReq(session_id="B")]
    )
    _enqueue(scheduler, _chunk("A", 0), _chunk("C", 0))

    assert scheduler._realtime_rr_sessions == ["A", "C"]
    assert _next(scheduler).request_id == "C-0"


def test_failed_release_still_cleans_cancelled_session_from_rotation():
    class Worker:
        def release_realtime_session(self, _session_id):
            raise RuntimeError("worker already stopped")

    scheduler = _scheduler("session_round_robin")
    scheduler.worker = Worker()
    scheduler._realtime_rr_sessions = ["A", "B"]
    scheduler._realtime_rr_last_session = "A"

    with pytest.raises(RuntimeError, match="already stopped"):
        scheduler._handle_release_realtime_session(
            [ReleaseRealtimeSessionReq(session_id="A")]
        )

    assert scheduler._realtime_rr_sessions == ["B"]


def test_realtime_round_robin_dispatch_never_enables_dynamic_batching():
    scheduler = _scheduler("session_round_robin")
    scheduler._dynamic_batching_enabled = lambda: True
    _enqueue(scheduler, _chunk("A", 0), _chunk("B", 0))

    items = scheduler.get_next_batch_to_run()

    assert items is not None and len(items) == 1
    assert items[0][1].request_id == "A-0"
    assert len(scheduler.waiting_queue) == 1


def test_queued_replacement_keeps_original_session_turn_and_uses_new_version():
    scheduler = _scheduler("session_round_robin")
    original = _chunk("A", 0)
    original.realtime_action_version = 1
    replacement = _chunk("A", 0)
    replacement.realtime_action_version = 3
    _enqueue(
        scheduler,
        original,
        _chunk("A", 1),
        _chunk("B", 0),
    )

    assert scheduler._replace_waiting_realtime_request(
        ReplaceQueuedRealtimeReq(
            session_id="A",
            generation_id="generation-A",
            chunk_index=0,
            request_id="A-0",
            replacement=replacement,
        )
    )

    first = _next(scheduler)
    assert first is replacement
    assert first.realtime_action_version == 3
    assert _next(scheduler).request_id == "B-0"
    assert _next(scheduler).request_id == "A-1"


def test_pending_replacement_before_request_does_not_create_phantom_ring_turn():
    scheduler = _scheduler("session_round_robin")
    replies = []
    scheduler.return_result = (
        lambda output, identity, should_not_return: replies.append(
            (output.output, identity, should_not_return)
        )
    )
    original = _chunk("A", 0)
    replacement = _chunk("A", 0)
    replacement.realtime_prompt_version = 2
    update = ReplaceQueuedRealtimeReq(
        session_id="A",
        generation_id="generation-A",
        chunk_index=0,
        request_id="A-0",
        replacement=replacement,
    )

    scheduler._enqueue_received_reqs([(b"update", update)], now=100.0)
    assert scheduler._realtime_rr_sessions == []

    scheduler._enqueue_received_reqs(
        [(b"B", _chunk("B", 0)), (b"A", original)],
        now=100.1,
    )

    assert scheduler._realtime_rr_sessions == ["B", "A"]
    assert _next(scheduler).request_id == "B-0"
    assert _next(scheduler) is replacement
    assert replacement.realtime_prompt_version == 2
    assert replies == [
        (
            {
                "replaced": False,
                "buffered": True,
                "too_late": False,
                "invalid": False,
            },
            b"update",
            False,
        )
    ]
