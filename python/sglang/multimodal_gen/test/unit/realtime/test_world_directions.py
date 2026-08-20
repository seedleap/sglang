# SPDX-License-Identifier: Apache-2.0
"""DirectionCoordinator baseline, rewrite, one-time revert, and supersede tests."""

import asyncio

from sglang.multimodal_gen.runtime.realtime.world_directions import (
    DirectionCoordinator,
    parse_init_directions,
)


class Harness:
    """Injected test harness with programmable rewrite delay and failures."""

    def __init__(self, baseline="base", schedule=(), revert_delay_s=0.01):
        self.dispatched = []  # [(prompt, event_id)]
        self.notified = []  # [(event_id, status)]
        self.rewrite_delay = 0.0
        self.rewrite_error = None
        self.change_type = "persistent"
        self.rewrite_calls = 0
        # Set to an asyncio.Lock to emulate the gateway upstream send lock: the
        # dispatch then yields for real, so two applies can interleave at the
        # dispatch point (what upstream backpressure looks like in production).
        self.dispatch_lock = None
        # (event_id, status) -> exception, to emulate a notify timing out while
        # the browser is under backpressure.
        self.notify_errors = {}
        self.dispatch_error = None
        self.coord = DirectionCoordinator(
            baseline=baseline,
            schedule=list(schedule),
            rewrite=self._rewrite,
            dispatch=self._dispatch,
            notify=self._notify,
            revert_delay_s=revert_delay_s,
        )

    async def _rewrite(self, text, baseline):
        self.rewrite_calls += 1
        # Latch the flag when the call starts, so a test flipping it between two
        # calls does not affect the one already in flight.
        error = self.rewrite_error
        if self.rewrite_delay:
            await asyncio.sleep(self.rewrite_delay)
        if error is not None:
            raise error
        return f"rw({text}|{baseline})", self.change_type

    async def _dispatch(self, prompt, event_id):
        if self.dispatch_lock is not None:
            async with self.dispatch_lock:
                await asyncio.sleep(0)
        if self.dispatch_error is not None:
            error, self.dispatch_error = self.dispatch_error, None
            raise error
        self.dispatched.append((prompt, event_id))

    async def _notify(self, event_id, status):
        self.notified.append((event_id, status))
        error = self.notify_errors.pop((event_id, status), None)
        if error is not None:
            raise error


def test_persistent_submit_updates_baseline_and_carries_event_id():
    async def run():
        h = Harness()
        await h.coord.submit(7, "add rain")
        assert h.notified == [(7, "rewriting")]
        assert h.dispatched == [("rw(add rain|base)", 7)]
        assert h.coord.baseline == "rw(add rain|base)"

    asyncio.run(run())


def test_one_time_reverts_to_baseline_without_event_id():
    async def run():
        h = Harness()
        h.change_type = "one_time"
        await h.coord.submit(1, "explosion")
        assert h.coord.baseline == "base"  # one-time effects keep baseline intact
        await asyncio.sleep(0.05)
        # Revert frame: return to baseline without an event_id.
        assert h.dispatched == [("rw(explosion|base)", 1), ("base", None)]

    asyncio.run(run())


def test_newer_submit_supersedes_inflight_rewrite():
    async def run():
        h = Harness()
        h.rewrite_delay = 0.03
        slow = asyncio.ensure_future(h.coord.submit(1, "old"))
        await asyncio.sleep(0.005)
        h.rewrite_delay = 0.0
        await h.coord.submit(2, "new")
        await slow
        # The old rewrite is stale when it completes; only new is dispatched.
        assert h.dispatched == [("rw(new|base)", 2)]
        assert (1, "superseded") in h.notified

    asyncio.run(run())


def test_rewrite_failure_keeps_pending_revert_alive():
    async def run():
        h = Harness(revert_delay_s=0.03)
        h.change_type = "one_time"
        await h.coord.submit(1, "flash")  # schedule a revert
        h.rewrite_error = RuntimeError("model down")
        await h.coord.submit(2, "broken")  # failure must not consume the revert
        assert (2, "failed") in h.notified
        await asyncio.sleep(0.06)
        assert h.dispatched[-1] == ("base", None)  # revert still fires on time

    asyncio.run(run())


def test_schedule_advances_baseline_and_revert_uses_latest():
    async def run():
        h = Harness(schedule=[(3, "storm"), (6, "calm")])
        h.change_type = "one_time"
        h.coord.observe_chunk(4)  # pass chunk 3
        assert h.coord.baseline == "storm"
        await h.coord.submit(1, "lightning")
        h.coord.observe_chunk(6)  # timeline advances while revert is pending
        await asyncio.sleep(0.05)
        assert h.dispatched[-1] == ("calm", None)  # revert reads latest baseline

    asyncio.run(run())


def test_skill_apply_supersedes_inflight_and_one_time_reverts():
    async def run():
        h = Harness()
        h.rewrite_delay = 0.03
        slow = asyncio.ensure_future(h.coord.submit(1, "old direction"))
        await asyncio.sleep(0.005)
        await h.coord.apply("skill prompt", "one_time", 2)  # player used a skill
        await slow
        assert (1, "superseded") in h.notified  # later action supersedes rewrite
        await asyncio.sleep(0.05)
        assert h.dispatched == [("skill prompt", 2), ("base", None)]

    asyncio.run(run())


def test_persistent_apply_cancels_pending_revert():
    async def run():
        h = Harness(revert_delay_s=0.02)
        await h.coord.apply("burst", "one_time", 1)
        await h.coord.apply("night mode", "persistent", 2)  # take over scene state
        await asyncio.sleep(0.05)
        assert ("base", None) not in h.dispatched  # old revert was canceled
        assert h.coord.baseline == "night mode"

    asyncio.run(run())


def test_close_cancels_revert_timer():
    async def run():
        h = Harness(revert_delay_s=0.02)
        await h.coord.apply("burst", "one_time", 1)
        h.coord.close()
        await asyncio.sleep(0.05)
        assert h.dispatched == [("burst", 1)]

    asyncio.run(run())


def test_bad_event_id_types_are_dropped_from_dispatch():
    async def run():
        h = Harness()
        await h.coord.apply("p", "persistent", "42")  # non-int would be rejected
        await h.coord.apply("q", "persistent", True)  # bool is rejected too
        assert h.dispatched == [("p", None), ("q", None)]

    asyncio.run(run())


def test_parse_init_directions():
    msg = {
        "prompt": "seed",
        "condition_inputs": {
            "minwm_prompt_schedule": [
                {"target_chunk": 6, "prompt": "b", "kind": "prompt"},
                {"target_chunk": 3, "prompt": "a", "kind": "prompt"},
                {"target_chunk": "x", "prompt": "bad"},
                "garbage",
            ]
        },
    }
    baseline, schedule = parse_init_directions(msg)
    assert baseline == "seed"
    assert schedule == [(6, "b"), (3, "a")]  # sorting happens in coordinator
    assert parse_init_directions({}) == ("", [])


def test_defensive_empty_schedule_sorted_in_coordinator():
    coord = DirectionCoordinator(
        baseline="s",
        schedule=[(6, "b"), (3, "a")],
        rewrite=None,
        dispatch=None,
        notify=None,
    )
    coord.observe_chunk(4)
    assert coord.baseline == "a"
    coord.observe_chunk(9)
    assert coord.baseline == "b"


# ---- Interleaving at the dispatch point ----


async def _interleave_two_applies(h, first, second):
    """Block the dispatch, start two applies, let both reach it, then release."""
    h.dispatch_lock = asyncio.Lock()
    await h.dispatch_lock.acquire()
    t1 = asyncio.ensure_future(h.coord.apply(*first))
    t2 = asyncio.ensure_future(h.coord.apply(*second))
    for _ in range(4):  # let both tasks reach the dispatch point
        await asyncio.sleep(0)
    h.dispatch_lock.release()
    await asyncio.gather(t1, t2)


def test_interleaved_one_time_applies_leave_single_revert():
    """Two one-time effects interleaving must leave exactly one revert timer.

    Registering after the dispatch means the later apply cancels None and the
    first timer becomes an orphan, firing a stray revert ten seconds later.
    """

    async def run():
        h = Harness(revert_delay_s=0.05)
        h.change_type = "one_time"
        await _interleave_two_applies(
            h, ("burstA", "one_time", 1), ("burstB", "one_time", 2)
        )
        await asyncio.sleep(0.12)
        reverts = [d for d in h.dispatched if d[1] is None]
        assert reverts == [("base", None)], h.dispatched

    asyncio.run(run())


def test_interleaved_persistent_apply_cancels_earlier_one_time():
    """Once a persistent prompt owns the scene there must be no revert frame."""

    async def run():
        h = Harness(revert_delay_s=0.05)
        await _interleave_two_applies(
            h, ("burst", "one_time", 1), ("night", "persistent", 2)
        )
        await asyncio.sleep(0.12)
        assert [d for d in h.dispatched if d[1] is None] == [], h.dispatched
        assert h.coord.baseline == "night"

    asyncio.run(run())


def test_close_after_interleaved_apply_leaves_no_timer():
    """No timer may outlive the session: orphans dispatch to a closed upstream."""

    async def run():
        h = Harness(revert_delay_s=0.05)
        await _interleave_two_applies(
            h, ("burstA", "one_time", 1), ("burstB", "one_time", 2)
        )
        h.coord.close()
        before = len(h.dispatched)
        await asyncio.sleep(0.12)
        assert len(h.dispatched) == before, h.dispatched[before:]

    asyncio.run(run())


def test_failed_dispatch_cancels_its_own_revert():
    """An effect frame that never went out must not get a revert frame."""

    async def run():
        h = Harness(revert_delay_s=0.03)
        h.dispatch_error = ConnectionError("upstream closed")
        try:
            await h.coord.apply("burst", "one_time", 1)
        except ConnectionError:
            pass
        await asyncio.sleep(0.08)
        assert h.dispatched == [], h.dispatched

    asyncio.run(run())


# ---- Single flight: a burst must not fan out into N model calls ----


def test_burst_submits_collapse_to_first_and_last():
    """Five rapid instructions must call the model twice: in-flight plus newest.

    The superseded ones in between skip the dispatch as well, which saves both
    intermediate frames and wasted model calls.
    """

    async def run():
        h = Harness()
        h.rewrite_delay = 0.01
        await asyncio.gather(*[h.coord.submit(i, f"t{i}") for i in range(1, 6)])
        assert h.rewrite_calls == 2, h.rewrite_calls
        assert h.dispatched == [("rw(t5|base)", 5)], h.dispatched
        # A queued instruction still needs an immediate "rewriting" so the input
        # box does not look unresponsive.
        assert (5, "rewriting") in h.notified
        for dropped in (1, 2, 3, 4):
            assert (dropped, "superseded") in h.notified, dropped

    asyncio.run(run())


def test_pending_survives_rewrite_failure():
    """A failing in-flight rewrite must not swallow the queued instruction."""

    async def run():
        h = Harness()
        h.rewrite_delay = 0.01
        h.rewrite_error = RuntimeError("model down")
        first = asyncio.ensure_future(h.coord.submit(1, "boom"))
        await asyncio.sleep(0.002)
        h.rewrite_error = None
        await h.coord.submit(2, "ok")
        await first
        assert (1, "failed") in h.notified
        assert h.dispatched == [("rw(ok|base)", 2)], h.dispatched

    asyncio.run(run())


def test_close_drops_pending():
    """Closing the session drops the queued instruction instead of running it."""

    async def run():
        h = Harness()
        h.rewrite_delay = 0.01
        first = asyncio.ensure_future(h.coord.submit(1, "a"))
        await asyncio.sleep(0.002)
        await h.coord.submit(2, "b")
        h.coord.close()
        await first
        assert h.rewrite_calls == 1, h.rewrite_calls
        assert h.dispatched == [], h.dispatched

    asyncio.run(run())


# ---- Ownership: no queued instruction while nobody is in flight ----


def test_notify_failure_does_not_leave_an_ownerless_pending():
    """A notify timeout that breaks the in-flight round must drop the queue.

    Otherwise the queued instruction never reaches a terminal status, and it
    supersedes the *newer* instruction before taking its place: the player's
    latest words lose to words from minutes ago.
    """

    async def run():
        h = Harness()
        h.rewrite_delay = 0.01
        h.notify_errors[(1, "superseded")] = TimeoutError("browser backpressure")
        first = asyncio.ensure_future(h.coord.submit(1, "a"))
        await asyncio.sleep(0.002)
        await h.coord.submit(2, "b")  # queued
        try:
            await first  # the notify timeout escapes here
        except TimeoutError:
            pass
        assert h.coord._pending is None, h.coord._pending

        # The next instruction must be the one that gets dispatched, rather than
        # losing to the stale queued one.
        await h.coord.submit(3, "c")
        assert h.dispatched == [("rw(c|base)", 3)], h.dispatched

    asyncio.run(run())


def test_dispatch_failure_does_not_leave_an_ownerless_pending():
    """Same for a failing dispatch with an instruction queued behind it."""

    async def run():
        h = Harness()
        h.dispatch_lock = asyncio.Lock()
        await h.dispatch_lock.acquire()  # hold #1 at the dispatch point
        first = asyncio.ensure_future(h.coord.submit(1, "a"))
        for _ in range(4):
            await asyncio.sleep(0)
        await h.coord.submit(2, "b")  # queued while #1 is blocked
        h.dispatch_error = ConnectionError("upstream closed")
        h.dispatch_lock.release()
        try:
            await first
        except ConnectionError:
            pass
        assert h.coord._pending is None, h.coord._pending

        h.dispatch_lock = None
        await h.coord.submit(3, "c")
        assert h.dispatched == [("rw(c|base)", 3)], h.dispatched

    asyncio.run(run())
