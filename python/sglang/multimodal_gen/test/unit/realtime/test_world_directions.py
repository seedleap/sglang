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
        self.coord = DirectionCoordinator(
            baseline=baseline,
            schedule=list(schedule),
            rewrite=self._rewrite,
            dispatch=self._dispatch,
            notify=self._notify,
            revert_delay_s=revert_delay_s,
        )

    async def _rewrite(self, text, baseline):
        if self.rewrite_delay:
            await asyncio.sleep(self.rewrite_delay)
        if self.rewrite_error is not None:
            raise self.rewrite_error
        return f"rw({text}|{baseline})", self.change_type

    async def _dispatch(self, prompt, event_id):
        self.dispatched.append((prompt, event_id))

    async def _notify(self, event_id, status):
        self.notified.append((event_id, status))


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
