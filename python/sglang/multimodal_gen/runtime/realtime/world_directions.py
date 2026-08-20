"""Player direction orchestration: baseline tracking, rewrite, and revert.

The engine only accepts the complete current visual state (kind:"prompt"), while
players send edit instructions such as "move the shark closer". Forwarding those
edits directly would erase any scene details not mentioned by the player. This
module keeps the session's current baseline prompt, sends the user instruction
and baseline to the rewrite service, dispatches the rewritten full prompt to the
engine, and automatically reverts one_time effects back to the baseline. Skill
activation uses the same apply path because skill prompts are authored as full
scene descriptions.

This module is pure orchestration over stdlib asyncio. Rewrite, dispatch, and
notification hooks are injected by the caller, so unit tests do not need to boot
the gateway.

Concurrency contract: one event loop, but submit and apply each run in their own
task and therefore interleave at every await. "Same loop" does not mean "cannot
interleave". State updates that span an await must be completed inside one
await-free block (claim the slot, cancel the old timer, register the new one),
or a task arriving later observes an unregistered intermediate state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

CHANGE_PERSISTENT = "persistent"
CHANGE_ONE_TIME = "one_time"

# Revert delay for one-time effects. This matches the production web UI
# restoreDelayMs=10000: long enough for the effect to land, but short enough to
# keep transient effects from becoming permanent scene state.
DEFAULT_REVERT_DELAY_S = 10.0


class DirectionCoordinator:
    """Direction orchestrator for one session.

    The object lives and dies with the session: it is created after init
    validation and closed in the session finally block.

    The baseline evolves from three sources, where the latest dispatched prompt
    wins:
    1. the init prompt seed;
    2. timeline entries dispatched by the engine according to schedule, observed
       by the gateway through chunk_telemetry chunk_index updates;
    3. persistent rewrites or skills, once apply succeeds.
    one_time effects do not mutate the baseline; they only schedule a revert.
    The revert reads the baseline when it fires, so timeline progress during the
    timer window does not revert to stale scene state.
    """

    def __init__(
        self,
        *,
        baseline: str,
        schedule: list[tuple[int, str]],
        rewrite: Callable[[str, str], Awaitable[tuple[str, str]]],
        dispatch: Callable[[str, int | None], Awaitable[None]],
        notify: Callable[[Any, str], Awaitable[None]],
        revert_delay_s: float = DEFAULT_REVERT_DELAY_S,
    ) -> None:
        self._baseline = baseline
        self._schedule = sorted(schedule)  # [(target_chunk, prompt)] ascending
        self._pos = 0
        self._rewrite = rewrite  # async (raw_text, baseline) -> (full_prompt, type)
        self._dispatch = dispatch  # async (full_prompt, event_id|None) -> engine
        self._notify = notify  # async (event_id, status) -> direction_status
        self._revert_delay_s = revert_delay_s
        self._gen = 0  # generation counter: newest action supersedes old rewrites
        self._revert_task: asyncio.Task | None = None
        self._closed = False
        # Single flight: one rewrite per session at a time, keeping only the
        # newest queued instruction. Only the last one can take effect anyway,
        # so calling the model for the ones in between is pure waste - and the
        # model account is shared with judging and world creation, so that waste
        # starves settlement.
        self._rewriting = False
        self._pending: tuple[Any, str] | None = None

    @property
    def baseline(self) -> str:
        return self._baseline

    def observe_chunk(self, chunk_index: int) -> None:
        """Fold dispatched timeline entries into the baseline up to chunk_index.

        Synthetic revert entries in the schedule are normal timeline entries, so
        "latest dispatched prompt is the current scene" applies to them too.
        """
        while (
            self._pos < len(self._schedule)
            and self._schedule[self._pos][0] <= chunk_index
        ):
            self._baseline = self._schedule[self._pos][1]
            self._pos += 1

    async def submit(self, event_id: Any, text: str) -> None:
        """Run one player instruction through rewriting and apply.

        Single flight with a one-slot tail queue: while a rewrite is in flight,
        only the newest instruction is queued and the one it replaces is closed
        out immediately. The observable semantics are identical to "newest
        wins"; the difference is that the replaced instructions never reach the
        model, so a burst of typing no longer fans out into N concurrent calls.
        """
        # Claiming the slot must sit in the same await-free block as the check.
        # With an await in between, two tasks both observe _rewriting=False and
        # both enter.
        if self._rewriting:
            dropped, self._pending = self._pending, (event_id, text)
            await self._notify(event_id, "rewriting")
            if dropped is not None:
                await self._notify(dropped[0], "superseded")
            return
        self._rewriting = True
        try:
            await self._notify(event_id, "rewriting")
            while True:
                await self._rewrite_once(event_id, text)
                if self._pending is None or self._closed:
                    return
                event_id, text = self._pending
                self._pending = None
        finally:
            # The queued instruction is owned by whoever holds the flight slot.
            # The invariant is "no pending instruction while nobody is in
            # flight". Both notify and dispatch inside the loop can raise (a
            # browser under backpressure makes notify time out); clearing only
            # _rewriting would leave an ownerless queued instruction that never
            # reaches a terminal status and later supersedes the *newer*
            # instruction before taking its place.
            self._rewriting = False
            self._pending = None

    async def _rewrite_once(self, event_id: Any, text: str) -> None:
        """Rewrite and land one instruction."""
        self._gen += 1
        gen = self._gen
        try:
            prompt, change_type = await self._rewrite(text, self._baseline)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed rewrite does not change the baseline or pending revert.
            # The revert still returns the scene to the baseline, which is the
            # intended fallback state. The rewrite provider logs the root cause.
            await self._notify(event_id, "failed")
            return
        if gen != self._gen or self._pending is not None or self._closed:
            # A newer player action arrived while rewriting; discard the old
            # result. When the newer one is already queued we also skip the
            # dispatch, which saves one intermediate frame.
            await self._notify(event_id, "superseded")
            return
        await self.apply(prompt, change_type, event_id)

    async def apply(self, prompt: str, change_type: str, event_id: Any) -> None:
        """Dispatch one full prompt and take ownership of scene state."""
        self._gen += 1  # newest action wins; in-flight old rewrites become stale
        self._cancel_revert()  # new content owns the scene; old one-time revert is stale
        mine: asyncio.Task | None = None
        if change_type == CHANGE_ONE_TIME:
            # Registration must share the await-free block with the cancel
            # above. Assigning after the dispatch means an interleaved apply
            # cancels None, and this timer becomes an orphan that nobody can
            # cancel and close() cannot reach: it fires a stray revert frame ten
            # seconds later and outlives the session. Registering early only
            # starts the sleep a few milliseconds sooner; _revert_later still
            # reads the baseline after sleeping, so it reads the latest one.
            mine = asyncio.get_running_loop().create_task(self._revert_later())
            self._revert_task = mine
        try:
            await self._dispatch(prompt, _as_event_id(event_id))
        except BaseException:
            # The effect frame never went out, so it must not get a revert
            # frame. Compare by task identity so an interleaved newer timer that
            # already took the slot is not cancelled by mistake.
            if mine is not None and self._revert_task is mine:
                self._cancel_revert()
            raise
        if change_type != CHANGE_ONE_TIME:
            # The baseline is assigned after the dispatch on purpose: only what
            # actually went out counts, so a failed dispatch never records a
            # prompt the engine never received.
            self._baseline = prompt

    async def _revert_later(self) -> None:
        await asyncio.sleep(self._revert_delay_s)
        # Revert frames carry no event_id: they are not the result of one player
        # input. Attaching an event_id would make the UI mark a completed input
        # as active again.
        await self._dispatch(self._baseline, None)

    def _cancel_revert(self) -> None:
        task, self._revert_task = self._revert_task, None
        if task is not None and not task.done():
            task.cancel()

    def close(self) -> None:
        self._closed = True
        # Drop the queued instruction too, otherwise the in-flight loop runs one
        # more rewrite after the session is gone.
        self._pending = None
        self._cancel_revert()


def _as_event_id(value: Any) -> int | None:
    # Engine RealtimeEvent.event_id is an optional int; pydantic rejects others.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_init_directions(init_message: dict) -> tuple[str, list[tuple[int, str]]]:
    """Extract baseline seed and timeline entries from the unsealed init message."""
    baseline = str(init_message.get("prompt") or "")
    schedule: list[tuple[int, str]] = []
    cond = init_message.get("condition_inputs")
    entries = cond.get("minwm_prompt_schedule") if isinstance(cond, dict) else None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        chunk = entry.get("target_chunk")
        prompt = entry.get("prompt")
        if isinstance(chunk, int) and isinstance(prompt, str) and prompt:
            schedule.append((chunk, prompt))
    return baseline, schedule
