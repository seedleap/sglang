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
the gateway. All methods run in one event loop, so no lock is required.
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
        """Run one player instruction through rewriting and apply."""
        self._gen += 1
        gen = self._gen
        await self._notify(event_id, "rewriting")
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
        if gen != self._gen or self._closed:
            # A newer player action arrived while rewriting; discard the old result.
            await self._notify(event_id, "superseded")
            return
        await self.apply(prompt, change_type, event_id)

    async def apply(self, prompt: str, change_type: str, event_id: Any) -> None:
        """Dispatch one full prompt and take ownership of scene state."""
        self._gen += 1  # newest action wins; in-flight old rewrites become stale
        self._cancel_revert()  # new content owns the scene; old one-time revert is stale
        await self._dispatch(prompt, _as_event_id(event_id))
        if change_type == CHANGE_ONE_TIME:
            self._revert_task = asyncio.get_running_loop().create_task(
                self._revert_later()
            )
        else:
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
