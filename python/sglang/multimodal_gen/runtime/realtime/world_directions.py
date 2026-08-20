"""玩家指令（direction）编排：基线跟踪、改写调度、一次性还原。

引擎只接受「完整的当前画面状态」（kind:"prompt"），而玩家发来的是编辑指令
（"让鲨鱼游近一点"）——直接转发会把画面里未提及的元素全部抹掉。本模块维护
会话内的**当前基线**（当前生效的完整画面描述），把玩家原话连同基线交给
改写服务换取新的完整描述再派发给引擎；一次性（one_time）效果到点自动
发回基线还原。技能释放走同一条 apply 通道（技能提示词在创作期已是完整
描述，等价于"预改写好的 direction"）。

纯编排、只依赖 stdlib asyncio：改写、派发、通知全部由构造方注入，
单测不需要拉起网关。所有方法都在同一事件循环内调用，不需要锁。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

CHANGE_PERSISTENT = "persistent"
CHANGE_ONE_TIME = "one_time"

# 一次性效果的还原延迟。与现网 webui 的 restoreDelayMs=10000 一致：
# 足够引擎把效果演完一拍，又不至于让"爆发"永久挂在画面上。
DEFAULT_REVERT_DELAY_S = 10.0


class DirectionCoordinator:
    """单个会话的指令编排器。

    生死同会话：init 验证通过后创建，会话 finally 里 close()。

    基线演进的三个来源（谁最后派发谁就是基线）：
    1. init 的 prompt（种子）；
    2. 时间轴条目 —— 引擎按 schedule 自主派发，网关通过路过的
       chunk_telemetry 的 chunk_index 观察推进（observe_chunk）；
    3. persistent 改写/技能 —— apply 成功即更新。
    one_time 不动基线，只排一个还原定时器；还原**到点才取基线**，
    这样定时期间时间轴推进过也不会还原到过时画面。
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
        self._schedule = sorted(schedule)  # [(target_chunk, prompt)] 升序
        self._pos = 0
        self._rewrite = rewrite  # async (原话, 基线) -> (新完整描述, change_type)
        self._dispatch = dispatch  # async (完整描述, event_id|None) -> 发给引擎
        self._notify = notify  # async (event_id, status) -> direction_status 给浏览器
        self._revert_delay_s = revert_delay_s
        self._gen = 0  # 代数：谁最新谁说了算（取代在途改写）
        self._revert_task: asyncio.Task | None = None
        self._closed = False

    @property
    def baseline(self) -> str:
        return self._baseline

    def observe_chunk(self, chunk_index: int) -> None:
        """引擎已推进到 chunk_index：把时间轴上已派发的条目滚进基线。

        schedule 里一次性条目的合成还原也是普通条目，"最后派发的就是
        当前画面"对它们一并成立，不需要区别对待。
        """
        while (
            self._pos < len(self._schedule)
            and self._schedule[self._pos][0] <= chunk_index
        ):
            self._baseline = self._schedule[self._pos][1]
            self._pos += 1

    async def submit(self, event_id: Any, text: str) -> None:
        """一次玩家输入的完整旅程：rewriting → 改写 → apply。"""
        self._gen += 1
        gen = self._gen
        await self._notify(event_id, "rewriting")
        try:
            prompt, change_type = await self._rewrite(text, self._baseline)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 失败不动基线、不动已排定的还原（还原会把画面拉回基线，
            # 正是失败后该有的样子）；具体原因由 rewrite 注入方记日志
            await self._notify(event_id, "failed")
            return
        if gen != self._gen or self._closed:
            # 改写期间玩家又说了新的：旧结果作废，画面听最新的
            await self._notify(event_id, "superseded")
            return
        await self.apply(prompt, change_type, event_id)

    async def apply(self, prompt: str, change_type: str, event_id: Any) -> None:
        """把一条完整描述派发给引擎并接管画面状态（direction 与技能共用）。"""
        self._gen += 1  # 最新动作胜出：在途的旧改写完成后自动作废
        self._cancel_revert()  # 新内容接管画面，旧的一次性还原不再有意义
        await self._dispatch(prompt, _as_event_id(event_id))
        if change_type == CHANGE_ONE_TIME:
            self._revert_task = asyncio.get_running_loop().create_task(
                self._revert_later()
            )
        else:
            self._baseline = prompt

    async def _revert_later(self) -> None:
        await asyncio.sleep(self._revert_delay_s)
        # 还原帧不带 event_id：它不是玩家某次输入的产物，
        # 带上会让前端把"已完成"的输入又标成"生效中"
        await self._dispatch(self._baseline, None)

    def _cancel_revert(self) -> None:
        task, self._revert_task = self._revert_task, None
        if task is not None and not task.done():
            task.cancel()

    def close(self) -> None:
        self._closed = True
        self._cancel_revert()


def _as_event_id(value: Any) -> int | None:
    # 引擎的 RealtimeEvent.event_id 是可选 int，其它类型会被 pydantic 拒掉
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_init_directions(init_message: dict) -> tuple[str, list[tuple[int, str]]]:
    """从（已解封的）init 消息里取基线种子与时间轴。容错：字段缺失按空处理。"""
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
