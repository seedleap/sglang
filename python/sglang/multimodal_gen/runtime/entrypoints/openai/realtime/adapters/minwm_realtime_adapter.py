# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_CAMERA_ACTIONS_CONDITION,
    MINWM_CAMERA_INTRINSICS_CONDITION,
    MINWM_PROMPT_UPDATED_CONDITION,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import ControlSignalQueue
from sglang.multimodal_gen.runtime.realtime.states import RealtimeCameraControlState
from sglang.multimodal_gen.runtime.utils.minwm_camera import MOTION_PRIMITIVES

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


MINWM_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 4
COMPOSITE_INPUT_EVENT_KIND = "composite_input"

# Browser KeyboardEvent.key aliases -> minWM motion primitive codes. Canonical
# minWM codes (w/s/a/d/u/dn/i/k/j/l) pass through unchanged.
MINWM_KEY_ALIASES = {
    "ArrowLeft": "j",
    "ArrowRight": "l",
    "ArrowUp": "i",
    "ArrowDown": "k",
    " ": "u",
    "q": "dn",
    "e": "u",
    "Control": "dn",
}


def _normalize_minwm_keys(actions: list[Any]) -> list[str]:
    normalized = []
    for key in actions:
        key = str(key)
        key = MINWM_KEY_ALIASES.get(key, key)
        if key in MOTION_PRIMITIVES:
            normalized.append(key)
    return normalized


class MinWMRealtimeState(RealtimeCameraControlState):
    def __init__(self):
        super().__init__(
            min_pulse_items=1,
            script_maxlen=512,
            max_transitions=512,
            normalize_state_actions=_normalize_minwm_keys,
        )
        self.prompt_queue = ControlSignalQueue(max_events={"prompt": 1})

    def clear(self) -> None:
        super().clear()
        self.prompt_queue.clear()

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        self.prompt_queue.push("prompt", prompt, event_id=event_id)

    def sample_prompt(self) -> str:
        prompt = self.prompt_queue.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        self.latest_sampled_event_id = self.prompt_queue.last_sampled_seq_id("prompt")
        return prompt

    def has_prompt(self) -> bool:
        return self.prompt_queue.has_events("prompt")


class MinWMRealtimeAdapter(BaseRealtimeModelAdapter):
    """Realtime adapter for the MinWM causal world model.

    Latest-wins semantics for both prompt and camera state, matching the
    private minWM serving stack: control state is sampled once per chunk, and
    the per-frame key lists integrate into cumulative camera poses inside the
    pipeline config (``prepare_minwm_camera_chunk``).
    """

    def create_state(self) -> MinWMRealtimeState:
        return MinWMRealtimeState()

    def _state(self, session: GenerateSession) -> MinWMRealtimeState:
        state = session.adapter_state
        if not isinstance(state, MinWMRealtimeState):
            raise TypeError("MinWM realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        condition_inputs = request.condition_inputs or {}
        camera_actions = condition_inputs.get(MINWM_CAMERA_ACTIONS_CONDITION)
        if camera_actions is not None:
            state = self._state(session)
            state.receive_camera_action_script(
                self._validate_camera_actions(camera_actions)
            )

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError(
                "minwm_camera_actions event payload must be list[list[str]]"
            )
        normalized = []
        for frame_actions in payload:
            if not isinstance(frame_actions, list):
                raise ValueError(
                    "minwm_camera_actions event payload must be list[list[str]]"
                )
            normalized.append(_normalize_minwm_keys(frame_actions))
        return normalized

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind == "camera_actions":
            return state.receive_camera_control_event_payload(
                event.payload,
                event_id=event.event_id,
                validate_camera_actions=self._validate_camera_actions,
            )
        elif event.kind == "prompt":
            prompt = self._validate_prompt_payload(event.payload)
            state.receive_prompt(prompt, event_id=event.event_id)
            return f"kind=prompt, prompt_len={len(prompt)}"
        elif event.kind == COMPOSITE_INPUT_EVENT_KIND:
            return self._ingest_composite_input(state, event.payload, event.event_id)
        raise ValueError(f"unsupported event kind: {event.kind}")

    @staticmethod
    def _validate_prompt_payload(payload: Any) -> str:
        if not isinstance(payload, str) or not payload:
            raise ValueError("prompt event payload must be a non-empty string")
        return payload

    def _ingest_composite_input(
        self,
        state: MinWMRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        if not isinstance(payload, dict):
            raise ValueError("composite_input event payload must be a map")
        input_types = payload.get("input_types")
        if not isinstance(input_types, list) or not input_types:
            raise ValueError(
                "composite_input event payload requires non-empty input_types"
            )
        input_logs = []
        for input_type in input_types:
            if input_type not in payload:
                raise ValueError(f"composite_input event payload requires {input_type}")
            if input_type == "camera_actions":
                input_logs.append(
                    state.receive_camera_control_event_payload(
                        payload[input_type],
                        event_id=event_id,
                        validate_camera_actions=self._validate_camera_actions,
                    )
                )
            elif input_type == "prompt":
                prompt = self._validate_prompt_payload(payload[input_type])
                state.receive_prompt(prompt, event_id=event_id)
                input_logs.append(f"kind=prompt, prompt_len={len(prompt)}")
            else:
                raise ValueError(f"unsupported composite_input type: {input_type}")
        return f"kind=composite_input, inputs={input_logs}"

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        prompt_updated = False
        if chunk.index == 0:
            prompt = request.prompt
        elif state.has_prompt():
            prompt = state.sample_prompt()
            request.prompt = prompt
            prompt_updated = True
        else:
            prompt = request.prompt

        condition_inputs: dict[str, Any] = {}
        if prompt_updated:
            condition_inputs[MINWM_PROMPT_UPDATED_CONDITION] = True

        request_conditions = request.condition_inputs or {}
        intrinsics = request_conditions.get(MINWM_CAMERA_INTRINSICS_CONDITION)
        if intrinsics is not None:
            condition_inputs[MINWM_CAMERA_INTRINSICS_CONDITION] = intrinsics

        camera_actions = state.sample_camera_actions(chunk_size)
        if camera_actions is None:
            # Camera pose must keep integrating even while no key is held —
            # the model is unconditionally camera-conditioned. Neutral frames
            # hold the current pose.
            camera_actions = [[] for _ in range(chunk_size)]
        condition_inputs[MINWM_CAMERA_ACTIONS_CONDITION] = camera_actions

        return RealtimeChunkInputs(prompt=prompt, condition_inputs=condition_inputs)

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        temporal_ratio = int(
            server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        num_frames = max(
            int(request.num_frames or 0), (chunk_size - 1) * temporal_ratio + 1
        )

        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=num_frames,
            num_inference_steps=(
                request.num_inference_steps
                or MINWM_REALTIME_DEFAULT_NUM_INFERENCE_STEPS
            ),
            chunk_size=chunk_size,
        )

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._state(session).latest_sampled_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, MinWMRealtimeState):
            state.clear()
