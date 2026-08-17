# SPDX-License-Identifier: Apache-2.0

"""Self-contained Gemini prompt rewriter for the realtime World Studio UI.

Credentials are read only by this server process. The browser API receives the
rewritten prompt and its lifetime classification, never credential material.
"""

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = (
    ROOT.parent / "realtime_webui_secrets" / "prompt-rewriter-vertex.json"
)
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_LOCATION = "global"
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class ChangeType(str, Enum):
    """How long the rewritten state should remain active."""

    PERSISTENT = "persistent"
    ONE_TIME = "one_time"


class PromptRewriteOutput(BaseModel):
    """Strict response sent back to the World Studio browser."""

    prompt: str = Field(
        min_length=1,
        description=(
            "Detailed standalone rewritten video-generation prompt in English, "
            "normally 180-320 words."
        ),
    )
    change_type: ChangeType

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"propertyOrdering": ["prompt", "change_type"]},
    )


class WorldRuleCompletionOutput(BaseModel):
    """A display label and prepared prompt completed from one rule input."""

    name: str = Field(
        min_length=1,
        max_length=28,
        description="Concise UI label for the skill button or achieved reward.",
    )
    prompt: str = Field(
        min_length=1,
        description="Detailed standalone rewritten video-generation prompt in English.",
    )
    change_type: ChangeType

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "propertyOrdering": ["name", "prompt", "change_type"]
        },
    )


SYSTEM_PROMPT = """Rewrite PREVIOUS PROMPT using EDIT INSTRUCTION and return the schema only.

Write one detailed, standalone English video prompt in this exact semantic order: viewpoint/camera first; main subject second; scene/environment third. Target 180-320 English words for a normal edit, using several substantial paragraphs when useful. Do not compress the result into a short summary. Favor explicit visual detail and continuity constraints over brevity, but avoid redundant restatement.

First determine the full visual state described by PREVIOUS PROMPT. Then apply only EDIT INSTRUCTION. Preserve every unspecified element, but do not merely say "everything else remains unchanged." Explicitly name the important unchanged elements and state how their identity, appearance, position, scale, motion, framing, and lighting remain stable when applicable. Never redesign an unchanged landmark, subject, prop, costume, or composition merely to fit the new environment.

For every affected element, state all applicable facts clearly: what is visibly present before the change; whether it is replaced, transformed, removed completely, newly added, or temporarily activated; exactly what it becomes or what replaces it; which old visual traits disappear completely and must no longer be visible; the new material, shape, color, texture, condition, position, scale, motion, lighting, or atmospheric traits; and how neighboring elements and the overall composition respond to the change.

Never leave a changed category implicit. For an environment edit, audit terrain/topography, ground surfaces, vegetation and trees, water and waterfalls, weather, air/mist/clouds/particles, architecture and landmarks, background depth, lighting, shadows, and color palette. For a subject edit, audit species/identity, anatomy, skin/fur/scales, head, limbs, wings/horns/tail, clothing/equipment/reins, pose, motion, silhouette, screen position, and interaction with the rider or scene. If a feature is removed, say it is completely absent and no longer visible. For an added action or event, describe its entrance, location, appearance, scale, motion, effects, and relationship to existing subjects without accidentally replacing them.

Resolve all contradictions. Do not preserve a trait that the requested change removes. Do not invent unrelated changes. The generated prompt must not use the words "reference," "source," "previous," "original," "input," or "instruction" anywhere. It must not describe the generation or comparison process. Describe visual transitions directly in present tense. Do not use headings, bullets, markdown, commentary, or alternatives inside the generated prompt.

Set change_type=persistent for a continuing state edit: environment, weather, time, lighting, viewpoint, composition, subject identity/type/appearance, clothing, equipment, style, layout, or another stable visual state. Set change_type=one_time for a transient shot action/event: summoning, casting or releasing a skill, attacking, jumping, exploding, or transforming as an action beat. Summons and skill releases are always one_time even when visible throughout this shot. Choose exactly one type."""


WORLD_RULE_SYSTEM_PROMPT = """Complete one interactive world rule and return the schema only.

RULE INPUT may be either a short interface label or a longer visual instruction. Infer whichever side is missing. Return both a concise name and one detailed standalone English video-generation prompt. Do not ask questions and do not return alternatives.

For kind=skill, name is the short action label shown on a gameplay button. For kind=goal, name is the concise reward or achievement shown in the success popup. Prefer 2-8 Chinese characters when RULE INPUT is Chinese; otherwise use 2-4 short words. Never use generic labels such as “技能”, “动作”, “目标”, or “成功”.

Write prompt in this semantic order: viewpoint/camera first; main subject second; scene/environment third. Preserve every unspecified visual fact in WORLD PROMPT. If RULE INPUT is only a name, invent the most direct visible action or reward manifestation that matches it. If it is already a detailed instruction, preserve its intent and infer only the concise name. For goal rules, make the named reward or achievement visibly appear in the scene so the trigger has a concrete visual result.

The prompt must be a complete current visual state rather than an edit command. State what changes and explicitly retain important unchanged subjects, composition, environment, lighting, and camera continuity. Resolve contradictions and do not invent unrelated changes. Do not use headings, bullets, markdown, commentary, or alternatives inside prompt.

Set change_type=persistent for continuing world, weather, lighting, viewpoint, composition, identity, appearance, equipment, or layout changes. Set change_type=one_time for a transient action, skill release, summon, attack, jump, explosion, transformation beat, reward appearance, or achievement event. Choose exactly one type."""


def build_user_message(previous_prompt: str, instruction: str) -> str:
    """Build a clearly delimited model input without altering either value."""

    normalized_prompt = previous_prompt.strip()
    normalized_instruction = instruction.strip()
    if not normalized_prompt:
        raise ValueError("previous_prompt must not be empty")
    if not normalized_instruction:
        raise ValueError("instruction must not be empty")
    return (
        "PREVIOUS PROMPT:\n"
        f"<previous_prompt>\n{normalized_prompt}\n</previous_prompt>\n\n"
        "EDIT INSTRUCTION:\n"
        f"<edit_instruction>\n{normalized_instruction}\n</edit_instruction>"
    )


def build_world_rule_message(
    previous_prompt: str, rule_input: str, kind: str
) -> str:
    """Build a delimited input for a skill or goal completion."""

    normalized_prompt = previous_prompt.strip()
    normalized_input = rule_input.strip()
    normalized_kind = kind.strip().lower()
    if not normalized_prompt:
        raise ValueError("previous_prompt must not be empty")
    if not normalized_input:
        raise ValueError("rule_input must not be empty")
    if normalized_kind not in {"skill", "goal"}:
        raise ValueError("kind must be skill or goal")
    return (
        f"RULE KIND: {normalized_kind}\n\n"
        "WORLD PROMPT:\n"
        f"<world_prompt>\n{normalized_prompt}\n</world_prompt>\n\n"
        "RULE INPUT:\n"
        f"<rule_input>\n{normalized_input}\n</rule_input>"
    )


def _extract_payload(response: Any) -> Any:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    text = getattr(response, "text", None)
    if text:
        return text
    raise ValueError("Gemini returned no readable structured content")


def _validate_output(payload: Any) -> PromptRewriteOutput:
    if isinstance(payload, PromptRewriteOutput):
        return payload
    elif isinstance(payload, BaseModel):
        return PromptRewriteOutput.model_validate(payload.model_dump())
    elif isinstance(payload, (bytes, bytearray)):
        return PromptRewriteOutput.model_validate_json(payload.decode("utf-8"))
    elif isinstance(payload, str):
        return PromptRewriteOutput.model_validate_json(payload)
    return PromptRewriteOutput.model_validate(payload)


def _validate_world_rule_output(payload: Any) -> WorldRuleCompletionOutput:
    if isinstance(payload, WorldRuleCompletionOutput):
        return payload
    if isinstance(payload, BaseModel):
        return WorldRuleCompletionOutput.model_validate(payload.model_dump())
    if isinstance(payload, (bytes, bytearray)):
        return WorldRuleCompletionOutput.model_validate_json(
            payload.decode("utf-8")
        )
    if isinstance(payload, str):
        return WorldRuleCompletionOutput.model_validate_json(payload)
    return WorldRuleCompletionOutput.model_validate(payload)


class PromptRewriter:
    """One warm Vertex Gemini client shared by all rewrite HTTP requests."""

    def __init__(
        self,
        *,
        model: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        credentials_path: str | Path | None = None,
        request_timeout_seconds: float | None = None,
        max_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get(
            "VIDEO_PROMPT_REWRITE_MODEL", DEFAULT_MODEL
        )
        self.location = location or os.environ.get(
            "VIDEO_PROMPT_REWRITE_VERTEX_LOCATION", DEFAULT_LOCATION
        )
        configured_credentials = credentials_path or os.environ.get(
            "VIDEO_PROMPT_REWRITE_CREDENTIALS"
        )
        self.credentials_path = Path(
            configured_credentials or DEFAULT_CREDENTIALS_PATH
        ).expanduser()
        self.project_id = project_id or os.environ.get(
            "VIDEO_PROMPT_REWRITE_PROJECT_ID", ""
        )
        self.request_timeout_seconds = float(
            request_timeout_seconds
            or os.environ.get("VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS", "20")
        )
        self.max_attempts = max(1, int(max_attempts))
        self._client = client
        self._client_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return self._client is not None or self.credentials_path.is_file()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            if not self.credentials_path.is_file():
                raise RuntimeError(
                    "Prompt rewriter credential is not configured in this project"
                )
            try:
                from google import genai
                from google.oauth2 import service_account
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is required for the prompt rewriter"
                ) from exc

            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=[VERTEX_SCOPE]
            )
            project_id = self.project_id or credentials.project_id
            if not project_id:
                try:
                    payload = json.loads(
                        self.credentials_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    payload = {}
                project_id = str(payload.get("project_id", "")).strip()
            if not project_id:
                raise RuntimeError("Vertex project ID is not configured")
            self.project_id = project_id
            self._client = genai.Client(
                vertexai=True,
                project=project_id,
                location=self.location,
                credentials=credentials,
            )
            return self._client

    async def rewrite(
        self, instruction: str, previous_prompt: str
    ) -> PromptRewriteOutput:
        """Rewrite once, retrying only malformed structured model output."""

        from google.genai import types

        client = await self._get_client()
        user_message = build_user_message(previous_prompt, instruction)
        attempt_message = user_message
        for _attempt in range(1, self.max_attempts + 1):
            # API failures and timeouts are latency-sensitive failures: surface them
            # immediately instead of silently paying for another model call.
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=[attempt_message],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                        response_json_schema=PromptRewriteOutput.model_json_schema(),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                            include_thoughts=False,
                        ),
                    ),
                ),
                timeout=self.request_timeout_seconds,
            )
            try:
                return _validate_output(_extract_payload(response))
            except (TypeError, ValueError) as exc:
                if _attempt == self.max_attempts:
                    raise RuntimeError(
                        "Gemini returned invalid structured output after "
                        f"{self.max_attempts} attempt(s)"
                    ) from exc
                attempt_message = (
                    user_message
                    + "\n\nSTRUCTURE CORRECTION FOR THE NEXT ATTEMPT:\n"
                    + str(exc)
                    + "\nReturn valid JSON with exactly prompt and change_type. "
                    + "change_type must be persistent or one_time."
                )
        raise AssertionError("unreachable")

    async def complete_world_rule(
        self, rule_input: str, previous_prompt: str, kind: str
    ) -> WorldRuleCompletionOutput:
        """Infer a rule label and prepared prompt in one latency-sensitive call."""

        from google.genai import types

        client = await self._get_client()
        user_message = build_world_rule_message(
            previous_prompt, rule_input, kind
        )
        attempt_message = user_message
        for attempt in range(1, self.max_attempts + 1):
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=[attempt_message],
                    config=types.GenerateContentConfig(
                        system_instruction=WORLD_RULE_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                        response_json_schema=(
                            WorldRuleCompletionOutput.model_json_schema()
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                            include_thoughts=False,
                        ),
                    ),
                ),
                timeout=self.request_timeout_seconds,
            )
            try:
                return _validate_world_rule_output(_extract_payload(response))
            except (TypeError, ValueError) as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        "Gemini returned invalid world rule output after "
                        f"{self.max_attempts} attempt(s)"
                    ) from exc
                attempt_message = (
                    user_message
                    + "\n\nSTRUCTURE CORRECTION FOR THE NEXT ATTEMPT:\n"
                    + str(exc)
                    + "\nReturn valid JSON with exactly name, prompt, and "
                    + "change_type. change_type must be persistent or one_time."
                )
        raise AssertionError("unreachable")
