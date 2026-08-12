# SPDX-License-Identifier: Apache-2.0

"""Self-contained Gemini prompt rewriter for the realtime World Studio UI.

Credentials are read only by this server process. The browser API receives the
rewritten prompt and its lifetime classification, never credential material.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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


SYSTEM_PROMPT = """Rewrite PREVIOUS PROMPT using EDIT INSTRUCTION and return the schema only.

Write one detailed, standalone English video prompt in this exact semantic order: viewpoint/camera first; main subject second; scene/environment third. Target 180-320 English words for a normal edit, using several substantial paragraphs when useful. Do not compress the result into a short summary. Favor explicit visual detail and continuity constraints over brevity, but avoid redundant restatement.

First determine the full visual state described by PREVIOUS PROMPT. Then apply only EDIT INSTRUCTION. Preserve every unspecified element, but do not merely say "everything else remains unchanged." Explicitly name the important unchanged elements and state how their identity, appearance, position, scale, motion, framing, and lighting remain stable when applicable. Never redesign an unchanged landmark, subject, prop, costume, or composition merely to fit the new environment.

For every affected element, state all applicable facts clearly: what is visibly present before the change; whether it is replaced, transformed, removed completely, newly added, or temporarily activated; exactly what it becomes or what replaces it; which old visual traits disappear completely and must no longer be visible; the new material, shape, color, texture, condition, position, scale, motion, lighting, or atmospheric traits; and how neighboring elements and the overall composition respond to the change.

Never leave a changed category implicit. For an environment edit, audit terrain/topography, ground surfaces, vegetation and trees, water and waterfalls, weather, air/mist/clouds/particles, architecture and landmarks, background depth, lighting, shadows, and color palette. For a subject edit, audit species/identity, anatomy, skin/fur/scales, head, limbs, wings/horns/tail, clothing/equipment/reins, pose, motion, silhouette, screen position, and interaction with the rider or scene. If a feature is removed, say it is completely absent and no longer visible. For an added action or event, describe its entrance, location, appearance, scale, motion, effects, and relationship to existing subjects without accidentally replacing them.

Resolve all contradictions. Do not preserve a trait that the requested change removes. Do not invent unrelated changes. The generated prompt must not use the words "reference," "source," "previous," "original," "input," or "instruction" anywhere. It must not describe the generation or comparison process. Describe visual transitions directly in present tense. Do not use headings, bullets, markdown, commentary, or alternatives inside the generated prompt.

Set change_type=persistent for a continuing state edit: environment, weather, time, lighting, viewpoint, composition, subject identity/type/appearance, clothing, equipment, style, layout, or another stable visual state. Set change_type=one_time for a transient shot action/event: summoning, casting or releasing a skill, attacking, jumping, exploding, or transforming as an action beat. Summons and skill releases are always one_time even when visible throughout this shot. Choose exactly one type."""


_FORBIDDEN_PROCESS_WORDS = re.compile(
    r"\b(reference|source|previous|original|input|instruction)\b",
    flags=re.IGNORECASE,
)
_PROCESS_WORD_REPLACEMENTS = {
    "reference": "depicted",
    "source": "origin",
    "previous": "prior",
    "original": "established",
    "input": "provided",
    "instruction": "request",
}


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
        result = payload
    elif isinstance(payload, BaseModel):
        result = PromptRewriteOutput.model_validate(payload.model_dump())
    elif isinstance(payload, (bytes, bytearray)):
        result = PromptRewriteOutput.model_validate_json(payload.decode("utf-8"))
    elif isinstance(payload, str):
        result = PromptRewriteOutput.model_validate_json(payload)
    else:
        result = PromptRewriteOutput.model_validate(payload)
    forbidden = sorted(
        {
            match.group(0).lower()
            for match in _FORBIDDEN_PROCESS_WORDS.finditer(result.prompt)
        }
    )
    if forbidden:
        raise ValueError(
            "remove these process-oriented words: " + ", ".join(forbidden)
        )
    return result


def _remove_forbidden_process_words(result: PromptRewriteOutput) -> PromptRewriteOutput:
    """Apply a narrow final repair when Gemini uses a forbidden transition word."""

    prompt = _FORBIDDEN_PROCESS_WORDS.sub(
        lambda match: _PROCESS_WORD_REPLACEMENTS[match.group(0).lower()],
        result.prompt,
    )
    return result.model_copy(update={"prompt": prompt})


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
        """Rewrite one instruction, retrying only invalid/failed model output."""

        from google.genai import types

        client = await self._get_client()
        user_message = build_user_message(previous_prompt, instruction)
        attempt_message = user_message
        last_error: Exception | None = None
        for _attempt in range(1, self.max_attempts + 1):
            try:
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
                payload = _extract_payload(response)
                try:
                    return _validate_output(payload)
                except ValueError:
                    if _attempt == self.max_attempts:
                        if isinstance(payload, PromptRewriteOutput):
                            result = payload
                        elif isinstance(payload, BaseModel):
                            result = PromptRewriteOutput.model_validate(
                                payload.model_dump()
                            )
                        elif isinstance(payload, str):
                            result = PromptRewriteOutput.model_validate_json(payload)
                        else:
                            result = PromptRewriteOutput.model_validate(payload)
                        return _validate_output(
                            _remove_forbidden_process_words(result)
                        )
                    raise
            except Exception as exc:  # SDK exception types change between releases.
                last_error = exc
                attempt_message = (
                    user_message
                    + "\n\nQUALITY CORRECTION FOR THE NEXT ATTEMPT:\n"
                    + str(exc)
                    + "\nRewrite the complete prompt and correct every listed issue."
                )
        raise RuntimeError(
            f"Gemini prompt rewrite failed after {self.max_attempts} attempt(s)"
        ) from last_error
