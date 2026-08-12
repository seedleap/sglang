# SPDX-License-Identifier: Apache-2.0

"""Create a complete realtime world from a short description or first frame."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from prompt_rewriter import DEFAULT_CREDENTIALS_PATH, VERTEX_SCOPE


ROOT = Path(__file__).resolve().parent
SECRETS_ROOT = ROOT.parent / "realtime_webui_secrets"
GENERATED_ROOT = ROOT.parent / "realtime_webui_generated"
DEFAULT_IMAGE_CONFIG_PATH = SECRETS_ROOT / "world-image-model-config.json"
DESCRIPTION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_IMAGE_SIZE = "1536x1024"
DEFAULT_IMAGE_QUALITY = "medium"
FINAL_IMAGE_SIZE = (1280, 720)
IMAGE_CONFIG_NAME = "azure/gpt-image-2"


class WorldDescriptionOutput(BaseModel):
    world_description: str = Field(min_length=1)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"propertyOrdering": ["world_description"]},
    )


class CompletedWorld(BaseModel):
    world_description: str
    image_bytes: bytes | None = None
    image_generated: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


WORLD_DESCRIPTION_SYSTEM_PROMPT = """Create one detailed, standalone English visual world description. Return the schema only.

Write in this exact semantic order:
1. Viewpoint and camera: explicitly name first-person, third-person, over-the-shoulder, aerial, eye-level, low-angle, or another precise perspective; state camera height, direction, framing, visible foreground elements, and motion when applicable.
2. Main subject: describe identity/species, anatomy or appearance, clothing/equipment, materials and textures, pose/action, scale, screen position, silhouette, and relationship to the camera.
3. Scene and environment: describe terrain and ground, vegetation, water, weather, atmosphere and particles, architecture or landmarks, foreground/midground/background depth, lighting direction and quality, shadows, color palette, mood, and visual style.

Preserve every explicit user detail and every visibly supported image detail. When both seed text and an image are supplied, combine them without contradicting the visible composition. Add coherent details needed to form a complete world, but do not invent named characters, brands, text, logos, or unrelated story events. If the seed explicitly specifies a viewpoint, preserve it exactly. If viewpoint is unspecified, select the single clearest viewpoint and state it explicitly.

Target 180-320 English words. Use concrete visible facts suitable both as an image-generation prompt and as the persistent description of the world. Do not use headings, bullets, markdown, alternatives, or explanatory commentary inside world_description."""


def _parse_world_description(response: Any) -> str:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        parsed = parsed.model_dump()
    if isinstance(parsed, dict):
        return WorldDescriptionOutput.model_validate(
            parsed
        ).world_description.strip()
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return WorldDescriptionOutput.model_validate_json(
            text
        ).world_description.strip()
    raise RuntimeError("Gemini returned no world description")


def _resource_endpoint(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Azure image endpoint must not be empty")
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid Azure image endpoint")
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalize_generated_image(image_bytes: bytes) -> bytes:
    """Crop the model's landscape response to the realtime 16:9 first frame."""

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        fitted = ImageOps.fit(
            rgb,
            FINAL_IMAGE_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = io.BytesIO()
        fitted.save(output, format="PNG", optimize=True)
        return output.getvalue()


class WorldCreator:
    """Warm Gemini description client plus project-local GPT Image 2 config."""

    def __init__(
        self,
        *,
        gemini_client_provider: Callable[[], Awaitable[Any]] | None = None,
        image_client: Any | None = None,
        credentials_path: str | Path | None = None,
        image_config_path: str | Path | None = None,
        generated_root: str | Path | None = None,
        request_timeout_seconds: float | None = None,
    ) -> None:
        self.description_model = os.environ.get(
            "CREATE_WORLD_DESCRIPTION_MODEL", DESCRIPTION_MODEL
        )
        self.vertex_location = os.environ.get(
            "CREATE_WORLD_VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION
        )
        self.credentials_path = Path(
            credentials_path
            or os.environ.get("CREATE_WORLD_CREDENTIALS", DEFAULT_CREDENTIALS_PATH)
        ).expanduser()
        self.image_config_path = Path(
            image_config_path
            or os.environ.get(
                "CREATE_WORLD_IMAGE_CONFIG", DEFAULT_IMAGE_CONFIG_PATH
            )
        ).expanduser()
        self.generated_root = Path(generated_root or GENERATED_ROOT).expanduser()
        self.request_timeout_seconds = float(
            request_timeout_seconds
            or os.environ.get("CREATE_WORLD_TIMEOUT_SECONDS", "180")
        )
        self._gemini_client_provider = gemini_client_provider
        self._gemini_client = None
        self._gemini_lock = asyncio.Lock()
        self._image_client = image_client
        self._image_config: dict[str, Any] | None = None

    @property
    def description_configured(self) -> bool:
        return self._gemini_client_provider is not None or self.credentials_path.is_file()

    @property
    def image_configured(self) -> bool:
        return self._image_client is not None or self.image_config_path.is_file()

    async def _get_gemini_client(self) -> Any:
        if self._gemini_client_provider is not None:
            return await self._gemini_client_provider()
        if self._gemini_client is not None:
            return self._gemini_client
        async with self._gemini_lock:
            if self._gemini_client is not None:
                return self._gemini_client
            if not self.credentials_path.is_file():
                raise RuntimeError("World description credential is not configured")
            from google import genai
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=[VERTEX_SCOPE]
            )
            project_id = os.environ.get("CREATE_WORLD_PROJECT_ID") or credentials.project_id
            if not project_id:
                raise RuntimeError("World description Vertex project is not configured")
            self._gemini_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=self.vertex_location,
                credentials=credentials,
            )
            return self._gemini_client

    async def describe(
        self,
        seed_text: str,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/png",
    ) -> str:
        from google.genai import types

        normalized_text = seed_text.strip()
        if not normalized_text and not image_bytes:
            raise ValueError("A world description or first frame is required")
        contents: list[Any] = []
        if normalized_text and image_bytes:
            contents.append(
                "Create a complete visual world by combining this seed text "
                "with every visible detail in the supplied first frame:\n"
                f"<world_seed>\n{normalized_text}\n</world_seed>"
            )
        elif normalized_text:
            contents.append(
                "Create the complete visual world described by this text:\n"
                f"<world_seed>\n{normalized_text}\n</world_seed>"
            )
        else:
            contents.append(
                "Describe the complete visible world as a standalone visual state."
            )
        if image_bytes:
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
            )
        client = await self._get_gemini_client()
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=self.description_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=WORLD_DESCRIPTION_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_json_schema=WorldDescriptionOutput.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                        include_thoughts=False,
                    ),
                ),
            ),
            timeout=min(self.request_timeout_seconds, 30),
        )
        return _parse_world_description(response)

    def _load_image_config(self) -> dict[str, Any]:
        if self._image_config is not None:
            return self._image_config
        if not self.image_config_path.is_file():
            raise RuntimeError("World image model is not configured in this project")
        try:
            payload = json.loads(self.image_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Unable to read the project world image config") from exc
        config = payload.get(IMAGE_CONFIG_NAME, payload)
        if not isinstance(config, dict):
            raise RuntimeError("World image model config must be a JSON object")
        self._image_config = config
        return config

    def _get_image_client(self) -> tuple[Any, dict[str, Any]]:
        config = self._load_image_config()
        if self._image_client is not None:
            return self._image_client, config
        from openai import AzureOpenAI

        api_key = os.environ.get("CREATE_WORLD_IMAGE_API_KEY") or config.get(
            "api_key"
        )
        base_url = os.environ.get("CREATE_WORLD_IMAGE_ENDPOINT") or (
            config.get("client_args") or {}
        ).get("base_url")
        api_version = os.environ.get("CREATE_WORLD_IMAGE_API_VERSION") or config.get(
            "api_version"
        )
        if not api_key or not base_url or not api_version:
            raise RuntimeError("World image model credentials are incomplete")
        self._image_client = AzureOpenAI(
            api_key=str(api_key),
            azure_endpoint=_resource_endpoint(str(base_url)),
            api_version=str(api_version),
            timeout=self.request_timeout_seconds,
        )
        return self._image_client, config

    def _generate_image_sync(self, world_description: str) -> bytes:
        client, config = self._get_image_client()
        image_prompt = (
            world_description
            + " Render a single cinematic opening frame in a wide 16:9 landscape "
            "composition. Keep important subjects and landmarks safely inside the "
            "central frame. Do not add text, captions, borders, or logos."
        )
        response = client.images.generate(
            model=str(config.get("model_name") or "gpt-image-2"),
            prompt=image_prompt,
            size=os.environ.get("CREATE_WORLD_IMAGE_SIZE", DEFAULT_IMAGE_SIZE),
            quality=os.environ.get(
                "CREATE_WORLD_IMAGE_QUALITY", DEFAULT_IMAGE_QUALITY
            ),
            n=1,
        )
        items = getattr(response, "data", None) or []
        if not items:
            raise RuntimeError("World image model returned no image")
        first = items[0]
        encoded = getattr(first, "b64_json", None)
        if not encoded and isinstance(first, dict):
            encoded = first.get("b64_json")
        if not encoded:
            raise RuntimeError("World image response did not contain image bytes")
        try:
            return _normalize_generated_image(base64.b64decode(encoded, validate=True))
        except ValueError as exc:
            raise RuntimeError("World image response contained invalid image data") from exc

    async def complete(
        self,
        seed_text: str,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/png",
    ) -> CompletedWorld:
        description = await self.describe(seed_text, image_bytes, image_mime_type)
        if image_bytes:
            return CompletedWorld(
                world_description=description,
                image_bytes=None,
                image_generated=False,
            )
        generated = await asyncio.to_thread(self._generate_image_sync, description)
        return CompletedWorld(
            world_description=description,
            image_bytes=generated,
            image_generated=True,
        )

    def save_generated_image(self, image_bytes: bytes) -> str:
        self.generated_root.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex
        target = self.generated_root / f"{image_id}.png"
        target.write_bytes(image_bytes)
        return image_id

    def generated_image_path(self, image_id: str) -> Path | None:
        if len(image_id) != 32 or any(c not in "0123456789abcdef" for c in image_id):
            return None
        path = self.generated_root / f"{image_id}.png"
        return path if path.is_file() else None
