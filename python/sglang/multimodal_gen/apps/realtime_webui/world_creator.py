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
from typing import Any, Awaitable, Callable, Literal
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
    camera_mode: Literal["first_person", "third_person"] = Field(
        description=(
            "For an uploaded image, third_person only when a clear external playable "
            "subject is visible; otherwise first_person."
        )
    )
    source_image_has_clear_external_subject: bool = Field(
        description=(
            "Whether the source image visibly contains a distinct external playable "
            "person, creature, rider, or vehicle separate from the camera."
        )
    )
    visible_first_person_body_parts: list[str] = Field(
        description=(
            "Only player body parts visibly present in a first-person source image; "
            "empty for third-person or when no body parts are visible."
        )
    )
    world_description: str = Field(min_length=1)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "propertyOrdering": [
                "source_image_has_clear_external_subject",
                "camera_mode",
                "visible_first_person_body_parts",
                "world_description",
            ]
        },
    )


class CompletedWorld(BaseModel):
    world_description: str
    image_bytes: bytes | None = None
    image_generated: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GameplayImageValidation(BaseModel):
    camera_mode_matches: bool
    avatar_orientation_valid: bool
    camera_distance_and_scale_valid: bool
    continuous_walkable_route: bool
    explorable_space: bool
    not_poster_or_portrait: bool
    passed: bool
    feedback: str

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "propertyOrdering": [
                "camera_mode_matches",
                "avatar_orientation_valid",
                "camera_distance_and_scale_valid",
                "continuous_walkable_route",
                "explorable_space",
                "not_poster_or_portrait",
                "passed",
                "feedback",
            ]
        },
    )


WORLD_DESCRIPTION_SYSTEM_PROMPT = """Create one detailed, standalone English description for the opening state of an explorable 3D game world. Return the schema only.

When a source image is supplied, visually classify its gameplay perspective before writing. Do not default every uploaded image to third-person:
- Set source_image_has_clear_external_subject=true only when the image visibly contains a clear external playable subject separate from the camera, such as a recognizable person, creature, rider, or vehicle that can serve as the controlled avatar. A landscape landmark, building, incidental crowd, distant indistinct figure, first-person hands/forearms/legs, held equipment, vehicle cockpit, handlebars, dashboard, or mount edge does not count as an external subject.
- If the source image has a clear external playable subject, use third_person. Preserve that subject's supported identity and appearance, but convert the gameplay composition so the subject is seen from behind and centered in the frame.
- If the source image has no clear external playable subject, use first_person. Do not invent a third-person avatar merely because the seed text mentions a person or role.
- Image evidence determines the perspective for an uploaded image. Seed text may clarify identity and world details but must not override the source image's first-person-versus-third-person evidence.
- When there is no source image, honor an explicitly requested perspective; otherwise retain the normal third-person default.

For visible_first_person_body_parts, return a concise list of only the player body parts actually visible in a first-person source image, including side/count/appearance when supported, for example ["both gloved hands", "left forearm"] or ["knees and boots"]. Do not include equipment or vehicle parts. Return [] when no player body part is visible or when camera_mode is third_person.

The camera must use exactly one standard gameplay perspective:
- first_person: an eye-level forward gameplay view. The world_description must explicitly state whether player body parts are visible. If visible_first_person_body_parts is empty, include the exact fact that no player body parts are visible in the frame. Otherwise, describe every listed visible body part, its screen-edge position, appearance, and relation to held equipment without inventing hidden anatomy. No external player avatar is visible.
- third_person: a trailing follow camera 4-7 meters behind the playable subject. The subject must be centered in the frame on the central vertical axis, occupy only about 18-28% of frame height, and face directly away from the camera toward the route ahead. Explicitly state that the subject is centered in the frame. Never show the subject's face, front, three-quarter front, portrait pose, or close-up.

Use an explicitly requested first- or third-person perspective. If none is requested, choose third_person. Convert cinematic, portrait, poster, front-facing, aerial, side-profile, or close-up framing into one of these two playable perspectives.

Every world must be immediately traversable on foot. Describe continuous walkable ground beginning at the bottom edge/player position and extending through the midground into the background. Include a clearly readable path, road, trail, corridor, street, bridge, or broad traversable terrain with forward space and at least one reachable area or route choice. Keep the path unobstructed and wide enough for movement. Do not place the player at a cliff edge, on an isolated pedestal, in open flight, behind an impassable wall, or facing only water/void. Compose landmarks around and beyond the route so the environment invites exploration rather than functioning as a static backdrop.

Write in this exact semantic order:
1. Viewpoint and camera: explicitly name first-person or third-person and follow the matching gameplay rules above; state camera height, direction, framing, visible foreground elements, and forward movement space. For first-person, explicitly say whether body parts are visible and identify every actually visible part. For third-person, explicitly say the subject is centered in the frame.
2. Main subject: for third-person, describe identity/species, anatomy or appearance, clothing/equipment, materials and textures, pose/action, scale, centered screen position, silhouette, and relationship to the camera. For first-person, describe only image-supported body parts and held equipment; if no body parts are visible, state that clearly and do not invent an avatar.
3. Traversal and environment: first describe the continuous walkable route from foreground to background and its reachable destinations or branches, then describe terrain and ground, vegetation, water, weather, atmosphere and particles, architecture or landmarks, foreground/midground/background depth, lighting direction and quality, shadows, color palette, mood, and visual style.

Preserve explicit user details and visibly supported image details unless they conflict with the mandatory gameplay camera or traversability rules. When both seed text and an image are supplied, combine them while converting unsuitable framing into a playable composition. Add coherent details needed to form a complete world, but do not invent named characters, brands, text, logos, or unrelated story events.

Target 180-320 English words. Use concrete visible facts suitable both as an image-generation prompt and as the persistent description of the world. Do not use headings, bullets, markdown, alternatives, or explanatory commentary inside world_description."""


FIRST_PERSON_IMAGE_RULES = """MANDATORY FIRST-PERSON GAMEPLAY CAMERA: Render an eye-level forward in-engine gameplay view from approximately 1.6 meters above the ground. Do not show an external third-person avatar, face, portrait, or posed hero. Player-attached hands, forearms, knees, lower legs, or feet may appear only when the world description explicitly says they are visible; preserve the stated body parts and place them naturally at the lower or peripheral frame without blocking the route. If the description says no player body parts are visible, show none. Held equipment or a vehicle edge may appear along the bottom edge. The playable ground begins at the bottom center and a broad unobstructed walkable route continues clearly through the midground toward reachable background destinations."""


THIRD_PERSON_IMAGE_RULES = """MANDATORY THIRD-PERSON GAMEPLAY CAMERA: Render a trailing follow-camera view 4-7 meters behind the playable subject. Show the subject's back facing the camera and their head, torso, and feet oriented directly toward the route ahead. Never show their face, front, three-quarter front, side-profile hero pose, or over-the-shoulder close-up. Keep the full-body subject precisely centered on the frame's vertical centerline, with the body midpoint close to the visual center, and relatively small at approximately 18-28% of frame height, leaving generous navigable space around them. The playable ground begins beneath the subject and a broad unobstructed walkable route continues clearly through the midground toward reachable background destinations."""


SHARED_GAMEPLAY_IMAGE_RULES = """This must look like a playable 3D game screenshot, not key art, concept art, a movie poster, a character portrait, or a scenic establishing shot. Prioritize spatial readability and traversal: show continuous ground, clear depth, a forward route wide enough to walk along, and at least one visible reachable area or branch to explore. Do not begin at a cliff edge, isolated platform, open flight, impassable barrier, water-only foreground, or dead end. Do not let any character dominate the frame. Use a wide landscape composition with no text, captions, UI, borders, or logos."""


GAMEPLAY_IMAGE_VALIDATION_PROMPT = """Strictly review this generated opening frame for an explorable 3D game world. Do not reward visual beauty. It passes only if every requirement is visibly satisfied.

For first_person: it must be an eye-level forward gameplay view with no external player avatar or visible face. Player-attached hands, forearms, knees, lower legs, or feet are allowed only when required by the world description and must match its explicit body-part visibility statement. If the description says no body parts are visible, any visible player anatomy fails.
For third_person: the full playable subject must be precisely centered on the frame's vertical centerline with its body midpoint close to the visual center, seen from behind, facing directly away toward the route, approximately 18-28% of frame height, with a clear 4-7 meter follow-camera feeling. An off-center subject, visible face, front, three-quarter front, dominant close-up, portrait pose, or hero pose fails.

For both modes: continuous walkable ground must begin at the player position/bottom of frame and lead through the midground toward a reachable background area. There must be a broad readable path, street, corridor, bridge, trail, or traversable terrain, plus enough free space and depth to explore. A cliff edge, isolated platform, open flight, water-only foreground, impassable barrier, scenic-only vista, poster, or portrait fails.

Set passed=true only when all six boolean checks are true. feedback must be a concise English image-regeneration instruction that names every visible failure; use an empty string when passed."""


def _parse_world_description(response: Any) -> WorldDescriptionOutput:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        parsed = parsed.model_dump()
    if isinstance(parsed, dict):
        result = WorldDescriptionOutput.model_validate(parsed)
        result.world_description = result.world_description.strip()
        return result
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        result = WorldDescriptionOutput.model_validate_json(text)
        result.world_description = result.world_description.strip()
        return result
    raise RuntimeError("Gemini returned no world description")


def _enforce_uploaded_image_perspective(
    result: WorldDescriptionOutput,
) -> WorldDescriptionOutput:
    """Make the visual subject classification authoritative for uploaded frames."""

    description = result.world_description.strip()
    if result.source_image_has_clear_external_subject:
        result.camera_mode = "third_person"
        result.visible_first_person_body_parts = []
        composition = (
            "A third-person trailing gameplay view keeps the clear external "
            "playable subject centered in the frame on the central vertical axis, "
            "seen from behind and facing the route ahead."
        )
    else:
        result.camera_mode = "first_person"
        parts = [
            part.strip()
            for part in result.visible_first_person_body_parts
            if part.strip()
        ]
        result.visible_first_person_body_parts = parts
        if parts:
            visible_parts = ", ".join(parts)
            body_visibility = (
                f"The player's visible body parts are {visible_parts}; no other "
                "player anatomy is visible."
            )
        else:
            body_visibility = "No player body parts are visible in the frame."
        composition = (
            "A first-person eye-level forward gameplay view has no external player "
            f"avatar. {body_visibility}"
        )
    result.world_description = f"{composition} {description}"
    return result


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
        self.image_generation_attempts = max(
            1, int(os.environ.get("CREATE_WORLD_IMAGE_ATTEMPTS", "2"))
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
    ) -> WorldDescriptionOutput:
        from google.genai import types

        normalized_text = seed_text.strip()
        if not normalized_text and not image_bytes:
            raise ValueError("A world description or first frame is required")
        contents: list[Any] = []
        if normalized_text and image_bytes:
            contents.append(
                "First inspect the supplied first frame and decide whether it has "
                "a clear external playable subject. Use third_person only when it "
                "does; otherwise use first_person. For first_person, inventory only "
                "the player body parts actually visible in the image and make the "
                "final description explicitly state whether any are visible. For "
                "third_person, make the external subject centered in the frame. "
                "Then create a complete visual world by combining the image with "
                "this seed text:\n"
                f"<world_seed>\n{normalized_text}\n</world_seed>"
            )
        elif normalized_text:
            contents.append(
                "Create the complete visual world described by this text:\n"
                f"<world_seed>\n{normalized_text}\n</world_seed>"
            )
        else:
            contents.append(
                "Inspect the supplied first frame before writing. Use third_person "
                "only if it contains a clear external playable subject; otherwise "
                "use first_person. In first_person, explicitly describe whether the "
                "player's body parts are visible and list only image-supported body "
                "parts. In third_person, keep the external subject centered in the "
                "frame. Describe the complete visible world as a standalone visual "
                "state."
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
        result = _parse_world_description(response)
        if image_bytes:
            result = _enforce_uploaded_image_perspective(result)
        return result

    async def validate_gameplay_image(
        self, image_bytes: bytes, camera_mode: str
    ) -> GameplayImageValidation:
        from google.genai import types

        client = await self._get_gemini_client()
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=self.description_model,
                contents=[
                    f"Required camera mode: {camera_mode}\n\n"
                    + GAMEPLAY_IMAGE_VALIDATION_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_json_schema=GameplayImageValidation.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                        include_thoughts=False,
                    ),
                ),
            ),
            timeout=min(self.request_timeout_seconds, 30),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump()
        if isinstance(parsed, dict):
            return GameplayImageValidation.model_validate(parsed)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return GameplayImageValidation.model_validate_json(text)
        raise RuntimeError("Gemini returned no gameplay image validation")

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

    def _generate_image_sync(
        self,
        world_description: str,
        camera_mode: str,
        retry_feedback: str = "",
    ) -> bytes:
        client, config = self._get_image_client()
        camera_rules = (
            FIRST_PERSON_IMAGE_RULES
            if camera_mode == "first_person"
            else THIRD_PERSON_IMAGE_RULES
        )
        image_prompt = (
            f"{camera_rules}\n\n{SHARED_GAMEPLAY_IMAGE_RULES}\n\n"
            "WORLD CONTENT:\n"
            + world_description
            + "\n\nObey the mandatory gameplay camera and traversal rules even if the "
            "world content suggests a conflicting portrait, close-up, aerial, "
            "front-facing, flying, or poster composition. Keep the route and player "
            "inside the central safe area so a 16:9 crop preserves both."
        )
        if retry_feedback:
            image_prompt += (
                "\n\nTHE PREVIOUS IMAGE FAILED THE GAMEPLAY QUALITY GATE. Fix every "
                f"issue in this review: {retry_feedback}"
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
                world_description=description.world_description,
                image_bytes=None,
                image_generated=False,
            )
        retry_feedback = ""
        for _attempt in range(self.image_generation_attempts):
            generated = await asyncio.to_thread(
                self._generate_image_sync,
                description.world_description,
                description.camera_mode,
                retry_feedback,
            )
            validation = await self.validate_gameplay_image(
                generated, description.camera_mode
            )
            if validation.passed:
                return CompletedWorld(
                    world_description=description.world_description,
                    image_bytes=generated,
                    image_generated=True,
                )
            retry_feedback = validation.feedback or (
                "Use the required gameplay camera, show the subject correctly, "
                "and add a continuous walkable route with explorable depth."
            )
        raise RuntimeError(
            "Generated first frame failed the gameplay composition quality gate"
        )

    def save_generated_image(self, image_bytes: bytes) -> str:
        self.generated_root.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex
        target = self.generated_root / f"{image_id}.png"
        target.write_bytes(image_bytes)
        return image_id

    def save_shared_image(self, image_bytes: bytes, suffix: str = ".png") -> str:
        """Persist one browser-supplied frame so an external model can fetch it."""

        normalized_suffix = suffix.lower()
        if normalized_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            normalized_suffix = ".png"
        self.generated_root.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex
        target = self.generated_root / f"{image_id}{normalized_suffix}"
        target.write_bytes(image_bytes)
        return target.name

    def generated_image_path(self, image_id: str) -> Path | None:
        stem = Path(image_id).stem
        suffix = Path(image_id).suffix.lower()
        if suffix and suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            return None
        if len(stem) != 32 or any(c not in "0123456789abcdef" for c in stem):
            return None
        path = self.generated_root / (image_id if suffix else f"{stem}.png")
        return path if path.is_file() else None
