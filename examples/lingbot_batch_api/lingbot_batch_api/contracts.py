"""Validation for the lightweight synchronous LingBot API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from .actions import ActionPair, select_action_pairs, validate_action_pair


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class VideoRequest:
    request_id: str
    source_id: str
    image_index: int
    variant_slot: int
    variants: int
    prompt: str
    negative_prompt: Optional[str]
    first_frame: str
    action_seed: int
    video_seed: int
    action_pair: ActionPair


def _required_string(payload: dict[str, Any], name: str, limit: int) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    value = value.strip()
    if len(value) > limit:
        raise ValidationError(f"{name} exceeds {limit} characters")
    return value


def parse_video_request(payload: Any) -> VideoRequest:
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    request_id = _required_string(payload, "request_id", 256)
    source_id = _required_string(payload, "source_id", 256)
    prompt = _required_string(payload, "prompt", 20_000)
    first_frame = _required_string(payload, "first_frame", 2_048)
    parsed = urlparse(first_frame)
    if parsed.scheme not in {"s3", "https"} or not parsed.netloc:
        raise ValidationError("first_frame must be an s3:// or https:// URL")

    negative_prompt = payload.get("negative_prompt")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValidationError("negative_prompt must be a string or null")

    integer_fields = {
        "image_index": payload.get("image_index"),
        "variant_slot": payload.get("variant_slot"),
        "variants": payload.get("variants", 5),
        "action_seed": payload.get("action_seed", 20260715),
        "video_seed": payload.get("video_seed", 0),
    }
    for name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValidationError(f"{name} must be a non-negative integer")
    variants = integer_fields["variants"]
    variant_slot = integer_fields["variant_slot"]
    if not 1 <= variants <= 16:
        raise ValidationError("variants must be between 1 and 16")
    if variant_slot >= variants:
        raise ValidationError("variant_slot must be smaller than variants")

    movement_key = payload.get("movement_key")
    camera_key = payload.get("camera_key")
    if movement_key is None and camera_key is None:
        pair = select_action_pairs(
            image_index=integer_fields["image_index"],
            variants=variants,
            action_seed=integer_fields["action_seed"],
        )[variant_slot]
    elif isinstance(movement_key, str) and isinstance(camera_key, str):
        try:
            pair = validate_action_pair(movement_key, camera_key)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    else:
        raise ValidationError(
            "movement_key and camera_key must either both be set or both be omitted"
        )

    return VideoRequest(
        request_id=request_id,
        source_id=source_id,
        image_index=integer_fields["image_index"],
        variant_slot=variant_slot,
        variants=variants,
        prompt=prompt,
        negative_prompt=negative_prompt,
        first_frame=first_frame,
        action_seed=integer_fields["action_seed"],
        video_seed=integer_fields["video_seed"],
        action_pair=pair,
    )
