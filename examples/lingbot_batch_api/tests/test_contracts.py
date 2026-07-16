import pytest

from lingbot_batch_api.contracts import ValidationError, parse_video_request


def valid_payload():
    return {
        "request_id": "dataset-a/image-42/variant-0",
        "source_id": "image-42",
        "image_index": 42,
        "variant_slot": 0,
        "variants": 5,
        "prompt": "Third-person gameplay view",
        "first_frame": "s3://example-bucket/images/image-42.png",
    }


def test_action_pair_is_derived_from_image_and_variant():
    first = parse_video_request(valid_payload())
    payload = valid_payload()
    payload["variant_slot"] = 1
    second = parse_video_request(payload)
    assert first.action_pair != second.action_pair


def test_explicit_action_pair_is_supported():
    payload = valid_payload()
    payload.update({"movement_key": "d", "camera_key": "j"})
    request = parse_video_request(payload)
    assert request.action_pair.movement_key == "d"
    assert request.action_pair.camera_key == "j"


def test_http_first_frame_must_use_https():
    payload = valid_payload()
    payload["first_frame"] = "http://example.test/image.png"
    with pytest.raises(ValidationError, match="s3:// or https://"):
        parse_video_request(payload)


def test_variant_slot_must_fit_variant_count():
    payload = valid_payload()
    payload["variant_slot"] = 5
    with pytest.raises(ValidationError, match="smaller"):
        parse_video_request(payload)
