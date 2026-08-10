from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_realtime_throughput import (  # noqa: E402
    record_frame_batch,
    validate_frame_batch,
)
from common import (  # noqa: E402
    action_label_sequence,
    build_minwm_message,
    is_realtime_trace_event,
    load_cases,
    materialize_first_frame,
)

DRAGON_CASES = Path(__file__).with_name("cases_dragon_ride_60s_832x480.json")
STEP1600_T2V_CASES = Path(__file__).with_name("cases_step1600_t2v_30s_832x480.json")


def _raw_header(**overrides):
    header = {
        "chunk_index": 0,
        "num_frames": 1,
        "content_type": "application/x-raw-rgb",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
        "raw_size": 12,
        "total_size": 12,
        "frame_batch_index": 0,
        "num_frame_batches": 1,
        "is_final_frame_batch": True,
    }
    header.update(overrides)
    return header


def test_streamed_frame_batch_allows_unknown_count_until_final_batch() -> None:
    payload = bytes(12)
    state = {"num_batches": None, "seen": set(), "frames": 0}

    for batch_index in range(3):
        header = _raw_header(
            frame_batch_index=batch_index,
            num_frame_batches=0,
            is_final_frame_batch=False,
        )
        index, count, frames = validate_frame_batch(header, payload, chunk_index=0)
        assert not record_frame_batch(
            state,
            chunk_index=0,
            batch_index=index,
            num_batches=count,
            batch_frames=frames,
            expected_frames=4,
        )

    final_header = _raw_header(frame_batch_index=3, num_frame_batches=4)
    index, count, frames = validate_frame_batch(final_header, payload, chunk_index=0)
    assert record_frame_batch(
        state,
        chunk_index=0,
        batch_index=index,
        num_batches=count,
        batch_frames=frames,
        expected_frames=4,
    )
    assert state == {"num_batches": 4, "seen": {0, 1, 2, 3}, "frames": 4}


def test_streamed_frame_batch_rejects_final_unknown_count() -> None:
    with pytest.raises(AssertionError, match="is_final_frame_batch"):
        validate_frame_batch(_raw_header(num_frame_batches=0), bytes(12), chunk_index=0)


def test_realtime_trace_events_are_out_of_band() -> None:
    assert is_realtime_trace_event({"type": "trace_event", "trace": {}})
    assert not is_realtime_trace_event({"type": "chunk_stats"})


def test_dragon_ride_contract_is_exactly_sixty_generated_seconds() -> None:
    manifest = load_cases(DRAGON_CASES)
    contract = manifest["contract"]
    case = manifest["cases"][0]

    assert contract["generated_pixel_frames"] == 1440
    assert contract["fps"] == 24
    assert contract["generated_pixel_frames"] / contract["fps"] == 60
    assert contract["chunks"] == 90
    labels = action_label_sequence(case, contract)
    assert labels == (
        [9] * 30 + [18] * 30 + [0] * 30 + [9] * 30 + [18] * 30 + [0] * 210
    )

    message = build_minwm_message(case, contract, Path("/tmp/dragon.png"))
    actions = message["messages"][1]["controls"][0]["actions"]
    assert actions[0] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert actions[119] == actions[0]
    assert actions[120] == [0, 0, 1, 0, 0, 0, 0, 0]
    assert actions[239] == actions[120]
    assert actions[240] == [0] * 8
    assert actions[359] == actions[240]
    assert actions[360] == actions[0]
    assert actions[480] == actions[120]
    assert actions[600] == [0] * 8
    assert actions[-1] == [0] * 8


def test_step1600_t2v_contract_preserves_first_regular_and_remainder_chunks() -> None:
    manifest = load_cases(STEP1600_T2V_CASES)
    default_contract = manifest["contract"]
    standard = manifest["cases"][1]
    short = manifest["cases"][2]

    assert default_contract["latent_chunk_sizes"] == [1] + [4] * 45 + [1]
    assert action_label_sequence(standard, default_contract) == (
        [0] + [9] * 30 + [2] * 60 + [1] * 60 + [27] * 31
    )
    assert short["contract"]["latent_chunk_sizes"] == [1] + [4] * 40 + [1]
    assert action_label_sequence(short, default_contract) == (
        [0] + [27] * 40 + [1] * 40 + [2] * 40 + [36] * 41
    )


def test_materialize_http_first_frame_verifies_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"stable-reference-image"
    expected = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    path = materialize_first_frame(
        {
            "id": "http-fixture",
            "first_frame": "https://example.invalid/reference.png",
            "first_frame_sha256": expected,
        },
        tmp_path,
    )

    assert path.read_bytes() == payload


def test_materialize_first_frame_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"changed"),
    )

    with pytest.raises(ValueError, match="does not match"):
        materialize_first_frame(
            {
                "id": "http-fixture",
                "first_frame": "https://example.invalid/reference.png",
                "first_frame_sha256": "0" * 64,
            },
            tmp_path,
        )
