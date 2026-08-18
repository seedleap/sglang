# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    AcceptDisposition,
    ChunkSequenceTracker,
    LatentChunkHeader,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    discard_async_shared_memory_payload,
    encode_message,
    materialize_async_payload_from_shared_memory,
    publish_async_shared_memory_error,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)


def _header(**overrides) -> LatentChunkHeader:
    values = {
        "session_id": "s1",
        "generation_id": "g2",
        "request_id": "r0",
        "chunk_index": 0,
        "dtype": "bfloat16",
        "shape": (1, 48, 1, 30, 52),
        "byte_length": 149_760,
        "checksum": "abc",
    }
    values.update(overrides)
    return LatentChunkHeader(**values)


def test_latent_header_rejects_stale_generation():
    tracker = ChunkSequenceTracker("s1", "g2")

    with pytest.raises(ProtocolViolation, match="stale generation"):
        tracker.accept(_header(generation_id="g1"))


def test_latent_header_accepts_next_chunk_and_deduplicates_retry():
    tracker = ChunkSequenceTracker("s1", "g2")

    assert tracker.accept(_header()) is AcceptDisposition.ACCEPT
    assert tracker.accept(_header()) is AcceptDisposition.DUPLICATE
    assert tracker.accept(_header(chunk_index=1)) is AcceptDisposition.ACCEPT


def test_latent_header_rejects_conflicting_duplicate_chunk():
    tracker = ChunkSequenceTracker("s1", "g2")
    tracker.accept(_header())

    with pytest.raises(ProtocolViolation, match="conflicting duplicate"):
        tracker.accept(_header(request_id="different-request"))


def test_latent_header_rejects_gap_and_wrong_session():
    tracker = ChunkSequenceTracker("s1", "g2")

    with pytest.raises(ProtocolViolation, match="out-of-order chunk"):
        tracker.accept(_header(chunk_index=2))
    with pytest.raises(ProtocolViolation, match="wrong session"):
        tracker.accept(_header(session_id="s2"))


def test_message_round_trip_keeps_binary_payload_and_checksum():
    payload = b"\x00\x01latent"
    wire = encode_message(
        "latent_chunk",
        header=_header(
            byte_length=len(payload),
            checksum=checksum_payload(payload),
        ),
        payload=payload,
    )

    message = decode_message(wire)

    assert message["type"] == "latent_chunk"
    assert message["header"]["shape"] == [1, 48, 1, 30, 52]
    assert message["payload"] == payload


def test_decode_rejects_oversized_wire_message():
    with pytest.raises(ProtocolViolation, match="message exceeds"):
        decode_message(b"x" * 33, max_message_bytes=32)


def test_async_shared_memory_payload_publishes_atomically_and_cleans(tmp_path):
    payload = bytearray(range(32))
    reference = reserve_async_shared_memory_payload(len(payload), root=tmp_path)
    path = Path(reference["path"])
    ready_path = Path(reference["ready_path"])

    def publish_and_reclaim():
        publish_async_shared_memory_payload(reference, payload, root=tmp_path)
        assert (
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
            == "ack"
        )

    producer = threading.Thread(target=publish_and_reclaim)
    producer.start()
    materialized = materialize_async_payload_from_shared_memory(
        reference,
        root=tmp_path,
        timeout_s=1,
    )
    producer.join()

    assert materialized == bytes(payload)
    assert not path.exists()
    assert not ready_path.exists()


def test_async_shared_memory_payload_propagates_error_and_cleans(tmp_path):
    reference = reserve_async_shared_memory_payload(16, root=tmp_path)

    def publish_error_and_reclaim():
        publish_async_shared_memory_error(
            reference,
            RuntimeError("copy failed"),
            root=tmp_path,
        )
        assert (
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
            == "ack"
        )

    producer = threading.Thread(target=publish_error_and_reclaim)
    producer.start()

    with pytest.raises(RuntimeError, match="copy failed"):
        materialize_async_payload_from_shared_memory(
            reference,
            root=tmp_path,
            timeout_s=1,
        )
    producer.join()

    assert not Path(reference["path"]).exists()
    assert not Path(reference["ready_path"]).exists()


def test_async_shared_memory_payload_timeout_cleans_reserved_paths(tmp_path):
    reference = reserve_async_shared_memory_payload(16, root=tmp_path)

    def reclaim_after_cancel():
        assert (
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
            == "cancel"
        )

    producer = threading.Thread(target=reclaim_after_cancel)
    producer.start()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        materialize_async_payload_from_shared_memory(
            reference,
            root=tmp_path,
            timeout_s=0.01,
            poll_interval_s=0.001,
        )
    producer.join()

    assert time.monotonic() - started < 1
    assert not Path(reference["path"]).exists()
    assert not Path(reference["ready_path"]).exists()


def test_async_shared_memory_payload_rejects_ready_path_escape(tmp_path):
    reference = reserve_async_shared_memory_payload(4, root=tmp_path / "shm")
    expected_ready_path = reference["ready_path"]
    outside = tmp_path / "outside.ready"
    reference["ready_path"] = str(outside)

    with pytest.raises(ProtocolViolation, match="escapes configured root"):
        publish_async_shared_memory_payload(
            reference,
            b"data",
            root=tmp_path / "shm",
        )

    assert not outside.exists()
    reference["ready_path"] = expected_ready_path
    discard_async_shared_memory_payload(reference, root=tmp_path / "shm")


def test_async_timeout_tombstone_prevents_late_publish_orphans(tmp_path):
    reference = reserve_async_shared_memory_payload(4, root=tmp_path)

    def publish_late_and_reclaim():
        time.sleep(0.05)
        assert not publish_async_shared_memory_payload(
            reference,
            b"data",
            root=tmp_path,
        )
        assert (
            wait_for_async_shared_memory_terminal(
                reference,
                root=tmp_path,
                timeout_s=1,
            )
            == "cancel"
        )

    producer = threading.Thread(target=publish_late_and_reclaim)
    producer.start()
    with pytest.raises(TimeoutError, match="timed out"):
        materialize_async_payload_from_shared_memory(
            reference,
            root=tmp_path,
            timeout_s=0.01,
            poll_interval_s=0.001,
        )
    producer.join()

    assert list(tmp_path.iterdir()) == []


def test_async_reservation_failure_leaves_no_paths(monkeypatch, tmp_path):
    def fail_fallocate(_fd, _offset, _length):
        raise OSError("insufficient shared memory")

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate, raising=False)

    with pytest.raises(OSError, match="insufficient shared memory"):
        reserve_async_shared_memory_payload(42_172_416, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_async_reservation_rejects_insufficient_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=1, f_frsize=4096),
    )

    with pytest.raises(OSError, match="insufficient shared-memory capacity"):
        reserve_async_shared_memory_payload(4096, root=tmp_path)

    assert list(tmp_path.iterdir()) == []
