# SPDX-License-Identifier: Apache-2.0

"""Wire primitives shared by the realtime gateway and remote VAE worker."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import msgspec.msgpack

PROTOCOL_VERSION = 2
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
SHARED_MEMORY_DIR_ENV = "SGLANG_REALTIME_VAE_SHM_DIR"
DEFAULT_SHARED_MEMORY_DIR = "/dev/shm/sglang-realtime-vae"
PAYLOAD_TRANSPORT_WEBSOCKET = "websocket"
PAYLOAD_TRANSPORT_SHARED_MEMORY = "shared_memory"


class ProtocolViolation(ValueError):
    """Raised when a peer sends an invalid or stale realtime VAE message."""


class AcceptDisposition(str, Enum):
    ACCEPT = "accept"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class LatentChunkHeader:
    session_id: str
    generation_id: str
    request_id: str
    chunk_index: int
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    checksum: str
    event_id: int | None = None
    action_version: int = 0
    prompt_version: int = 0
    deadline_epoch_ms: int = 0
    has_reference: bool = False
    is_final_chunk: bool = False

    def validate(self) -> None:
        if not self.session_id or not self.generation_id or not self.request_id:
            raise ProtocolViolation("session, generation, and request IDs are required")
        if self.chunk_index < 0:
            raise ProtocolViolation("chunk index must be non-negative")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ProtocolViolation(f"unsupported latent dtype: {self.dtype}")
        if len(self.shape) != 5 or any(int(dim) <= 0 for dim in self.shape):
            raise ProtocolViolation(
                "latent shape must contain five positive dimensions"
            )
        if self.byte_length <= 0:
            raise ProtocolViolation("latent byte length must be positive")
        if not self.checksum:
            raise ProtocolViolation("latent checksum is required")


class ChunkSequenceTracker:
    """Accept exactly-once, monotonically ordered chunks for one generation."""

    def __init__(self, session_id: str, generation_id: str) -> None:
        self.session_id = session_id
        self.generation_id = generation_id
        self.next_chunk_index = 0
        self._accepted_headers: dict[int, LatentChunkHeader] = {}
        self.final_chunk_index: int | None = None

    def accept(self, header: LatentChunkHeader) -> AcceptDisposition:
        header.validate()
        if header.session_id != self.session_id:
            raise ProtocolViolation("wrong session")
        if header.generation_id != self.generation_id:
            raise ProtocolViolation("stale generation")
        if (
            self.final_chunk_index is not None
            and header.chunk_index > self.final_chunk_index
        ):
            raise ProtocolViolation("chunk received after final chunk")
        if header.chunk_index == self.next_chunk_index:
            self._accepted_headers[header.chunk_index] = header
            if header.is_final_chunk:
                self.final_chunk_index = header.chunk_index
            self.next_chunk_index += 1
            return AcceptDisposition.ACCEPT
        if header.chunk_index < self.next_chunk_index:
            if self._accepted_headers.get(header.chunk_index) != header:
                raise ProtocolViolation("conflicting duplicate chunk")
            return AcceptDisposition.DUPLICATE
        raise ProtocolViolation("out-of-order chunk")


def checksum_payload(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_loopback_url(url: str) -> bool:
    return urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}


def _shared_memory_root(root: str | Path | None = None) -> Path:
    return Path(
        root or os.environ.get(SHARED_MEMORY_DIR_ENV, DEFAULT_SHARED_MEMORY_DIR)
    ).resolve()


def store_payload_in_shared_memory(
    payload: bytes,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("shared-memory payload must be bytes")
    shared_root = _shared_memory_root(root)
    shared_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = shared_root / f"{os.getpid()}-{uuid4().hex}.bin"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "num_bytes": len(payload)}


def materialize_payload_from_shared_memory(
    reference: dict[str, Any],
    *,
    root: str | Path | None = None,
    max_bytes: int | None = None,
) -> bytes:
    shared_root = _shared_memory_root(root)
    expected_bytes = int(reference.get("num_bytes", -1))
    path = Path(str(reference.get("path") or "")).resolve(strict=True)
    if path.parent != shared_root:
        raise ProtocolViolation("shared-memory payload path escapes configured root")
    try:
        if expected_bytes < 0:
            raise ProtocolViolation("shared-memory payload size is invalid")
        if max_bytes is not None and expected_bytes > max_bytes:
            raise ProtocolViolation("shared-memory payload exceeds protocol limit")
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise ProtocolViolation(
                "shared-memory payload size mismatch: "
                f"expected {expected_bytes}, got {len(payload)}"
            )
        return payload
    finally:
        path.unlink(missing_ok=True)


def discard_shared_memory_payload(
    reference: dict[str, Any] | None,
    *,
    root: str | Path | None = None,
) -> None:
    if not reference:
        return
    shared_root = _shared_memory_root(root)
    path = Path(str(reference.get("path") or "")).resolve()
    if path.parent == shared_root:
        path.unlink(missing_ok=True)


def encode_message(
    message_type: str,
    *,
    header: LatentChunkHeader | dict[str, Any] | None = None,
    payload: bytes | None = None,
    **fields: Any,
) -> bytes:
    message: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        **fields,
    }
    if header is not None:
        message["header"] = (
            asdict(header) if isinstance(header, LatentChunkHeader) else header
        )
    if payload is not None:
        message["payload"] = payload
    return msgspec.msgpack.encode(message)


def decode_message(
    wire: bytes,
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    if len(wire) > max_message_bytes:
        raise ProtocolViolation(
            f"message exceeds {max_message_bytes} byte protocol limit"
        )
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError as exc:
        raise ProtocolViolation("invalid MessagePack message") from exc
    if not isinstance(message, dict):
        raise ProtocolViolation("protocol message must be a map")
    if message.get("version") != PROTOCOL_VERSION:
        raise ProtocolViolation("unsupported protocol version")
    if not isinstance(message.get("type"), str):
        raise ProtocolViolation("protocol message type is required")
    return message


def latent_header_from_message(message: dict[str, Any]) -> LatentChunkHeader:
    raw = message.get("header")
    if not isinstance(raw, dict):
        raise ProtocolViolation("latent header is required")
    try:
        header = LatentChunkHeader(
            **{
                **raw,
                "shape": tuple(raw.get("shape", ())),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation("invalid latent header") from exc
    header.validate()
    return header


def validate_payload(header: LatentChunkHeader, payload: bytes) -> None:
    if len(payload) != header.byte_length:
        raise ProtocolViolation("latent payload length mismatch")
    if checksum_payload(payload) != header.checksum:
        raise ProtocolViolation("latent payload checksum mismatch")
