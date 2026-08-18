# SPDX-License-Identifier: Apache-2.0

"""Wire primitives shared by the realtime gateway and remote VAE worker."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import time
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
ASYNC_SHARED_MEMORY_READY_TIMEOUT_S = 60.0
ASYNC_SHARED_MEMORY_OWNER_TIMEOUT_S = 75.0
# Keep headroom for the owner, ready, and terminal marker files after the
# payload itself is fallocated. This prevents a successful reservation from
# failing later merely because the tmpfs had room for data but not the protocol.
ASYNC_SHARED_MEMORY_METADATA_RESERVE_BYTES = 64 * 1024


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


def _validated_shared_memory_path(
    reference: dict[str, Any],
    key: str,
    *,
    root: str | Path | None = None,
    strict: bool = False,
) -> Path:
    shared_root = _shared_memory_root(root)
    path = Path(str(reference.get(key) or "")).resolve(strict=strict)
    if path.parent != shared_root:
        raise ProtocolViolation("shared-memory payload path escapes configured root")
    return path


def _validated_async_shared_memory_paths(
    reference: dict[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Path]:
    shared_root = _shared_memory_root(root)
    if str(reference.get("shared_memory_root") or "") != str(shared_root):
        raise ProtocolViolation("shared-memory producer and consumer roots differ")
    path = _validated_shared_memory_path(reference, "path", root=root)
    paths = {
        "path": path,
        "ready_path": _validated_shared_memory_path(reference, "ready_path", root=root),
        "ready_tmp_path": _validated_shared_memory_path(
            reference, "ready_tmp_path", root=root
        ),
        "ack_path": _validated_shared_memory_path(reference, "ack_path", root=root),
        "cancel_path": _validated_shared_memory_path(
            reference, "cancel_path", root=root
        ),
        "owner_path": _validated_shared_memory_path(reference, "owner_path", root=root),
    }
    expected_names = {
        "ready_path": f"{path.name}.ready",
        "ready_tmp_path": f"{path.name}.ready.tmp",
        "ack_path": f"{path.name}.ack",
        "cancel_path": f"{path.name}.cancel",
        "owner_path": f"{path.name}.owner",
    }
    for key, expected_name in expected_names.items():
        if paths[key] == path or paths[key].name != expected_name:
            raise ProtocolViolation(f"shared-memory {key} is invalid")
    return paths


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
    expected_bytes = int(reference.get("num_bytes", -1))
    path = _validated_shared_memory_path(reference, "path", root=root, strict=True)
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


def reserve_async_shared_memory_payload(
    num_bytes: int,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Reserve serializable paths for a payload published by a background worker."""
    if num_bytes <= 0:
        raise ValueError("shared-memory payload size must be positive")
    shared_root = _shared_memory_root(root)
    shared_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fs_stats = os.statvfs(shared_root)
    available_bytes = fs_stats.f_bavail * fs_stats.f_frsize
    required_bytes = num_bytes + ASYNC_SHARED_MEMORY_METADATA_RESERVE_BYTES
    if available_bytes < required_bytes:
        raise OSError(
            errno.ENOSPC,
            "insufficient shared-memory capacity for asynchronous payload",
            str(shared_root),
        )
    name = f"{os.getpid()}-{uuid4().hex}.bin"
    reference = {
        "path": str(shared_root / name),
        "ready_path": str(shared_root / f"{name}.ready"),
        "ready_tmp_path": str(shared_root / f"{name}.ready.tmp"),
        "ack_path": str(shared_root / f"{name}.ack"),
        "cancel_path": str(shared_root / f"{name}.cancel"),
        "owner_path": str(shared_root / f"{name}.owner"),
        "num_bytes": int(num_bytes),
        "producer_pid": os.getpid(),
        "shared_memory_root": str(shared_root),
    }
    path = Path(reference["path"])
    owner_path = Path(reference["owner_path"])
    try:
        payload_fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(payload_fd, 0, num_bytes)
            else:
                os.ftruncate(payload_fd, num_bytes)
        finally:
            os.close(payload_fd)
        fd = os.open(owner_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb", buffering=0) as handle:
            handle.write(b"active")
    except Exception:
        path.unlink(missing_ok=True)
        owner_path.unlink(missing_ok=True)
        raise
    return reference


def publish_async_shared_memory_payload(
    reference: dict[str, Any],
    payload: Any,
    *,
    root: str | Path | None = None,
) -> bool:
    """Publish a buffer and then atomically expose its ready marker."""
    expected_bytes = int(reference.get("num_bytes", -1))
    payload_view = memoryview(payload).cast("B")
    if payload_view.nbytes != expected_bytes:
        raise ProtocolViolation(
            "shared-memory payload size mismatch: "
            f"expected {expected_bytes}, got {payload_view.nbytes}"
        )

    paths = _validated_async_shared_memory_paths(reference, root=root)
    path = paths["path"]
    if paths["cancel_path"].exists():
        return False
    try:
        fd = os.open(path, os.O_WRONLY)
        with os.fdopen(fd, "wb", buffering=0) as handle:
            written = 0
            while written < payload_view.nbytes:
                write_size = handle.write(payload_view[written:])
                if write_size <= 0:
                    raise OSError("shared-memory payload write made no progress")
                written += write_size
        if paths["cancel_path"].exists():
            return False
        return _publish_async_ready_status(reference, {"ok": True}, root=root)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def publish_async_shared_memory_error(
    reference: dict[str, Any],
    error: BaseException,
    *,
    root: str | Path | None = None,
) -> bool:
    """Wake a waiting consumer after an asynchronous producer failure."""
    paths = _validated_async_shared_memory_paths(reference, root=root)
    paths["path"].unlink(missing_ok=True)
    return _publish_async_ready_status(
        reference,
        {"ok": False, "error": f"{type(error).__name__}: {error}"},
        root=root,
    )


def materialize_async_payload_from_shared_memory(
    reference: dict[str, Any],
    *,
    root: str | Path | None = None,
    timeout_s: float = ASYNC_SHARED_MEMORY_READY_TIMEOUT_S,
    poll_interval_s: float = 0.001,
) -> bytes:
    """Wait for a ready marker, read the payload, and ACK producer ownership."""
    if timeout_s <= 0:
        raise ValueError("shared-memory ready timeout must be positive")
    paths = _validated_async_shared_memory_paths(reference, root=root)
    deadline = time.monotonic() + timeout_s
    while not paths["ready_path"].exists():
        if paths["cancel_path"].exists():
            raise RuntimeError("asynchronous shared-memory payload was cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancel_async_shared_memory_payload(reference, root=root)
            raise TimeoutError("timed out waiting for shared-memory payload")
        time.sleep(min(poll_interval_s, remaining))

    try:
        status = json.loads(paths["ready_path"].read_text(encoding="utf-8"))
        if not status.get("ok"):
            raise RuntimeError(
                "asynchronous shared-memory producer failed: "
                f"{status.get('error', 'unknown error')}"
            )
        expected_bytes = int(reference.get("num_bytes", -1))
        payload = paths["path"].read_bytes()
        if len(payload) != expected_bytes:
            raise ProtocolViolation(
                "shared-memory payload size mismatch: "
                f"expected {expected_bytes}, got {len(payload)}"
            )
        return payload
    finally:
        acknowledge_async_shared_memory_payload(reference, root=root)


def acknowledge_async_shared_memory_payload(
    reference: dict[str, Any] | None,
    *,
    root: str | Path | None = None,
) -> bool:
    """ACK a consumed reference without taking file ownership from the producer."""
    return _signal_async_shared_memory_terminal(reference, "ack_path", root=root)


def cancel_async_shared_memory_payload(
    reference: dict[str, Any] | None,
    *,
    root: str | Path | None = None,
) -> bool:
    """Cancel an unconsumed reference; producer remains responsible for cleanup."""
    signalled = _signal_async_shared_memory_terminal(
        reference, "cancel_path", root=root
    )
    if reference and not _process_is_alive(int(reference.get("producer_pid", -1))):
        discard_async_shared_memory_payload(reference, root=root)
    return signalled


def wait_for_async_shared_memory_terminal(
    reference: dict[str, Any],
    *,
    root: str | Path | None = None,
    timeout_s: float = ASYNC_SHARED_MEMORY_OWNER_TIMEOUT_S,
    poll_interval_s: float = 0.001,
) -> str:
    """Wait for consumer ACK/CANCEL, then reclaim all producer-owned paths."""
    if timeout_s <= 0:
        raise ValueError("shared-memory owner timeout must be positive")
    paths = _validated_async_shared_memory_paths(reference, root=root)
    deadline = time.monotonic() + timeout_s
    terminal = "owner_timeout"
    try:
        while True:
            if paths["cancel_path"].exists():
                terminal = "cancel"
                break
            if paths["ack_path"].exists():
                terminal = "ack"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_s, remaining))
        return terminal
    finally:
        discard_async_shared_memory_payload(reference, root=root)


def discard_async_shared_memory_payload(
    reference: dict[str, Any] | None,
    *,
    root: str | Path | None = None,
) -> None:
    """Reclaim an async reference. Only its producer may call this function."""
    if not reference:
        return
    paths = _validated_async_shared_memory_paths(reference, root=root)
    owner_path = paths["owner_path"]
    try:
        owner_fd = os.open(owner_path, os.O_RDWR)
    except FileNotFoundError:
        owner_fd = None
    if owner_fd is None:
        for key, path in paths.items():
            if key != "owner_path":
                path.unlink(missing_ok=True)
        return

    with os.fdopen(owner_fd, "r+b", buffering=0) as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        try:
            owner.seek(0)
            owner.truncate()
            owner.write(b"closed")
            for key, path in paths.items():
                if key != "owner_path":
                    path.unlink(missing_ok=True)
            owner_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)


def _signal_async_shared_memory_terminal(
    reference: dict[str, Any] | None,
    terminal_key: str,
    *,
    root: str | Path | None = None,
) -> bool:
    if not reference:
        return False
    paths = _validated_async_shared_memory_paths(reference, root=root)
    try:
        owner_fd = os.open(paths["owner_path"], os.O_RDWR)
    except FileNotFoundError:
        return False
    with os.fdopen(owner_fd, "r+b", buffering=0) as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        try:
            owner.seek(0)
            if owner.read() != b"active":
                return False
            marker = paths[terminal_key]
            try:
                marker_fd = os.open(
                    marker,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                return True
            os.close(marker_fd)
            return True
        finally:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _publish_async_ready_status(
    reference: dict[str, Any],
    status: dict[str, Any],
    *,
    root: str | Path | None = None,
) -> bool:
    paths = _validated_async_shared_memory_paths(reference, root=root)
    if paths["cancel_path"].exists():
        return False
    status_tmp = paths["ready_tmp_path"]
    try:
        fd = os.open(status_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(status, handle)
        if paths["cancel_path"].exists():
            status_tmp.unlink(missing_ok=True)
            return False
        os.replace(status_tmp, paths["ready_path"])
        return True
    except Exception:
        status_tmp.unlink(missing_ok=True)
        raise


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
