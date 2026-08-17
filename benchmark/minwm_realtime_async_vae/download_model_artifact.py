#!/usr/bin/env python3
"""Stage an immutable, parameterized model release with the AWS CRT client."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
S3_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_control_object(client: Any, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    content_length = response.get("ContentLength")
    if content_length is not None and content_length > MAX_CONTROL_FILE_BYTES:
        raise ValueError(f"control object is unexpectedly large: s3://{bucket}/{key}")
    payload = response["Body"].read(MAX_CONTROL_FILE_BYTES + 1)
    if len(payload) > MAX_CONTROL_FILE_BYTES:
        raise ValueError(f"control object is unexpectedly large: s3://{bucket}/{key}")
    return payload


def _load_json(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description} JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return value


def _safe_relative_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest file path must be a non-empty string")
    if "\\" in raw_path:
        raise ValueError(f"manifest file path contains a backslash: {raw_path!r}")
    pure_path = PurePosixPath(raw_path)
    if (
        pure_path.is_absolute()
        or raw_path != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError(f"unsafe manifest file path: {raw_path!r}")
    if raw_path in {"_READY", "artifact-manifest.json", "info.json"}:
        raise ValueError(f"manifest cannot overwrite control file: {raw_path!r}")
    return Path(*pure_path.parts)


def _safe_model_component(name: str, value: str) -> str:
    if not MODEL_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{name} is not a safe model path component")
    return value


def _safe_s3_prefix(prefix: str) -> str:
    normalized = prefix.strip("/")
    if not normalized or "\\" in normalized:
        raise ValueError("S3 prefix cannot be empty or contain a backslash")
    parsed = PurePosixPath(normalized)
    if normalized != parsed.as_posix() or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError("S3 prefix is not a safe object-key prefix")
    return normalized


def resolve_model_source(
    *,
    bucket: str | None,
    prefix: str | None,
    model_s3_uri: str | None,
    model_name: str,
    model_version: str,
    model_release_id: str,
) -> tuple[str, str, str]:
    """Resolve an explicit rollback URI or the versioned serving convention."""
    _safe_model_component("model-name", model_name)
    _safe_model_component("model-version", model_version)
    _safe_model_component("model-release-id", model_release_id)

    if model_s3_uri:
        parsed = urlsplit(model_s3_uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
        ):
            raise ValueError(
                "model-s3-uri must be an unambiguous s3://bucket/prefix URI"
            )
        if not S3_BUCKET_RE.fullmatch(parsed.netloc):
            raise ValueError("model-s3-uri has an invalid S3 bucket name")
        resolved_prefix = _safe_s3_prefix(parsed.path)
        prefix_parts = PurePosixPath(resolved_prefix).parts
        if len(prefix_parts) != 4 or prefix_parts[:3] != (
            "models",
            model_name,
            model_version,
        ):
            raise ValueError(
                "model-s3-uri must follow "
                "models/<model-name>/<model-version>/<release-id>"
            )
        rollback_release_id = _safe_model_component(
            "model-s3-uri release-id", prefix_parts[3]
        )
        return parsed.netloc, resolved_prefix, rollback_release_id

    if not bucket:
        raise ValueError("bucket is required when model-s3-uri is not set")
    if not S3_BUCKET_RE.fullmatch(bucket):
        raise ValueError("bucket is not a valid S3 bucket name")
    if prefix:
        return bucket, _safe_s3_prefix(prefix), model_release_id
    return (
        bucket,
        f"models/{model_name}/{model_version}/{model_release_id}",
        model_release_id,
    )


def validate_control_files(
    ready_body: bytes,
    manifest_body: bytes,
    expected_revision: str | None,
    *,
    require_canonical_ready_revision: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ready = _load_json(ready_body, "_READY")
    manifest = _load_json(manifest_body, "artifact manifest")

    manifest_sha256 = ready.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("_READY has no valid manifest_sha256")
    if _sha256_bytes(manifest_body) != manifest_sha256:
        raise ValueError("artifact manifest SHA256 does not match _READY")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema_version")

    manifest_revision = manifest.get("revision")
    manifest_resolved_revision = manifest.get("resolved_revision")
    if (
        manifest_revision is not None
        and manifest_resolved_revision is not None
        and manifest_revision != manifest_resolved_revision
    ):
        raise ValueError("artifact manifest revision fields differ")
    revision = manifest_revision or manifest_resolved_revision
    if not isinstance(revision, str) or not revision:
        raise ValueError("artifact manifest has no revision")
    ready_revision = ready.get("revision")
    ready_resolved_revision = ready.get("resolved_revision")
    if require_canonical_ready_revision:
        if ready.get("schema_version") != 1:
            raise ValueError("unsupported _READY schema_version")
        if ready_revision != revision:
            raise ValueError("_READY has no canonical revision")
    if (
        ready_revision is not None
        and ready_resolved_revision is not None
        and ready_revision != ready_resolved_revision
    ):
        raise ValueError("_READY revision fields differ")
    if (ready_revision or ready_resolved_revision) != revision:
        raise ValueError("_READY and artifact manifest revisions differ")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"artifact revision {revision!r} does not match expected "
            f"revision {expected_revision!r}"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("artifact manifest files must be a non-empty list")

    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict):
            raise TypeError("artifact manifest file entry must be an object")
        relative_path = _safe_relative_path(raw_entry.get("path"))
        path_string = relative_path.as_posix()
        if path_string in seen_paths:
            raise ValueError(f"duplicate artifact manifest path: {path_string}")
        seen_paths.add(path_string)

        size = raw_entry.get("size")
        digest = raw_entry.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid size for artifact file {path_string}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid SHA256 for artifact file {path_string}")
        files.append({"path": path_string, "size": size, "sha256": digest})

    return manifest, files


def validate_release_controls(
    info_body: bytes,
    ready_body: bytes,
    manifest_body: bytes,
    *,
    expected_model_name: str,
    expected_model_version: str,
    expected_release_id: str,
    expected_revision: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate the release identity and immutable artifact authorization chain."""
    info = _load_json(info_body, "model info")
    ready = _load_json(ready_body, "_READY")
    if info.get("schema_version") != 1:
        raise ValueError("unsupported model info schema_version")

    expected_identity = {
        "model_name": expected_model_name,
        "model_version": expected_model_version,
        "release_id": expected_release_id,
    }
    for field, expected in expected_identity.items():
        actual = info.get(field)
        if actual != expected:
            raise ValueError(
                f"model info {field} {actual!r} does not match expected {expected!r}"
            )

    info_sha256 = ready.get("info_sha256")
    if not isinstance(info_sha256, str) or not SHA256_RE.fullmatch(info_sha256):
        raise ValueError("_READY has no valid info_sha256")
    if _sha256_bytes(info_body) != info_sha256:
        raise ValueError("model info SHA256 does not match _READY")
    if ready.get("release_id") != expected_release_id:
        raise ValueError("_READY release_id does not match model info")

    info_manifest_sha256 = info.get("artifact_manifest_sha256")
    if (
        not isinstance(info_manifest_sha256, str)
        or not SHA256_RE.fullmatch(info_manifest_sha256)
        or info_manifest_sha256 != ready.get("manifest_sha256")
    ):
        raise ValueError("model info does not authorize the artifact manifest")

    source_revision = info.get("source_revision")
    source_uri = info.get("source_uri")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("model info has no source_revision")
    if not isinstance(source_uri, str) or not source_uri.startswith("s3://"):
        raise ValueError("model info has no valid source_uri")

    manifest, files = validate_control_files(
        ready_body,
        manifest_body,
        expected_revision,
        require_canonical_ready_revision=True,
    )
    revision = manifest.get("revision") or manifest.get("resolved_revision")
    if revision != source_revision:
        raise ValueError("artifact revision does not match model info source_revision")
    artifact_object_count = info.get("artifact_object_count")
    if artifact_object_count is not None and artifact_object_count != len(files):
        raise ValueError("model info artifact_object_count differs from manifest")
    artifact_bytes = info.get("artifact_bytes")
    if artifact_bytes is not None and artifact_bytes != sum(
        entry["size"] for entry in files
    ):
        raise ValueError("model info artifact_bytes differs from manifest")
    for field in ("release_spec_sha256", "publisher_bundle_sha256"):
        info_value = info.get(field)
        ready_value = ready.get(field)
        if info_value is None and ready_value is None:
            continue
        if not isinstance(info_value, str) or not SHA256_RE.fullmatch(info_value):
            raise ValueError(f"model info has no valid {field}")
        if ready_value != info_value:
            raise ValueError(f"model info and _READY {field} differ")
    return info, manifest, files


def _cache_matches(
    destination: Path,
    info_body: bytes,
    ready_body: bytes,
    manifest_body: bytes,
    files: list[dict[str, Any]],
) -> bool:
    try:
        if (destination / "_READY").read_bytes() != ready_body:
            return False
        if (destination / "info.json").read_bytes() != info_body:
            return False
        if (destination / "artifact-manifest.json").read_bytes() != manifest_body:
            return False
        return all(
            (destination / entry["path"]).is_file()
            and (destination / entry["path"]).stat().st_size == entry["size"]
            and _sha256_file(destination / entry["path"]) == entry["sha256"]
            for entry in files
        )
    except OSError:
        return False


def create_crt_transfer_manager(client: Any, concurrency: int, part_size: int) -> Any:
    """Create a transfer manager and reject any non-CRT fallback."""
    from boto3.s3.transfer import TransferConfig, create_transfer_manager
    from s3transfer.crt import CRTTransferManager

    config = TransferConfig(
        multipart_threshold=part_size,
        multipart_chunksize=part_size,
        max_concurrency=concurrency,
        preferred_transfer_client="crt",
    )
    manager = create_transfer_manager(client, config)
    if not isinstance(manager, CRTTransferManager):
        manager.shutdown(cancel=True)
        raise TypeError("Boto3 did not create a CRTTransferManager; refusing fallback")
    return manager


def _download_files(
    client: Any,
    manager_factory: Callable[[Any, int, int], Any],
    bucket: str,
    prefix: str,
    staging: Path,
    files: list[dict[str, Any]],
    concurrency: int,
    part_size: int,
) -> None:
    manager = manager_factory(client, concurrency, part_size)
    with manager:
        futures = []
        streams = []
        try:
            for entry in files:
                destination = staging / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = destination.open("wb")
                streams.append(stream)
                # Passing a file object avoids an s3transfer CRT completion race
                # where the native future resolves before its temporary-file
                # rename callback has published the requested destination path.
                future = manager.download(
                    bucket,
                    f"{prefix}/model/{entry['path']}",
                    stream,
                )
                futures.append((entry, destination, stream, future))

            # Submit every object first. CRT then schedules byte-range GETs across
            # the complete artifact instead of serializing large model shards.
            for entry, destination, stream, future in futures:
                future.result()
                stream.flush()
                stream.close()
                actual_size = destination.stat().st_size
                if actual_size != entry["size"]:
                    raise ValueError(
                        f"size mismatch for {entry['path']}: "
                        f"expected {entry['size']}, got {actual_size}"
                    )
        finally:
            for stream in streams:
                stream.close()


def _verify_sha256(staging: Path, files: list[dict[str, Any]]) -> None:
    def verify(entry: dict[str, Any]) -> None:
        actual_digest = _sha256_file(staging / entry["path"])
        if actual_digest != entry["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {entry['path']}: "
                f"expected {entry['sha256']}, got {actual_digest}"
            )

    with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
        for future in [executor.submit(verify, entry) for entry in files]:
            future.result()


def _activate_staging(staging: Path, destination: Path) -> None:
    backup = destination.parent / f".{destination.name}.old.{uuid.uuid4().hex}"
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if had_destination:
            os.replace(backup, destination)
        raise
    if had_destination:
        shutil.rmtree(backup)


def stage_model(
    *,
    client: Any,
    bucket: str,
    prefix: str,
    destination: Path,
    lock_path: Path,
    expected_model_name: str,
    expected_model_version: str,
    expected_release_id: str,
    expected_revision: str | None,
    concurrency: int,
    part_size: int,
    manager_factory: Callable[[Any, int, int], Any] = create_crt_transfer_manager,
) -> dict[str, Any]:
    prefix = _safe_s3_prefix(prefix)
    _safe_model_component("model-name", expected_model_name)
    _safe_model_component("model-version", expected_model_version)
    _safe_model_component("model-release-id", expected_release_id)
    if concurrency < 1 or concurrency > 128:
        raise ValueError("concurrency must be between 1 and 128")
    if part_size < 8 * 1024 * 1024 or part_size > 512 * 1024 * 1024:
        raise ValueError("part size must be between 8 MiB and 512 MiB")

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        # Read immutable metadata and the manifest first. _READY is read last so
        # a partially published release can never authorize a download.
        info_body = _read_control_object(client, bucket, f"{prefix}/info.json")
        manifest_body = _read_control_object(
            client, bucket, f"{prefix}/artifact-manifest.json"
        )
        ready_body = _read_control_object(client, bucket, f"{prefix}/_READY")
        info, manifest, files = validate_release_controls(
            info_body,
            ready_body,
            manifest_body,
            expected_model_name=expected_model_name,
            expected_model_version=expected_model_version,
            expected_release_id=expected_release_id,
            expected_revision=expected_revision,
        )
        total_bytes = sum(entry["size"] for entry in files)

        if _cache_matches(destination, info_body, ready_body, manifest_body, files):
            return {
                "backend": "awscrt",
                "bytes": total_bytes,
                "cache_hit": True,
                "destination": str(destination),
                "model_name": info["model_name"],
                "model_version": info["model_version"],
                "release_id": info["release_id"],
                "revision": info["source_revision"],
            }

        available_bytes = shutil.disk_usage(destination.parent).free
        if available_bytes < total_bytes:
            raise ValueError(
                "insufficient disk space for model staging: "
                f"required {total_bytes} bytes, available {available_bytes} bytes"
            )

        staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
        staging.mkdir(parents=False)
        started = time.monotonic()
        try:
            download_started = time.monotonic()
            _download_files(
                client,
                manager_factory,
                bucket,
                prefix,
                staging,
                files,
                concurrency,
                part_size,
            )
            download_elapsed_seconds = time.monotonic() - download_started
            verify_started = time.monotonic()
            _verify_sha256(staging, files)
            verify_elapsed_seconds = time.monotonic() - verify_started
            (staging / "artifact-manifest.json").write_bytes(manifest_body)
            (staging / "info.json").write_bytes(info_body)
            # The completion marker is deliberately written last.
            (staging / "_READY").write_bytes(ready_body)
            _activate_staging(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        elapsed_seconds = time.monotonic() - started
        return {
            "backend": "awscrt",
            "bytes": total_bytes,
            "cache_hit": False,
            "concurrency": concurrency,
            "destination": str(destination),
            "download_elapsed_seconds": round(download_elapsed_seconds, 3),
            "download_throughput_gbps": round(
                total_bytes * 8 / download_elapsed_seconds / 1e9, 3
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "end_to_end_throughput_gbps": round(
                total_bytes * 8 / elapsed_seconds / 1e9, 3
            ),
            "part_size_bytes": part_size,
            "model_name": info["model_name"],
            "model_version": info["model_version"],
            "release_id": info["release_id"],
            "revision": info["source_revision"],
            "verify_elapsed_seconds": round(verify_elapsed_seconds, 3),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--model-s3-uri")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-release-id", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-2"))
    parser.add_argument("--expected-revision")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--part-size-mib", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    import boto3
    from botocore.config import Config

    args = parse_args()
    bucket, prefix, expected_release_id = resolve_model_source(
        bucket=args.bucket,
        prefix=args.prefix,
        model_s3_uri=args.model_s3_uri,
        model_name=args.model_name,
        model_version=args.model_version,
        model_release_id=args.model_release_id,
    )
    client = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 8}),
    )
    result = stage_model(
        client=client,
        bucket=bucket,
        prefix=prefix,
        destination=args.destination,
        lock_path=args.lock_path,
        expected_model_name=args.model_name,
        expected_model_version=args.model_version,
        expected_release_id=expected_release_id,
        expected_revision=args.expected_revision,
        concurrency=args.concurrency,
        part_size=args.part_size_mib * 1024 * 1024,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
