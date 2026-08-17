#!/usr/bin/env python3
"""Plan, copy, and verify one version-pinned immutable model release."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from download_model_artifact import validate_control_files, validate_release_controls

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PUBLISHER_BUNDLE_FILES = (
    "build_model_release_spec.py",
    "copy_model_release.py",
    "download_model_artifact.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checksum_b64(hex_digest: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_digest)).decode("ascii")


def publisher_script_inventory(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the exact generic builder, copy/verify, and init/rollback scripts."""
    root = root or Path(__file__).resolve().parent
    inventory: dict[str, dict[str, Any]] = {}
    for relative_path in PUBLISHER_BUNDLE_FILES:
        payload = (root / relative_path).read_bytes()
        inventory[relative_path] = {
            "size": len(payload),
            "sha256": _sha256(payload),
        }
    return inventory


def publisher_bundle_sha256(root: Path | None = None) -> str:
    inventory = publisher_script_inventory(root)
    body = json.dumps(
        inventory,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256(body)


def render_destination_control_bodies(
    release: dict[str, Any],
    release_spec_sha256: str,
    manifest_body: bytes,
    manifest_files: dict[str, Any],
) -> tuple[bytes, bytes]:
    """Render deterministic info and _READY bodies for the three-control contract."""
    if not SHA256_RE.fullmatch(release_spec_sha256):
        raise ValueError("release_spec_sha256 is invalid")
    model = release["model"]
    total_bytes = sum(entry["size"] for entry in manifest_files.values())
    info = {
        "schema_version": 1,
        "model_name": model["serving_name"],
        "model_version": model["version"],
        "release_id": release["release_id"],
        "created_at": release["release_manifest_created_at"],
        "source_revision": model["revision"],
        "source_uri": (
            f"s3://{release['source']['bucket']}/"
            f"{release['source']['prefix'].strip('/')}/"
        ),
        "artifact_manifest_sha256": _sha256(manifest_body),
        "artifact_object_count": len(manifest_files),
        "artifact_bytes": total_bytes,
        "source_ready_sha256": release["source"]["ready"]["sha256"],
        "publisher_bundle_sha256": release["publisher_bundle_sha256"],
        "release_spec_sha256": release_spec_sha256,
        "rollback_release": model.get("rollback_release"),
    }
    info_body = json.dumps(
        info,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    ready = {
        "schema_version": 1,
        "revision": model["revision"],
        "release_id": release["release_id"],
        "manifest_sha256": _sha256(manifest_body),
        "info_sha256": _sha256(info_body),
        "publisher_bundle_sha256": release["publisher_bundle_sha256"],
        "release_spec_sha256": release_spec_sha256,
    }
    ready_body = json.dumps(
        ready,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return info_body, ready_body


def destination_control_hashes(
    manifest_body: bytes, info_body: bytes, ready_body: bytes
) -> dict[str, str]:
    return {
        "artifact_manifest_sha256": _sha256(manifest_body),
        "info_sha256": _sha256(info_body),
        "ready_sha256": _sha256(ready_body),
    }


def _object_bytes(client: Any, bucket: str, key: str, version_id: str) -> bytes:
    return client.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
    )["Body"].read()


def _source_control_key(source: dict[str, Any], control: str, default_name: str) -> str:
    value = source[control]
    key = value.get("key") if isinstance(value, dict) else None
    if key is None:
        key = f"{source['prefix'].strip('/')}/{default_name}"
    if not isinstance(key, str) or not key or key.startswith("/") or "\\" in key:
        raise ValueError(f"source.{control}.key is not a safe S3 object key")
    parsed = PurePosixPath(key)
    if key != parsed.as_posix() or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError(f"source.{control}.key is not a safe S3 object key")
    return key


def load_and_validate_release(
    release_path: Path,
    source_client: Any,
) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != 1:
        raise ValueError("unsupported release schema_version")
    expected_scripts = release.get("publisher_scripts")
    actual_scripts = publisher_script_inventory()
    if expected_scripts != actual_scripts:
        raise ValueError("runtime publisher scripts do not match release spec")
    expected_bundle_sha256 = release.get("publisher_bundle_sha256")
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise ValueError("release has no valid publisher_bundle_sha256")
    if publisher_bundle_sha256() != expected_bundle_sha256:
        raise ValueError("runtime publisher bundle does not match release spec")

    source = release["source"]
    ready_spec = source["ready"]
    manifest_spec = source["artifact_manifest"]
    for name, control_spec in (
        ("ready", ready_spec),
        ("artifact_manifest", manifest_spec),
    ):
        if (
            not isinstance(control_spec, dict)
            or not isinstance(control_spec.get("version_id"), str)
            or not control_spec["version_id"]
            or not isinstance(control_spec.get("size"), int)
            or isinstance(control_spec["size"], bool)
            or control_spec["size"] < 0
            or not isinstance(control_spec.get("sha256"), str)
            or not SHA256_RE.fullmatch(control_spec["sha256"])
        ):
            raise ValueError(f"source.{name} is not fully version/hash pinned")
    ready_key = _source_control_key(source, "ready", "_READY")
    manifest_key = _source_control_key(
        source, "artifact_manifest", "artifact-manifest.json"
    )
    ready_body = _object_bytes(
        source_client,
        source["bucket"],
        ready_key,
        ready_spec["version_id"],
    )
    manifest_body = _object_bytes(
        source_client,
        source["bucket"],
        manifest_key,
        manifest_spec["version_id"],
    )

    if (
        len(ready_body) != ready_spec["size"]
        or _sha256(ready_body) != ready_spec["sha256"]
    ):
        raise ValueError("version-pinned source _READY does not match release spec")
    if (
        len(manifest_body) != manifest_spec["size"]
        or _sha256(manifest_body) != manifest_spec["sha256"]
    ):
        raise ValueError("version-pinned artifact manifest does not match release spec")

    ready = json.loads(ready_body)
    expected_revision = (release.get("model") or {}).get("revision")
    manifest, validated_files = validate_control_files(
        ready_body,
        manifest_body,
        expected_revision,
        require_canonical_ready_revision=True,
    )
    if ready.get("manifest_sha256") != manifest_spec["sha256"]:
        raise ValueError("source _READY does not authorize artifact manifest")
    expected_count = source.get("object_count")
    expected_bytes = source.get("bytes")
    if expected_count is not None and expected_count != len(validated_files):
        raise ValueError("source object_count does not match artifact manifest")
    actual_bytes = sum(entry["size"] for entry in validated_files)
    if expected_bytes is not None and expected_bytes != actual_bytes:
        raise ValueError("source bytes does not match artifact manifest")

    manifest_files = {entry["path"]: entry for entry in validated_files}
    version_ids = source.get("object_version_ids")
    if not isinstance(version_ids, dict) or set(version_ids) != set(manifest_files):
        raise ValueError("source object VersionIds do not exactly match manifest files")
    if not all(isinstance(value, str) and value for value in version_ids.values()):
        raise ValueError("every source object must have a VersionId")
    return release, ready_body, manifest_body, manifest_files


def execution_bundle_document(
    release: dict[str, Any],
    manifest_body: bytes,
    manifest_files: dict[str, Any],
    release_spec_sha256: str,
) -> dict[str, Any]:
    """Render the complete approval object for one immutable publication."""
    if not SHA256_RE.fullmatch(release_spec_sha256):
        raise ValueError("release_spec_sha256 is invalid")
    source = release["source"]
    payloads = []
    for path, entry in sorted(manifest_files.items()):
        payloads.append(
            {
                "path": path,
                "size": entry["size"],
                "sha256": entry["sha256"],
                "source_version_id": source["object_version_ids"][path],
            }
        )
    info_body, destination_ready_body = render_destination_control_bodies(
        release, release_spec_sha256, manifest_body, manifest_files
    )
    return {
        "schema_version": 1,
        "release_spec_sha256": release_spec_sha256,
        "release_id": release["release_id"],
        "model": release["model"],
        "publisher_scripts": release["publisher_scripts"],
        "publisher_bundle_sha256": release["publisher_bundle_sha256"],
        "source": {
            "bucket": source["bucket"],
            "region": source["region"],
            "prefix": source["prefix"],
            "ready": {
                **source["ready"],
                "key": _source_control_key(source, "ready", "_READY"),
            },
            "artifact_manifest": {
                **source["artifact_manifest"],
                "key": _source_control_key(
                    source, "artifact_manifest", "artifact-manifest.json"
                ),
            },
        },
        "payloads": payloads,
        "destination": release["destination"],
        "destination_controls": destination_control_hashes(
            manifest_body, info_body, destination_ready_body
        ),
    }


def execution_bundle_sha256(
    release: dict[str, Any],
    manifest_body: bytes,
    manifest_files: dict[str, Any],
    release_spec_sha256: str,
) -> str:
    body = json.dumps(
        execution_bundle_document(
            release, manifest_body, manifest_files, release_spec_sha256
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256(body)


def validate_source_objects(
    release: dict[str, Any],
    manifest_files: dict[str, Any],
    source_client: Any,
) -> None:
    """Validate every pinned source object's size and full content SHA256."""
    source = release["source"]
    prefix = source["prefix"].strip("/")
    for path, entry in manifest_files.items():
        version_id = source["object_version_ids"][path]
        response = source_client.head_object(
            Bucket=source["bucket"],
            Key=f"{prefix}/{path}",
            VersionId=version_id,
            ChecksumMode="ENABLED",
        )
        if response["ContentLength"] != entry["size"]:
            raise ValueError(f"source size mismatch for {path}")
        returned_version_id = response.get("VersionId")
        if returned_version_id is not None and returned_version_id != version_id:
            raise ValueError(f"source VersionId mismatch for {path}")

        checksum = response.get("ChecksumSHA256")
        checksum_type = response.get("ChecksumType")
        if checksum is not None and checksum_type in {None, "FULL_OBJECT"}:
            if checksum != _checksum_b64(entry["sha256"]):
                raise ValueError(f"source SHA256 mismatch for {path}")
            continue

        object_response = source_client.get_object(
            Bucket=source["bucket"],
            Key=f"{prefix}/{path}",
            VersionId=version_id,
            ChecksumMode="ENABLED",
        )
        content_length = object_response.get("ContentLength")
        if content_length is not None and content_length != entry["size"]:
            raise ValueError(f"source size mismatch for {path}")
        digest = hashlib.sha256()
        body = object_response["Body"]
        try:
            for block in iter(lambda: body.read(8 * 1024 * 1024), b""):
                digest.update(block)
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"source SHA256 mismatch for {path}")


def _destination_head(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    from botocore.exceptions import ClientError

    try:
        return client.head_object(
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise


def _verify_destination(
    response: dict[str, Any],
    path: str,
    expected_size: int,
    expected_sha256: str,
    expected_source_version_id: str | None = None,
) -> None:
    if response["ContentLength"] != expected_size:
        raise ValueError(f"destination size mismatch for {path}")
    if expected_source_version_id is None:
        if response.get("ChecksumSHA256") != _checksum_b64(expected_sha256):
            raise ValueError(f"destination SHA256 mismatch for {path}")
        return

    metadata = response.get("Metadata")
    if not isinstance(metadata, dict) or metadata.get("sha256") != expected_sha256:
        raise ValueError(f"destination SHA256 metadata mismatch for {path}")
    if metadata.get("source-version-id") != expected_source_version_id:
        raise ValueError(f"destination source VersionId metadata mismatch for {path}")
    checksum = response.get("ChecksumSHA256")
    checksum_type = response.get("ChecksumType")
    if (
        checksum is not None
        and checksum_type in {None, "FULL_OBJECT"}
        and checksum != _checksum_b64(expected_sha256)
    ):
        raise ValueError(f"destination SHA256 mismatch for {path}")


def _list_destination_keys(client: Any, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    continuation_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": f"{prefix.strip('/')}/"}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        for entry in response.get("Contents", []):
            key = entry.get("Key")
            if not isinstance(key, str):
                raise ValueError("destination listing returned an invalid object key")
            keys.add(key)
        if not response.get("IsTruncated"):
            return keys
        continuation_token = response.get("NextContinuationToken")
        if not isinstance(continuation_token, str) or not continuation_token:
            raise ValueError(
                "destination listing is truncated without a continuation token"
            )


def _expected_destination_keys(prefix: str, manifest_files: dict[str, Any]) -> set[str]:
    prefix = prefix.strip("/")
    return {
        f"{prefix}/artifact-manifest.json",
        f"{prefix}/info.json",
        f"{prefix}/_READY",
        *(f"{prefix}/model/{path}" for path in manifest_files),
    }


def _validate_destination_before_copy(
    release: dict[str, Any],
    manifest_files: dict[str, Any],
    destination_client: Any,
) -> None:
    """Reject foreign keys and validate all reusable payloads before any copy."""
    source = release["source"]
    destination = release["destination"]
    prefix = destination["prefix"].strip("/")
    existing_keys = _list_destination_keys(
        destination_client, destination["bucket"], prefix
    )
    unexpected = existing_keys - _expected_destination_keys(prefix, manifest_files)
    if unexpected:
        raise ValueError(
            "destination release prefix has unexpected keys: "
            + ", ".join(sorted(unexpected))
        )
    if f"{prefix}/_READY" in existing_keys:
        raise RuntimeError("destination _READY already exists")
    for path, entry in manifest_files.items():
        key = f"{prefix}/model/{path}"
        if key not in existing_keys:
            continue
        response = _destination_head(destination_client, destination["bucket"], key)
        if response is None:
            raise RuntimeError(
                f"destination object disappeared during preflight: {path}"
            )
        _verify_destination(
            response,
            path,
            entry["size"],
            entry["sha256"],
            source["object_version_ids"][path],
        )


def _copy_one(
    release: dict[str, Any],
    path: str,
    entry: dict[str, Any],
    source_client: Any,
    destination_client: Any,
) -> dict[str, Any]:
    source = release["source"]
    destination = release["destination"]
    source_key = f"{source['prefix'].strip('/')}/{path}"
    destination_key = f"{destination['prefix'].strip('/')}/model/{path}"

    existing = _destination_head(
        destination_client, destination["bucket"], destination_key
    )
    if existing is not None:
        _verify_destination(
            existing,
            path,
            entry["size"],
            entry["sha256"],
            source["object_version_ids"][path],
        )
        return {
            "path": path,
            "size": entry["size"],
            "sha256": entry["sha256"],
            "source_version_id": source["object_version_ids"][path],
            "destination_version_id": existing["VersionId"],
            "reused": True,
        }

    from boto3.s3.transfer import TransferConfig

    # The managed copy automatically switches to multipart copy above 5 GiB.
    # Outer object concurrency is already bounded by execute_copy, so multipart
    # parts are intentionally sequential per object to avoid a thread explosion.
    destination_client.copy(
        {
            "Bucket": source["bucket"],
            "Key": source_key,
            "VersionId": source["object_version_ids"][path],
        },
        destination["bucket"],
        destination_key,
        ExtraArgs={
            "ChecksumAlgorithm": "SHA256",
            "MetadataDirective": "REPLACE",
            "Metadata": {
                "sha256": entry["sha256"],
                "source-version-id": source["object_version_ids"][path],
            },
        },
        SourceClient=source_client,
        Config=TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=1,
            use_threads=False,
        ),
    )
    copied = destination_client.head_object(
        Bucket=destination["bucket"],
        Key=destination_key,
        ChecksumMode="ENABLED",
    )
    _verify_destination(
        copied,
        path,
        entry["size"],
        entry["sha256"],
        source["object_version_ids"][path],
    )
    return {
        "path": path,
        "size": entry["size"],
        "sha256": entry["sha256"],
        "source_version_id": source["object_version_ids"][path],
        "destination_version_id": copied["VersionId"],
        "reused": False,
    }


def _put_control_object(
    client: Any,
    bucket: str,
    key: str,
    body: bytes,
    *,
    refuse_existing: bool = False,
) -> str:
    digest = _sha256(body)
    existing = _destination_head(client, bucket, key)
    if existing is not None:
        if refuse_existing:
            raise RuntimeError(f"immutable control already exists: s3://{bucket}/{key}")
        _verify_destination(existing, key, len(body), digest)
        version_id = existing["VersionId"]
    else:
        response = client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=_checksum_b64(digest),
            IfNoneMatch="*",
        )
        version_id = response["VersionId"]
    readback = client.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ChecksumMode="ENABLED",
    )["Body"].read()
    if readback != body:
        raise ValueError(f"control object readback mismatch for {key}")
    return version_id


def offline_plan(
    release: dict[str, Any], *, release_spec_sha256: str | None = None
) -> dict[str, Any]:
    """Render a zero-network plan, including unresolved inventory explicitly."""
    source = release["source"]
    destination = release["destination"]
    version_ids = source.get("object_version_ids")
    version_ids = version_ids if isinstance(version_ids, dict) else {}
    release_id = release.get("release_id")
    unresolved = []
    if not isinstance(release_id, str) or release_id.startswith("REPLACE_"):
        unresolved.append("release_id")
    if not version_ids:
        unresolved.append("source.object_version_ids")
    for control in ("ready", "artifact_manifest"):
        value = source.get(control)
        if not isinstance(value, dict) or not value.get("version_id"):
            unresolved.append(f"source.{control}.version_id")
    bundle_sha256 = release.get("publisher_bundle_sha256")
    if (
        not isinstance(bundle_sha256, str)
        or not SHA256_RE.fullmatch(bundle_sha256)
        or bundle_sha256 != publisher_bundle_sha256()
    ):
        unresolved.append("publisher_bundle_sha256")
    if release.get("publisher_scripts") != publisher_script_inventory():
        unresolved.append("publisher_scripts")
    return {
        "action": "copy-version-pinned-release",
        "execute": False,
        "network": False,
        "source": f"s3://{source['bucket']}/{source['prefix'].strip('/')}/",
        "destination": (
            f"s3://{destination['bucket']}/{destination['prefix'].strip('/')}/"
        ),
        "release_id": release_id,
        "model_object_count": source.get("object_count", len(version_ids) or None),
        "bytes": source.get("bytes"),
        "publisher_bundle_sha256": bundle_sha256,
        "release_spec_sha256": release_spec_sha256,
        "unresolved": sorted(unresolved),
        "release_spec_ready": not unresolved,
    }


def verify_destination_release(
    release: dict[str, Any],
    destination_client: Any,
    release_spec_sha256: str,
) -> dict[str, Any]:
    destination = release["destination"]
    bucket = destination["bucket"]
    prefix = destination["prefix"].strip("/")

    ready_body = destination_client.get_object(Bucket=bucket, Key=f"{prefix}/_READY")[
        "Body"
    ].read()
    info_body = destination_client.get_object(Bucket=bucket, Key=f"{prefix}/info.json")[
        "Body"
    ].read()
    manifest_body = destination_client.get_object(
        Bucket=bucket, Key=f"{prefix}/artifact-manifest.json"
    )["Body"].read()
    model = release["model"]
    _, _, validated_files = validate_release_controls(
        info_body,
        ready_body,
        manifest_body,
        expected_model_name=model["serving_name"],
        expected_model_version=model["version"],
        expected_release_id=release["release_id"],
        expected_revision=model["revision"],
    )
    expected_objects = {entry["path"]: entry for entry in validated_files}
    expected_info_body, expected_ready_body = render_destination_control_bodies(
        release, release_spec_sha256, manifest_body, expected_objects
    )
    if info_body != expected_info_body:
        raise ValueError("destination info.json does not match release spec")
    if ready_body != expected_ready_body:
        raise ValueError("destination _READY does not match release spec")
    expected_keys = _expected_destination_keys(prefix, expected_objects)
    actual_keys = _list_destination_keys(destination_client, bucket, prefix)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "destination release key set differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    source_versions = release["source"]["object_version_ids"]
    for path, entry in expected_objects.items():
        response = destination_client.head_object(
            Bucket=bucket,
            Key=f"{prefix}/model/{path}",
            ChecksumMode="ENABLED",
        )
        _verify_destination(
            response,
            path,
            entry["size"],
            entry["sha256"],
            source_versions[path],
        )
    return {
        "bucket": bucket,
        "prefix": prefix,
        "release_id": release["release_id"],
        "object_count": len(expected_objects) + 3,
        "payload_object_count": len(expected_objects),
        "bytes": sum(entry["size"] for entry in expected_objects.values()),
        "verified": True,
    }


def execute_copy(
    release: dict[str, Any],
    ready_body: bytes,
    manifest_body: bytes,
    manifest_files: dict[str, Any],
    source_client: Any,
    destination_client: Any,
    concurrency: int,
    release_spec_sha256: str,
    confirmed_execution_bundle_sha256: str,
) -> dict[str, Any]:
    destination = release["destination"]
    destination_prefix = destination["prefix"].strip("/")
    ready_key = f"{destination_prefix}/_READY"
    if _destination_head(destination_client, destination["bucket"], ready_key):
        raise RuntimeError(
            f"immutable release is already complete: s3://{destination['bucket']}/{ready_key}"
        )
    if not SHA256_RE.fullmatch(release_spec_sha256):
        raise ValueError("release_spec_sha256 is invalid")
    actual_execution_bundle_sha256 = execution_bundle_sha256(
        release, manifest_body, manifest_files, release_spec_sha256
    )
    if confirmed_execution_bundle_sha256 != actual_execution_bundle_sha256:
        raise ValueError("confirmed execution bundle SHA256 does not match")
    _validate_destination_before_copy(release, manifest_files, destination_client)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _copy_one,
                release,
                path,
                entry,
                source_client,
                destination_client,
            )
            for path, entry in sorted(manifest_files.items())
        ]
        copied = [future.result() for future in futures]

    # Revalidate every payload after all copy futures complete and before the
    # first control write. This also catches concurrent foreign-key creation.
    _validate_destination_before_copy(release, manifest_files, destination_client)

    info_body, destination_ready_body = render_destination_control_bodies(
        release, release_spec_sha256, manifest_body, manifest_files
    )

    _put_control_object(
        destination_client,
        destination["bucket"],
        f"{destination_prefix}/artifact-manifest.json",
        manifest_body,
    )
    total_bytes = sum(entry["size"] for entry in manifest_files.values())
    _put_control_object(
        destination_client,
        destination["bucket"],
        f"{destination_prefix}/info.json",
        info_body,
    )

    pre_ready_keys = _list_destination_keys(
        destination_client, destination["bucket"], destination_prefix
    )
    expected_pre_ready_keys = _expected_destination_keys(
        destination_prefix, manifest_files
    ) - {ready_key}
    if pre_ready_keys != expected_pre_ready_keys:
        raise ValueError("destination key set is not exact before _READY publication")

    ready_version_id = _put_control_object(
        destination_client,
        destination["bucket"],
        ready_key,
        destination_ready_body,
        refuse_existing=True,
    )
    return {
        "bucket": destination["bucket"],
        "prefix": destination_prefix,
        "object_count": len(copied) + 3,
        "payload_object_count": len(copied),
        "bytes": total_bytes,
        "execution_bundle_sha256": actual_execution_bundle_sha256,
        "ready_version_id": ready_version_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--offline-plan", action="store_true")
    parser.add_argument("--confirm-release-id")
    parser.add_argument("--confirm-release-spec-sha256")
    parser.add_argument("--confirm-execution-bundle-sha256")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1 or args.concurrency > 32:
        raise ValueError("concurrency must be between 1 and 32")
    release_body = args.release.read_bytes()
    release_spec_sha256 = _sha256(release_body)
    release_document = json.loads(release_body)
    if args.offline_plan:
        print(
            json.dumps(
                offline_plan(release_document, release_spec_sha256=release_spec_sha256),
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.execute and args.confirm_release_id != release_document.get("release_id"):
        raise ValueError(
            "--execute requires --confirm-release-id matching the reviewed release"
        )
    if args.execute and args.confirm_release_spec_sha256 != release_spec_sha256:
        raise ValueError(
            "--execute requires --confirm-release-spec-sha256 matching the "
            "reviewed release spec bytes"
        )

    import boto3
    from botocore.config import Config

    if args.verify:
        if release_document.get("publisher_scripts") != publisher_script_inventory():
            raise ValueError("runtime publisher scripts do not match release spec")
        if release_document.get("publisher_bundle_sha256") != publisher_bundle_sha256():
            raise ValueError("runtime publisher bundle does not match release spec")
        destination_session = boto3.Session(
            region_name=release_document["destination"]["region"]
        )
        destination_client = destination_session.client(
            "s3", config=Config(retries={"mode": "adaptive", "max_attempts": 8})
        )
        result = verify_destination_release(
            release_document, destination_client, release_spec_sha256
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return

    source_session = boto3.Session(region_name=release_document["source"]["region"])
    source_client = source_session.client(
        "s3", config=Config(retries={"mode": "adaptive", "max_attempts": 8})
    )
    release, ready_body, manifest_body, manifest_files = load_and_validate_release(
        args.release, source_client
    )
    validate_source_objects(release, manifest_files, source_client)
    reviewed_execution_bundle_sha256 = execution_bundle_sha256(
        release, manifest_body, manifest_files, release_spec_sha256
    )

    plan = {
        "action": "copy-version-pinned-release",
        "execute": args.execute,
        "network": True,
        "source": f"s3://{release['source']['bucket']}/{release['source']['prefix']}/",
        "destination": f"s3://{release['destination']['bucket']}/{release['destination']['prefix']}/",
        "model_object_count": len(manifest_files),
        "control_object_count": 3,
        "bytes": sum(entry["size"] for entry in manifest_files.values()),
        "publisher_bundle_sha256": release["publisher_bundle_sha256"],
        "release_spec_sha256": release_spec_sha256,
        "execution_bundle_sha256": reviewed_execution_bundle_sha256,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.confirm_execution_bundle_sha256 != reviewed_execution_bundle_sha256:
        raise ValueError(
            "--execute requires --confirm-execution-bundle-sha256 matching the "
            "reviewed controls/scripts/payload/source/destination bundle"
        )

    destination_session = boto3.Session(region_name=release["destination"]["region"])
    destination_client = destination_session.client(
        "s3", config=Config(retries={"mode": "adaptive", "max_attempts": 8})
    )
    result = execute_copy(
        release,
        ready_body,
        manifest_body,
        manifest_files,
        source_client,
        destination_client,
        args.concurrency,
        release_spec_sha256,
        args.confirm_execution_bundle_sha256,
    )
    verification = verify_destination_release(
        release, destination_client, release_spec_sha256
    )
    print(
        json.dumps({"copy": result, "verification": verification}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
