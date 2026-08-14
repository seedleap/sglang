#!/usr/bin/env python3
"""Plan, copy, and verify one version-pinned immutable model release."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from download_model_artifact import validate_control_files


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checksum_b64(hex_digest: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_digest)).decode("ascii")


def _object_bytes(client: Any, bucket: str, key: str, version_id: str) -> bytes:
    return client.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
    )["Body"].read()


def load_and_validate_release(
    release_path: Path,
    source_client: Any,
) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != 1:
        raise ValueError("unsupported release schema_version")

    source = release["source"]
    source_prefix = source["prefix"].strip("/")
    ready_spec = source["ready"]
    manifest_spec = source["artifact_manifest"]
    ready_body = _object_bytes(
        source_client,
        source["bucket"],
        f"{source_prefix}/_READY",
        ready_spec["version_id"],
    )
    manifest_body = _object_bytes(
        source_client,
        source["bucket"],
        f"{source_prefix}/artifact-manifest.json",
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
    manifest, validated_files = validate_control_files(
        ready_body, manifest_body, ready.get("revision")
    )
    if ready.get("manifest_sha256") != manifest_spec["sha256"]:
        raise ValueError("source _READY does not authorize artifact manifest")
    if ready.get("revision") != manifest.get("revision"):
        raise ValueError("source control file revisions differ")

    manifest_files = {entry["path"]: entry for entry in validated_files}
    version_ids = source.get("object_version_ids")
    if not isinstance(version_ids, dict) or set(version_ids) != set(manifest_files):
        raise ValueError("source object VersionIds do not exactly match manifest files")
    if not all(isinstance(value, str) and value for value in version_ids.values()):
        raise ValueError("every source object must have a VersionId")
    return release, ready_body, manifest_body, manifest_files


def validate_source_objects(
    release: dict[str, Any],
    manifest_files: dict[str, Any],
    source_client: Any,
) -> None:
    source = release["source"]
    prefix = source["prefix"].strip("/")
    for path, entry in manifest_files.items():
        response = source_client.head_object(
            Bucket=source["bucket"],
            Key=f"{prefix}/{path}",
            VersionId=source["object_version_ids"][path],
        )
        if response["ContentLength"] != entry["size"]:
            raise ValueError(f"source size mismatch for {path}")


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
) -> None:
    if response["ContentLength"] != expected_size:
        raise ValueError(f"destination size mismatch for {path}")
    if response.get("ChecksumSHA256") != _checksum_b64(expected_sha256):
        raise ValueError(f"destination SHA256 mismatch for {path}")


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
    destination_key = f"{destination['prefix'].strip('/')}/{path}"

    existing = _destination_head(
        destination_client, destination["bucket"], destination_key
    )
    if existing is not None:
        _verify_destination(existing, path, entry["size"], entry["sha256"])
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
        ExtraArgs={"ChecksumAlgorithm": "SHA256", "MetadataDirective": "COPY"},
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
    _verify_destination(copied, path, entry["size"], entry["sha256"])
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
) -> str:
    digest = _sha256(body)
    existing = _destination_head(client, bucket, key)
    if existing is not None:
        _verify_destination(existing, key, len(body), digest)
        return existing["VersionId"]
    response = client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ChecksumAlgorithm="SHA256",
        ChecksumSHA256=_checksum_b64(digest),
        IfNoneMatch="*",
    )
    return response["VersionId"]


def offline_plan(release: dict[str, Any]) -> dict[str, Any]:
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
        "unresolved": sorted(unresolved),
    }


def verify_destination_release(
    release: dict[str, Any], destination_client: Any
) -> dict[str, Any]:
    destination = release["destination"]
    bucket = destination["bucket"]
    prefix = destination["prefix"].strip("/")
    release_root = prefix.rsplit("/model", 1)[0]

    ready_body = destination_client.get_object(
        Bucket=bucket, Key=f"{prefix}/_READY"
    )["Body"].read()
    manifest_body = destination_client.get_object(
        Bucket=bucket, Key=f"{prefix}/artifact-manifest.json"
    )["Body"].read()
    release_manifest_body = destination_client.get_object(
        Bucket=bucket, Key=f"{release_root}/release-manifest.json"
    )["Body"].read()
    ready = json.loads(ready_body)
    release_manifest = json.loads(release_manifest_body)
    if ready.get("manifest_sha256") != _sha256(manifest_body):
        raise ValueError("destination _READY does not authorize artifact manifest")
    if ready.get("release_manifest_sha256") != _sha256(release_manifest_body):
        raise ValueError("destination _READY does not authorize release-manifest")
    if ready.get("release_id") != release["release_id"]:
        raise ValueError("destination _READY has the wrong release_id")
    if release_manifest.get("release_id") != release["release_id"]:
        raise ValueError("destination release-manifest has the wrong release_id")

    objects = release_manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("destination release-manifest has no model objects")
    for entry in objects:
        path = entry["path"]
        response = destination_client.head_object(
            Bucket=bucket,
            Key=f"{prefix}/{path}",
            VersionId=entry["destination_version_id"],
            ChecksumMode="ENABLED",
        )
        _verify_destination(response, path, entry["size"], entry["sha256"])
    return {
        "bucket": bucket,
        "prefix": prefix,
        "release_id": release["release_id"],
        "object_count": len(objects),
        "bytes": sum(entry["size"] for entry in objects),
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
) -> dict[str, Any]:
    destination = release["destination"]
    destination_prefix = destination["prefix"].strip("/")
    ready_key = f"{destination_prefix}/_READY"
    if _destination_head(destination_client, destination["bucket"], ready_key):
        raise RuntimeError(
            f"immutable release is already complete: s3://{destination['bucket']}/{ready_key}"
        )

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

    artifact_manifest_version_id = _put_control_object(
        destination_client,
        destination["bucket"],
        f"{destination_prefix}/artifact-manifest.json",
        manifest_body,
    )
    release_root = destination_prefix.rsplit("/model", 1)[0]
    release_manifest = {
        "schema_version": 1,
        "release_id": release["release_id"],
        "created_at": release["release_manifest_created_at"],
        "source": release["source"],
        "destination": destination,
        "model": release.get("model"),
        "rollback_release": (release.get("model") or {}).get("rollback_release"),
        "object_count": len(copied),
        "bytes": sum(entry["size"] for entry in manifest_files.values()),
        "artifact_manifest_sha256": _sha256(manifest_body),
        "artifact_manifest_destination_version_id": artifact_manifest_version_id,
        "objects": copied,
    }
    release_manifest_body = json.dumps(
        release_manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    release_manifest_version_id = _put_control_object(
        destination_client,
        destination["bucket"],
        f"{release_root}/release-manifest.json",
        release_manifest_body,
    )
    destination_ready = json.loads(ready_body)
    destination_ready.update(
        {
            "release_id": release["release_id"],
            "release_manifest_sha256": _sha256(release_manifest_body),
            "release_manifest_version_id": release_manifest_version_id,
        }
    )
    destination_ready_body = json.dumps(
        destination_ready,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    ready_version_id = _put_control_object(
        destination_client,
        destination["bucket"],
        ready_key,
        destination_ready_body,
    )
    return {
        "bucket": destination["bucket"],
        "prefix": destination_prefix,
        "object_count": len(copied) + 3,
        "bytes": sum(entry["size"] for entry in manifest_files.values()),
        "release_manifest_version_id": release_manifest_version_id,
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
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1 or args.concurrency > 32:
        raise ValueError("concurrency must be between 1 and 32")
    release_document = json.loads(args.release.read_text(encoding="utf-8"))
    if args.offline_plan:
        print(json.dumps(offline_plan(release_document), sort_keys=True), flush=True)
        return
    if args.execute and args.confirm_release_id != release_document.get("release_id"):
        raise ValueError(
            "--execute requires --confirm-release-id matching the reviewed release"
        )

    import boto3
    from botocore.config import Config

    if args.verify:
        destination_session = boto3.Session(
            region_name=release_document["destination"]["region"]
        )
        destination_client = destination_session.client(
            "s3", config=Config(retries={"mode": "adaptive", "max_attempts": 8})
        )
        result = verify_destination_release(release_document, destination_client)
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

    plan = {
        "action": "copy-version-pinned-release",
        "execute": args.execute,
        "network": True,
        "source": f"s3://{release['source']['bucket']}/{release['source']['prefix']}/",
        "destination": f"s3://{release['destination']['bucket']}/{release['destination']['prefix']}/",
        "model_object_count": len(manifest_files),
        "control_object_count": 3,
        "bytes": sum(entry["size"] for entry in manifest_files.values()),
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return

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
    )
    verification = verify_destination_release(release, destination_client)
    print(
        json.dumps({"copy": result, "verification": verification}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
