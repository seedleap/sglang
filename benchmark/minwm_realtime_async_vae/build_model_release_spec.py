#!/usr/bin/env python3
"""Build a version-pinned model release spec using read-only S3 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from download_model_artifact import validate_control_files


MODEL_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


def _read_versioned_object(
    client: Any, bucket: str, key: str
) -> tuple[bytes, str]:
    response = client.get_object(Bucket=bucket, Key=key)
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise ValueError(f"source object is not version-pinned: s3://{bucket}/{key}")
    return response["Body"].read(), version_id


def build_release_spec(
    *,
    client: Any,
    source_bucket: str,
    source_region: str,
    source_prefix: str,
    source_manifest_key: str | None,
    destination_bucket: str,
    destination_region: str,
    model_family: str,
    model_id: str,
    revision: str,
    rollback_release: str | None,
    created_at: datetime,
) -> dict[str, Any]:
    """Inventory an existing immutable source without writing any S3 object."""
    for name, value in (("model-family", model_family), ("model-id", model_id)):
        if not MODEL_COMPONENT_RE.fullmatch(value):
            raise ValueError(f"{name} is not a safe release path component")
    if not REVISION_RE.fullmatch(revision):
        raise ValueError("revision is not a safe release path component")
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")

    prefix = source_prefix.strip("/")
    ready_key = f"{prefix}/_READY"
    manifest_key = (
        source_manifest_key.strip("/")
        if source_manifest_key
        else f"{prefix}/artifact-manifest.json"
    )
    ready_body, ready_version_id = _read_versioned_object(
        client, source_bucket, ready_key
    )
    manifest_body, manifest_version_id = _read_versioned_object(
        client, source_bucket, manifest_key
    )
    manifest, files = validate_control_files(ready_body, manifest_body, revision)

    object_version_ids: dict[str, str] = {}
    for entry in files:
        path = entry["path"]
        response = client.head_object(
            Bucket=source_bucket,
            Key=f"{prefix}/{path}",
        )
        version_id = response.get("VersionId")
        if not isinstance(version_id, str) or not version_id:
            raise ValueError(f"source object has no VersionId: {path}")
        if response["ContentLength"] != entry["size"]:
            raise ValueError(f"source size mismatch for {path}")
        object_version_ids[path] = version_id

    manifest_sha256 = hashlib.sha256(manifest_body).hexdigest()
    release_timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"{release_timestamp}-{manifest_sha256[:8]}"
    destination_prefix = (
        f"models/{model_family}/{model_id}/{revision}/releases/{release_id}/model"
    )
    return {
        "schema_version": 1,
        "release_id": release_id,
        "release_manifest_created_at": created_at.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model": {
            "family": model_family,
            "id": model_id,
            "revision": revision,
            "rollback_release": rollback_release,
        },
        "source": {
            "bucket": source_bucket,
            "region": source_region,
            "prefix": prefix,
            "ready": {
                "key": ready_key,
                "version_id": ready_version_id,
                "size": len(ready_body),
                "sha256": hashlib.sha256(ready_body).hexdigest(),
            },
            "artifact_manifest": {
                "key": manifest_key,
                "version_id": manifest_version_id,
                "size": len(manifest_body),
                "sha256": manifest_sha256,
            },
            "object_version_ids": object_version_ids,
            "object_count": len(files),
            "bytes": sum(entry["size"] for entry in files),
            "manifest_revision": manifest.get("revision")
            or manifest["resolved_revision"],
        },
        "destination": {
            "bucket": destination_bucket,
            "region": destination_region,
            "prefix": destination_prefix,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bucket", required=True)
    parser.add_argument("--source-region", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument(
        "--source-manifest-key",
        help=(
            "source manifest object key; defaults to "
            "<source-prefix>/artifact-manifest.json"
        ),
    )
    parser.add_argument("--destination-bucket", required=True)
    parser.add_argument("--destination-region", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--rollback-release")
    return parser.parse_args()


def main() -> None:
    import boto3
    from botocore.config import Config

    args = parse_args()
    client = boto3.client(
        "s3",
        region_name=args.source_region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 8}),
    )
    result = build_release_spec(
        client=client,
        source_bucket=args.source_bucket,
        source_region=args.source_region,
        source_prefix=args.source_prefix,
        source_manifest_key=args.source_manifest_key,
        destination_bucket=args.destination_bucket,
        destination_region=args.destination_region,
        model_family=args.model_family,
        model_id=args.model_id,
        revision=args.revision,
        rollback_release=args.rollback_release,
        created_at=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
