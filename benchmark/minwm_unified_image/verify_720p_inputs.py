#!/usr/bin/env python3
"""Verify that current S3 objects still match the pinned 720p release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).with_name("inputs_720p.json")
STAGED_DONOR_COMPONENTS = {"scheduler", "text_encoder", "tokenizer", "vae"}


def _etag(value: Any) -> str | None:
    return value.strip('"') if isinstance(value, str) else None


def _aws_json(
    arguments: list[str], *, profile: str | None, region: str
) -> dict[str, Any]:
    command = ["aws"]
    if profile:
        command.extend(("--profile", profile))
    command.extend(("--region", region, *arguments, "--output", "json"))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _head(
    expected: dict[str, Any],
    *,
    profile: str | None,
    region: str,
    pinned: bool,
) -> dict[str, Any]:
    arguments = [
        "s3api",
        "head-object",
        "--bucket",
        expected["bucket"],
        "--key",
        expected["key"],
        "--checksum-mode",
        "ENABLED",
    ]
    if pinned:
        arguments.extend(("--version-id", expected["version_id"]))
    return _aws_json(arguments, profile=profile, region=region)


def _validate_head(
    label: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    errors: list[str],
) -> None:
    checks = {
        "VersionId": expected["version_id"],
        "ContentLength": expected["content_length"],
        "ETag": expected["etag"],
        "ChecksumCRC64NVME": expected["checksum_crc64nvme"],
    }
    for field, wanted in checks.items():
        got = _etag(actual.get(field)) if field == "ETag" else actual.get(field)
        if got != wanted:
            errors.append(f"{label} {field}: expected {wanted!r}, got {got!r}")


def verify(manifest_path: Path, *, profile: str | None, region: str) -> dict[str, Any]:
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    errors: list[str] = []
    evidence: dict[str, Any] = {
        "schema_version": "minwm-720p-release-input-preflight/v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "region": region,
        "profile": profile,
    }

    object_evidence = {}
    for label in ("checkpoint", "first_frame"):
        expected = manifest[label]
        current = _head(expected, profile=profile, region=region, pinned=False)
        pinned = _head(expected, profile=profile, region=region, pinned=True)
        _validate_head(f"{label}.current", expected, current, errors)
        _validate_head(f"{label}.pinned", expected, pinned, errors)
        object_evidence[label] = {"current": current, "pinned": pinned}
    evidence["objects"] = object_evidence

    donor = manifest["donor"]
    versions = _aws_json(
        [
            "s3api",
            "list-object-versions",
            "--bucket",
            donor["bucket"],
            "--prefix",
            donor["prefix"],
        ],
        profile=profile,
        region=region,
    )
    if versions.get("IsTruncated"):
        errors.append("donor version listing was truncated")
    latest = {
        item["Key"]: item
        for item in versions.get("Versions", [])
        if item.get("IsLatest")
    }
    expected_keys = {donor["prefix"] + item["path"] for item in donor["files"]}
    actual_staged_keys = {
        key
        for key in latest
        if key.startswith(donor["prefix"])
        and key[len(donor["prefix"]) :].split("/", 1)[0] in STAGED_DONOR_COMPONENTS
    }
    if actual_staged_keys != expected_keys:
        errors.append(
            "donor staged-key set drifted: "
            f"missing={sorted(expected_keys - actual_staged_keys)}, "
            f"unexpected={sorted(actual_staged_keys - expected_keys)}"
        )
    checked_donor = []
    for expected in donor["files"]:
        key = donor["prefix"] + expected["path"]
        actual = latest.get(key)
        if actual is None:
            continue
        for field, wanted in (
            ("VersionId", expected["version_id"]),
            ("Size", expected["content_length"]),
            ("ETag", expected["etag"]),
        ):
            got = _etag(actual.get(field)) if field == "ETag" else actual.get(field)
            if got != wanted:
                errors.append(
                    f"donor {expected['path']} {field}: "
                    f"expected {wanted!r}, got {got!r}"
                )
        expected_object = {
            **expected,
            "bucket": donor["bucket"],
            "key": key,
        }
        current = _head(expected_object, profile=profile, region=region, pinned=False)
        pinned = _head(expected_object, profile=profile, region=region, pinned=True)
        _validate_head(
            f"donor.{expected['path']}.current", expected_object, current, errors
        )
        _validate_head(
            f"donor.{expected['path']}.pinned", expected_object, pinned, errors
        )
        checked_donor.append(
            {"path": expected["path"], "current": current, "pinned": pinned}
        )
    evidence["donor_latest_versions"] = checked_donor
    evidence["errors"] = errors
    evidence["status"] = "pass" if not errors else "fail"
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        evidence = verify(args.manifest, profile=args.profile, region=args.region)
    except Exception as exc:
        evidence = {
            "schema_version": "minwm-720p-release-input-preflight/v1",
            "status": "fail",
            "errors": [f"preflight raised {exc!r}"],
        }
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if evidence["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
