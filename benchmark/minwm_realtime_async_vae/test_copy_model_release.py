from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from copy_model_release import (
    load_and_validate_release,
    offline_plan,
    verify_destination_release,
)

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "model_releases/20260810T042157Z-c302d572/source-release.json"
LINGBOT2_RELEASE_ROOT = (
    ROOT
    / "model_releases/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2"
)


class FakeSourceClient:
    def __init__(self, ready_body: bytes, manifest_body: bytes):
        self.ready_body = ready_body
        self.manifest_body = manifest_body

    def get_object(self, *, Bucket: str, Key: str, VersionId: str):
        del Bucket, VersionId
        payload = self.ready_body if Key.endswith("/_READY") else self.manifest_body
        return {"Body": io.BytesIO(payload)}


def _source_control_files():
    manifest = {
        "schema_version": 1,
        "revision": "gs3200-ema-student-v1",
        "files": [
            {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
            {"path": "transformer/model", "size": 3, "sha256": "2" * 64},
        ],
    }
    manifest_body = json.dumps(manifest, sort_keys=True).encode()
    ready_body = json.dumps(
        {
            "revision": manifest["revision"],
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    return ready_body, manifest_body


def test_checked_in_release_pins_every_manifest_file_version():
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    versions = release["source"]["object_version_ids"]

    assert release["release_id"] == "20260810T042157Z-c302d572"
    assert len(versions) == 19
    assert all(versions.values())
    assert release["destination"]["region"] == "us-east-2"
    assert release["destination"]["prefix"].endswith(
        "/releases/20260810T042157Z-c302d572/model"
    )


def test_control_validation_rejects_a_version_map_that_does_not_match_manifest(
    tmp_path,
):
    ready_body, manifest_body = _source_control_files()

    release = {
        "schema_version": 1,
        "source": {
            "bucket": "source",
            "prefix": "release/model",
            "ready": {
                "version_id": "ready-version",
                "size": len(ready_body),
                "sha256": hashlib.sha256(ready_body).hexdigest(),
            },
            "artifact_manifest": {
                "version_id": "manifest-version",
                "size": len(manifest_body),
                "sha256": hashlib.sha256(manifest_body).hexdigest(),
            },
            "object_version_ids": {"model_index.json": "only-one-version"},
        },
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release), encoding="utf-8")

    with pytest.raises(ValueError, match="do not exactly match"):
        load_and_validate_release(path, FakeSourceClient(ready_body, manifest_body))


def test_offline_plan_never_uses_network_and_marks_unresolved_inventory():
    plan = offline_plan(
        {
            "release_id": "REPLACE_AFTER_READONLY_INVENTORY",
            "source": {
                "bucket": "source",
                "prefix": "legacy/model",
                "ready": {},
                "artifact_manifest": {},
                "object_version_ids": {},
            },
            "destination": {
                "bucket": "serving",
                "prefix": "models/lingbot2/model/revision/releases/REPLACE/model",
            },
        }
    )

    assert plan["network"] is False
    assert plan["execute"] is False
    assert plan["model_object_count"] is None
    assert plan["unresolved"] == [
        "release_id",
        "source.artifact_manifest.version_id",
        "source.object_version_ids",
        "source.ready.version_id",
    ]


def test_lingbot2_offline_dry_run_matches_reviewed_golden():
    release = json.loads(
        (LINGBOT2_RELEASE_ROOT / "release-spec.template.json").read_text()
    )
    expected = json.loads(
        (LINGBOT2_RELEASE_ROOT / "offline-dry-run.golden.json").read_text()
    )

    assert offline_plan(release) == expected
    assert release["model"]["revision"] == (
        "59cccf49f2d2dd27418ae7a04b82b10868d455c2"
    )
    assert release["destination"]["bucket"] == (
        "leap-world-model-serving-829115578968-us-east-2"
    )


def test_destination_verifier_checks_ready_manifest_and_versioned_objects():
    manifest_body = b'{"schema_version":1}'
    release_manifest_body = json.dumps(
        {
            "release_id": "20260814T010203Z-12345678",
            "objects": [
                {
                    "path": "transformer/model",
                    "size": 3,
                    "sha256": "2" * 64,
                    "destination_version_id": "object-v1",
                }
            ],
        }
    ).encode()
    ready_body = json.dumps(
        {
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
            "release_id": "20260814T010203Z-12345678",
            "release_manifest_sha256": hashlib.sha256(
                release_manifest_body
            ).hexdigest(),
        }
    ).encode()

    class DestinationClient:
        def get_object(self, *, Bucket, Key):
            del Bucket
            if Key.endswith("/_READY"):
                body = ready_body
            elif Key.endswith("/artifact-manifest.json"):
                body = manifest_body
            else:
                body = release_manifest_body
            return {"Body": io.BytesIO(body)}

        def head_object(self, **kwargs):
            assert kwargs["VersionId"] == "object-v1"
            return {
                "ContentLength": 3,
                "ChecksumSHA256": base64.b64encode(bytes.fromhex("2" * 64)).decode(),
            }

    result = verify_destination_release(
        {
            "release_id": "20260814T010203Z-12345678",
            "destination": {"bucket": "serving", "prefix": "release/model"},
        },
        DestinationClient(),
    )

    assert result == {
        "bucket": "serving",
        "prefix": "release/model",
        "release_id": "20260814T010203Z-12345678",
        "object_count": 1,
        "bytes": 3,
        "verified": True,
    }
