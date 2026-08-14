from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import copy_model_release
import pytest
from copy_model_release import (
    execute_copy,
    load_and_validate_release,
    offline_plan,
    verify_destination_release,
)

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "model_releases/20260810T042157Z-c302d572/source-release.json"
LINGBOT2_RELEASE_ROOT = (
    ROOT / "model_releases/lingbot2/59cccf49f2d2dd27418ae7a04b82b10868d455c2"
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
    assert release["destination"]["prefix"].endswith("/20260810T042157Z-c302d572")


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
                "prefix": "models/lingbot2-denoiser/model-v1/REPLACE",
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
    assert plan["release_spec_ready"] is False


def test_lingbot2_offline_dry_run_matches_reviewed_golden():
    assert not list(LINGBOT2_RELEASE_ROOT.glob("*template*"))
    release = json.loads((LINGBOT2_RELEASE_ROOT / "release-spec.json").read_text())
    expected = json.loads(
        (LINGBOT2_RELEASE_ROOT / "offline-dry-run.golden.json").read_text()
    )

    assert offline_plan(release) == expected
    assert release["model"]["revision"] == ("59cccf49f2d2dd27418ae7a04b82b10868d455c2")
    assert release["destination"]["bucket"] == (
        "leap-world-model-serving-829115578968-us-east-2"
    )
    assert release["release_id"] == "20260814T054118Z-e0650875"
    assert release["source"]["object_count"] == 26
    assert release["source"]["bytes"] == 86071995490
    assert len(release["source"]["object_version_ids"]) == 26


def test_destination_verifier_checks_ready_manifest_and_versioned_objects():
    manifest_body = json.dumps(
        {
            "schema_version": 1,
            "revision": "source-revision-1",
            "files": [
                {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
                {"path": "transformer/model", "size": 3, "sha256": "2" * 64},
            ],
        },
        sort_keys=True,
    ).encode()
    release_manifest_body = json.dumps(
        {
            "release_id": "20260814T010203Z-12345678",
            "objects": [
                {
                    "path": "model_index.json",
                    "size": 2,
                    "sha256": "1" * 64,
                    "destination_version_id": "object-v0",
                },
                {
                    "path": "transformer/model",
                    "size": 3,
                    "sha256": "2" * 64,
                    "destination_version_id": "object-v1",
                },
            ],
        }
    ).encode()
    info_body = json.dumps(
        {
            "schema_version": 1,
            "model_name": "minwm-async-denoiser-0",
            "model_version": "model-v1",
            "release_id": "20260814T010203Z-12345678",
            "source_revision": "source-revision-1",
            "source_uri": "s3://source/model/",
            "artifact_manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    ready_body = json.dumps(
        {
            "revision": "source-revision-1",
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
            "info_sha256": hashlib.sha256(info_body).hexdigest(),
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
            elif Key.endswith("/info.json"):
                body = info_body
            elif Key.endswith("/artifact-manifest.json"):
                body = manifest_body
            else:
                body = release_manifest_body
            return {"Body": io.BytesIO(body)}

        def head_object(self, **kwargs):
            is_model_index = kwargs["Key"].endswith("model_index.json")
            assert kwargs["VersionId"] == (
                "object-v0" if is_model_index else "object-v1"
            )
            return {
                "ContentLength": 2 if is_model_index else 3,
                "ChecksumSHA256": base64.b64encode(
                    bytes.fromhex(("1" if is_model_index else "2") * 64)
                ).decode(),
            }

    result = verify_destination_release(
        {
            "release_id": "20260814T010203Z-12345678",
            "destination": {"bucket": "serving", "prefix": "release"},
            "model": {
                "serving_name": "minwm-async-denoiser-0",
                "version": "model-v1",
                "revision": "source-revision-1",
            },
        },
        DestinationClient(),
    )

    assert result == {
        "bucket": "serving",
        "prefix": "release",
        "release_id": "20260814T010203Z-12345678",
        "object_count": 2,
        "bytes": 5,
        "verified": True,
    }


def test_execute_writes_info_before_ready_and_keeps_payload_under_model(
    monkeypatch,
):
    manifest_body = json.dumps(
        {
            "schema_version": 1,
            "revision": "source-revision-1",
            "files": [
                {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
                {"path": "transformer/model", "size": 3, "sha256": "2" * 64},
            ],
        },
        sort_keys=True,
    ).encode()
    source_ready = json.dumps(
        {
            "revision": "source-revision-1",
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        }
    ).encode()
    release = {
        "release_id": "20260814T010203Z-12345678",
        "release_manifest_created_at": "2026-08-14T01:02:03Z",
        "model": {
            "serving_name": "minwm-async-denoiser-0",
            "version": "model-v1",
            "family": "minwm",
            "id": "source-model",
            "revision": "source-revision-1",
            "rollback_release": None,
        },
        "source": {
            "bucket": "source-bucket",
            "prefix": "source/model",
            "object_version_ids": {
                "model_index.json": "source-v0",
                "transformer/model": "source-v1",
            },
        },
        "destination": {"bucket": "serving-bucket", "prefix": "models/name/v/r"},
    }
    files = {
        "model_index.json": {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
        "transformer/model": {
            "path": "transformer/model",
            "size": 3,
            "sha256": "2" * 64,
        },
    }
    written = []

    monkeypatch.setattr(copy_model_release, "_destination_head", lambda *_: None)

    def fake_copy_one(release, path, entry, *_):
        return {
            **entry,
            "source_version_id": release["source"]["object_version_ids"][path],
            "destination_version_id": f"destination-{path}",
            "reused": False,
        }

    def fake_put(_client, bucket, key, body):
        written.append((bucket, key, body))
        return f"version-{len(written)}"

    monkeypatch.setattr(copy_model_release, "_copy_one", fake_copy_one)
    monkeypatch.setattr(copy_model_release, "_put_control_object", fake_put)
    result = execute_copy(
        release,
        source_ready,
        manifest_body,
        files,
        object(),
        object(),
        concurrency=2,
    )

    assert [key for _, key, _ in written] == [
        "models/name/v/r/artifact-manifest.json",
        "models/name/v/r/release-manifest.json",
        "models/name/v/r/info.json",
        "models/name/v/r/_READY",
    ]
    info = json.loads(written[-2][2])
    ready = json.loads(written[-1][2])
    assert info["model_name"] == "minwm-async-denoiser-0"
    assert ready["info_sha256"] == hashlib.sha256(written[-2][2]).hexdigest()
    assert result["object_count"] == 6
