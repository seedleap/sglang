from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

import pytest
from build_model_release_spec import build_release_spec, parse_created_at
from copy_model_release import publisher_bundle_sha256


class FakeClient:
    def __init__(self):
        manifest = {
            "schema_version": 1,
            "revision": "revision-1",
            "files": [
                {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
                {"path": "transformer/model", "size": 3, "sha256": "2" * 64},
            ],
        }
        self.manifest_body = json.dumps(manifest, sort_keys=True).encode()
        self.ready_body = json.dumps(
            {
                "schema_version": 1,
                "revision": "revision-1",
                "manifest_sha256": hashlib.sha256(self.manifest_body).hexdigest(),
            },
            sort_keys=True,
        ).encode()

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        if Key.endswith("/_READY"):
            return {"Body": io.BytesIO(self.ready_body), "VersionId": "ready-v1"}
        return {"Body": io.BytesIO(self.manifest_body), "VersionId": "manifest-v1"}

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        size = 2 if Key.endswith("model_index.json") else 3
        return {"ContentLength": size, "VersionId": f"version-{size}"}


def test_builds_frozen_release_path_from_manifest_sha_and_utc_time():
    result = build_release_spec(
        client=FakeClient(),
        source_bucket="source",
        source_region="us-west-2",
        source_prefix="legacy/model/",
        source_manifest_key=None,
        destination_bucket="serving",
        destination_region="us-east-2",
        serving_model_name="lingbot2-denoiser",
        model_version="lingbot-world-v2",
        model_family="lingbot2",
        model_id="lingbot-world-v2",
        revision="revision-1",
        rollback_release=None,
        created_at=datetime(2026, 8, 14, 1, 2, 3, tzinfo=timezone.utc),
    )

    manifest_sha = hashlib.sha256(FakeClient().manifest_body).hexdigest()
    assert result["release_id"] == f"20260814T010203Z-{manifest_sha[:8]}"
    assert result["source"]["object_count"] == 2
    assert result["source"]["bytes"] == 5
    assert result["publisher_bundle_sha256"] == publisher_bundle_sha256()
    assert result["source"]["ready"]["key"] == "legacy/model/_READY"
    assert result["source"]["artifact_manifest"]["key"] == (
        "legacy/model/artifact-manifest.json"
    )
    assert result["destination"]["prefix"] == (
        "models/lingbot2-denoiser/lingbot-world-v2/"
        f"20260814T010203Z-{manifest_sha[:8]}"
    )


def test_supports_legacy_manifest_key_and_resolved_manifest_revision():
    client = FakeClient()
    manifest = json.loads(client.manifest_body)
    manifest["resolved_revision"] = manifest.pop("revision")
    client.manifest_body = json.dumps(manifest, sort_keys=True).encode()
    client.ready_body = json.dumps(
        {
            "schema_version": 1,
            "revision": "revision-1",
            "manifest_sha256": hashlib.sha256(client.manifest_body).hexdigest(),
        },
        sort_keys=True,
    ).encode()

    result = build_release_spec(
        client=client,
        source_bucket="source",
        source_region="us-west-2",
        source_prefix="legacy/model",
        source_manifest_key="legacy/manifest.json",
        destination_bucket="serving",
        destination_region="us-east-2",
        serving_model_name="lingbot2-denoiser",
        model_version="lingbot-world-v2",
        model_family="lingbot2",
        model_id="lingbot-world-v2",
        revision="revision-1",
        rollback_release=None,
        created_at=datetime(2026, 8, 14, 1, 2, 3, tzinfo=timezone.utc),
    )

    assert result["source"]["artifact_manifest"]["key"] == "legacy/manifest.json"
    assert result["source"]["manifest_revision"] == "revision-1"


def test_created_at_can_reproduce_a_reviewed_release_id():
    assert parse_created_at("2026-08-14T05:41:18Z") == datetime(
        2026, 8, 14, 5, 41, 18, tzinfo=timezone.utc
    )
    assert parse_created_at("2026-08-14T13:41:18+08:00") == datetime(
        2026, 8, 14, 5, 41, 18, tzinfo=timezone.utc
    )

    with pytest.raises(ValueError, match="include a timezone"):
        parse_created_at("2026-08-14T05:41:18")
    with pytest.raises(ValueError, match="RFC3339"):
        parse_created_at("not-a-timestamp")
