from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

from build_model_release_spec import build_release_spec


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
        destination_bucket="serving",
        destination_region="us-east-2",
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
    assert result["destination"]["prefix"] == (
        "models/lingbot2/lingbot-world-v2/revision-1/releases/"
        f"20260814T010203Z-{manifest_sha[:8]}/model"
    )
