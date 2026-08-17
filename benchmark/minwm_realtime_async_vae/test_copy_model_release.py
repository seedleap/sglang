from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import copy_model_release
import pytest
from copy_model_release import (
    _put_control_object,
    _validate_destination_before_copy,
    execute_copy,
    execution_bundle_sha256,
    load_and_validate_release,
    offline_plan,
    publisher_bundle_sha256,
    publisher_script_inventory,
    render_destination_control_bodies,
    validate_source_objects,
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
            "schema_version": 1,
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
        "publisher_scripts": publisher_script_inventory(),
        "publisher_bundle_sha256": publisher_bundle_sha256(),
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


def test_release_spec_rejects_a_different_runtime_publisher_bundle(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publisher_scripts": publisher_script_inventory(),
                "publisher_bundle_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime publisher bundle"):
        load_and_validate_release(path, object())


def test_release_spec_rejects_script_and_source_control_tampering(tmp_path):
    ready_body, manifest_body = _source_control_files()
    files = json.loads(manifest_body)["files"]
    release = {
        "schema_version": 1,
        "publisher_scripts": publisher_script_inventory(),
        "publisher_bundle_sha256": publisher_bundle_sha256(),
        "model": {"revision": "gs3200-ema-student-v1"},
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
            "object_version_ids": {
                entry["path"]: f"payload-version-{index}"
                for index, entry in enumerate(files)
            },
        },
    }

    scripts_tampered = json.loads(json.dumps(release))
    scripts_tampered["publisher_scripts"]["copy_model_release.py"]["sha256"] = "0" * 64
    path = tmp_path / "scripts-tampered.json"
    path.write_text(json.dumps(scripts_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="publisher scripts"):
        load_and_validate_release(path, object())

    control_sha_tampered = json.loads(json.dumps(release))
    control_sha_tampered["source"]["ready"]["sha256"] = "0" * 64
    path = tmp_path / "control-sha-tampered.json"
    path.write_text(json.dumps(control_sha_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source _READY does not match"):
        load_and_validate_release(path, FakeSourceClient(ready_body, manifest_body))

    class VersionAwareSource(FakeSourceClient):
        def get_object(self, *, Bucket, Key, VersionId):
            if Key.endswith("/_READY") and VersionId != "ready-version":
                return {"Body": io.BytesIO(b"wrong-version")}
            return super().get_object(Bucket=Bucket, Key=Key, VersionId=VersionId)

    control_version_tampered = json.loads(json.dumps(release))
    control_version_tampered["source"]["ready"]["version_id"] = "wrong-version"
    path = tmp_path / "control-version-tampered.json"
    path.write_text(json.dumps(control_version_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source _READY does not match"):
        load_and_validate_release(path, VersionAwareSource(ready_body, manifest_body))

    noncanonical_ready = json.loads(ready_body)
    noncanonical_ready["resolved_revision"] = noncanonical_ready.pop("revision")
    noncanonical_ready_body = json.dumps(noncanonical_ready, sort_keys=True).encode()
    noncanonical_source = json.loads(json.dumps(release))
    noncanonical_source["source"]["ready"].update(
        size=len(noncanonical_ready_body),
        sha256=hashlib.sha256(noncanonical_ready_body).hexdigest(),
    )
    path = tmp_path / "noncanonical-ready.json"
    path.write_text(json.dumps(noncanonical_source), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical revision"):
        load_and_validate_release(
            path, FakeSourceClient(noncanonical_ready_body, manifest_body)
        )


def test_source_validation_hashes_version_pinned_content_before_copy():
    payload = b"arbitrary-model-payload"
    digest = hashlib.sha256(payload).hexdigest()
    release = {
        "source": {
            "bucket": "source",
            "prefix": "legacy/model",
            "object_version_ids": {"acoustic/weights.bin": "source-v1"},
        }
    }
    files = {
        "acoustic/weights.bin": {
            "path": "acoustic/weights.bin",
            "size": len(payload),
            "sha256": digest,
        }
    }

    class SourceClient:
        def __init__(self, body):
            self.body = body

        def head_object(self, **kwargs):
            assert kwargs["VersionId"] == "source-v1"
            assert kwargs["ChecksumMode"] == "ENABLED"
            return {"ContentLength": len(payload), "VersionId": "source-v1"}

        def get_object(self, **kwargs):
            assert kwargs["VersionId"] == "source-v1"
            assert kwargs["ChecksumMode"] == "ENABLED"
            return {"ContentLength": len(payload), "Body": io.BytesIO(self.body)}

    validate_source_objects(release, files, SourceClient(payload))
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        validate_source_objects(release, files, SourceClient(b"x" * len(payload)))


def test_execution_bundle_binds_scripts_controls_allowlist_versions_and_destination():
    manifest_body = b'{"schema_version":1,"revision":"r1"}'
    files = {"weights.bin": {"path": "weights.bin", "size": 7, "sha256": "1" * 64}}
    release = {
        "release_id": "release-1",
        "release_manifest_created_at": "2026-08-14T01:02:03Z",
        "model": {"serving_name": "model", "version": "v1", "revision": "r1"},
        "publisher_scripts": publisher_script_inventory(),
        "publisher_bundle_sha256": publisher_bundle_sha256(),
        "source": {
            "bucket": "source",
            "region": "us-west-2",
            "prefix": "legacy/model",
            "ready": {
                "key": "legacy/model/_READY",
                "version_id": "ready-v1",
                "size": 100,
                "sha256": "2" * 64,
            },
            "artifact_manifest": {
                "key": "legacy/manifest.json",
                "version_id": "manifest-v1",
                "size": 200,
                "sha256": "3" * 64,
            },
            "object_version_ids": {"weights.bin": "payload-v1"},
        },
        "destination": {
            "bucket": "serving",
            "region": "us-east-2",
            "prefix": "models/model/v1/release-1",
        },
    }
    baseline = execution_bundle_sha256(release, manifest_body, files, "a" * 64)

    variants = []
    for mutation in (
        lambda value: value["publisher_scripts"]["copy_model_release.py"].update(
            sha256="6" * 64
        ),
        lambda value: value["source"]["ready"].update(sha256="7" * 64),
        lambda value: value["source"]["object_version_ids"].update(
            {"weights.bin": "payload-v2"}
        ),
        lambda value: value["destination"].update(prefix="models/model/v1/release-2"),
    ):
        candidate = json.loads(json.dumps(release))
        mutation(candidate)
        variants.append(
            execution_bundle_sha256(candidate, manifest_body, files, "a" * 64)
        )

    variants.append(
        execution_bundle_sha256(release, manifest_body + b" ", files, "a" * 64)
    )

    assert all(value != baseline for value in variants)


def test_control_put_requires_exact_readback(monkeypatch):
    body = b'{"schema_version":1}'

    class Client:
        def put_object(self, **kwargs):
            return {"VersionId": "control-v1"}

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(body + b" ")}

    monkeypatch.setattr(copy_model_release, "_destination_head", lambda *_: None)
    with pytest.raises(ValueError, match="readback mismatch"):
        _put_control_object(Client(), "bucket", "release/info.json", body)

    monkeypatch.setattr(
        copy_model_release,
        "_destination_head",
        lambda *_: {"ContentLength": len(body), "VersionId": "ready-v1"},
    )
    with pytest.raises(RuntimeError, match="immutable control already exists"):
        _put_control_object(
            Client(), "bucket", "release/_READY", body, refuse_existing=True
        )


def test_destination_preflight_rejects_unexpected_keys_and_bad_reuse_metadata():
    release = {
        "source": {"object_version_ids": {"weights.bin": "source-v1"}},
        "destination": {"bucket": "serving", "prefix": "models/name/v/r"},
    }
    files = {
        "weights.bin": {
            "path": "weights.bin",
            "size": 7,
            "sha256": "1" * 64,
        }
    }

    class DestinationClient:
        def __init__(self, keys, metadata=None):
            self.keys = keys
            self.metadata = metadata or {}

        def list_objects_v2(self, **kwargs):
            return {
                "IsTruncated": False,
                "Contents": [{"Key": key} for key in self.keys],
            }

        def head_object(self, **kwargs):
            return {
                "ContentLength": 7,
                "Metadata": self.metadata,
            }

    with pytest.raises(ValueError, match="unexpected keys"):
        _validate_destination_before_copy(
            release,
            files,
            DestinationClient({"models/name/v/r/foreign-object"}),
        )

    with pytest.raises(RuntimeError, match="_READY already exists"):
        _validate_destination_before_copy(
            release,
            files,
            DestinationClient({"models/name/v/r/_READY"}),
        )

    with pytest.raises(ValueError, match="SHA256 metadata mismatch"):
        _validate_destination_before_copy(
            release,
            files,
            DestinationClient(
                {"models/name/v/r/model/weights.bin"},
                {"sha256": "2" * 64, "source-version-id": "source-v1"},
            ),
        )


def test_offline_plan_never_uses_network_and_marks_unresolved_inventory():
    plan = offline_plan(
        {
            "release_id": "REPLACE_AFTER_READONLY_INVENTORY",
            "publisher_scripts": publisher_script_inventory(),
            "publisher_bundle_sha256": publisher_bundle_sha256(),
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


def test_legacy_lingbot2_offline_dry_run_remains_blocked_by_new_bundle_contract():
    assert not list(LINGBOT2_RELEASE_ROOT.glob("*template*"))
    release_body = (LINGBOT2_RELEASE_ROOT / "release-spec.json").read_bytes()
    release = json.loads(release_body)
    expected = json.loads(
        (LINGBOT2_RELEASE_ROOT / "offline-dry-run.golden.json").read_text()
    )

    assert (
        offline_plan(
            release, release_spec_sha256=hashlib.sha256(release_body).hexdigest()
        )
        == expected
    )
    assert release["model"]["revision"] == ("59cccf49f2d2dd27418ae7a04b82b10868d455c2")
    assert release["destination"]["bucket"] == (
        "leap-world-model-serving-829115578968-us-east-2"
    )
    assert release["release_id"] == "20260814T054118Z-e0650875"
    assert release["source"]["object_count"] == 26
    assert release["source"]["bytes"] == 86071995490
    assert len(release["source"]["object_version_ids"]) == 26
    assert expected["release_spec_ready"] is False
    assert expected["unresolved"] == [
        "publisher_bundle_sha256",
        "publisher_scripts",
    ]


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
    manifest_files = {
        entry["path"]: entry for entry in json.loads(manifest_body)["files"]
    }
    release_spec_sha256 = "a" * 64
    release = {
        "release_id": "20260814T010203Z-12345678",
        "release_manifest_created_at": "2026-08-14T01:02:03Z",
        "publisher_bundle_sha256": publisher_bundle_sha256(),
        "destination": {"bucket": "serving", "prefix": "release"},
        "source": {
            "bucket": "source",
            "prefix": "model",
            "ready": {"sha256": "f" * 64},
            "object_version_ids": {
                "model_index.json": "source-v0",
                "transformer/model": "source-v1",
            },
        },
        "model": {
            "serving_name": "minwm-async-denoiser-0",
            "version": "model-v1",
            "revision": "source-revision-1",
            "rollback_release": None,
        },
    }
    info_body, ready_body = render_destination_control_bodies(
        release, release_spec_sha256, manifest_body, manifest_files
    )

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
                raise AssertionError(Key)
            return {"Body": io.BytesIO(body)}

        def list_objects_v2(self, **kwargs):
            assert kwargs["Prefix"] == "release/"
            return {
                "IsTruncated": False,
                "Contents": [
                    {"Key": "release/_READY"},
                    {"Key": "release/artifact-manifest.json"},
                    {"Key": "release/info.json"},
                    {"Key": "release/model/model_index.json"},
                    {"Key": "release/model/transformer/model"},
                ],
            }

        def head_object(self, **kwargs):
            is_model_index = kwargs["Key"].endswith("model_index.json")
            source_version = "source-v0" if is_model_index else "source-v1"
            return {
                "ContentLength": 2 if is_model_index else 3,
                "ChecksumSHA256": base64.b64encode(
                    bytes.fromhex(("1" if is_model_index else "2") * 64)
                ).decode(),
                "ChecksumType": "FULL_OBJECT",
                "Metadata": {
                    "sha256": ("1" if is_model_index else "2") * 64,
                    "source-version-id": source_version,
                },
            }

    result = verify_destination_release(
        release, DestinationClient(), release_spec_sha256
    )

    assert result == {
        "bucket": "serving",
        "prefix": "release",
        "release_id": "20260814T010203Z-12345678",
        "object_count": 5,
        "payload_object_count": 2,
        "bytes": 5,
        "verified": True,
    }

    exact_keys = {
        "release/_READY",
        "release/artifact-manifest.json",
        "release/info.json",
        "release/model/model_index.json",
        "release/model/transformer/model",
    }
    for bad_keys in (
        exact_keys - {"release/model/transformer/model"},
        exact_keys | {"release/unexpected.json"},
    ):
        client = DestinationClient()
        client.list_objects_v2 = lambda **_: {
            "IsTruncated": False,
            "Contents": [{"Key": key} for key in sorted(bad_keys)],
        }
        with pytest.raises(ValueError, match="key set differs"):
            verify_destination_release(release, client, release_spec_sha256)


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
        "publisher_scripts": publisher_script_inventory(),
        "publisher_bundle_sha256": publisher_bundle_sha256(),
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
            "region": "us-west-2",
            "prefix": "source/model",
            "ready": {
                "key": "source/model/_READY",
                "version_id": "source-ready-v1",
                "size": len(source_ready),
                "sha256": hashlib.sha256(source_ready).hexdigest(),
            },
            "artifact_manifest": {
                "key": "source/model/artifact-manifest.json",
                "version_id": "source-manifest-v1",
                "size": len(manifest_body),
                "sha256": hashlib.sha256(manifest_body).hexdigest(),
            },
            "object_version_ids": {
                "model_index.json": "source-v0",
                "transformer/model": "source-v1",
            },
        },
        "destination": {
            "bucket": "serving-bucket",
            "region": "us-east-2",
            "prefix": "models/name/v/r",
        },
    }
    files = {
        "model_index.json": {"path": "model_index.json", "size": 2, "sha256": "1" * 64},
        "transformer/model": {
            "path": "transformer/model",
            "size": 3,
            "sha256": "2" * 64,
        },
    }
    release_spec_sha256 = "a" * 64
    expected_info_body, expected_ready_body = render_destination_control_bodies(
        release, release_spec_sha256, manifest_body, files
    )
    confirmed_execution_bundle_sha256 = execution_bundle_sha256(
        release, manifest_body, files, release_spec_sha256
    )
    written = []

    monkeypatch.setattr(copy_model_release, "_destination_head", lambda *_: None)
    monkeypatch.setattr(
        copy_model_release, "_validate_destination_before_copy", lambda *_: None
    )
    monkeypatch.setattr(
        copy_model_release,
        "_list_destination_keys",
        lambda *_: {
            "models/name/v/r/model/model_index.json",
            "models/name/v/r/model/transformer/model",
            "models/name/v/r/artifact-manifest.json",
            "models/name/v/r/info.json",
        },
    )

    def fake_copy_one(release, path, entry, *_):
        return {
            **entry,
            "source_version_id": release["source"]["object_version_ids"][path],
            "destination_version_id": f"destination-{path}",
            "reused": False,
        }

    def fake_put(_client, bucket, key, body, **kwargs):
        if key.endswith("/_READY"):
            assert kwargs == {"refuse_existing": True}
        else:
            assert kwargs == {}
        written.append((bucket, key, body))
        return f"version-{len(written)}"

    monkeypatch.setattr(copy_model_release, "_copy_one", fake_copy_one)
    monkeypatch.setattr(copy_model_release, "_put_control_object", fake_put)
    with pytest.raises(ValueError, match="confirmed execution bundle"):
        execute_copy(
            release,
            source_ready,
            manifest_body,
            files,
            object(),
            object(),
            concurrency=2,
            release_spec_sha256=release_spec_sha256,
            confirmed_execution_bundle_sha256="0" * 64,
        )
    result = execute_copy(
        release,
        source_ready,
        manifest_body,
        files,
        object(),
        object(),
        concurrency=2,
        release_spec_sha256=release_spec_sha256,
        confirmed_execution_bundle_sha256=confirmed_execution_bundle_sha256,
    )

    assert [key for _, key, _ in written] == [
        "models/name/v/r/artifact-manifest.json",
        "models/name/v/r/info.json",
        "models/name/v/r/_READY",
    ]
    info = json.loads(written[-2][2])
    ready = json.loads(written[-1][2])
    assert info["model_name"] == "minwm-async-denoiser-0"
    assert ready["info_sha256"] == hashlib.sha256(written[-2][2]).hexdigest()
    assert ready["revision"] == "source-revision-1"
    assert info["release_spec_sha256"] == release_spec_sha256
    assert ready["release_spec_sha256"] == release_spec_sha256
    assert info["publisher_bundle_sha256"] == ready["publisher_bundle_sha256"]
    assert written[-2][2] == expected_info_body
    assert written[-1][2] == expected_ready_body
    assert result["object_count"] == 5
    assert result["execution_bundle_sha256"] == confirmed_execution_bundle_sha256


def test_execute_refuses_to_rewrite_an_existing_ready_marker(monkeypatch):
    monkeypatch.setattr(
        copy_model_release,
        "_destination_head",
        lambda *_: {"VersionId": "already-ready"},
    )
    with pytest.raises(RuntimeError, match="already complete"):
        execute_copy(
            {
                "destination": {
                    "bucket": "serving",
                    "prefix": "models/model/v1/release-1",
                }
            },
            b"{}",
            b"{}",
            {},
            object(),
            object(),
            concurrency=1,
            release_spec_sha256="a" * 64,
            confirmed_execution_bundle_sha256="b" * 64,
        )
