from __future__ import annotations

import hashlib
import io
import json

import pytest
from download_model_artifact import (
    resolve_model_source,
    stage_model,
    validate_control_files,
    validate_release_controls,
)

MODEL_NAME = "minwm"
MODEL_VERSION = "gs3200-ema-student-v1"
RELEASE_ID = "20260810T042157Z-c302d572"
PREFIX = f"models/{MODEL_NAME}/{MODEL_VERSION}/{RELEASE_ID}"


def _artifact() -> tuple[dict[str, bytes], bytes, bytes, bytes]:
    objects = {
        "model_index.json": b"{}",
        "transformer/config.json": b'{"layers": 1}',
        "transformer/model.safetensors": b"model-weights",
    }
    manifest = {
        "schema_version": 1,
        "revision": "gs3200-ema-student-v1",
        "files": [
            {
                "path": path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(objects.items())
        ],
    }
    manifest_body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    info_body = json.dumps(
        {
            "schema_version": 1,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "release_id": RELEASE_ID,
            "source_revision": "gs3200-ema-student-v1",
            "source_uri": "s3://source-bucket/source/model/",
            "artifact_manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
            "release_spec_sha256": "a" * 64,
            "publisher_bundle_sha256": "b" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    ready_body = json.dumps(
        {
            "schema_version": 1,
            "revision": manifest["revision"],
            "release_id": RELEASE_ID,
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
            "info_sha256": hashlib.sha256(info_body).hexdigest(),
            "release_spec_sha256": "a" * 64,
            "publisher_bundle_sha256": "b" * 64,
        },
        sort_keys=True,
    ).encode()
    return objects, info_body, manifest_body, ready_body


class FakeClient:
    def __init__(self, info_body: bytes, manifest_body: bytes, ready_body: bytes):
        self.control_objects = {
            f"{PREFIX}/info.json": info_body,
            f"{PREFIX}/artifact-manifest.json": manifest_body,
            f"{PREFIX}/_READY": ready_body,
        }
        self.requested_keys = []

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        self.requested_keys.append(Key)
        payload = self.control_objects[Key]
        return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}


class FakeFuture:
    def __init__(self, manager, key: str, destination):
        self.manager = manager
        self.key = key
        self.destination = destination

    def result(self):
        assert len(self.manager.submitted) == self.manager.expected_submissions
        relative_path = self.key.removeprefix(f"{PREFIX}/model/")
        self.destination.write(self.manager.objects[relative_path])


class FakeManager:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.expected_submissions = len(objects)
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def download(self, bucket: str, key: str, destination: str):
        self.submitted.append((bucket, key, destination))
        return FakeFuture(self, key, destination)


def test_stages_all_objects_before_waiting_and_then_hits_cache(tmp_path):
    objects, info_body, manifest_body, ready_body = _artifact()
    client = FakeClient(info_body, manifest_body, ready_body)
    managers = []

    def factory(client, concurrency, part_size):
        del client
        assert concurrency == 128
        assert part_size == 16 * 1024 * 1024
        manager = FakeManager(objects)
        managers.append(manager)
        return manager

    kwargs = {
        "client": client,
        "bucket": "model-bucket",
        "prefix": PREFIX,
        "destination": tmp_path / "model",
        "lock_path": tmp_path / "model.lock",
        "expected_model_name": MODEL_NAME,
        "expected_model_version": MODEL_VERSION,
        "expected_release_id": RELEASE_ID,
        "expected_revision": MODEL_VERSION,
        "concurrency": 128,
        "part_size": 16 * 1024 * 1024,
        "manager_factory": factory,
    }
    downloaded = stage_model(**kwargs)

    assert downloaded["backend"] == "awscrt"
    assert downloaded["cache_hit"] is False
    assert len(managers) == 1
    assert client.requested_keys[:3] == [
        f"{PREFIX}/info.json",
        f"{PREFIX}/artifact-manifest.json",
        f"{PREFIX}/_READY",
    ]
    assert (tmp_path / "model/_READY").read_bytes() == ready_body
    assert (tmp_path / "model/info.json").read_bytes() == info_body
    assert (tmp_path / "model/artifact-manifest.json").read_bytes() == manifest_body
    assert downloaded["model_name"] == MODEL_NAME
    assert downloaded["model_version"] == MODEL_VERSION
    assert downloaded["release_id"] == RELEASE_ID
    for relative_path, payload in objects.items():
        assert (tmp_path / "model" / relative_path).read_bytes() == payload

    cached = stage_model(**kwargs)
    assert cached["cache_hit"] is True
    assert len(managers) == 1

    # A same-size payload corruption must not be accepted as a cache hit.
    (tmp_path / "model/model_index.json").write_bytes(b"[]")
    repaired = stage_model(**kwargs)
    assert repaired["cache_hit"] is False
    assert len(managers) == 2
    assert (tmp_path / "model/model_index.json").read_bytes() == b"{}"


def test_rejects_manifest_path_traversal():
    objects, info_body, manifest_body, ready_body = _artifact()
    del objects, info_body
    manifest = json.loads(manifest_body)
    manifest["files"][0]["path"] = "../escape"
    invalid_body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ready = json.loads(ready_body)
    ready["manifest_sha256"] = hashlib.sha256(invalid_body).hexdigest()

    with pytest.raises(ValueError, match="unsafe manifest file path"):
        validate_control_files(
            json.dumps(ready, sort_keys=True).encode(),
            invalid_body,
            "gs3200-ema-student-v1",
        )


def test_rejects_manifest_not_authorized_by_ready_marker():
    objects, info_body, manifest_body, ready_body = _artifact()
    del objects, info_body
    tampered = json.loads(manifest_body)
    tampered["source_uri"] = "s3://unexpected"
    tampered_body = json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="does not match _READY"):
        validate_control_files(
            ready_body,
            tampered_body,
            "gs3200-ema-student-v1",
        )


def test_rejects_missing_wrong_or_noncanonical_destination_revision():
    _, info_body, manifest_body, ready_body = _artifact()

    missing_revision_manifest = json.loads(manifest_body)
    del missing_revision_manifest["revision"]
    missing_revision_body = json.dumps(
        missing_revision_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    ready = json.loads(ready_body)
    ready["manifest_sha256"] = hashlib.sha256(missing_revision_body).hexdigest()
    with pytest.raises(ValueError, match="has no revision"):
        validate_control_files(
            json.dumps(ready, sort_keys=True).encode(),
            missing_revision_body,
            None,
        )

    wrong_ready = json.loads(ready_body)
    wrong_ready["revision"] = "wrong-revision"
    with pytest.raises(ValueError, match="revisions differ"):
        validate_control_files(
            json.dumps(wrong_ready, sort_keys=True).encode(), manifest_body, None
        )

    resolved_only_ready = json.loads(ready_body)
    resolved_only_ready["resolved_revision"] = resolved_only_ready.pop("revision")
    with pytest.raises(ValueError, match="canonical revision"):
        validate_release_controls(
            info_body,
            json.dumps(resolved_only_ready, sort_keys=True).encode(),
            manifest_body,
            expected_model_name=MODEL_NAME,
            expected_model_version=MODEL_VERSION,
            expected_release_id=RELEASE_ID,
            expected_revision=MODEL_VERSION,
        )


def test_accepts_an_arbitrary_model_file_layout():
    objects = {
        "acoustic/weights.bin": b"speech-weights",
        "vocabulary.txt": b"hello\nworld\n",
    }
    manifest = {
        "schema_version": 1,
        "revision": "speech-revision-7",
        "files": [
            {
                "path": path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(objects.items())
        ],
    }
    manifest_body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ready_body = json.dumps(
        {
            "revision": "speech-revision-7",
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        },
        sort_keys=True,
    ).encode()

    _, files = validate_control_files(
        ready_body,
        manifest_body,
        "speech-revision-7",
    )

    assert [entry["path"] for entry in files] == [
        "acoustic/weights.bin",
        "vocabulary.txt",
    ]


def test_failed_hash_never_publishes_ready_marker(tmp_path):
    objects, info_body, manifest_body, ready_body = _artifact()
    corrupt_objects = {**objects, "transformer/model.safetensors": b"corrupt"}
    client = FakeClient(info_body, manifest_body, ready_body)

    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        stage_model(
            client=client,
            bucket="model-bucket",
            prefix=PREFIX,
            destination=tmp_path / "model",
            lock_path=tmp_path / "model.lock",
            expected_model_name=MODEL_NAME,
            expected_model_version=MODEL_VERSION,
            expected_release_id=RELEASE_ID,
            expected_revision=MODEL_VERSION,
            concurrency=128,
            part_size=16 * 1024 * 1024,
            manager_factory=lambda *_: FakeManager(corrupt_objects),
        )

    assert not (tmp_path / "model/_READY").exists()
    assert not list(tmp_path.glob(".model.staging.*"))


def test_insufficient_disk_fails_before_crt_download(tmp_path, monkeypatch):
    objects, info_body, manifest_body, ready_body = _artifact()
    client = FakeClient(info_body, manifest_body, ready_body)
    manager_started = False

    def factory(*_):
        nonlocal manager_started
        manager_started = True
        raise AssertionError("CRT transfer manager must not start")

    monkeypatch.setattr(
        "download_model_artifact.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": sum(map(len, objects.values())) - 1})(),
    )
    with pytest.raises(ValueError, match="insufficient disk space"):
        stage_model(
            client=client,
            bucket="model-bucket",
            prefix=PREFIX,
            destination=tmp_path / "model",
            lock_path=tmp_path / "model.lock",
            expected_model_name=MODEL_NAME,
            expected_model_version=MODEL_VERSION,
            expected_release_id=RELEASE_ID,
            expected_revision=MODEL_VERSION,
            concurrency=128,
            part_size=16 * 1024 * 1024,
            manager_factory=factory,
        )

    assert manager_started is False
    assert not list(tmp_path.glob(".model.staging.*"))


def test_resolves_versioned_default_and_explicit_rollback_uri():
    kwargs = {
        "bucket": "serving-bucket",
        "prefix": None,
        "model_s3_uri": None,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_release_id": RELEASE_ID,
    }
    assert resolve_model_source(**kwargs) == ("serving-bucket", PREFIX, RELEASE_ID)
    rollback_release_id = "20260801T010203Z-12345678"
    assert resolve_model_source(
        **{
            **kwargs,
            "model_s3_uri": (
                f"s3://rollback-bucket/models/{MODEL_NAME}/{MODEL_VERSION}/"
                f"{rollback_release_id}/"
            ),
        }
    ) == (
        "rollback-bucket",
        f"models/{MODEL_NAME}/{MODEL_VERSION}/{rollback_release_id}",
        rollback_release_id,
    )

    with pytest.raises(ValueError, match="must follow"):
        resolve_model_source(
            **{
                **kwargs,
                "model_s3_uri": "s3://rollback-bucket/models/other/v1/release",
            }
        )


def test_rejects_wrong_info_before_starting_crt_download(tmp_path):
    objects, info_body, manifest_body, ready_body = _artifact()
    del objects
    info = json.loads(info_body)
    info["release_id"] = "wrong-release"
    tampered_info_body = json.dumps(
        info, sort_keys=True, separators=(",", ":")
    ).encode()
    ready = json.loads(ready_body)
    ready["info_sha256"] = hashlib.sha256(tampered_info_body).hexdigest()
    client = FakeClient(
        tampered_info_body,
        manifest_body,
        json.dumps(ready, sort_keys=True).encode(),
    )
    manager_started = False

    def factory(*_):
        nonlocal manager_started
        manager_started = True
        raise AssertionError("CRT transfer manager must not start")

    with pytest.raises(ValueError, match="model info release_id"):
        stage_model(
            client=client,
            bucket="model-bucket",
            prefix=PREFIX,
            destination=tmp_path / "model",
            lock_path=tmp_path / "model.lock",
            expected_model_name=MODEL_NAME,
            expected_model_version=MODEL_VERSION,
            expected_release_id=RELEASE_ID,
            expected_revision=MODEL_VERSION,
            concurrency=128,
            part_size=16 * 1024 * 1024,
            manager_factory=factory,
        )

    assert manager_started is False


def test_accepts_unbound_migration_controls_and_rejects_mismatched_bindings():
    _, info_body, manifest_body, ready_body = _artifact()
    info = json.loads(info_body)
    ready = json.loads(ready_body)

    del info["release_spec_sha256"]
    del info["publisher_bundle_sha256"]
    del ready["release_spec_sha256"]
    del ready["publisher_bundle_sha256"]
    unbound_info_body = json.dumps(info, sort_keys=True, separators=(",", ":")).encode()
    ready["info_sha256"] = hashlib.sha256(unbound_info_body).hexdigest()
    validate_release_controls(
        unbound_info_body,
        json.dumps(ready, sort_keys=True).encode(),
        manifest_body,
        expected_model_name=MODEL_NAME,
        expected_model_version=MODEL_VERSION,
        expected_release_id=RELEASE_ID,
        expected_revision=MODEL_VERSION,
    )

    info = json.loads(info_body)
    ready = json.loads(ready_body)
    ready["publisher_bundle_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="publisher_bundle_sha256 differ"):
        validate_release_controls(
            info_body,
            json.dumps(ready, sort_keys=True).encode(),
            manifest_body,
            expected_model_name=MODEL_NAME,
            expected_model_version=MODEL_VERSION,
            expected_release_id=RELEASE_ID,
            expected_revision=MODEL_VERSION,
        )
