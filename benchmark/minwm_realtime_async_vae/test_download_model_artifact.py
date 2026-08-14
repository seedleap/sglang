from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from download_model_artifact import stage_model, validate_control_files


def _artifact() -> tuple[dict[str, bytes], bytes, bytes]:
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
    ready_body = json.dumps(
        {
            "revision": manifest["revision"],
            "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    return objects, manifest_body, ready_body


class FakeClient:
    def __init__(self, manifest_body: bytes, ready_body: bytes):
        self.control_objects = {
            "release/model/_READY": ready_body,
            "release/model/artifact-manifest.json": manifest_body,
        }

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        payload = self.control_objects[Key]
        return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}


class FakeFuture:
    def __init__(self, manager, key: str, destination):
        self.manager = manager
        self.key = key
        self.destination = destination

    def result(self):
        assert len(self.manager.submitted) == self.manager.expected_submissions
        relative_path = self.key.removeprefix("release/model/")
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
    objects, manifest_body, ready_body = _artifact()
    client = FakeClient(manifest_body, ready_body)
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
        "prefix": "release/model",
        "destination": tmp_path / "model",
        "lock_path": tmp_path / "model.lock",
        "expected_revision": "gs3200-ema-student-v1",
        "concurrency": 128,
        "part_size": 16 * 1024 * 1024,
        "manager_factory": factory,
    }
    downloaded = stage_model(**kwargs)

    assert downloaded["backend"] == "awscrt"
    assert downloaded["cache_hit"] is False
    assert len(managers) == 1
    assert (tmp_path / "model/_READY").read_bytes() == ready_body
    assert (tmp_path / "model/artifact-manifest.json").read_bytes() == manifest_body
    for relative_path, payload in objects.items():
        assert (tmp_path / "model" / relative_path).read_bytes() == payload

    cached = stage_model(**kwargs)
    assert cached["cache_hit"] is True
    assert len(managers) == 1


def test_rejects_manifest_path_traversal():
    objects, manifest_body, ready_body = _artifact()
    del objects
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
    objects, manifest_body, ready_body = _artifact()
    del objects
    tampered = json.loads(manifest_body)
    tampered["source_uri"] = "s3://unexpected"
    tampered_body = json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="does not match _READY"):
        validate_control_files(
            ready_body,
            tampered_body,
            "gs3200-ema-student-v1",
        )


def test_failed_hash_never_publishes_ready_marker(tmp_path):
    objects, manifest_body, ready_body = _artifact()
    corrupt_objects = {**objects, "transformer/model.safetensors": b"corrupt"}
    client = FakeClient(manifest_body, ready_body)

    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        stage_model(
            client=client,
            bucket="model-bucket",
            prefix="release/model",
            destination=tmp_path / "model",
            lock_path=tmp_path / "model.lock",
            expected_revision="gs3200-ema-student-v1",
            concurrency=128,
            part_size=16 * 1024 * 1024,
            manager_factory=lambda *_: FakeManager(corrupt_objects),
        )

    assert not (tmp_path / "model/_READY").exists()
    assert not list(tmp_path.glob(".model.staging.*"))
