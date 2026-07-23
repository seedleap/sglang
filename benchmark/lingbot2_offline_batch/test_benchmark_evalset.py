import sys
from types import SimpleNamespace


def test_load_first_frame_payload_reads_s3_uri_as_bytes(monkeypatch):
    msgpack = SimpleNamespace(encode=lambda value: value, decode=lambda value: value)
    monkeypatch.setitem(sys.modules, "msgspec", SimpleNamespace(msgpack=msgpack))
    monkeypatch.setitem(sys.modules, "msgspec.msgpack", msgpack)
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace())
    from benchmark_evalset import load_first_frame_payload

    class Body:
        def read(self):
            return b"image-bytes"

    class FakeS3Client:
        def __init__(self):
            self.calls = []

        def get_object(self, **kwargs):
            self.calls.append(kwargs)
            return {"Body": Body()}

    client = FakeS3Client()

    import asyncio

    payload = asyncio.run(
        load_first_frame_payload("s3://bucket/path/to/image.png", s3_client=client)
    )

    assert payload == b"image-bytes"
    assert client.calls == [{"Bucket": "bucket", "Key": "path/to/image.png"}]


def test_parse_args_defaults_websocket_close_timeout_to_short_cleanup_window(
    tmp_path, monkeypatch
):
    msgpack = SimpleNamespace(encode=lambda value: value, decode=lambda value: value)
    monkeypatch.setitem(sys.modules, "msgspec", SimpleNamespace(msgpack=msgpack))
    monkeypatch.setitem(sys.modules, "msgspec.msgpack", msgpack)
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace())
    from benchmark_evalset import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_evalset.py",
            "--messages",
            str(tmp_path / "messages.jsonl.gz"),
            "--image-urls",
            str(tmp_path / "image-urls.json"),
            "--urls",
            "ws://127.0.0.1:30000/v1/realtime_video/generate",
            "--output-dir",
            str(tmp_path / "cases"),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )

    args = parse_args()

    assert args.timeout == 1200.0
    assert args.close_timeout == 10.0
