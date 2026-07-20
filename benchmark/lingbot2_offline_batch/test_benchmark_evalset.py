import sys
from types import SimpleNamespace


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
