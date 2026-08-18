# SPDX-License-Identifier: Apache-2.0

import json

from sglang.multimodal_gen.runtime.realtime import critical_path_metrics as metrics


def test_histogram_has_50ms_buckets_through_one_second():
    expected = tuple(round(index * 0.05, 2) for index in range(1, 21))
    actual = tuple(
        bucket for bucket in metrics.CRITICAL_PATH_BUCKETS if 0.05 <= bucket <= 1.0
    )

    assert actual == expected


def test_model_contract_collapses_revisions_paths_and_legacy_names():
    assert metrics.infer_model_label("lingbot2") == "lingbot2"
    assert metrics.infer_model_label("robbyant-lingbot-world-v2") == "lingbot2"
    assert metrics.infer_model_label("minwm-denoiser") == "wan"
    assert metrics.infer_model_label("zing") == "wan"
    assert metrics.infer_model_label("wan22-5b-stage3") == "wan"
    assert metrics.infer_model_label("/model-cache/model") is None
    assert metrics.infer_model_label("59cccf49f2d2dd27418ae7a04b82b10868d455c2") is None
    assert metrics.infer_model_label("unknown") is None


def test_codec_contract_includes_webp_and_normalizes_wire_names():
    assert metrics.codec_label(None) == "none"
    assert metrics.codec_label("rgb24") == "none"
    assert metrics.codec_label("image/webp") == "webp"
    assert metrics.codec_label("avc") == "h264"
    assert metrics.codec_label("image/jpeg") == "jpeg"
    assert metrics.codec_label("arbitrary-codec") is None


def test_all_critical_path_stages_emit_structured_stdout_events(monkeypatch, capfd):
    monkeypatch.setenv("WORLD_MODEL_METRIC_STRUCTURED_LOGS", "true")
    monkeypatch.setenv("WORLD_MODEL_METRIC_MODEL", "wan")

    for stage in sorted(metrics.VALID_STAGES):
        assert metrics.observe_stage_seconds(
            stage,
            0.0125,
            service="contract-test",
            codec="webp",
            scope="frame" if "frame" in stage else "request",
        )

    events = [json.loads(line) for line in capfd.readouterr().out.splitlines()]
    assert {event["stage"] for event in events} == metrics.VALID_STAGES
    assert all(event["event"] == "world_model_metric" for event in events)
    assert all(event["schema_version"] == 1 for event in events)
    assert all(event["model"] == "wan" for event in events)
    assert all(event["codec"] == "webp" for event in events)
    assert all(event["duration_ms"] == 12.5 for event in events)
    assert all("cluster" not in event for event in events)
    assert all("business_line" not in event for event in events)


def test_unrecognized_model_is_dropped_without_polluting_histogram(monkeypatch):
    monkeypatch.delenv("WORLD_MODEL_METRIC_MODEL", raising=False)
    monkeypatch.delenv("WORLD_MODEL_METRIC_SERVICE", raising=False)
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    monkeypatch.setenv("HOSTNAME", "worker-without-model-family")

    assert not metrics.observe_stage_seconds(
        "denoiser_compute",
        0.1,
        service="contract-test-invalid-model",
        model="59cccf49f2d2dd27418ae7a04b82b10868d455c2",
    )
    payload = metrics.prometheus_latest().decode()
    assert 'service="contract-test-invalid-model"' not in payload
    assert 'reason="invalid_model"' in payload


def test_structured_metric_events_can_be_disabled(monkeypatch, capfd):
    monkeypatch.setenv("WORLD_MODEL_METRIC_STRUCTURED_LOGS", "false")
    monkeypatch.setenv("WORLD_MODEL_METRIC_MODEL", "lingbot2")

    assert metrics.observe_stage_seconds(
        "scheduler_queue",
        0.005,
        service="contract-test-disabled-json",
    )
    assert capfd.readouterr().out == ""
