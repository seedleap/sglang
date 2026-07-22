import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _write_result_root(
    root: Path, *, latency: float, measured_wall: float, enabled: bool
):
    cases = root / "cases"
    (cases / "videos" / "G1").mkdir(parents=True)
    (cases / "videos" / "G1" / "case-1.mp4").write_bytes(b"mp4")
    summary = {
        "summary": {
            "selected_samples": 1,
            "successful_samples": 1,
            "failed_samples": 0,
            "failure_rate": 0.0,
            "warmup_wall_sec": 2.0,
            "measured_wall_sec": measured_wall,
            "node_videos_per_hour_this_run": 3600 / measured_wall,
            "videos_per_gpu_hour_this_run": 3600 / measured_wall / 8,
            "request_persisted_end_to_end_sec": {"p50": latency, "p95": latency},
        },
        "results": [
            {
                "sample_id": "testset100_v2/G1/case-1",
                "output": str(cases / "videos" / "G1" / "case-1.mp4"),
                "success": True,
                "persisted_end_to_end_sec": latency,
                "trajectory": "w*2",
            }
        ],
    }
    (cases / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "server-startup-seconds").write_text("12\n", encoding="utf-8")
    (root / "taehv-runtime.json").write_text(
        json.dumps(
            {"enabled": enabled, "checkpoint_sha256": "abc" if enabled else None}
        ),
        encoding="utf-8",
    )


def test_build_report_pairs_artifacts_and_exposes_prompt_action_and_delta(
    tmp_path, monkeypatch
):
    benchmark_dir = Path(__file__).parent
    monkeypatch.syspath_prepend(str(benchmark_dir))
    fake_client = SimpleNamespace(
        generate_presigned_url=lambda _method, Params, ExpiresIn: (
            f"https://signed.example/{Params['Key']}?ttl={ExpiresIn}"
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=lambda **_kwargs: SimpleNamespace(client=lambda _name: fake_client)
        ),
    )
    from build_taehv_ab_report import build_report

    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "sample_id": "testset100_v2/G1/case-1",
                "metadata": {"image_id": "p01"},
                "messages": [
                    {"role": "user", "type": "text", "content": "a test prompt"},
                    {
                        "role": "target",
                        "type": "video",
                        "uri": "s3://source-bucket/images/p01.png",
                        "controls": [
                            {
                                "type": "keyboard_direction_frame_interval",
                                "action_keys": ["w", "a"],
                                "actions": [[1, 0], [1, 0], [0, 1]],
                            }
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_result_root(baseline, latency=10.0, measured_wall=100.0, enabled=False)
    _write_result_root(candidate, latency=9.0, measured_wall=90.0, enabled=True)

    result = build_report(
        baseline_root=baseline,
        candidate_root=candidate,
        fixture_path=fixture,
        bucket="test-bucket",
        output_prefix="world-model/eval/taehv-ab/run-1",
        output_dir=tmp_path / "report",
        profile="test",
        region="us-east-2",
        expires_in=60,
    )

    html = (tmp_path / "report" / "index.html").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "report" / "performance.json").read_text())
    assert result["paired_successes"] == 1
    assert "a test prompt" in html
    assert "w x2" in html
    assert "a x1" in html
    assert "baseline/cases/videos/G1/case-1.mp4" in html
    assert "taehv/cases/videos/G1/case-1.mp4" in html
    assert report["metrics"]["measured_wall_sec"]["improvement_percent"] == 10.0
    assert report["variants"]["taehv"]["enabled"] is True
