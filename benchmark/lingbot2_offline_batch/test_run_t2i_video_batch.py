import json
import sys
from types import SimpleNamespace
import urllib.request

from run_t2i_video_batch import (
    _make_s3_client,
    build_runtime_inputs,
    collect_upload_results,
    parse_s3_uri,
    post_callback,
    run_benchmark,
    upload_report,
)
from t2i_video_batch import build_case_records


def _traj(traj_id: str, first_key: str, last_key: str | None = None) -> dict:
    last_key = last_key or first_key
    return {
        "traj_id": traj_id,
        "fps": 24,
        "num_frames": 129,
        "traj_type": "test_traj",
        "condition_inputs": {
            "camera_actions": (
                [[first_key] for _ in range(64)]
                + [[last_key] for _ in range(65)]
            )
        },
        "segments": [
            {"key": first_key, "start_frame": 0, "end_frame": 63, "num_frames": 64},
            {"key": last_key, "start_frame": 64, "end_frame": 128, "num_frames": 65},
        ],
    }


def _trajs() -> tuple[dict, ...]:
    return (
        _traj("traj-0", "w", "a"),
        _traj("traj-1", "a", "s"),
        _traj("traj-2", "s", "d"),
        _traj("traj-3", "d", "w"),
    )


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads = []
        self.objects = []

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        assert operation == "get_object"
        return f"https://signed.example.com/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "extra_args": ExtraArgs,
            }
        )

    def put_object(self, **kwargs):
        self.objects.append(kwargs)


def _request() -> dict:
    return {
        "schema_version": "sglang-video-batch.v1",
        "generation_job_id": "gen_t2i_001",
        "input": {
            "video_manifest_uri": "s3://bucket/t2i/sglang_video_manifest.jsonl",
        },
        "output": {
            "video_s3_prefix": "s3://bucket/t2i/videos",
            "report_s3_uri": "s3://bucket/t2i/reports/sglang_video_report.json",
        },
        "video": {
            "videos_per_image": 1,
            "fps": 24,
            "width": 1280,
            "height": 704,
            "action_seed": 20260715,
        },
    }


def _rows() -> list[dict]:
    return [
        {
            "item_id": "img001",
            "image_uri": "s3://bucket/t2i/images/img001.png",
            "image_prompt": "A quiet workshop.",
            "video_prompt": "A quiet workshop.",
            "video_prompt_source": "image_prompt_fallback",
        }
    ]


def test_build_runtime_inputs_writes_messages_and_presigned_image_urls(tmp_path):
    cases = build_case_records(_request(), _rows(), action_trajectories=_trajs())

    runtime = build_runtime_inputs(
        request=_request(),
        cases=cases,
        work_dir=tmp_path,
        s3_client=FakeS3Client(),
    )

    message_rows = [
        json.loads(line)
        for line in runtime.messages_path.read_text(encoding="utf-8").splitlines()
    ]
    image_urls = json.loads(runtime.image_urls_path.read_text(encoding="utf-8"))
    assert len(message_rows) == 1
    assert message_rows[0]["sample_id"] == cases[0]["sample_id"]
    assert image_urls == {
        "img001": "https://signed.example.com/bucket/t2i/images/img001.png?ttl=604800"
    }


def test_make_s3_client_forces_sigv4_presigned_urls(monkeypatch):
    calls = []

    class FakeBoto3:
        def client(self, service_name, **kwargs):
            calls.append((service_name, kwargs))
            return object()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3())

    _make_s3_client()

    assert calls[0][0] == "s3"
    assert calls[0][1]["region_name"] == "us-east-2"
    assert calls[0][1]["config"].signature_version == "s3v4"


def test_run_benchmark_forwards_request_video_dimensions_to_runner(tmp_path, monkeypatch):
    request = _request()
    request["video"].update({"width": 1024, "height": 576, "fps": 12})
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())
    runtime = build_runtime_inputs(
        request=request,
        cases=cases,
        work_dir=tmp_path,
        s3_client=FakeS3Client(),
    )
    captured = {}

    def fake_run(command, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        summary_path = runtime.results_root / "cases" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text('{"results":[]}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("run_t2i_video_batch.subprocess.run", fake_run)

    summary = run_benchmark(runtime, request=request)

    assert summary == {"results": []}
    assert captured["check"] is False
    assert captured["env"]["SGLANG_VIDEO_WIDTH"] == "1024"
    assert captured["env"]["SGLANG_VIDEO_HEIGHT"] == "576"
    assert captured["env"]["SGLANG_VIDEO_FPS"] == "12"


def test_collect_upload_results_uploads_successes_and_reports_failures(tmp_path):
    cases = build_case_records(
        _request(),
        _rows()
        + [
            {
                "item_id": "img002",
                "image_uri": "s3://bucket/t2i/images/img002.png",
                "image_prompt": "A red robot.",
                "video_prompt": "A red robot.",
                "video_prompt_source": "image_prompt_fallback",
            }
        ],
        action_trajectories=_trajs(),
    )
    successful_output = tmp_path / "video.mp4"
    successful_output.write_bytes(b"mp4")
    summary = {
        "results": [
            {
                "sample_id": cases[0]["sample_id"],
                "success": True,
                "output": str(successful_output),
            },
            {
                "sample_id": cases[1]["sample_id"],
                "success": False,
                "error": "RuntimeError: boom",
            },
        ]
    }
    fake_s3 = FakeS3Client()

    results = collect_upload_results(cases, summary, fake_s3)
    bucket, key = parse_s3_uri(cases[0]["video_s3_uri"])

    assert fake_s3.uploads == [
        {
            "filename": str(successful_output),
            "bucket": bucket,
            "key": key,
            "extra_args": {"ContentType": "video/mp4"},
        }
    ]
    assert results[0]["status"] == "succeeded"
    assert results[0]["video_uri"] == cases[0]["video_s3_uri"]
    assert results[0]["traj_id"] == cases[0]["traj_id"]
    assert results[0]["action_source"] == "trajs.jsonl"
    assert results[1]["status"] == "failed"
    assert results[1]["error"] == "RuntimeError: boom"


def test_upload_report_writes_report_to_requested_s3_uri():
    fake_s3 = FakeS3Client()

    upload_report({"summary": {"video_status": "succeeded"}}, _request(), fake_s3)

    assert fake_s3.objects[0]["Bucket"] == "bucket"
    assert fake_s3.objects[0]["Key"] == "t2i/reports/sglang_video_report.json"
    assert fake_s3.objects[0]["ContentType"] == "application/json"
    assert b'"video_status": "succeeded"' in fake_s3.objects[0]["Body"]


def test_post_callback_uses_generation_progress_put_with_bearer_token(monkeypatch):
    sent = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        sent["method"] = request.get_method()
        sent["url"] = request.full_url
        sent["authorization"] = request.headers.get("Authorization")
        sent["content_type"] = request.headers.get("Content-type")
        sent["body"] = request.data
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("SGLANG_VIDEO_CALLBACK_TOKEN", "callback-token")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    post_callback(
        {
            "callback": {
                "url": "https://pipeline.example.com/api/v1/generation/jobs/gen/progress"
            }
        },
        {"status": "succeeded"},
    )

    assert sent["method"] == "PUT"
    assert sent["url"].endswith("/api/v1/generation/jobs/gen/progress")
    assert sent["authorization"] == "Bearer callback-token"
    assert sent["content_type"] == "application/json"
    assert sent["body"] == b'{"status": "succeeded"}'
    assert sent["timeout"] == 60
