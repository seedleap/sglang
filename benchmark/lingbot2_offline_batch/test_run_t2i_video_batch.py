import json
import urllib.request

from run_t2i_video_batch import (
    build_runtime_inputs,
    collect_upload_results,
    parse_s3_uri,
    post_callback,
    upload_report,
)
from t2i_video_batch import build_case_records


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
            "videos_per_image": 5,
            "fps": 24,
            "width": 1280,
            "height": 720,
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
    cases = build_case_records(_request(), _rows())

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
    assert len(message_rows) == 5
    assert message_rows[0]["sample_id"] == cases[0]["sample_id"]
    assert image_urls == {
        "img001": "https://signed.example.com/bucket/t2i/images/img001.png?ttl=604800"
    }


def test_collect_upload_results_uploads_successes_and_reports_failures(tmp_path):
    cases = build_case_records(_request(), _rows())
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
