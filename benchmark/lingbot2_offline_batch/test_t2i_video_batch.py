from t2i_video_batch import (
    build_callback_progress_payload,
    build_case_records,
)


def _request() -> dict:
    return {
        "schema_version": "sglang-video-batch.v1",
        "request_id": "req-video-001",
        "generation_job_id": "gen_t2i_001",
        "idempotency_key": "gen_t2i_001:sglang-video:v1",
        "input": {
            "video_manifest_uri": "s3://bucket/t2i/sglang_video_manifest.jsonl",
        },
        "output": {
            "video_s3_prefix": "s3://bucket/t2i/videos",
            "metadata_s3_prefix": "s3://bucket/t2i/video_metadata",
            "report_s3_uri": "s3://bucket/t2i/reports/sglang_video_report.json",
        },
        "video": {
            "videos_per_image": 5,
            "frames": 129,
            "fps": 24,
            "width": 1280,
            "height": 720,
            "model": "lingbot2",
            "action_seed": 20260715,
        },
        "callback": {
            "url": "https://pipeline.example.com/api/v1/generation/jobs/gen_t2i_001/progress",
            "job_id": "gen_t2i_001",
            "auth_secret_name": "lwdp-generation-callback-token",
        },
        "limits": {
            "max_active_gpus": 32,
            "gpu_per_pod": 8,
            "job_parallelism": 4,
            "timeout_seconds": 21600,
        },
    }


def test_build_case_records_expands_each_t2i_image_to_five_prompted_cases():
    rows = [
        {
            "item_id": "img001",
            "image_uri": "s3://bucket/t2i/images/img001.png",
            "image_prompt": "A quiet workshop with warm practical lighting.",
            "video_prompt": "A quiet workshop with warm practical lighting.",
            "video_prompt_source": "image_prompt_fallback",
            "orientation": "横图",
        },
        {
            "item_id": "img002",
            "image_uri": "s3://bucket/t2i/images/img002.png",
            "image_prompt": "A red robot on a desk.",
            "video_prompt": "The camera slowly orbits around the red robot.",
            "video_prompt_source": "explicit",
            "orientation": "方图",
        },
    ]

    cases = build_case_records(_request(), rows)

    assert len(cases) == 10
    assert [case["case_index"] for case in cases] == list(range(10))
    first_image_pairs = {
        (case["movement_key"], case["camera_key"]) for case in cases[:5]
    }
    assert len(first_image_pairs) == 5
    assert cases[0]["image_id"] == "img001"
    assert cases[0]["image_uri"] == "s3://bucket/t2i/images/img001.png"
    assert cases[0]["messages"][0]["content"] == "A quiet workshop with warm practical lighting."
    assert cases[0]["metadata"]["video_prompt_source"] == "image_prompt_fallback"
    assert cases[5]["messages"][0]["content"] == "The camera slowly orbits around the red robot."
    assert cases[5]["metadata"]["video_prompt_source"] == "explicit"
    assert cases[0]["video_s3_uri"].startswith("s3://bucket/t2i/videos/img001/")
    assert cases[0]["action_pattern"] == "57 movement + 15 noop + 57 camera"
    assert len(cases[0]["messages"][1]["controls"][0]["actions"]) == 128


def test_callback_progress_payload_groups_five_videos_per_image():
    cases = build_case_records(
        _request(),
        [
            {
                "item_id": "img001",
                "image_uri": "s3://bucket/t2i/images/img001.png",
                "image_prompt": "A quiet workshop.",
                "video_prompt": "A quiet workshop.",
                "video_prompt_source": "image_prompt_fallback",
            }
        ],
    )
    results = [
        {
            "case_id": case["case_id"],
            "status": "succeeded",
            "video_uri": case["video_s3_uri"],
            "movement_key": case["movement_key"],
            "camera_key": case["camera_key"],
            "action_seed": case["action_seed"],
            "action_pattern": case["action_pattern"],
        }
        for case in cases
    ]

    payload = build_callback_progress_payload(_request(), cases, results)

    assert payload["status"] == "succeeded"
    assert payload["stage"] == "sglang_video_generation"
    assert payload["summary"]["video_status"] == "succeeded"
    assert payload["summary"]["video_succeeded_count"] == 5
    assert payload["items"][0]["item_id"] == "img001"
    assert payload["items"][0]["metadata"]["video_status"] == "succeeded"
    assert len(payload["items"][0]["metadata"]["videos"]) == 5
    assert payload["items"][0]["metadata"]["videos"][0]["video_uri"].startswith(
        "s3://bucket/t2i/videos/img001/"
    )
