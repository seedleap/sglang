from t2i_video_batch import (
    build_callback_progress_payload,
    build_case_records,
)


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
            "videos_per_image": 1,
            "frames": 129,
            "fps": 24,
            "width": 1280,
            "height": 704,
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


def test_build_case_records_expands_each_t2i_image_to_one_random_traj_case():
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

    cases = build_case_records(_request(), rows, action_trajectories=_trajs())

    assert len(cases) == 2
    assert [case["case_index"] for case in cases] == [0, 1]
    assert all(case["camera_key"] == "" for case in cases)
    assert all(case["metadata"]["camera_key"] == "" for case in cases)
    assert cases[0]["image_id"] == "img001"
    assert cases[0]["image_uri"] == "s3://bucket/t2i/images/img001.png"
    assert cases[0]["messages"][0]["content"] == "A quiet workshop with warm practical lighting."
    assert cases[0]["metadata"]["video_prompt_source"] == "image_prompt_fallback"
    assert cases[1]["messages"][0]["content"] == "The camera slowly orbits around the red robot."
    assert cases[1]["metadata"]["video_prompt_source"] == "explicit"
    assert cases[0]["video_s3_uri"].startswith("s3://bucket/t2i/videos/img001/")
    assert cases[0]["video_s3_uri"].endswith(
        f"_{cases[0]['traj_id']}.mp4"
    )
    assert cases[0]["traj_id"] == "traj-1"
    assert cases[0]["metadata"]["traj_id"] == "traj-1"
    assert cases[0]["metadata"]["action_source"] == "trajs.jsonl"
    assert cases[0]["action_pattern"] == "trajs.jsonl:test_traj"
    assert len(cases[0]["messages"][1]["controls"][0]["actions"]) == 128
    assert cases[0]["messages"][1]["controls"][0]["action_keys"] == ["w", "a", "s", "d"]
    assert all(
        len(action) == 4
        for action in cases[0]["messages"][1]["controls"][0]["actions"]
    )


def test_build_case_records_defaults_to_1280x704_video_size():
    request = _request()
    request["video"].pop("width")
    request["video"].pop("height")

    cases = build_case_records(
        request,
        [
            {
                "item_id": "img001",
                "image_uri": "s3://bucket/t2i/images/img001.png",
                "image_prompt": "A quiet workshop.",
                "video_prompt": "A quiet workshop.",
                "video_prompt_source": "image_prompt_fallback",
            }
        ],
        action_trajectories=_trajs(),
    )

    target = cases[0]["messages"][1]
    assert target["output"]["width"] == 1280
    assert target["output"]["height"] == 704


def test_callback_progress_payload_groups_one_video_per_image():
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
        action_trajectories=_trajs(),
    )
    results = [
        {
            "case_id": case["case_id"],
            "status": "succeeded",
            "video_uri": case["video_s3_uri"],
            "movement_key": case["movement_key"],
            "ending_movement_key": case["ending_movement_key"],
            "camera_key": case["camera_key"],
            "traj_id": case["traj_id"],
            "traj_type": case["traj_type"],
            "action_source": case["action_source"],
            "action_seed": case["action_seed"],
            "action_pattern": case["action_pattern"],
        }
        for case in cases
    ]

    payload = build_callback_progress_payload(_request(), cases, results)

    assert payload["status"] == "succeeded"
    assert payload["stage"] == "sglang_video_generation"
    assert payload["summary"]["video_status"] == "succeeded"
    assert payload["summary"]["video_succeeded_count"] == 1
    assert payload["items"][0]["item_id"] == "img001"
    assert payload["items"][0]["metadata"]["video_status"] == "succeeded"
    assert len(payload["items"][0]["metadata"]["videos"]) == 1
    assert payload["items"][0]["metadata"]["videos"][0]["video_uri"].startswith(
        "s3://bucket/t2i/videos/img001/"
    )
    assert payload["items"][0]["metadata"]["videos"][0]["camera_key"] == ""
    assert payload["items"][0]["metadata"]["videos"][0]["traj_id"] == cases[0]["traj_id"]
