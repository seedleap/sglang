import json
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

from run_t2i_video_batch import (
    _make_s3_client,
    ProgressUploadWatcher,
    RuntimeInputs,
    build_runtime_inputs,
    case_checkpoint_s3_uri,
    collect_upload_results,
    case_metadata_s3_uri,
    main,
    parse_s3_uri,
    post_callback,
    read_action_trajectories,
    run_benchmark,
    run_video_batch,
    split_completed_cases,
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
        self.existing = set()
        self.head_calls = 0

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
        self.existing.add((bucket, key))

    def put_object(self, **kwargs):
        self.objects.append(kwargs)

    def head_object(self, **kwargs):
        self.head_calls += 1
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.existing:
            error = Exception("Not Found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {"ContentLength": 1024}


def _request() -> dict:
    return {
        "schema_version": "sglang-video-batch.v1",
        "generation_job_id": "gen_t2i_001",
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


def test_read_action_trajectories_always_uses_bundled_trajs(tmp_path, monkeypatch):
    bundled_trajs = tmp_path / "trajs.jsonl"
    bundled_trajs.write_text(
        json.dumps(_traj("bundled-traj", "w", "d")) + "\n",
        encoding="utf-8",
    )

    class FailingS3Client:
        def get_object(self, **kwargs):
            raise AssertionError("request action_trajs_uri should not be read")

    monkeypatch.setattr("run_t2i_video_batch.DEFAULT_ACTION_TRAJS_PATH", bundled_trajs)
    monkeypatch.setenv("SGLANG_VIDEO_ACTION_TRAJS_URI", "s3://bucket/env-trajs.jsonl")

    trajs = read_action_trajectories(
        {"input": {"action_trajs_uri": "s3://bucket/request-trajs.jsonl"}},
        FailingS3Client(),
    )

    assert [traj["traj_id"] for traj in trajs] == ["bundled-traj"]


def test_read_action_trajectories_supports_compressed_bundled_trajs(tmp_path, monkeypatch):
    bundled_trajs = tmp_path / "trajs.jsonl.gz"
    import gzip

    with gzip.open(bundled_trajs, "wt", encoding="utf-8") as file:
        file.write(json.dumps(_traj("compressed-traj", "w", "d")) + "\n")

    monkeypatch.setattr("run_t2i_video_batch.DEFAULT_ACTION_TRAJS_PATH", bundled_trajs)

    trajs = read_action_trajectories({}, object())

    assert [traj["traj_id"] for traj in trajs] == ["compressed-traj"]


def test_progress_callback_does_not_block_upload_watcher(tmp_path):
    started = threading.Event()
    unblock = threading.Event()
    watcher = ProgressUploadWatcher(
        runtime=RuntimeInputs(tmp_path / "messages", tmp_path / "urls", tmp_path / "results"),
        cases=[],
        request=_request(),
        s3_client=FakeS3Client(),
        delete_local_outputs=True,
        progress_callback=lambda _payload: (started.set(), unblock.wait(5)),
    )

    started_at = time.monotonic()
    watcher._emit_progress_callback(force=True)

    assert time.monotonic() - started_at < 0.1
    assert started.wait(1)
    unblock.set()


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

    class FakeProcess:
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            raise AssertionError("completed runner should not be terminated")

        def kill(self):
            raise AssertionError("completed runner should not be killed")

    def fake_popen(command, env, start_new_session):
        captured["command"] = command
        captured["env"] = env
        captured["start_new_session"] = start_new_session
        summary_path = runtime.results_root / "cases" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text('{"results":[]}\n', encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr("run_t2i_video_batch.subprocess.Popen", fake_popen)

    summary = run_benchmark(runtime, request=request)

    assert summary == {"results": []}
    assert captured["start_new_session"] is True
    assert captured["env"]["SGLANG_VIDEO_WIDTH"] == "1024"
    assert captured["env"]["SGLANG_VIDEO_HEIGHT"] == "576"
    assert captured["env"]["SGLANG_VIDEO_FPS"] == "12"


def test_run_benchmark_returns_summary_when_runner_exits_nonzero(tmp_path, monkeypatch):
    request = _request()
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())
    runtime = build_runtime_inputs(
        request=request,
        cases=cases,
        work_dir=tmp_path,
        s3_client=FakeS3Client(),
    )

    class FakeProcess:
        returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            raise AssertionError("completed runner should not be terminated")

        def kill(self):
            raise AssertionError("completed runner should not be killed")

    def fake_popen(command, env, start_new_session):
        summary_path = runtime.results_root / "cases" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "summary": {"successful_samples": 1, "failed_samples": 1},
                    "results": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("run_t2i_video_batch.subprocess.Popen", fake_popen)

    summary = run_benchmark(runtime, request=request)

    assert summary["summary"]["successful_samples"] == 1
    assert summary["summary"]["failed_samples"] == 1
    assert summary["summary"]["benchmark_exit_code"] == 1
    assert summary["summary"]["benchmark_error"] == "benchmark failed with exit code 1"


def test_run_benchmark_returns_terminal_summary_when_runner_hangs_after_generation(
    tmp_path, monkeypatch
):
    request = _request()
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())
    runtime = build_runtime_inputs(
        request=request,
        cases=cases,
        work_dir=tmp_path,
        s3_client=FakeS3Client(),
    )
    summary_path = runtime.results_root / "cases" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "summary": {
                    "selected_samples": 1,
                    "successful_samples": 1,
                    "failed_samples": 0,
                },
                "results": [{"sample_id": "sample/img001", "success": True}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class HangingProcess:
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise TimeoutError("still running")
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    process = HangingProcess()

    def fake_popen(command, env, start_new_session):
        return process

    monkeypatch.setenv("SGLANG_VIDEO_BENCHMARK_EXIT_GRACE_SECONDS", "0")
    monkeypatch.setenv("SGLANG_VIDEO_BENCHMARK_POLL_SECONDS", "0")
    monkeypatch.setattr("run_t2i_video_batch.subprocess.Popen", fake_popen)

    summary = run_benchmark(runtime, request=request)

    assert process.terminated is True
    assert summary["summary"]["successful_samples"] == 1
    assert summary["summary"]["benchmark_terminated_after_terminal_summary"] is True


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


def test_collect_upload_results_accepts_stream_uploaded_deleted_outputs(tmp_path):
    cases = build_case_records(_request(), _rows(), action_trajectories=_trajs())
    uploaded_output = tmp_path / "streamed.mp4"
    uploaded_output.write_bytes(b"mp4")
    uploaded_output.unlink()
    fake_s3 = FakeS3Client()
    bucket, key = parse_s3_uri(cases[0]["video_s3_uri"])
    fake_s3.existing.add((bucket, key))

    results = collect_upload_results(
        cases,
        {
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(uploaded_output),
                }
            ]
        },
        fake_s3,
        request=_request(),
    )

    assert fake_s3.uploads == []
    assert results == [
        {
            "case_id": cases[0]["case_id"],
            "status": "succeeded",
            "video_uri": cases[0]["video_s3_uri"],
            "movement_key": cases[0]["movement_key"],
            "ending_movement_key": cases[0]["ending_movement_key"],
            "movement_pair": cases[0]["movement_pair"],
            "camera_key": cases[0]["camera_key"],
            "traj_id": cases[0]["traj_id"],
            "traj_type": cases[0]["traj_type"],
            "action_source": cases[0]["action_source"],
            "action_index": cases[0]["action_index"],
            "action_seed": cases[0]["action_seed"],
            "action_pattern": cases[0]["action_pattern"],
        }
    ]


def test_collect_upload_results_writes_metadata_and_deletes_local_video(tmp_path):
    cases = build_case_records(_request(), _rows(), action_trajectories=_trajs())
    successful_output = tmp_path / "video.mp4"
    successful_output.write_bytes(b"mp4")
    fake_s3 = FakeS3Client()

    collect_upload_results(
        cases,
        {
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(successful_output),
                }
            ]
        },
        fake_s3,
        request=_request(),
        delete_local_outputs=True,
    )

    assert not successful_output.exists()
    checkpoint_bucket, checkpoint_key = parse_s3_uri(
        case_checkpoint_s3_uri(_request(), cases[0])
    )
    metadata_bucket, metadata_key = parse_s3_uri(case_metadata_s3_uri(_request(), cases[0]))
    assert any(
        obj["Bucket"] == checkpoint_bucket and obj["Key"] == checkpoint_key
        for obj in fake_s3.objects
    )
    metadata_objects = [
        obj
        for obj in fake_s3.objects
        if obj["Bucket"] == metadata_bucket and obj["Key"] == metadata_key
    ]
    assert metadata_objects
    assert b'"video_prompt": "A quiet workshop."' in metadata_objects[0]["Body"]
    assert b'"video_uri":' in metadata_objects[0]["Body"]


def test_run_video_batch_streams_progress_uploads_before_benchmark_returns(
    tmp_path,
    monkeypatch,
):
    request = _request()
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())
    fake_s3 = FakeS3Client()

    def fake_run_benchmark(runtime, request):
        output = runtime.results_root / "cases" / "videos" / "video.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        progress = runtime.results_root / "cases" / "progress.jsonl"
        progress.write_text(
            json.dumps(
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(output),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 2.0
        while output.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not output.exists()
        return {
            "summary": {"successful_samples": 1, "failed_samples": 0},
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(output),
                }
            ],
        }

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_STREAM_UPLOAD", "true")
    monkeypatch.setattr("run_t2i_video_batch.run_benchmark", fake_run_benchmark)

    result = run_video_batch(
        request=request,
        cases=cases,
        work_dir=tmp_path / "work",
        s3_client=fake_s3,
    )

    assert result.results[0]["status"] == "succeeded"
    assert len(fake_s3.uploads) == 1
    # One preflight probe and one streaming-upload probe are sufficient. The
    # terminal collector must reuse the watcher's persisted result.
    assert fake_s3.head_calls == 2


def test_run_video_batch_reports_preflight_and_attempt_stage_timings(tmp_path, monkeypatch):
    request = _request()
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_STREAM_UPLOAD", "false")
    monkeypatch.setattr(
        "run_t2i_video_batch.run_benchmark",
        lambda runtime, request: {
            "summary": {"successful_samples": 1, "failed_samples": 0},
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(tmp_path / "video.mp4"),
                }
            ],
        },
    )

    output = tmp_path / "video.mp4"
    output.write_bytes(b"mp4")
    result = run_video_batch(
        request=request,
        cases=cases,
        work_dir=tmp_path / "work",
        s3_client=FakeS3Client(),
    )

    assert result.timings["resume_preflight_sec"] >= 0
    assert result.attempts[0]["runtime_input_sec"] >= 0
    assert result.attempts[0]["benchmark_sec"] >= 0
    assert result.attempts[0]["finalize_sec"] >= 0


def test_run_video_batch_posts_incremental_callback_after_stream_upload(
    tmp_path,
    monkeypatch,
):
    request = _request()
    rows = _rows() + [
        {
            "item_id": "img002",
            "image_uri": "s3://bucket/t2i/images/img002.png",
            "image_prompt": "A red robot.",
            "video_prompt": "A red robot.",
            "video_prompt_source": "image_prompt_fallback",
        }
    ]
    cases = build_case_records(request, rows, action_trajectories=_trajs())
    fake_s3 = FakeS3Client()
    callbacks = []

    def fake_run_benchmark(runtime, request):
        output = runtime.results_root / "cases" / "videos" / "video.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        progress = runtime.results_root / "cases" / "progress.jsonl"
        progress.write_text(
            json.dumps(
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(output),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 2.0
        while not callbacks and time.monotonic() < deadline:
            time.sleep(0.01)
        assert callbacks
        return {
            "summary": {"successful_samples": 1, "failed_samples": 1},
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(output),
                },
                {
                    "sample_id": cases[1]["sample_id"],
                    "success": False,
                    "error": "RuntimeError: invalid generate request",
                },
            ],
        }

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_STREAM_UPLOAD", "true")
    monkeypatch.setenv("SGLANG_VIDEO_CALLBACK_PROGRESS_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("SGLANG_VIDEO_BATCH_MAX_ATTEMPTS", "1")
    monkeypatch.setattr("run_t2i_video_batch.run_benchmark", fake_run_benchmark)

    run_video_batch(
        request=request,
        cases=cases,
        work_dir=tmp_path / "work",
        s3_client=fake_s3,
        progress_callback=callbacks.append,
    )

    assert callbacks[0]["status"] == "running"
    assert callbacks[0]["summary"]["video_expected_count"] == 2
    assert callbacks[0]["summary"]["video_succeeded_count"] == 1
    assert callbacks[0]["summary"]["video_failed_count"] == 0
    assert callbacks[0]["summary"]["video_running_count"] == 1


def test_run_video_batch_retries_only_failed_cases_and_preserves_successes(
    tmp_path,
    monkeypatch,
):
    request = _request()
    rows = _rows() + [
        {
            "item_id": "img002",
            "image_uri": "s3://bucket/t2i/images/img002.png",
            "image_prompt": "A red robot.",
            "video_prompt": "A red robot.",
            "video_prompt_source": "image_prompt_fallback",
        }
    ]
    cases = build_case_records(request, rows, action_trajectories=_trajs())
    fake_s3 = FakeS3Client()
    attempts = []

    def fake_run_benchmark(runtime, request):
        attempts.append(runtime)
        message_rows = [
            json.loads(line)
            for line in runtime.messages_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(attempts) == 1:
            assert [row["sample_id"] for row in message_rows] == [
                cases[0]["sample_id"],
                cases[1]["sample_id"],
            ]
            output = tmp_path / "first.mp4"
            output.write_bytes(b"mp4-first")
            return {
                "summary": {"successful_samples": 1, "failed_samples": 1},
                "results": [
                    {
                        "sample_id": cases[0]["sample_id"],
                        "success": True,
                        "output": str(output),
                    },
                    {
                        "sample_id": cases[1]["sample_id"],
                        "success": False,
                        "error": "RuntimeError: invalid generate request",
                    },
                ],
            }

        assert [row["sample_id"] for row in message_rows] == [cases[1]["sample_id"]]
        output = tmp_path / "second.mp4"
        output.write_bytes(b"mp4-second")
        return {
            "summary": {"successful_samples": 1, "failed_samples": 0},
            "results": [
                {
                    "sample_id": cases[1]["sample_id"],
                    "success": True,
                    "output": str(output),
                }
            ],
        }

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_MAX_ATTEMPTS", "2")
    monkeypatch.setattr("run_t2i_video_batch.run_benchmark", fake_run_benchmark)

    result = run_video_batch(
        request=request,
        cases=cases,
        work_dir=tmp_path / "work",
        s3_client=fake_s3,
    )

    assert len(attempts) == 2
    assert [attempt.results_root.name for attempt in attempts] == ["results", "results"]
    assert [attempt.results_root.parent.name for attempt in attempts] == [
        "attempt-001",
        "attempt-002",
    ]
    assert [row["status"] for row in result.results] == ["succeeded", "succeeded"]
    assert [attempt["status"] for attempt in result.attempts] == ["partial", "succeeded"]
    assert [upload["filename"] for upload in fake_s3.uploads] == [
        str(tmp_path / "first.mp4"),
        str(tmp_path / "second.mp4"),
    ]


def test_split_completed_cases_skips_existing_video_outputs_and_reports_resume_success(tmp_path):
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
    fake_s3 = FakeS3Client()
    bucket, key = parse_s3_uri(cases[0]["video_s3_uri"])
    fake_s3.existing.add((bucket, key))
    checkpoint_bucket, checkpoint_key = parse_s3_uri(case_checkpoint_s3_uri(_request(), cases[0]))
    fake_s3.existing.add((checkpoint_bucket, checkpoint_key))

    pending, resumed_results = split_completed_cases(cases, _request(), fake_s3)

    assert [case["case_id"] for case in pending] == [cases[1]["case_id"]]
    assert resumed_results == [
        {
            "case_id": cases[0]["case_id"],
            "status": "succeeded",
            "video_uri": cases[0]["video_s3_uri"],
            "movement_key": cases[0]["movement_key"],
            "ending_movement_key": cases[0]["ending_movement_key"],
            "movement_pair": cases[0]["movement_pair"],
            "camera_key": cases[0]["camera_key"],
            "traj_id": cases[0]["traj_id"],
            "traj_type": cases[0]["traj_type"],
            "action_source": cases[0]["action_source"],
            "action_index": cases[0]["action_index"],
            "action_seed": cases[0]["action_seed"],
            "action_pattern": cases[0]["action_pattern"],
            "resumed": True,
        }
    ]


def test_collect_upload_results_writes_case_checkpoint_after_success(tmp_path):
    cases = build_case_records(_request(), _rows(), action_trajectories=_trajs())
    successful_output = tmp_path / "video.mp4"
    successful_output.write_bytes(b"mp4")
    fake_s3 = FakeS3Client()

    collect_upload_results(
        cases,
        {
            "results": [
                {
                    "sample_id": cases[0]["sample_id"],
                    "success": True,
                    "output": str(successful_output),
                }
            ]
        },
        fake_s3,
        request=_request(),
    )

    checkpoint_uri = case_checkpoint_s3_uri(_request(), cases[0])
    bucket, key = parse_s3_uri(checkpoint_uri)
    assert any(
        obj["Bucket"] == bucket
        and obj["Key"] == key
        and b'"status": "succeeded"' in obj["Body"]
        and b'"video_uri":' in obj["Body"]
        for obj in fake_s3.objects
    )


def test_upload_report_writes_report_to_requested_s3_uri():
    fake_s3 = FakeS3Client()

    upload_report({"summary": {"video_status": "succeeded"}}, _request(), fake_s3)

    assert fake_s3.objects[0]["Bucket"] == "bucket"
    assert fake_s3.objects[0]["Key"] == "t2i/reports/sglang_video_report.json"
    assert fake_s3.objects[0]["ContentType"] == "application/json"
    assert b'"video_status": "succeeded"' in fake_s3.objects[0]["Body"]


def test_post_callback_uses_generation_progress_put_with_callback_token_headers(monkeypatch):
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
        sent["x_lwdp_token"] = request.headers.get("X-lwdp-token")
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
    assert sent["x_lwdp_token"] == "callback-token"
    assert sent["content_type"] == "application/json"
    assert sent["body"] == b'{"status": "succeeded"}'
    assert sent["timeout"] == 15.0


def test_post_callback_retries_transient_http_failures(monkeypatch):
    attempts = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        attempts.append((request.full_url, timeout))
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                502,
                "Bad Gateway",
                hdrs=None,
                fp=None,
            )
        return FakeResponse()

    monkeypatch.setenv("SGLANG_VIDEO_CALLBACK_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("SGLANG_VIDEO_CALLBACK_RETRY_BASE_SECONDS", "0")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    post_callback(
        {
            "callback": {
                "url": "https://pipeline.example.com/api/v1/generation/jobs/gen/progress"
            }
        },
        {"status": "succeeded"},
    )

    assert len(attempts) == 3


def test_main_accepts_completed_batch_with_failed_videos_and_keeps_failure_workdir(
    tmp_path,
    monkeypatch,
):
    request = _request()
    cases = build_case_records(
        request,
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
    callback_payloads = []
    reports = []
    cleaned = []

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr("run_t2i_video_batch._load_request", lambda: request)
    monkeypatch.setattr("run_t2i_video_batch._make_s3_client", FakeS3Client)
    monkeypatch.setattr("run_t2i_video_batch.read_jsonl_uri", lambda *_args: _rows())
    monkeypatch.setattr("run_t2i_video_batch.read_action_trajectories", lambda *_args: _trajs())
    monkeypatch.setattr(
        "run_t2i_video_batch.build_case_records",
        lambda *_args, **_kwargs: cases,
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.run_video_batch",
        lambda **_kwargs: SimpleNamespace(
            attempts=[{"attempt": 1, "status": "partial"}],
            results=[
                {"case_id": cases[0]["case_id"], "status": "succeeded"},
                {
                    "case_id": cases[1]["case_id"],
                    "status": "failed",
                    "error": "RuntimeError: invalid generate request",
                },
            ],
        ),
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.upload_report",
        lambda report, *_args: reports.append(report),
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.post_callback",
        lambda _request, payload: callback_payloads.append(payload),
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.cleanup_work_dir",
        lambda work_dir: cleaned.append(work_dir),
    )

    main()

    assert callback_payloads[0]["status"] == "completed"
    assert callback_payloads[0]["summary"]["video_status"] == "completed_with_failures"
    assert reports[0]["summary"]["video_failed_count"] == 1
    assert cleaned == []


def test_main_can_defer_final_callback_to_controller(tmp_path, monkeypatch):
    request = _request()
    cases = build_case_records(request, _rows(), action_trajectories=_trajs())
    callback_payloads = []
    reports = []

    monkeypatch.setenv("SGLANG_VIDEO_BATCH_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("SGLANG_VIDEO_BATCH_DEFER_FINAL_CALLBACK", "true")
    monkeypatch.setattr("run_t2i_video_batch._load_request", lambda: request)
    monkeypatch.setattr("run_t2i_video_batch._make_s3_client", FakeS3Client)
    monkeypatch.setattr("run_t2i_video_batch.read_jsonl_uri", lambda *_args: _rows())
    monkeypatch.setattr("run_t2i_video_batch.read_action_trajectories", lambda *_args: _trajs())
    monkeypatch.setattr(
        "run_t2i_video_batch.build_case_records",
        lambda *_args, **_kwargs: cases,
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.run_video_batch",
        lambda **_kwargs: SimpleNamespace(
            attempts=[{"attempt": 1, "status": "succeeded"}],
            results=[{"case_id": cases[0]["case_id"], "status": "succeeded"}],
            timings={"resume_preflight_sec": 0.01},
        ),
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.upload_report",
        lambda report, *_args: reports.append(report),
    )
    monkeypatch.setattr(
        "run_t2i_video_batch.post_callback",
        lambda _request, payload: callback_payloads.append(payload),
    )

    main()

    assert reports[0]["summary"]["video_succeeded_count"] == 1
    assert callback_payloads == []
