#!/usr/bin/env python3
"""Run one t2i-triggered LingBot/SGLang video batch job."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t2i_video_batch import (
    build_callback_progress_payload,
    build_case_records,
    manifest_requires_action_trajectories,
    write_jsonl,
)


DEFAULT_PRESIGN_SECONDS = 7 * 24 * 60 * 60
DEFAULT_RUN_SCRIPT = "/opt/bench/run_capacity_smoke_720p.sh"
DEFAULT_S3_REGION = "us-east-2"
DEFAULT_ACTION_TRAJS_PATH = Path(__file__).with_name("trajs.jsonl")


@dataclass(frozen=True)
class RuntimeInputs:
    messages_path: Path
    image_urls_path: Path
    results_root: Path


@dataclass(frozen=True)
class BatchRunResult:
    results: list[dict[str, Any]]
    attempts: list[dict[str, Any]]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    bucket_key = uri[5:]
    if "/" not in bucket_key:
        raise ValueError(f"S3 URI has no key: {uri}")
    bucket, key = bucket_key.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"invalid S3 URI: {uri}")
    return bucket, key


def read_jsonl_uri(uri: str, s3_client: Any) -> list[dict[str, Any]]:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        text = body.decode("utf-8")
    elif uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=300) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(uri).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_action_trajectories(request: dict[str, Any], s3_client: Any) -> list[dict[str, Any]]:
    if not DEFAULT_ACTION_TRAJS_PATH.exists():
        raise FileNotFoundError(f"bundled action trajectories not found: {DEFAULT_ACTION_TRAJS_PATH}")
    return read_jsonl_uri(str(DEFAULT_ACTION_TRAJS_PATH), s3_client)


def _presigned_image_url(image_uri: str, s3_client: Any, expires_in: int) -> str:
    if not image_uri.startswith("s3://"):
        return image_uri
    bucket, key = parse_s3_uri(image_uri)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def build_runtime_inputs(
    *,
    request: dict[str, Any],
    cases: list[dict[str, Any]],
    work_dir: Path,
    s3_client: Any,
) -> RuntimeInputs:
    messages_path = work_dir / "inputs" / "messages.jsonl"
    image_urls_path = work_dir / "inputs" / "image-urls.json"
    results_root = work_dir / "results"
    write_jsonl(messages_path, cases)

    expires_in = int(request.get("input", {}).get("image_url_expires_in") or DEFAULT_PRESIGN_SECONDS)
    image_urls = {}
    for case in cases:
        image_id = case["image_id"]
        image_urls.setdefault(
            image_id,
            _presigned_image_url(case["image_uri"], s3_client, expires_in),
        )
    image_urls_path.parent.mkdir(parents=True, exist_ok=True)
    image_urls_path.write_text(
        json.dumps(image_urls, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return RuntimeInputs(
        messages_path=messages_path,
        image_urls_path=image_urls_path,
        results_root=results_root,
    )


def benchmark_run_script() -> Path:
    configured = Path(os.environ.get("SGLANG_VIDEO_BATCH_RUN_SCRIPT", DEFAULT_RUN_SCRIPT))
    if configured.exists():
        return configured
    return Path(__file__).with_name("run_capacity_smoke_720p.sh")


def _video_env(request: dict[str, Any]) -> dict[str, str]:
    video = request.get("video") if isinstance(request.get("video"), dict) else {}
    env: dict[str, str] = {}
    if video.get("width") not in (None, ""):
        env["SGLANG_VIDEO_WIDTH"] = str(video["width"])
    if video.get("height") not in (None, ""):
        env["SGLANG_VIDEO_HEIGHT"] = str(video["height"])
    if video.get("fps") not in (None, ""):
        env["SGLANG_VIDEO_FPS"] = str(video["fps"])
    return env


def run_benchmark(runtime: RuntimeInputs, *, request: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "RESULTS_ROOT": str(runtime.results_root),
            "MESSAGES_PATH": str(runtime.messages_path),
            "IMAGE_URLS_PATH": str(runtime.image_urls_path),
            "STREAM_UPLOAD": "false",
            "RESUME": env.get("RESUME", "false"),
        }
    )
    env.update(_video_env(request))
    completed = subprocess.run(
        ["bash", str(benchmark_run_script())],
        env=env,
        check=False,
    )
    summary_path = runtime.results_root / "cases" / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"benchmark failed with exit code {completed.returncode}; "
            f"summary not found: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if completed.returncode != 0:
        summary.setdefault("summary", {})["benchmark_exit_code"] = completed.returncode
        summary["summary"]["benchmark_error"] = (
            f"benchmark failed with exit code {completed.returncode}"
        )
    return summary


def _upload_video(path: Path, s3_uri: str, s3_client: Any) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def _s3_object_exists(s3_uri: str, s3_client: Any) -> bool:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"} or "Not Found" in str(error):
            return False
        raise
    return int(response.get("ContentLength") or 0) > 0


def _video_state_prefix(request: dict[str, Any]) -> str:
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    configured = str(output.get("video_state_s3_prefix") or "").rstrip("/")
    if configured:
        return configured
    video_prefix = str(output.get("video_s3_prefix") or "").rstrip("/")
    if video_prefix.endswith("/videos"):
        return video_prefix[: -len("/videos")] + "/video_state/cases"
    return video_prefix + "/video_state/cases"


def case_checkpoint_s3_uri(request: dict[str, Any], case: dict[str, Any]) -> str:
    return f"{_video_state_prefix(request)}/{case['case_id']}.json"


def _case_result_from_case(case: dict[str, Any], *, resumed: bool = False) -> dict[str, Any]:
    result = {
        "case_id": case["case_id"],
        "status": "succeeded",
        "video_uri": case["video_s3_uri"],
        "movement_key": case["movement_key"],
        "ending_movement_key": case["ending_movement_key"],
        "movement_pair": case["movement_pair"],
        "camera_key": case["camera_key"],
        "traj_id": case["traj_id"],
        "traj_type": case["traj_type"],
        "action_source": case["action_source"],
        "action_index": case["action_index"],
        "action_seed": case["action_seed"],
        "action_pattern": case["action_pattern"],
    }
    if resumed:
        result["resumed"] = True
    return result


def _write_case_checkpoint(
    request: dict[str, Any],
    case: dict[str, Any],
    result: dict[str, Any],
    s3_client: Any,
) -> None:
    checkpoint_uri = case_checkpoint_s3_uri(request, case)
    bucket, key = parse_s3_uri(checkpoint_uri)
    body = {
        "case_id": case["case_id"],
        "status": result["status"],
        "video_uri": result.get("video_uri") or case["video_s3_uri"],
        "movement_key": result.get("movement_key"),
        "ending_movement_key": result.get("ending_movement_key"),
        "movement_pair": result.get("movement_pair"),
        "camera_key": result.get("camera_key"),
        "traj_id": result.get("traj_id"),
        "traj_type": result.get("traj_type"),
        "action_source": result.get("action_source"),
        "action_index": result.get("action_index"),
        "action_seed": result.get("action_seed"),
        "action_pattern": result.get("action_pattern"),
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(body, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def split_completed_cases(
    cases: list[dict[str, Any]],
    request: dict[str, Any],
    s3_client: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    resumed_results: list[dict[str, Any]] = []
    for case in cases:
        if _s3_object_exists(case["video_s3_uri"], s3_client):
            result = _case_result_from_case(case, resumed=True)
            if not _s3_object_exists(case_checkpoint_s3_uri(request, case), s3_client):
                _write_case_checkpoint(request, case, result, s3_client)
            resumed_results.append(result)
            continue
        pending.append(case)
    return pending, resumed_results


def collect_upload_results(
    cases: list[dict[str, Any]],
    benchmark_summary: dict[str, Any],
    s3_client: Any,
    request: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cases_by_sample_id = {case["sample_id"]: case for case in cases}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in benchmark_summary.get("results", []):
        sample_id = row["sample_id"]
        if sample_id not in cases_by_sample_id:
            raise ValueError(f"unknown benchmark sample_id: {sample_id}")
        case = cases_by_sample_id[sample_id]
        seen.add(sample_id)
        if row.get("success"):
            output = Path(row["output"])
            _upload_video(output, case["video_s3_uri"], s3_client)
            result = _case_result_from_case(case)
            results.append(result)
            if request is not None:
                _write_case_checkpoint(request, case, result, s3_client)
        else:
            results.append(
                {
                    "case_id": case["case_id"],
                    "status": "failed",
                    "movement_key": case["movement_key"],
                    "ending_movement_key": case["ending_movement_key"],
                    "movement_pair": case["movement_pair"],
                    "camera_key": case["camera_key"],
                    "traj_id": case["traj_id"],
                    "traj_type": case["traj_type"],
                    "action_source": case["action_source"],
                    "action_index": case["action_index"],
                    "action_seed": case["action_seed"],
                    "action_pattern": case["action_pattern"],
                    "error": row.get("error") or "benchmark failed",
                }
            )

    for case in cases:
        if case["sample_id"] in seen:
            continue
        results.append(
            {
                "case_id": case["case_id"],
                "status": "failed",
                "movement_key": case["movement_key"],
                "ending_movement_key": case["ending_movement_key"],
                "movement_pair": case["movement_pair"],
                "camera_key": case["camera_key"],
                "traj_id": case["traj_id"],
                "traj_type": case["traj_type"],
                "action_source": case["action_source"],
                "action_index": case["action_index"],
                "action_seed": case["action_seed"],
                "action_pattern": case["action_pattern"],
                "error": "missing benchmark result",
            }
        )
    return results


def _max_attempts() -> int:
    raw_value = os.environ.get("SGLANG_VIDEO_BATCH_MAX_ATTEMPTS", "3")
    try:
        attempts = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"SGLANG_VIDEO_BATCH_MAX_ATTEMPTS must be an integer: {raw_value}"
        ) from error
    return max(1, attempts)


def _failed_case_result(case: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "status": "failed",
        "movement_key": case["movement_key"],
        "ending_movement_key": case["ending_movement_key"],
        "movement_pair": case["movement_pair"],
        "camera_key": case["camera_key"],
        "traj_id": case["traj_id"],
        "traj_type": case["traj_type"],
        "action_source": case["action_source"],
        "action_index": case["action_index"],
        "action_seed": case["action_seed"],
        "action_pattern": case["action_pattern"],
        "error": error,
    }


def run_video_batch(
    *,
    request: dict[str, Any],
    cases: list[dict[str, Any]],
    work_dir: Path,
    s3_client: Any,
) -> BatchRunResult:
    pending_cases, resumed_results = split_completed_cases(cases, request, s3_client)
    completed_by_case_id = {
        str(result["case_id"]): result for result in resumed_results
    }
    failed_by_case_id: dict[str, dict[str, Any]] = {}
    attempt_summaries: list[dict[str, Any]] = []

    for attempt in range(1, _max_attempts() + 1):
        if not pending_cases:
            break

        runtime = build_runtime_inputs(
            request=request,
            cases=pending_cases,
            work_dir=work_dir / f"attempt-{attempt:03d}",
            s3_client=s3_client,
        )
        try:
            benchmark_summary = run_benchmark(runtime, request=request)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            attempt_summaries.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "pending_count": len(pending_cases),
                    "error": error_text,
                }
            )
            for case in pending_cases:
                failed_by_case_id[case["case_id"]] = _failed_case_result(
                    case,
                    error_text,
                )
        else:
            attempt_results = collect_upload_results(
                pending_cases,
                benchmark_summary,
                s3_client,
                request=request,
            )
            succeeded_count = 0
            failed_count = 0
            for result in attempt_results:
                case_id = str(result["case_id"])
                if result.get("status") == "succeeded":
                    completed_by_case_id[case_id] = result
                    failed_by_case_id.pop(case_id, None)
                    succeeded_count += 1
                else:
                    failed_by_case_id[case_id] = result
                    failed_count += 1
            attempt_summaries.append(
                {
                    "attempt": attempt,
                    "status": "succeeded" if failed_count == 0 else "partial",
                    "pending_count": len(pending_cases),
                    "succeeded_count": succeeded_count,
                    "failed_count": failed_count,
                    "benchmark_summary": benchmark_summary.get("summary", {}),
                }
            )

        pending_cases = [
            case
            for case in cases
            if case["case_id"] not in completed_by_case_id
        ]

    final_results = [
        completed_by_case_id[case["case_id"]]
        for case in cases
        if case["case_id"] in completed_by_case_id
    ]
    final_results.extend(
        failed_by_case_id.get(case["case_id"])
        or _failed_case_result(case, "batch attempts exhausted")
        for case in cases
        if case["case_id"] not in completed_by_case_id
    )
    return BatchRunResult(results=final_results, attempts=attempt_summaries)


def upload_report(report: dict[str, Any], request: dict[str, Any], s3_client: Any) -> None:
    report_s3_uri = request.get("output", {}).get("report_s3_uri")
    if not report_s3_uri:
        return
    bucket, key = parse_s3_uri(str(report_s3_uri))
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def post_callback(request: dict[str, Any], payload: dict[str, Any]) -> None:
    callback = request.get("callback") if isinstance(request.get("callback"), dict) else {}
    url = callback.get("url")
    if not url:
        return
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("SGLANG_VIDEO_CALLBACK_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    http_request = urllib.request.Request(
        str(url),
        data=body,
        headers=headers,
        method="PUT",
    )
    with urllib.request.urlopen(http_request, timeout=60) as response:
        if response.status >= 300:
            raise RuntimeError(f"callback failed with HTTP {response.status}")


def _load_request() -> dict[str, Any]:
    request_json = os.environ.get("SGLANG_VIDEO_BATCH_REQUEST_JSON")
    request_path = os.environ.get("SGLANG_VIDEO_BATCH_REQUEST_PATH")
    if request_json:
        return json.loads(request_json)
    if request_path:
        return json.loads(Path(request_path).read_text(encoding="utf-8"))
    raise ValueError("SGLANG_VIDEO_BATCH_REQUEST_JSON or _PATH is required")


def _make_s3_client() -> Any:
    import boto3
    from botocore.config import Config

    region_name = os.environ.get("SGLANG_VIDEO_S3_REGION") or DEFAULT_S3_REGION
    return boto3.client(
        "s3",
        region_name=region_name,
        config=Config(signature_version="s3v4"),
    )


def cleanup_work_dir(work_dir: Path) -> None:
    resolved = work_dir.resolve()
    if str(resolved) in {"/", "/fsx", "/tmp"}:
        raise ValueError(f"refusing to cleanup unsafe work_dir: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def main() -> None:
    request = _load_request()
    work_dir = Path(
        os.environ.get("SGLANG_VIDEO_BATCH_WORK_DIR")
        or f"/fsx/sglang-video/{request.get('generation_job_id', 'job')}"
    )
    s3_client = _make_s3_client()
    manifest_rows = read_jsonl_uri(
        str(request.get("input", {}).get("video_manifest_uri")),
        s3_client,
    )
    action_trajectories = (
        read_action_trajectories(request, s3_client)
        if manifest_requires_action_trajectories(request, manifest_rows)
        else None
    )
    cases = build_case_records(
        request,
        manifest_rows,
        action_trajectories=action_trajectories,
    )

    report: dict[str, Any]
    completed_successfully = False
    try:
        batch_result = run_video_batch(
            request=request,
            cases=cases,
            work_dir=work_dir,
            s3_client=s3_client,
        )
        upload_results = batch_result.results
        callback_payload = build_callback_progress_payload(request, cases, upload_results)
        report = {
            "request": request,
            "summary": callback_payload["summary"],
            "counters": callback_payload["counters"],
            "attempts": batch_result.attempts,
            "results": upload_results,
        }
        report_path = work_dir / "results" / "sglang-video-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        upload_report(report, request, s3_client)
        post_callback(request, callback_payload)
        completed_successfully = callback_payload["status"] == "succeeded"
        if not completed_successfully:
            failed = callback_payload["summary"]["video_failed_count"]
            total = callback_payload["summary"]["video_expected_count"]
            raise RuntimeError(f"video batch failed after retries: {failed}/{total} failed")
    finally:
        cleanup_enabled = _env_flag("SGLANG_VIDEO_BATCH_CLEANUP", True)
        cleanup_on_failure = _env_flag("SGLANG_VIDEO_BATCH_CLEANUP_ON_FAILURE", False)
        if cleanup_enabled and (completed_successfully or cleanup_on_failure):
            cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
