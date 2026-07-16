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
    write_jsonl,
)


DEFAULT_PRESIGN_SECONDS = 7 * 24 * 60 * 60
DEFAULT_RUN_SCRIPT = "/opt/bench/run_capacity_smoke_720p.sh"


@dataclass(frozen=True)
class RuntimeInputs:
    messages_path: Path
    image_urls_path: Path
    results_root: Path


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


def run_benchmark(runtime: RuntimeInputs) -> dict[str, Any]:
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
    completed = subprocess.run(
        ["bash", str(benchmark_run_script())],
        env=env,
        check=False,
    )
    summary_path = runtime.results_root / "cases" / "summary.json"
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed with exit code {completed.returncode}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _upload_video(path: Path, s3_uri: str, s3_client: Any) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )


def collect_upload_results(
    cases: list[dict[str, Any]],
    benchmark_summary: dict[str, Any],
    s3_client: Any,
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
            results.append(
                {
                    "case_id": case["case_id"],
                    "status": "succeeded",
                    "video_uri": case["video_s3_uri"],
                    "movement_key": case["movement_key"],
                    "camera_key": case["camera_key"],
                    "action_seed": case["action_seed"],
                    "action_pattern": case["action_pattern"],
                }
            )
        else:
            results.append(
                {
                    "case_id": case["case_id"],
                    "status": "failed",
                    "movement_key": case["movement_key"],
                    "camera_key": case["camera_key"],
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
                "camera_key": case["camera_key"],
                "action_seed": case["action_seed"],
                "action_pattern": case["action_pattern"],
                "error": "missing benchmark result",
            }
        )
    return results


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

    return boto3.client("s3")


def cleanup_work_dir(work_dir: Path) -> None:
    resolved = work_dir.resolve()
    if str(resolved) in {"/", "/fsx", "/tmp"}:
        raise ValueError(f"refusing to cleanup unsafe work_dir: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


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
    cases = build_case_records(request, manifest_rows)
    runtime = build_runtime_inputs(
        request=request,
        cases=cases,
        work_dir=work_dir,
        s3_client=s3_client,
    )

    report: dict[str, Any]
    try:
        benchmark_summary = run_benchmark(runtime)
        upload_results = collect_upload_results(cases, benchmark_summary, s3_client)
        callback_payload = build_callback_progress_payload(request, cases, upload_results)
        report = {
            "request": request,
            "summary": callback_payload["summary"],
            "counters": callback_payload["counters"],
            "results": upload_results,
        }
        report_path = runtime.results_root / "sglang-video-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        upload_report(report, request, s3_client)
        post_callback(request, callback_payload)
    finally:
        cleanup_enabled = os.environ.get("SGLANG_VIDEO_BATCH_CLEANUP", "true").lower()
        if cleanup_enabled in {"1", "true", "yes"}:
            cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
