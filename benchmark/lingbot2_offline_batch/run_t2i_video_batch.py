#!/usr/bin/env python3
"""Run one t2i-triggered LingBot/SGLang video batch job."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    timings: dict[str, float]


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


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    return max(minimum, _int_value(os.environ.get(name), default))


def _benchmark_summary_is_terminal(summary: dict[str, Any]) -> bool:
    summary_counts = (
        summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
    )
    selected = _int_value(summary_counts.get("selected_samples"))
    if selected <= 0:
        selected = len(summary.get("results") or [])
    succeeded = _int_value(summary_counts.get("successful_samples"))
    failed = _int_value(summary_counts.get("failed_samples"))
    return selected > 0 and succeeded + failed >= selected


def _terminate_process(process: subprocess.Popen[Any], *, timeout: float) -> None:
    pid = getattr(process, "pid", None)
    try:
        if pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()

    try:
        process.wait(timeout=max(timeout, 0.0))
        return
    except (subprocess.TimeoutExpired, TimeoutError):
        pass

    try:
        if pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    except Exception:
        process.kill()
    process.wait()


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
    summary_path = runtime.results_root / "cases" / "summary.json"
    process = subprocess.Popen(
        ["bash", str(benchmark_run_script())],
        env=env,
        start_new_session=True,
    )
    terminal_summary_seen_at: float | None = None
    terminated_after_terminal_summary = False
    exit_grace_seconds = _float_env("SGLANG_VIDEO_BENCHMARK_EXIT_GRACE_SECONDS", 30.0)
    poll_seconds = _float_env("SGLANG_VIDEO_BENCHMARK_POLL_SECONDS", 1.0)
    terminate_timeout = _float_env("SGLANG_VIDEO_BENCHMARK_TERMINATE_TIMEOUT_SECONDS", 15.0)
    while process.poll() is None:
        summary = _read_json_file(summary_path)
        if summary is not None and _benchmark_summary_is_terminal(summary):
            now = time.monotonic()
            if terminal_summary_seen_at is None:
                terminal_summary_seen_at = now
            elif now - terminal_summary_seen_at >= exit_grace_seconds:
                _terminate_process(process, timeout=terminate_timeout)
                terminated_after_terminal_summary = True
                break
        if poll_seconds > 0:
            time.sleep(poll_seconds)
        else:
            time.sleep(0)

    if not summary_path.exists():
        raise RuntimeError(
            f"benchmark failed with exit code {process.returncode}; "
            f"summary not found: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if terminated_after_terminal_summary:
        summary.setdefault("summary", {})[
            "benchmark_terminated_after_terminal_summary"
        ] = True
        summary["summary"]["benchmark_exit_code"] = process.returncode
    elif process.returncode != 0:
        summary.setdefault("summary", {})["benchmark_exit_code"] = process.returncode
        summary["summary"]["benchmark_error"] = (
            f"benchmark failed with exit code {process.returncode}"
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


def _metadata_prefix(request: dict[str, Any]) -> str:
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    configured = str(output.get("metadata_s3_prefix") or "").rstrip("/")
    if configured:
        return configured
    video_prefix = str(output.get("video_s3_prefix") or "").rstrip("/")
    if video_prefix.endswith("/videos"):
        return video_prefix[: -len("/videos")] + "/video_metadata"
    return video_prefix + "/video_metadata"


def case_metadata_s3_uri(request: dict[str, Any], case: dict[str, Any]) -> str:
    metadata_prefix = _metadata_prefix(request)
    output = request.get("output") if isinstance(request.get("output"), dict) else {}
    video_prefix = str(output.get("video_s3_prefix") or "").rstrip("/")
    video_uri = str(case.get("video_s3_uri") or "")
    if video_prefix and video_uri.startswith(video_prefix + "/"):
        relative = video_uri[len(video_prefix) + 1 :]
        if relative.endswith(".mp4"):
            relative = relative[: -len(".mp4")] + ".json"
    else:
        relative = f"{case['case_id']}.json"
    return f"{metadata_prefix}/{relative}"


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


def _write_case_metadata(
    request: dict[str, Any],
    case: dict[str, Any],
    result: dict[str, Any],
    s3_client: Any,
) -> None:
    metadata_uri = case_metadata_s3_uri(request, case)
    bucket, key = parse_s3_uri(metadata_uri)
    target_message = next(
        (
            message
            for message in case.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "target"
        ),
        {},
    )
    target_metadata = (
        target_message.get("metadata")
        if isinstance(target_message.get("metadata"), dict)
        else {}
    )
    body = {
        **(case.get("metadata") if isinstance(case.get("metadata"), dict) else {}),
        "sample_id": case["sample_id"],
        "case_id": case["case_id"],
        "status": result["status"],
        "image_id": case["image_id"],
        "image_uri": case["image_uri"],
        "video_uri": result.get("video_uri") or case["video_s3_uri"],
        "negative_prompt": target_metadata.get("negative_prompt"),
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


def _persist_successful_case_output(
    *,
    case: dict[str, Any],
    row: dict[str, Any],
    s3_client: Any,
    request: dict[str, Any] | None = None,
    delete_local_output: bool = False,
) -> dict[str, Any]:
    output = Path(str(row.get("output") or ""))
    exists_in_s3 = _s3_object_exists(case["video_s3_uri"], s3_client)
    if not exists_in_s3:
        if not output.is_file():
            raise FileNotFoundError(
                f"completed MP4 is missing and not uploaded: {output}"
            )
        _upload_video(output, case["video_s3_uri"], s3_client)

    result = _case_result_from_case(case)
    if request is not None:
        _write_case_checkpoint(request, case, result, s3_client)
        _write_case_metadata(request, case, result, s3_client)

    if delete_local_output and output.is_file():
        output.unlink()
    return result


def split_completed_cases(
    cases: list[dict[str, Any]],
    request: dict[str, Any],
    s3_client: Any,
    *,
    max_workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def check_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not _s3_object_exists(case["video_s3_uri"], s3_client):
            return case, None
        result = _case_result_from_case(case, resumed=True)
        if not _s3_object_exists(case_checkpoint_s3_uri(request, case), s3_client):
            _write_case_checkpoint(request, case, result, s3_client)
        return case, result

    pending: list[dict[str, Any]] = []
    resumed_results: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(cases) or 1))
    if workers == 1:
        checked_cases = (check_case(case) for case in cases)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-resume") as pool:
            checked_cases = list(pool.map(check_case, cases))

    for case, result in checked_cases:
        if result is None:
            pending.append(case)
        else:
            resumed_results.append(result)
    return pending, resumed_results


def collect_upload_results(
    cases: list[dict[str, Any]],
    benchmark_summary: dict[str, Any],
    s3_client: Any,
    request: dict[str, Any] | None = None,
    delete_local_outputs: bool = False,
    already_persisted_by_case_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases_by_sample_id = {case["sample_id"]: case for case in cases}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    already_persisted = already_persisted_by_case_id or {}

    for row in benchmark_summary.get("results", []):
        sample_id = row["sample_id"]
        if sample_id not in cases_by_sample_id:
            raise ValueError(f"unknown benchmark sample_id: {sample_id}")
        case = cases_by_sample_id[sample_id]
        seen.add(sample_id)
        persisted = already_persisted.get(str(case["case_id"]))
        if persisted is not None:
            results.append(persisted)
            continue
        if row.get("success"):
            try:
                results.append(
                    _persist_successful_case_output(
                        case=case,
                        row=row,
                        s3_client=s3_client,
                        request=request,
                        delete_local_output=delete_local_outputs,
                    )
                )
            except Exception as error:
                results.append(
                    _failed_case_result(case, f"{type(error).__name__}: {error}")
                )
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
        persisted = already_persisted.get(str(case["case_id"]))
        if persisted is not None:
            results.append(persisted)
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


class ProgressUploadWatcher:
    def __init__(
        self,
        *,
        runtime: RuntimeInputs,
        cases: list[dict[str, Any]],
        request: dict[str, Any],
        s3_client: Any,
        delete_local_outputs: bool,
        poll_seconds: float = 0.25,
        all_cases: list[dict[str, Any]] | None = None,
        initial_results: list[dict[str, Any]] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_callback_min_interval_seconds: float = 30.0,
    ) -> None:
        self.progress_path = runtime.results_root / "cases" / "progress.jsonl"
        self.cases_by_sample_id = {case["sample_id"]: case for case in cases}
        self.cases = all_cases or cases
        self.request = request
        self.s3_client = s3_client
        self.delete_local_outputs = delete_local_outputs
        self.poll_seconds = poll_seconds
        self.progress_callback = progress_callback
        self.progress_callback_min_interval_seconds = max(
            0.0,
            progress_callback_min_interval_seconds,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._uploaded_by_case_id: dict[str, dict[str, Any]] = {
            str(result["case_id"]): result for result in (initial_results or [])
        }
        self._errors: list[str] = []
        self._last_callback_at = 0.0

    def start(self) -> None:
        self._thread.start()
        if self._uploaded_by_case_id:
            self._emit_progress_callback(force=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=60)
        if self._thread.is_alive():
            with self._lock:
                self._errors.append("progress upload watcher did not stop within 60s")
        self._emit_progress_callback(force=True)

    def uploaded_results(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._uploaded_by_case_id.values())

    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._errors.append(f"{type(error).__name__}: {error}")

    def _emit_progress_callback(self, *, force: bool = False) -> None:
        if self.progress_callback is None:
            return
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and self._last_callback_at
                and now - self._last_callback_at
                < self.progress_callback_min_interval_seconds
            ):
                return
            results = list(self._uploaded_by_case_id.values())
            self._last_callback_at = now
        payload = build_callback_progress_payload(self.request, self.cases, results)
        try:
            self.progress_callback(payload)
        except Exception as error:
            self._record_error(error)

    def _handle_row(self, row: dict[str, Any], submitted: set[str]) -> None:
        sample_id = str(row.get("sample_id") or "")
        if not row.get("success") or sample_id in submitted:
            return
        case = self.cases_by_sample_id.get(sample_id)
        if case is None:
            raise ValueError(f"unknown progress sample_id: {sample_id}")
        result = _persist_successful_case_output(
            case=case,
            row=row,
            s3_client=self.s3_client,
            request=self.request,
            delete_local_output=self.delete_local_outputs,
        )
        with self._lock:
            self._uploaded_by_case_id[case["case_id"]] = result
        submitted.add(sample_id)
        self._emit_progress_callback()

    def _run(self) -> None:
        offset = 0
        partial = ""
        submitted: set[str] = set()
        while True:
            try:
                if self.progress_path.exists():
                    with self.progress_path.open(encoding="utf-8") as progress:
                        progress.seek(offset)
                        chunk = progress.read()
                        offset = progress.tell()
                    if chunk:
                        partial += chunk
                        lines = partial.split("\n")
                        partial = lines.pop()
                        for line in lines:
                            if line.strip():
                                self._handle_row(json.loads(line), submitted)
                if self._stop.is_set():
                    if partial.strip():
                        self._handle_row(json.loads(partial), submitted)
                    return
            except Exception as error:
                self._record_error(error)
            if self._stop.wait(self.poll_seconds):
                continue


def _max_attempts() -> int:
    raw_value = os.environ.get("SGLANG_VIDEO_BATCH_MAX_ATTEMPTS", "3")
    try:
        attempts = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"SGLANG_VIDEO_BATCH_MAX_ATTEMPTS must be an integer: {raw_value}"
        ) from error
    return max(1, attempts)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BatchRunResult:
    total_started_at = time.perf_counter()
    preflight_started_at = time.perf_counter()
    pending_cases, resumed_results = split_completed_cases(
        cases,
        request,
        s3_client,
        max_workers=_int_env("SGLANG_VIDEO_BATCH_RESUME_WORKERS", 16),
    )
    resume_preflight_sec = time.perf_counter() - preflight_started_at
    print(
        json.dumps(
            {
                "event": "resume_preflight_complete",
                "total_cases": len(cases),
                "pending_cases": len(pending_cases),
                "resumed_cases": len(resumed_results),
                "elapsed_sec": round(resume_preflight_sec, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed_by_case_id = {
        str(result["case_id"]): result for result in resumed_results
    }
    failed_by_case_id: dict[str, dict[str, Any]] = {}
    attempt_summaries: list[dict[str, Any]] = []
    stream_upload = _env_flag("SGLANG_VIDEO_BATCH_STREAM_UPLOAD", True)
    delete_local_outputs = _env_flag("SGLANG_VIDEO_BATCH_DELETE_UPLOADED_LOCAL", True)
    progress_callback_min_interval_seconds = _float_env(
        "SGLANG_VIDEO_CALLBACK_PROGRESS_INTERVAL_SECONDS",
        30.0,
    )

    for attempt in range(1, _max_attempts() + 1):
        if not pending_cases:
            break

        runtime_started_at = time.perf_counter()
        runtime = build_runtime_inputs(
            request=request,
            cases=pending_cases,
            work_dir=work_dir / f"attempt-{attempt:03d}",
            s3_client=s3_client,
        )
        runtime_input_sec = time.perf_counter() - runtime_started_at
        watcher = (
            ProgressUploadWatcher(
                runtime=runtime,
                cases=pending_cases,
                request=request,
                s3_client=s3_client,
                delete_local_outputs=delete_local_outputs,
                all_cases=cases,
                initial_results=list(completed_by_case_id.values()),
                progress_callback=progress_callback,
                progress_callback_min_interval_seconds=progress_callback_min_interval_seconds,
            )
            if stream_upload
            else None
        )
        if watcher is not None:
            watcher.start()
        benchmark_started_at = time.perf_counter()
        try:
            benchmark_summary = run_benchmark(runtime, request=request)
        except Exception as error:
            benchmark_sec = time.perf_counter() - benchmark_started_at
            if watcher is not None:
                watcher.stop()
                for result in watcher.uploaded_results():
                    completed_by_case_id[str(result["case_id"])] = result
            error_text = f"{type(error).__name__}: {error}"
            attempt_summaries.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "pending_count": len(pending_cases),
                    "error": error_text,
                    "runtime_input_sec": runtime_input_sec,
                    "benchmark_sec": benchmark_sec,
                    "stream_upload_errors": watcher.errors() if watcher is not None else [],
                }
            )
            for case in pending_cases:
                if case["case_id"] in completed_by_case_id:
                    continue
                failed_by_case_id[case["case_id"]] = _failed_case_result(
                    case,
                    error_text,
                )
        else:
            benchmark_sec = time.perf_counter() - benchmark_started_at
            finalize_started_at = time.perf_counter()
            streamed_by_case_id: dict[str, dict[str, Any]] = {}
            if watcher is not None:
                watcher.stop()
                streamed_by_case_id = {
                    str(result["case_id"]): result for result in watcher.uploaded_results()
                }
                completed_by_case_id.update(streamed_by_case_id)
            attempt_results = collect_upload_results(
                pending_cases,
                benchmark_summary,
                s3_client,
                request=request,
                delete_local_outputs=delete_local_outputs,
                already_persisted_by_case_id=streamed_by_case_id,
            )
            finalize_sec = time.perf_counter() - finalize_started_at
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
                    "runtime_input_sec": runtime_input_sec,
                    "benchmark_sec": benchmark_sec,
                    "finalize_sec": finalize_sec,
                    "stream_uploaded_count": len(watcher.uploaded_results())
                    if watcher is not None
                    else 0,
                    "stream_upload_errors": watcher.errors() if watcher is not None else [],
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
    return BatchRunResult(
        results=final_results,
        attempts=attempt_summaries,
        timings={
            "resume_preflight_sec": resume_preflight_sec,
            "total_sec": time.perf_counter() - total_started_at,
        },
    )


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


def _callback_retry_attempts() -> int:
    raw_value = os.environ.get("SGLANG_VIDEO_CALLBACK_RETRY_ATTEMPTS", "5")
    try:
        attempts = int(raw_value)
    except ValueError:
        attempts = 5
    return max(1, attempts)


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
    attempts = _callback_retry_attempts()
    base_sleep = _float_env("SGLANG_VIDEO_CALLBACK_RETRY_BASE_SECONDS", 1.0)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(http_request, timeout=60) as response:
                if response.status >= 300:
                    raise RuntimeError(f"callback failed with HTTP {response.status}")
                return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            if attempt >= attempts:
                raise
            sleep_seconds = base_sleep * min(2 ** (attempt - 1), 8)
            print(
                "generation progress callback retry "
                f"attempt={attempt} max_attempts={attempts} error={type(error).__name__}: {error}",
                flush=True,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


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
            progress_callback=lambda payload: post_callback(request, payload),
        )
        upload_results = batch_result.results
        callback_payload = build_callback_progress_payload(request, cases, upload_results)
        report = {
            "request": request,
            "summary": callback_payload["summary"],
            "counters": callback_payload["counters"],
            "attempts": batch_result.attempts,
            "timings": getattr(batch_result, "timings", {}),
            "results": upload_results,
            "callback_payload": callback_payload,
        }
        report_path = work_dir / "results" / "sglang-video-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        upload_report(report, request, s3_client)
        if _env_flag("SGLANG_VIDEO_BATCH_DEFER_FINAL_CALLBACK", False):
            print(
                json.dumps(
                    {
                        "event": "final_callback_deferred_to_controller",
                        "generation_job_id": request.get("generation_job_id"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            post_callback(request, callback_payload)
        completed_successfully = callback_payload["status"] == "succeeded"
    finally:
        cleanup_enabled = _env_flag("SGLANG_VIDEO_BATCH_CLEANUP", True)
        cleanup_on_failure = _env_flag("SGLANG_VIDEO_BATCH_CLEANUP_ON_FAILURE", False)
        if cleanup_enabled and (completed_successfully or cleanup_on_failure):
            cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
