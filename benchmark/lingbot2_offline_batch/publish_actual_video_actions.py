#!/usr/bin/env python3
"""Publish actual-only LingBot2 video/action manifests for the three TPV batches."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig


BUCKET = "leap-world-us-east-2"
REGION = "us-east-2"
KUBE_CONTEXT = "leap-world-aws03-usw2"
NAMESPACE = "default"
CURRENT_JOB = "codex-lingbot2-tpvremain3699x5-720p"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
PUBLISH_ROOT = (
    "world-model/eval/lingbot2/eval_results/minWM/video_action_manifests"
)


@dataclass(frozen=True)
class PhysicalBatch:
    name: str
    local_result: str
    fsx_root: str
    s3_prefix: str


@dataclass(frozen=True)
class LogicalBatch:
    name: str
    description: str
    physical_names: tuple[str, ...]
    output_slug: str


PHYSICAL_BATCHES = {
    "trajs_00000_04999": PhysicalBatch(
        name="trajs_00000_04999",
        local_result="2026-07-15-thirdperson1000x5-720p-129f",
        fsx_root=(
            "/fsx/world-model/eval/platform/eval_sets/minWM/"
            "third_person_all_1000x5_720p_129f_20260715"
        ),
        s3_prefix=(
            "world-model/eval/platform/eval_results/minWM/"
            "third_person_all_1000x5_720p_129f_20260715"
        ),
    ),
    "trajs_05000_09999": PhysicalBatch(
        name="trajs_05000_09999",
        local_result=(
            "2026-07-15-thirdperson1000x5-actions05000-09999-720p-129f"
        ),
        fsx_root=(
            "/fsx/world-model/eval/platform/eval_sets/minWM/"
            "third_person_all_1000x5_actions_05000_09999_720p_129f_20260715"
        ),
        s3_prefix=(
            "world-model/eval/platform/eval_results/minWM/"
            "third_person_all_1000x5_actions_05000_09999_720p_129f_20260715"
        ),
    ),
    "trajs_10000_14999": PhysicalBatch(
        name="trajs_10000_14999",
        local_result=(
            "2026-07-15-thirdperson1000x5-actions10000-14999-720p-129f"
        ),
        fsx_root=(
            "/fsx/world-model/eval/platform/eval_sets/minWM/"
            "third_person_all_1000x5_actions_10000_14999_720p_129f_20260715"
        ),
        s3_prefix=(
            "world-model/eval/platform/eval_results/minWM/"
            "third_person_all_1000x5_actions_10000_14999_720p_129f_20260715"
        ),
    ),
    "trajs_15000_19999": PhysicalBatch(
        name="trajs_15000_19999",
        local_result=(
            "2026-07-15-thirdperson1000x5-actions15000-19999-720p-129f"
        ),
        fsx_root=(
            "/fsx/world-model/eval/platform/eval_sets/minWM/"
            "third_person_all_1000x5_actions_15000_19999_720p_129f_20260715"
        ),
        s3_prefix=(
            "world-model/eval/platform/eval_results/minWM/"
            "third_person_all_1000x5_actions_15000_19999_720p_129f_20260715"
        ),
    ),
    "generated_3699x5": PhysicalBatch(
        name="generated_3699x5",
        local_result="2026-07-15-thirdperson-remaining3699x5-720p-129f",
        fsx_root=(
            "/fsx/world-model/eval/lingbot2/eval_sets/minWM/"
            "third_person_remaining_3699x5_720p_129f_20260715"
        ),
        s3_prefix=(
            "world-model/eval/lingbot2/eval_results/minWM/"
            "third_person_remaining_3699x5_720p_129f_20260715"
        ),
    ),
}

LOGICAL_BATCHES = {
    "batch1": LogicalBatch(
        name="batch1",
        description="1000 images x 5 videos using trajs rows 00000-04999",
        physical_names=("trajs_00000_04999",),
        output_slug="batch1_1000x5_trajs_00000_04999",
    ),
    "batch2": LogicalBatch(
        name="batch2",
        description="stopped continuation using trajs rows 05000-19999",
        physical_names=(
            "trajs_05000_09999",
            "trajs_10000_14999",
            "trajs_15000_19999",
        ),
        output_slug="batch2_1000x5_trajs_continuation_partial",
    ),
    "batch3": LogicalBatch(
        name="batch3",
        description="3699 images x 5 deterministic generated wasd+ijkl actions",
        physical_names=("generated_3699x5",),
        output_slug="batch3_3699x5_generated_wasd_ijkl",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        action="append",
        choices=sorted(LOGICAL_BATCHES),
        help="Logical batch to publish; repeat as needed. Defaults to all three.",
    )
    parser.add_argument("--aws-profile", default="spot")
    parser.add_argument("--kube-context", default=KUBE_CONTEXT)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--current-job", default=CURRENT_JOB)
    parser.add_argument("--no-fsx", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def collect_s3_status(
    s3: Any, physical: PhysicalBatch
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    benchmark: dict[str, dict[str, Any]] = {}
    uploads: dict[str, dict[str, Any]] = {}
    prefix = f"{physical.s3_prefix}/status/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(
                ("-benchmark-summary.json", "-upload-summary.json")
            ):
                continue
            payload = json.loads(
                s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            )
            target = uploads if key.endswith("-upload-summary.json") else benchmark
            for row in payload.get("results", []):
                if row.get("success"):
                    target[row["sample_id"]] = row
    return benchmark, uploads


def find_running_pod(context: str, namespace: str, job: str) -> str | None:
    payload = json.loads(
        subprocess.check_output(
            [
                "kubectl",
                "--context",
                context,
                "-n",
                namespace,
                "get",
                "pods",
                "-l",
                f"job-name={job}",
                "-o",
                "json",
            ],
            text=True,
        )
    )
    names = sorted(
        item["metadata"]["name"]
        for item in payload["items"]
        if item.get("status", {}).get("phase") == "Running"
    )
    return names[0] if names else None


def collect_fsx_status(
    context: str,
    namespace: str,
    pod: str,
    physicals: list[PhysicalBatch],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    remote_code = r'''
import glob, json, sys
roots = json.loads(sys.argv[1])
result = {}
for root in roots:
    progress = {}
    uploads = {}
    for path in glob.glob(root + '/shard-*/cases/progress.jsonl') + glob.glob(root + '/cases/progress.jsonl'):
        with open(path, encoding='utf-8') as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('success'):
                    progress[row['sample_id']] = row
    for path in glob.glob(root + '/shard-*/upload.log') + glob.glob(root + '/upload.log'):
        with open(path, encoding='utf-8') as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('success'):
                    uploads[row['sample_id']] = row
    result[root] = {'progress': progress, 'uploads': uploads}
print(json.dumps(result, separators=(',', ':')))
'''
    raw = subprocess.check_output(
        [
            "kubectl",
            "--context",
            context,
            "-n",
            namespace,
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            remote_code,
            json.dumps([item.fsx_root for item in physicals]),
        ],
        text=True,
    )
    payload = json.loads(raw)
    return {
        root: (value["progress"], value["uploads"])
        for root, value in payload.items()
    }


def build_frame_actions(segments: list[dict[str, Any]]) -> list[list[str]]:
    actions: list[list[str]] = [[] for _ in range(129)]
    for segment in segments:
        key = segment.get("key")
        if key is None:
            continue
        for index in range(segment["start_frame"], segment["end_frame"] + 1):
            actions[index] = [key]
    return actions


def build_physical_rows(
    physical: PhysicalBatch,
    progress: dict[str, dict[str, Any]],
    uploads: dict[str, dict[str, Any]],
    snapshot_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_root = RESULTS_ROOT / physical.local_result / "input"
    cases = {row["sample_id"]: row for row in read_jsonl(input_root / "case-index.jsonl")}
    outputs = {
        row["sample_id"]: row for row in read_jsonl(input_root / "output-s3-index.jsonl")
    }
    actual_ids = set(progress) & set(uploads)
    actual_shards = {cases[sample_id]["shard"] for sample_id in actual_ids}
    rows: list[dict[str, Any]] = []
    for messages_path in sorted(input_root.glob("messages-shard-*.jsonl.gz")):
        shard = int(messages_path.name.removeprefix("messages-shard-").split(".")[0])
        if shard not in actual_shards:
            continue
        with gzip.open(messages_path, "rt", encoding="utf-8") as file:
            for line in file:
                message = json.loads(line)
                sample_id = message["sample_id"]
                if sample_id not in actual_ids:
                    continue
                case = cases[sample_id]
                output = outputs[sample_id]
                metadata = message["metadata"]
                target = next(
                    item for item in message["messages"] if item["role"] == "target"
                )
                control = target["controls"][0]
                generated = progress[sample_id]
                uploaded = uploads[sample_id]
                balance = metadata.get("trajectory_balance", {})
                movement_key = metadata.get("movement_key", balance.get("movement_key"))
                camera_key = metadata.get("camera_key", balance.get("camera_key"))
                segments = metadata.get("source_segments", [])
                action_source = (
                    "generated_wasd_ijkl"
                    if "action_seed" in metadata
                    else "trajs.jsonl"
                )
                rows.append(
                    {
                        "schema_version": 2,
                        "physical_batch": physical.name,
                        "sample_id": sample_id,
                        "case_id": case["case_id"],
                        "case_index": case["case_index"],
                        "image": {
                            "image_id": case["image_id"],
                            "image_index": case["image_index"],
                            "s3_uri": case["image_uri"],
                        },
                        "video": {
                            "s3_uri": output["s3_uri"],
                            "http_url": output["http_url"],
                            "width": generated.get("media", {}).get("width", 1280),
                            "height": generated.get("media", {}).get("height", 720),
                            "fps": 24,
                            "frames": generated.get("media", {}).get("frames", 129),
                            "duration_seconds": generated.get("media", {}).get(
                                "duration_sec", 5.375
                            ),
                        },
                        "action_trajectory": {
                            "source": action_source,
                            "trajectory_id": metadata.get(
                                "source_trajectory_id", metadata.get("action_id")
                            ),
                            "source_trajectory_index": metadata.get(
                                "source_trajectory_index"
                            ),
                            "movement_key": movement_key,
                            "camera_key": camera_key,
                            "action_seed": metadata.get("action_seed"),
                            "action_pattern": metadata.get(
                                "action_pattern", balance.get("layout")
                            ),
                            "fps": 24,
                            "num_video_frames": 129,
                            "frame_actions": build_frame_actions(segments),
                            "segments": segments,
                        },
                        "model_controls": {
                            "type": control["type"],
                            "num_generated_frame_actions": len(control["actions"]),
                            "generated_frame_actions": control["actions"],
                            "temporal_compression": 4,
                            "latent_camera_actions": metadata.get(
                                "latent_camera_actions", []
                            ),
                            "quantization": metadata.get("action_quantization"),
                        },
                        "actual_output": {
                            "snapshot_at": snapshot_at,
                            "generated_success": True,
                            "s3_uploaded_success": True,
                            "upload_attempts": uploaded.get("attempts"),
                            "upload_status": uploaded.get("status"),
                            "generation_delivery_sec": generated.get(
                                "generation_delivery_sec"
                            ),
                            "persisted_end_to_end_sec": generated.get(
                                "persisted_end_to_end_sec"
                            ),
                            "scheduler_generated_fps": generated.get(
                                "scheduler_generated_fps"
                            ),
                            "delivered_and_persisted_fps": generated.get(
                                "delivered_and_persisted_fps"
                            ),
                            "realtime_factor": generated.get("realtime_factor"),
                            "media": generated.get("media"),
                        },
                    }
                )
    if len(rows) != len(actual_ids):
        found = {row["sample_id"] for row in rows}
        raise RuntimeError(
            f"{physical.name}: {len(actual_ids - found)} actual IDs missing from inputs"
        )
    rows.sort(key=lambda row: row["case_index"])
    return rows, {
        "physical_batch": physical.name,
        "generated_success": len(progress),
        "s3_uploaded_success": len(uploads),
        "actual_intersection": len(rows),
        "s3_video_prefix": f"s3://{BUCKET}/{physical.s3_prefix}/videos/",
    }


def upload_file(s3: Any, path: Path, key: str, content_type: str) -> None:
    s3.upload_file(
        str(path),
        BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
        Config=TransferConfig(
            max_concurrency=32,
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=32 * 1024 * 1024,
            use_threads=True,
        ),
    )


def main() -> None:
    args = parse_args()
    selected = args.batch or sorted(LOGICAL_BATCHES)
    session = boto3.Session(profile_name=args.aws_profile, region_name=REGION)
    s3 = session.client("s3")
    physical_names = {
        name
        for logical_name in selected
        for name in LOGICAL_BATCHES[logical_name].physical_names
    }
    statuses: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name in sorted(physical_names):
        statuses[name] = collect_s3_status(s3, PHYSICAL_BATCHES[name])

    pod = None if args.no_fsx else find_running_pod(
        args.kube_context, args.namespace, args.current_job
    )
    if pod:
        physicals = [PHYSICAL_BATCHES[name] for name in sorted(physical_names)]
        live = collect_fsx_status(
            args.kube_context, args.namespace, pod, physicals
        )
        for physical in physicals:
            benchmark, uploads = statuses[physical.name]
            live_benchmark, live_uploads = live[physical.fsx_root]
            benchmark.update(live_benchmark)
            uploads.update(live_uploads)

    snapshot_at = dt.datetime.now(dt.timezone.utc).isoformat()
    published = []
    for logical_name in selected:
        logical = LOGICAL_BATCHES[logical_name]
        rows: list[dict[str, Any]] = []
        physical_summaries = []
        for physical_name in logical.physical_names:
            physical = PHYSICAL_BATCHES[physical_name]
            physical_rows, physical_summary = build_physical_rows(
                physical, *statuses[physical_name], snapshot_at
            )
            for row in physical_rows:
                row["logical_batch"] = logical.name
            rows.extend(physical_rows)
            physical_summaries.append(physical_summary)

        output_dir = RESULTS_ROOT / "actual_video_actions" / logical.output_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "video_action.actual.jsonl"
        summary_path = output_dir / "video_action.actual.summary.json"
        with manifest_path.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        manifest_key = f"{PUBLISH_ROOT}/{logical.output_slug}/video_action.actual.jsonl"
        summary_key = (
            f"{PUBLISH_ROOT}/{logical.output_slug}/video_action.actual.summary.json"
        )
        summary = {
            "schema_version": 1,
            "logical_batch": logical.name,
            "description": logical.description,
            "selection": "generation success AND S3 upload success",
            "snapshot_at": snapshot_at,
            "actual_rows": len(rows),
            "physical_batches": physical_summaries,
            "manifest_s3_uri": f"s3://{BUCKET}/{manifest_key}",
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "fsx_snapshot_pod": pod,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not args.no_upload:
            upload_file(s3, manifest_path, manifest_key, "application/x-ndjson")
            upload_file(s3, summary_path, summary_key, "application/json")
        published.append(summary)
    print(json.dumps(published, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
