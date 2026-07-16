#!/usr/bin/env python3
"""Unsuspend a follow-up Kubernetes Job when its source finishes before a deadline."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
CA_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def request_json(method: str, path: str, body: Optional[dict] = None) -> dict:
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    token = TOKEN_PATH.read_text(encoding="utf-8")
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if method == "PATCH":
        headers["Content-Type"] = "application/merge-patch+json"
    elif data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"https://{host}:{port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    context = ssl.create_default_context(cafile=str(CA_PATH))
    with urllib.request.urlopen(req, context=context, timeout=30) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def delete_followup(path: str, reason: str) -> None:
    try:
        request_json(
            "DELETE",
            path,
            {
                "kind": "DeleteOptions",
                "apiVersion": "v1",
                "propagationPolicy": "Background",
            },
        )
        print(json.dumps({"action": "delete_followup", "reason": reason}), flush=True)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


def main() -> None:
    source_job = os.environ["SOURCE_JOB"]
    followup_job = os.environ["FOLLOWUP_JOB"]
    deadline = parse_timestamp(os.environ["SOURCE_DEADLINE"])
    poll_seconds = int(os.environ.get("POLL_SECONDS", "60"))
    namespace = NAMESPACE_PATH.read_text(encoding="utf-8").strip()
    root = f"/apis/batch/v1/namespaces/{namespace}/jobs"
    source_path = f"{root}/{source_job}"
    followup_path = f"{root}/{followup_job}"
    last_report = 0.0

    while True:
        source = request_json("GET", source_path)
        conditions = {
            item.get("type"): item
            for item in source.get("status", {}).get("conditions", [])
            if item.get("status") == "True"
        }
        if "Complete" in conditions:
            completed_at = parse_timestamp(
                conditions["Complete"].get("lastTransitionTime")
                or conditions["Complete"].get("lastProbeTime")
            )
            if completed_at < deadline:
                followup = request_json("PATCH", followup_path, {"spec": {"suspend": False}})
                print(
                    json.dumps(
                        {
                            "action": "unsuspend_followup",
                            "source_job": source_job,
                            "source_completed_at": completed_at.isoformat(),
                            "deadline": deadline.isoformat(),
                            "followup_job": followup_job,
                            "followup_suspended": followup.get("spec", {}).get("suspend"),
                        }
                    ),
                    flush=True,
                )
                return
            delete_followup(
                followup_path,
                f"source completed at {completed_at.isoformat()}, after {deadline.isoformat()}",
            )
            return
        if "Failed" in conditions:
            delete_followup(followup_path, "source job failed")
            raise SystemExit(1)

        now = time.monotonic()
        if now - last_report >= 300 or last_report == 0:
            status = source.get("status", {})
            print(
                json.dumps(
                    {
                        "action": "wait",
                        "source_job": source_job,
                        "deadline": deadline.isoformat(),
                        "active": status.get("active", 0),
                        "succeeded": status.get("succeeded", 0),
                        "completed_indexes": status.get("completedIndexes", ""),
                    }
                ),
                flush=True,
            )
            last_report = now
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
