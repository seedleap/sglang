#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SGLANG_SOURCE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
CURRENT_LANE_DIR="${RESULT_ROOT}/abba-preflight"

export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export MINWM_S0_KV_CACHE_NUM_FRAMES=45
export MINWM_S0_PROFILER_OFF_ONLY=1
export MINWM_S0_OFF_REPEAT_COUNT=1
export MINWM_S0_SP_DEGREES=4
export MINWM_S0_RUN_BITWISE=0

assert_no_nsys() {
  if pgrep -x nsys >/dev/null; then
    echo "S1 SP4 ABBA refuses to run while an nsys process exists" >&2
    pgrep -a -x nsys >&2 || true
    exit 2
  fi
}

mark_failed_lane() {
  local status="$1" lane_root="$2" timestamp marker
  if [[ -z "${lane_root}" ]]; then
    return
  fi
  mkdir -p "${lane_root}"
  if compgen -G "${lane_root}/invalid-marker*.json" >/dev/null; then
    return
  fi
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  marker="${lane_root}/invalid-marker-${timestamp}.json"
  python3 - "${lane_root}" "${marker}" "${status}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
marker = Path(sys.argv[2]).resolve()
status = int(sys.argv[3])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.resolve() == marker:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    files.append(
        {
            "original_path": str(path),
            "preserved_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "recoverable": True,
        }
    )
record = {
    "schema_version": "minwm-realtime-invalid-attempt/v1",
    "reason": (
        f"S1 SP4 ABBA exited non-zero (status={status}); "
        "artifacts preserved in place"
    ),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "original_root": str(root),
    "preserved_root": str(root),
    "recoverability": "preserved_in_place",
    "files": files,
}
marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
}

on_exit() {
  local status="$?"
  trap - EXIT INT TERM
  if (( status != 0 )); then
    mark_failed_lane "${status}" "${CURRENT_LANE_DIR}" || true
  fi
  exit "${status}"
}

main() {
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "${SOURCE_ROOT}" "${SCRIPT_DIR}/run_s0_measurement.sh" <<'PY'
import os
import re
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
source_python = str(source_root / "python")
if source_python not in os.environ["PYTHONPATH"].split(os.pathsep):
    raise SystemExit(f"source tree missing from PYTHONPATH: {source_python}")
import sglang.test.ci.ci_register  # noqa: F401, E402

runner = Path(sys.argv[2]).read_text(encoding="utf-8")
guards = (
    r'if \[\[ "\$\{PROFILER_OFF_ONLY\}" != "1" \]\]; then\s+install_nsys\s+fi',
    r'if \[\[ "\$\{PROFILER_OFF_ONLY\}" != "1" \]\]; then\s+run_profiler_on',
)
for guard in guards:
    if not re.search(guard, runner):
        raise SystemExit(f"profiler-off-only static guard missing: {guard}")
if 'OFF_REPEAT_COUNT="${MINWM_S0_OFF_REPEAT_COUNT:-2}"' not in runner:
    raise SystemExit("single-repeat profiler-off control is missing")
if runner.count("assert_no_nsys_processes") < 4:
    raise SystemExit("profiler-off-only runtime nsys assertions are incomplete")
print(f"S1 ABBA source-registration and off-only preflight passed: {source_python}")
PY

assert_no_nsys
python3 -m pytest -q \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py" \
  -k 'hoisted_timestep_modulation_matches_compiled_cuda_index'

run_position() {
  local label="$1" hoist="$2"
  export MINWM_S0_RUN_LABEL="${label}"
  export MINWM_HOIST_TIMESTEP_MODULATION="${hoist}"
  CURRENT_LANE_DIR="${RESULT_ROOT}/${label}/sp4"
  assert_no_nsys
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  assert_no_nsys
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${RESULT_ROOT}/${label}/sp4/profiler-off-repeat1.json"
  python3 "${SCRIPT_DIR}/assert_latency_counts.py" \
    "${RESULT_ROOT}/${label}/sp4/profiler-off-repeat1.json"
  CURRENT_LANE_DIR=""
}

run_position temb-hoist-abba-a1-candidate 1
run_position temb-hoist-abba-b1-legacy 0
run_position temb-hoist-abba-b2-legacy 0
run_position temb-hoist-abba-a2-candidate 1

CURRENT_LANE_DIR="${RESULT_ROOT}/temb-hoist-abba-aggregate"
python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
  "${RESULT_ROOT}/temb-hoist-abba-a1-candidate/sp4/profiler-off-repeat1.json" \
  "${RESULT_ROOT}/temb-hoist-abba-a2-candidate/sp4/profiler-off-repeat1.json" \
  --output "${RESULT_ROOT}/temb-hoist-abba-candidate-repeat-summary.json"
python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
  "${RESULT_ROOT}/temb-hoist-abba-b1-legacy/sp4/profiler-off-repeat1.json" \
  "${RESULT_ROOT}/temb-hoist-abba-b2-legacy/sp4/profiler-off-repeat1.json" \
  --output "${RESULT_ROOT}/temb-hoist-abba-legacy-repeat-summary.json"

python3 - "${RESULT_ROOT}" <<'PY'
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

root = Path(sys.argv[1])
positions = (
    ("a1", "candidate", "temb-hoist-abba-a1-candidate"),
    ("b1", "legacy", "temb-hoist-abba-b1-legacy"),
    ("b2", "legacy", "temb-hoist-abba-b2-legacy"),
    ("a2", "candidate", "temb-hoist-abba-a2-candidate"),
)
records = {}
sessions = {}
startup = {}
for position, variant, label in positions:
    lane = root / label / "sp4"
    records[position] = json.loads(
        (lane / "profiler-off-repeat1.json").read_text()
    )
    cache_lines = []
    execution_profiles = set()
    server_settings = {}
    by_session = defaultdict(
        lambda: {"first_ms": None, "last_ms": None, "chunks": set()}
    )
    for line in (lane / "profiler-off-server.log").open(errors="replace"):
        if "torch.compile cache:" in line:
            cache_lines.append(line.strip())
        if "MinWM execution profile:" in line:
            execution_profiles.add(
                line.split("MinWM execution profile:", 1)[1].strip()
            )
        if "server_args: {" in line and not server_settings:
            settings = json.loads(line.split("server_args: ", 1)[1])
            server_settings = {
                "enable_torch_compile": settings["enable_torch_compile"],
                "warmup_mode": settings["warmup_mode"],
                "server_warmup": settings["server_warmup"],
            }
        if "realtime_trace {" not in line:
            continue
        try:
            event = json.loads(line.split("realtime_trace ", 1)[1])
        except json.JSONDecodeError:
            continue
        session_id = event.get("session_id")
        epoch_ms = event.get("server_epoch_ms")
        if session_id is None or epoch_ms is None or "chunk_index" not in event:
            continue
        session = by_session[session_id]
        session["first_ms"] = (
            epoch_ms
            if session["first_ms"] is None
            else min(session["first_ms"], epoch_ms)
        )
        session["last_ms"] = (
            epoch_ms
            if session["last_ms"] is None
            else max(session["last_ms"], epoch_ms)
        )
        session["chunks"].add(event["chunk_index"])
    complete = [value for value in by_session.values() if len(value["chunks"]) == 220]
    if len(complete) != 1:
        raise SystemExit(f"{position}: expected one 220-chunk session, got {len(complete)}")
    sessions[position] = complete[0]
    reason_names = (
        "SW Power Cap",
        "HW Slowdown",
        "HW Thermal Slowdown",
        "HW Power Brake Slowdown",
        "SW Thermal Slowdown",
    )
    reason_counts = {name: Counter() for name in reason_names}
    for line in (root / label / "nvidia-smi-q.txt").open(errors="replace"):
        stripped = line.strip()
        for name in reason_names:
            prefix = f"{name}"
            if stripped.startswith(prefix) and ":" in stripped:
                reason_counts[name][stripped.rsplit(":", 1)[1].strip()] += 1
    startup[position] = {
        "record_timestamp_utc": records[position]["timestamp_utc"],
        "init_send_start_to_first_payload_complete_ms": records[position][
            "client"
        ]["init_send_start_to_first_payload_complete_ms"],
        "server_settings": server_settings,
        "torch_compile_cache_lines": cache_lines,
        "execution_profiles": sorted(execution_profiles),
        "startup_clock_event_reason_counts": {
            name: dict(counts) for name, counts in reason_counts.items()
        },
    }

def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]

telemetry = {}
for position, variant, label in positions:
    lane = root / label / "sp4"
    session = sessions[position]
    rows = []
    with (lane / "profiler-off-gpu-telemetry.csv").open(newline="") as source:
        for raw in csv.reader(source):
            if len(raw) != 8:
                continue
            stamp = datetime.strptime(
                raw[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=timezone.utc)
            epoch_ms = stamp.timestamp() * 1000
            gpu = int(raw[1])
            if session["first_ms"] <= epoch_ms <= session["last_ms"] and gpu < 4:
                rows.append(
                    {
                        "gpu": gpu,
                        "utilization_pct": float(raw[2]),
                        "clock_mhz": float(raw[3]),
                        "pstate": raw[4].strip(),
                        "power_w": float(raw[5]),
                        "temperature_c": float(raw[6]),
                        "memory_mib": float(raw[7]),
                    }
                )
    if not rows:
        raise SystemExit(f"{position}: no active-GPU telemetry rows")
    item = {
        "variant": variant,
        "label": label,
        "session_duration_s": (
            session["last_ms"] - session["first_ms"]
        )
        / 1000,
        "sample_count": len(rows),
        "samples_per_gpu": dict(Counter(row["gpu"] for row in rows)),
        "pstates": dict(Counter(row["pstate"] for row in rows)),
    }
    for metric in (
        "utilization_pct",
        "clock_mhz",
        "power_w",
        "temperature_c",
        "memory_mib",
    ):
        values = [row[metric] for row in rows]
        item[metric] = {
            "mean": fmean(values),
            "min": min(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    telemetry[position] = item

candidate = json.loads(
    (root / "temb-hoist-abba-candidate-repeat-summary.json").read_text()
)
legacy = json.loads(
    (root / "temb-hoist-abba-legacy-repeat-summary.json").read_text()
)
if not candidate["acceptance"]["passes_cv_target"]:
    raise SystemExit("candidate ABBA repeat CV contract failed")
if not legacy["acceptance"]["passes_cv_target"]:
    raise SystemExit("legacy ABBA repeat CV contract failed")

metrics = (
    "client_fps",
    "scheduler_fps",
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
)
comparisons = {}
for metric in metrics:
    legacy_value = legacy["metrics"][metric]["mean"]
    candidate_value = candidate["metrics"][metric]["mean"]
    delta = (candidate_value / legacy_value - 1.0) * 100.0
    comparisons[metric] = {
        "legacy": legacy_value,
        "candidate": candidate_value,
        "candidate_delta_pct": delta,
        "improvement_pct": delta if metric.endswith("fps") else -delta,
    }

summary = {
    "schema_version": "minwm-temb-hoist-sp4-abba/v1",
    "order": [position for position, _, _ in positions],
    "variants": {position: variant for position, variant, _ in positions},
    "comparison_contract": {
        "sp_degree": 4,
        "kv_cache_num_frames": 45,
        "warmup_chunks": 20,
        "measured_chunks": 200,
        "precision": "bf16",
        "dmd_forwards_per_chunk": 4,
        "clean_cache_forwards_per_chunk": 1,
    },
    "run_ids": {
        position: record["run_id"] for position, record in records.items()
    },
    "comparisons": comparisons,
    "startup_and_compile_cache": startup,
    "telemetry": telemetry,
}
(root / "temb-hoist-sp4-abba-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

assert_no_nsys
CURRENT_LANE_DIR=""
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/temb-hoist-sp4-abba-complete.txt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
