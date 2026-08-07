#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${MINWM_RESUME_SOURCE_ROOT:?set MINWM_RESUME_SOURCE_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SGLANG_SOURCE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
CASE_ID="00_forward_080_pottery_720p"
LEGACY_SP2_DIR="${MINWM_RESUME_SOURCE_ROOT%/}/temb-hoist-legacy/sp2"
LEGACY_BITWISE="${MINWM_RESUME_SOURCE_ROOT%/}/temb-hoist-bitwise/cases/${CASE_ID}/temb-hoist-legacy.npy"
CANDIDATE_BITWISE="${RESULT_ROOT}/temb-hoist-bitwise/cases/${CASE_ID}/temb-hoist-candidate.npy"
CURRENT_LANE_DIR="${RESULT_ROOT}/resume-preflight"

export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export MINWM_S0_KV_CACHE_NUM_FRAMES=45
export MINWM_S0_PROFILER_OFF_ONLY=1

assert_no_nsys() {
  if pgrep -x nsys >/dev/null; then
    echo "S1 off-only resume refuses to run while an nsys process exists" >&2
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
  timestamp="$(date --utc +%Y%m%dT%H%M%S%NZ)"
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
        f"S1 off-only resume exited non-zero (status={status}); "
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
  trap - EXIT ERR INT TERM
  if (( status != 0 )); then
    mark_failed_lane "${status}" "${CURRENT_LANE_DIR}" || true
  fi
  exit "${status}"
}

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
if runner.count("assert_no_nsys_processes") < 4:
    raise SystemExit("profiler-off-only runtime nsys assertions are incomplete")
print(f"S1 source-registration and off-only static preflight passed: {source_python}")
PY

assert_no_nsys
for result in profiler-off-repeat1.json profiler-off-repeat2.json; do
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate "${LEGACY_SP2_DIR}/${result}"
  python3 "${SCRIPT_DIR}/assert_latency_counts.py" "${LEGACY_SP2_DIR}/${result}"
done

python3 - "${LEGACY_SP2_DIR}" "${LEGACY_BITWISE}" "${SGLANG_GIT_REF}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

lane = Path(sys.argv[1])
bitwise = Path(sys.argv[2])
sglang_ref = sys.argv[3]
records = [
    json.loads((lane / f"profiler-off-repeat{repeat}.json").read_text())
    for repeat in (1, 2)
]
summary = json.loads((lane / "repeat-summary.json").read_text())
for record in records:
    assert record["comparison_contract"]["kv_cache_num_frames"] == 45
    assert record["provenance"]["sglang_commit"] == sglang_ref
    assert record["workload"]["sp_degree"] == 2
    assert record["workload"]["warmup_chunks"] == 20
    assert record["workload"]["measured_chunks"] == 200
    assert record["workload"]["precision"] == "bf16"
    assert record["workload"]["dmd_forwards_per_chunk"] == 4
    assert record["workload"]["clean_cache_forwards_per_chunk"] == 1
assert summary["acceptance"]["passes_cv_target"] is True

digest = hashlib.sha256()
with bitwise.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 << 20), b""):
        digest.update(chunk)
expected = json.loads(bitwise.with_suffix(".json").read_text())["frames_sha256"]
if digest.hexdigest() != expected:
    raise SystemExit("legacy bitwise input hash does not match its JSON evidence")
print(f"S1 resume input validated: {lane}; legacy_bitwise_sha256={expected}")
PY

CURRENT_LANE_DIR="${RESULT_ROOT}/cuda-correctness"

python3 -m pytest -q \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py" \
  -k 'hoisted_timestep_modulation_matches_compiled_cuda_index'

export MINWM_HOIST_TIMESTEP_MODULATION=0
export MINWM_S0_RUN_LABEL=temb-hoist-legacy
export MINWM_S0_SP_DEGREES=4
export MINWM_S0_RUN_BITWISE=0
CURRENT_LANE_DIR=""
assert_no_nsys
bash "${SCRIPT_DIR}/run_s0_measurement.sh"
assert_no_nsys

export MINWM_HOIST_TIMESTEP_MODULATION=1
export MINWM_S0_RUN_LABEL=temb-hoist-candidate
export MINWM_S0_SP_DEGREES="2 4"
export MINWM_S0_RUN_BITWISE=1
CURRENT_LANE_DIR=""
bash "${SCRIPT_DIR}/run_s0_measurement.sh"
assert_no_nsys

CURRENT_LANE_DIR="${RESULT_ROOT}/temb-hoist-quality"
python3 - \
  "${LEGACY_BITWISE}" \
  "${CANDIDATE_BITWISE}" \
  "${LEGACY_SP2_DIR}/repeat-summary.json" \
  "${RESULT_ROOT}/temb-hoist-legacy/sp4/repeat-summary.json" \
  "${RESULT_ROOT}/temb-hoist-candidate/sp2/repeat-summary.json" \
  "${RESULT_ROOT}/temb-hoist-candidate/sp4/repeat-summary.json" \
  "${RESULT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

legacy_bitwise = Path(sys.argv[1])
candidate_bitwise = Path(sys.argv[2])
summaries = {
    "sp2": {
        "legacy": json.loads(Path(sys.argv[3]).read_text()),
        "candidate": json.loads(Path(sys.argv[5]).read_text()),
    },
    "sp4": {
        "legacy": json.loads(Path(sys.argv[4]).read_text()),
        "candidate": json.loads(Path(sys.argv[6]).read_text()),
    },
}
result_root = Path(sys.argv[7])

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()

legacy = np.load(legacy_bitwise, mmap_mode="r")
candidate = np.load(candidate_bitwise, mmap_mode="r")
array_equal = legacy.shape == candidate.shape
max_abs_diff = 0
if array_equal:
    for frame_index in range(legacy.shape[0]):
        lhs = legacy[frame_index]
        rhs = candidate[frame_index]
        if not np.array_equal(lhs, rhs):
            array_equal = False
            max_abs_diff = max(
                max_abs_diff,
                int(np.abs(lhs.astype(np.int16) - rhs.astype(np.int16)).max()),
            )

bitwise = {
    "schema_version": "minwm-temb-hoist-bitwise/v1",
    "case": legacy_bitwise.parent.name,
    "kv_cache_num_frames": 45,
    "legacy": {"path": str(legacy_bitwise), "sha256": digest(legacy_bitwise)},
    "candidate": {
        "path": str(candidate_bitwise),
        "sha256": digest(candidate_bitwise),
    },
    "shape": list(legacy.shape),
    "array_equal": array_equal,
    "max_abs_diff_uint8": max_abs_diff,
}
(result_root / "temb-hoist-bitwise-summary.json").write_text(
    json.dumps(bitwise, indent=2, sort_keys=True) + "\n"
)
if not array_equal:
    raise SystemExit("legacy/candidate frame arrays are not bitwise equal")

metrics = (
    "client_fps",
    "scheduler_fps",
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
)
comparisons = {}
for sp, lanes in summaries.items():
    if not all(
        lane["acceptance"]["passes_cv_target"] for lane in lanes.values()
    ):
        raise SystemExit(f"{sp} repeat CV contract failed")
    comparisons[sp] = {}
    for metric in metrics:
        legacy_value = lanes["legacy"]["metrics"][metric]["mean"]
        candidate_value = lanes["candidate"]["metrics"][metric]["mean"]
        raw_delta_pct = (candidate_value / legacy_value - 1.0) * 100.0
        comparisons[sp][metric] = {
            "legacy": legacy_value,
            "candidate": candidate_value,
            "candidate_delta_pct": raw_delta_pct,
            "improvement_pct": (
                raw_delta_pct if metric.endswith("fps") else -raw_delta_pct
            ),
        }

record = {
    "schema_version": "minwm-temb-hoist-profiler-off-ab/v1",
    "comparison_contract": {
        "kv_cache_num_frames": 45,
        "warmup_chunks": 20,
        "measured_chunks": 200,
        "precision": "bf16",
        "dmd_forwards_per_chunk": 4,
        "clean_cache_forwards_per_chunk": 1,
    },
    "summaries": {
        sp: {lane: value["run_ids"] for lane, value in lanes.items()}
        for sp, lanes in summaries.items()
    },
    "comparisons": comparisons,
    "bitwise": bitwise,
}
(result_root / "temb-hoist-profiler-off-ab-summary.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(record, indent=2, sort_keys=True))
PY

assert_no_nsys
CURRENT_LANE_DIR=""
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/temb-hoist-off-resume-complete.txt"
