#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${MINWM_S1_LEGACY_OFF_ROOT:?set MINWM_S1_LEGACY_OFF_ROOT}"
: "${MINWM_S1_CANDIDATE_OFF_ROOT:?set MINWM_S1_CANDIDATE_OFF_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SGLANG_SOURCE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASE_RUN_ID="${MINWM_RUN_ID}"
BASE_WORK_ROOT="/work/minwm-realtime/${BASE_RUN_ID}"
BASE_MODEL_DIR="${BASE_WORK_ROOT}/sglang-model"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}"
CURRENT_LANE_DIR="${RESULT_ROOT}/nsys-preflight"
mkdir -p "${CURRENT_LANE_DIR}"

export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export MINWM_S0_SP_DEGREES=2
export MINWM_S0_PROFILE_PRECONDITION_CHUNKS=20
export MINWM_S0_PROFILE_DISCARD_CHUNKS=1
export MINWM_S0_PROFILE_MEASURED_CHUNKS=10
export MINWM_S0_KV_CACHE_NUM_FRAMES=45
unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR

mark_failed_lane() {
  local status="$1" lane_root="$2" timestamp
  if [[ -z "${lane_root}" || ! -d "${lane_root}" ]]; then
    return
  fi
  if compgen -G "${lane_root}/invalid-marker*.json" >/dev/null; then
    return
  fi
  timestamp="$(date --utc +%Y%m%dT%H%M%S%NZ)"
  python3 "${SCRIPT_DIR}/measurement_tool.py" mark-invalid \
    --root "${lane_root}" \
    --reason "S1 timestep-hoist Nsight wrapper exited non-zero (status=${status})" \
    --marker "${lane_root}/invalid-marker-${timestamp}.json" || true
}

on_exit() {
  local status="$?"
  trap - EXIT INT TERM
  if (( status != 0 )); then
    mark_failed_lane "${status}" "${CURRENT_LANE_DIR}"
  fi
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "${SOURCE_ROOT}" "${SCRIPT_DIR}" <<'PY'
import os
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
script_dir = Path(sys.argv[2]).resolve()
source_python = str(source_root / "python")
if source_python not in os.environ["PYTHONPATH"].split(os.pathsep):
    raise SystemExit(f"source tree missing from PYTHONPATH: {source_python}")
import sglang.test.ci.ci_register  # noqa: F401, E402

required_by_file = {
    "run_s0_measurement.sh": (
        '--gpu-metrics-devices=all',
        '--require-complete-stable-nsys',
        'SGLANG_REALTIME_NSYS_WARMUP_CHUNKS="${PROFILE_DISCARD_CHUNKS}"',
        'SGLANG_REALTIME_NSYS_MEASURED_CHUNKS="${PROFILE_MEASURED_CHUNKS}"',
    ),
    "measurement.py": (
        "API_BOUNDARY_ATTRIBUTION_POLICY",
        "boundary_included_by_start_count",
        "aggregation_mode",
    ),
    "nsys_metrics.py": (
        "_discrete_event_start_attribution",
        "boundary_event_examples",
        "streaming selected metricId rows",
    ),
}
for name, required in required_by_file.items():
    contents = (script_dir / name).read_text()
    for text in required:
        if text not in contents:
            raise SystemExit(f"canonical Nsight guard missing from {name}: {text}")
print(f"S1 Nsight source-registration/canonical preflight passed: {source_python}")
PY

if pgrep -x nsys >/dev/null; then
  echo "S1 Nsight wrapper refuses a pre-existing nsys process" >&2
  pgrep -a -x nsys >&2 || true
  exit 2
fi
if [[ ! -f "${BASE_MODEL_DIR}/minwm_conversion_manifest.json" ]]; then
  echo "setup model is missing: ${BASE_MODEL_DIR}" >&2
  exit 2
fi

validate_off_source() {
  local variant="$1" root="$2"
  local lane="${root%/}/sp2"
  if [[ "${root}" != *"temb-hoist-${variant}" ]]; then
    echo "${variant} resume root has an unexpected label: ${root}" >&2
    return 1
  fi
  for repeat in 1 2; do
    python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
      "${lane}/profiler-off-repeat${repeat}.json"
  done
  python3 - "${lane}" "${SGLANG_GIT_REF}" <<'PY'
import json
import sys
from pathlib import Path

lane = Path(sys.argv[1])
sglang_ref = sys.argv[2]
for repeat in (1, 2):
    record = json.loads((lane / f"profiler-off-repeat{repeat}.json").read_text())
    assert record["mode"] == "profiler_off"
    assert record["provenance"]["sglang_commit"] == sglang_ref
    assert record["provenance"]["gpu"]["count"] == 2
    assert record["provenance"]["gpu"]["allocated_count"] == 8
    assert record["workload"]["sp_degree"] == 2
    assert record["workload"]["precision"] == "bf16"
    assert record["workload"]["warmup_chunks"] == 20
    assert record["workload"]["measured_chunks"] == 200
    assert record["comparison_contract"]["kv_cache_num_frames"] == 45
summary = json.loads((lane / "repeat-summary.json").read_text())
assert summary["acceptance"]["passes_cv_target"] is True
print(f"validated profiler-off resume source: {lane}")
PY
}

validate_off_source legacy "${MINWM_S1_LEGACY_OFF_ROOT}"
validate_off_source candidate "${MINWM_S1_CANDIDATE_OFF_ROOT}"

CURRENT_LANE_DIR="${RESULT_ROOT}/cuda-correctness"
mkdir -p "${CURRENT_LANE_DIR}"
python3 -m pytest -q \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py" \
  -k 'hoisted_timestep_modulation_matches_compiled_cuda_index'

validate_profile_result() {
  local variant="$1" measurement="$2"
  python3 - "${SCRIPT_DIR}" "${measurement}" "${variant}" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
measurement = Path(sys.argv[2])
variant = sys.argv[3]
sys.path.insert(0, str(script_dir))

from compare_temb_hoist_nsys import _assert_contract, _load_record

record = _load_record(measurement)
_assert_contract(record)
print(f"validated exact formal Nsight result: variant={variant} path={measurement}")
PY
}

run_variant() {
  local variant="$1" hoist="$2" source_root="$3"
  local variant_run_id="${BASE_RUN_ID}-${variant}"
  local variant_work_root="/work/minwm-realtime/${variant_run_id}"
  local variant_result_root="${MINWM_RESULTS_ROOT%/}/${variant_run_id}/s0-measurement"
  local profile_dir="${variant_result_root}/sp2/profiler-on"
  local preflight_dir="${RESULT_ROOT}/${variant}-preflight"

  CURRENT_LANE_DIR="${preflight_dir}"
  mkdir -p "${CURRENT_LANE_DIR}"
  mkdir -p "${variant_work_root}"
  if [[ -e "${variant_work_root}/sglang-model" ]]; then
    echo "refusing pre-existing variant model path: ${variant_work_root}/sglang-model" >&2
    return 1
  fi
  ln -s "${BASE_MODEL_DIR}" "${variant_work_root}/sglang-model"

  export MINWM_RUN_ID="${variant_run_id}"
  export MINWM_S0_RESUME_PROFILER_OFF_ROOT="${source_root}"
  export MINWM_HOIST_TIMESTEP_MODULATION="${hoist}"
  # S0 owns creation and failure markers under its canonical result root. Do not
  # pre-create profile_dir: prepare_result_root treats any content as a stale run.
  CURRENT_LANE_DIR="${profile_dir}"
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${profile_dir}/measurement.json" \
    --require-complete-stable-nsys
  validate_profile_result "${variant}" "${profile_dir}/measurement.json"
  if pgrep -x nsys >/dev/null; then
    echo "${variant}: nsys remained alive after completed capture" >&2
    pgrep -a -x nsys >&2 || true
    return 1
  fi
  CURRENT_LANE_DIR=""
}

run_variant legacy 0 "${MINWM_S1_LEGACY_OFF_ROOT}"

NSYS_BIN="$(find "/work/minwm-realtime/${BASE_RUN_ID}-legacy/nsight-systems" \
  -type f -name nsys -perm -111 -print -quit 2>/dev/null || true)"
if [[ -n "${NSYS_BIN}" ]]; then
  export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
fi

run_variant candidate 1 "${MINWM_S1_CANDIDATE_OFF_ROOT}"

LEGACY_PROFILE="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}-legacy/s0-measurement/sp2/profiler-on"
CANDIDATE_PROFILE="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}-candidate/s0-measurement/sp2/profiler-on"
CURRENT_LANE_DIR="${RESULT_ROOT}/nsys-comparison"
mkdir -p "${CURRENT_LANE_DIR}"
python3 "${SCRIPT_DIR}/compare_temb_hoist_nsys.py" \
  --legacy "${LEGACY_PROFILE}/measurement.json" \
  --candidate "${CANDIDATE_PROFILE}/measurement.json" \
  --legacy-sqlite "${LEGACY_PROFILE}/sp2.sqlite" \
  --candidate-sqlite "${CANDIDATE_PROFILE}/sp2.sqlite" \
  --output "${RESULT_ROOT}/temb-hoist-nsys-comparison.json"

CURRENT_LANE_DIR=""
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/temb-hoist-nsys-complete.txt"
