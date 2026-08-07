#!/usr/bin/env bash
set -Eeuo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${MINWM_S3_OFF_ROOT:?set MINWM_S3_OFF_ROOT}"
: "${MINWM_S3_OFF_RUNTIME_REF:?set MINWM_S3_OFF_RUNTIME_REF}"
: "${MINWM_S3_PRODUCT_GIT_REF:?set MINWM_S3_PRODUCT_GIT_REF}"
: "${MINWM_S3_RUNNER_REF:?set MINWM_S3_RUNNER_REF}"
: "${MINWM_S0_TOOL_REF:?set MINWM_S0_TOOL_REF}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SGLANG_SOURCE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASE_RUN_ID="${MINWM_RUN_ID}"
BASE_WORK_ROOT="/work/minwm-realtime/${BASE_RUN_ID}"
BASE_MODEL_DIR="${BASE_WORK_ROOT}/sglang-model"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}/s3-post-nsys"
CURRENT_LANE_DIR="${RESULT_ROOT}/nsys-preflight"
mkdir -p "${CURRENT_LANE_DIR}"

export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
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
    --reason "S3 post-A2A Nsight wrapper exited non-zero (status=${status})" \
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
        "--gpu-metrics-devices=all",
        "--require-complete-stable-nsys",
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
print(f"S3 Nsight source-registration/canonical preflight passed: {source_python}")
PY

[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SGLANG_GIT_REF}" ]]
[[ "${SGLANG_GIT_REF}" == "${MINWM_S3_RUNNER_REF}" ]]
git -C "${SOURCE_ROOT}" merge-base --is-ancestor \
  "${MINWM_S0_TOOL_REF}" "${MINWM_S3_RUNNER_REF}"
if pgrep -x nsys >/dev/null; then
  echo "S3 Nsight wrapper refuses a pre-existing nsys process" >&2
  pgrep -a -x nsys >&2 || true
  exit 2
fi
if [[ ! -f "${BASE_MODEL_DIR}/minwm_conversion_manifest.json" ]]; then
  echo "setup model is missing: ${BASE_MODEL_DIR}" >&2
  exit 2
fi

validate_off_source() {
  local degree="$1" lane="$2" enabled="$3"
  local source="${MINWM_S3_OFF_ROOT%/}/sp${degree}/lane${lane}"
  for repeat in 1 2; do
    python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
      "${source}/profiler-off-repeat${repeat}.json"
  done
  python3 - \
    "${source}" "${degree}" "${lane}" "${enabled}" \
    "${MINWM_S3_OFF_RUNTIME_REF}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
degree = int(sys.argv[2])
lane = sys.argv[3]
enabled = sys.argv[4]
off_runtime_ref = sys.argv[5]
contract = (source / "contract.txt").read_text()
required_contract = (
    f"sglang_commit={off_runtime_ref}",
    f"sp_degree={degree}",
    f"lane={lane}",
    f"fused_post_a2a_rope_cache={enabled}",
    "off_window=20+200",
    "kv_cache_num_frames=45",
)
for text in required_contract:
    if text not in contract:
        raise SystemExit(f"{source}: missing profiler-off contract: {text}")
for repeat in (1, 2):
    record = json.loads((source / f"profiler-off-repeat{repeat}.json").read_text())
    assert record["mode"] == "profiler_off"
    assert record["provenance"]["sglang_commit"] == off_runtime_ref
    assert record["provenance"]["gpu"]["count"] == degree
    assert record["provenance"]["gpu"]["allocated_count"] == 8
    assert record["workload"]["sp_degree"] == degree
    assert record["workload"]["precision"] == "bf16"
    assert record["workload"]["warmup_chunks"] == 20
    assert record["workload"]["measured_chunks"] == 200
    assert record["comparison_contract"]["kv_cache_num_frames"] == 45
    for name in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"):
        metric = record["metrics"]["profiler_off"][name]
        assert metric["status"] == "available"
        assert metric["value"]["count"] == 200
summary = json.loads((source / "repeat-summary.json").read_text())
assert summary["acceptance"]["passes_cv_target"] is True
print(f"validated profiler-off resume source: {source}")
PY
}

for degree in 2 4; do
  validate_off_source "${degree}" 00 0
  validate_off_source "${degree}" 01 1
done

{
  echo "sglang_runner_commit=${SGLANG_GIT_REF}"
  echo "s3_product_commit=${MINWM_S3_PRODUCT_GIT_REF}"
  echo "s3_off_runtime_commit=${MINWM_S3_OFF_RUNTIME_REF}"
  echo "s0_tool_commit=${MINWM_S0_TOOL_REF}"
  echo "sp_degrees=2 4"
  echo "lanes=00 01"
  echo "nsys_window=20 precondition + 1 discarded + 10 stable"
  echo "kv_cache_num_frames=45"
  echo "gpu_metrics_devices=all"
  echo "torch_profiler_concurrent=false"
  echo "profiler_off_root=${MINWM_S3_OFF_ROOT}"
  echo "started_utc=$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${RESULT_ROOT}/contract.txt"

CURRENT_LANE_DIR="${RESULT_ROOT}/cuda-correctness"
mkdir -p "${CURRENT_LANE_DIR}"
python3 -m pytest -q \
  "${SOURCE_ROOT}/test/registered/jit/diffusion/test_minwm_ulysses_fused.py" \
  | tee "${CURRENT_LANE_DIR}/post-a2a-bitwise.log"
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${CURRENT_LANE_DIR}/COMPLETE"

prepare_resume_root() {
  local degree="$1" lane="$2"
  local source="${MINWM_S3_OFF_ROOT%/}/sp${degree}/lane${lane}"
  local adapter="${BASE_WORK_ROOT}/resume-sp${degree}-lane${lane}"
  local adapter_lane="${adapter}/sp${degree}"
  if [[ -e "${adapter}" ]]; then
    echo "refusing pre-existing profiler-off adapter: ${adapter}" >&2
    return 1
  fi
  mkdir -p "${adapter_lane}"
  cp -- "${source}"/profiler-off-repeat*.json "${adapter_lane}/"
  cp -- "${source}/repeat-summary.json" "${adapter_lane}/"
  printf '%s\n' "${source}" > "${adapter_lane}/original-source.txt"
  printf '%s\n' "${adapter}"
}

validate_stage_trace() {
  local measurement="$1" degree="$2" lane="$3"
  python3 - "${measurement}" "${degree}" "${lane}" <<'PY'
import json
import sys

path = sys.argv[1]
degree = int(sys.argv[2])
lane = sys.argv[3]
record = json.load(open(path))
assert record["mode"] == "profiler_on"
workload = record["workload"]
assert workload["sp_degree"] == degree
assert workload["precondition_warmup_chunks"] == 20
assert workload["warmup_chunks"] == 1
assert workload["measured_chunks"] == 10
assert record["comparison_contract"]["kv_cache_num_frames"] == 45
on = record["metrics"]["profiler_on"]
for name in ("scheduler_chunk_wall_ms", "dit_wall_ms", "vae_wall_ms"):
    metric = on["observed_wall_with_profiler_overhead"][name]
    assert metric["status"] == "available", (name, metric)
    assert metric["value"]["count"] == 10, (name, metric)
for name in ("dit_cuda_ms", "vae_cuda_ms"):
    metric = on[name]
    assert metric["status"] == "available", (name, metric)
    assert metric["value"]["count"] == 10, (name, metric)
stable = on["stable_window_coverage"]
assert stable["status"] == "available"
assert stable["value"]["expected_stable_chunk_indices"] == list(range(1, 11))
assert stable["value"]["observed_stable_chunk_indices"] == list(range(1, 11))
assert stable["value"]["normalization_denominator"] == 10
for name in ("sm_active", "tensor_active"):
    metric = on["gpu_metrics"][name]
    assert metric["status"] == "available", (name, metric)
    value = metric["value"]
    assert value["collected_target_count"] == 8
    assert value["allocated_target_count"] == 8
    assert value["active_target_count"] == degree
    assert value["active_cuda_device_ids"] == list(range(degree))
    assert len(value["active_pw_gpu_ids"]) == degree
dram = on["gpu_metrics"]["dram"]
assert dram["status"] == "available" or (
    dram["status"] == "unavailable" and dram["reason"] == "metric_not_exposed"
), dram
print(f"stage-trace-complete sp={degree} lane={lane}: {path}")
PY
}

run_variant() {
  local degree="$1" lane="$2" label="$3" enabled="$4"
  local variant_run_id="${BASE_RUN_ID}-sp${degree}-${label}"
  local variant_work_root="/work/minwm-realtime/${variant_run_id}"
  local variant_result_root="${MINWM_RESULTS_ROOT%/}/${variant_run_id}/s0-measurement"
  local profile_dir="${variant_result_root}/sp${degree}/profiler-on"
  local source="${MINWM_S3_OFF_ROOT%/}/sp${degree}/lane${lane}"
  local adapter
  adapter="$(prepare_resume_root "${degree}" "${lane}")"

  CURRENT_LANE_DIR="${profile_dir}"
  mkdir -p "${variant_work_root}"
  if [[ -e "${variant_work_root}/sglang-model" ]]; then
    echo "refusing pre-existing variant model path: ${variant_work_root}/sglang-model" >&2
    return 1
  fi
  ln -s "${BASE_MODEL_DIR}" "${variant_work_root}/sglang-model"

  export MINWM_RUN_ID="${variant_run_id}"
  export MINWM_S0_SP_DEGREES="${degree}"
  export MINWM_S0_RESUME_PROFILER_OFF_ROOT="${adapter}"
  export MINWM_FUSED_POST_A2A_ROPE_CACHE="${enabled}"
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${profile_dir}/measurement.json" \
    --require-complete-stable-nsys
  validate_stage_trace "${profile_dir}/measurement.json" "${degree}" "${lane}"
  {
    echo "sp_degree=${degree}"
    echo "lane=${lane}"
    echo "label=${label}"
    echo "fused_post_a2a_rope_cache=${enabled}"
    echo "original_profiler_off_source=${source}"
    echo "stage_trace_counts=10"
  } > "${profile_dir}/s3-lane-contract.txt"
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${profile_dir}/S3_LANE_COMPLETE"
  if pgrep -x nsys >/dev/null; then
    echo "SP${degree} ${label}: nsys remained alive after completed capture" >&2
    pgrep -a -x nsys >&2 || true
    return 1
  fi
  CURRENT_LANE_DIR=""
}

compare_degree() {
  local degree="$1"
  local baseline_root="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}-sp${degree}-baseline/s0-measurement/sp${degree}"
  local candidate_root="${MINWM_RESULTS_ROOT%/}/${BASE_RUN_ID}-sp${degree}-candidate/s0-measurement/sp${degree}"
  local comparison_dir="${RESULT_ROOT}/comparisons/sp${degree}"
  CURRENT_LANE_DIR="${comparison_dir}"
  mkdir -p "${comparison_dir}"
  python3 "${SCRIPT_DIR}/compare_s3_post_nsys.py" \
    --degree "${degree}" \
    --baseline "${baseline_root}/profiler-on/measurement.json" \
    --candidate "${candidate_root}/profiler-on/measurement.json" \
    --baseline-sqlite "${baseline_root}/profiler-on/sp${degree}.sqlite" \
    --candidate-sqlite "${candidate_root}/profiler-on/sp${degree}.sqlite" \
    --baseline-off-summary "${baseline_root}/repeat-summary.json" \
    --candidate-off-summary "${candidate_root}/repeat-summary.json" \
    --output "${comparison_dir}/comparison.json" \
    | tee "${comparison_dir}/comparison.log"
  date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${comparison_dir}/COMPLETE"
  CURRENT_LANE_DIR=""
}

for degree in 2 4; do
  run_variant "${degree}" 00 baseline 0
  NSYS_BIN="$(find "/work/minwm-realtime/${BASE_RUN_ID}-sp${degree}-baseline/nsight-systems" \
    -type f -name nsys -perm -111 -print -quit 2>/dev/null || true)"
  if [[ -n "${NSYS_BIN}" ]]; then
    export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
  fi
  run_variant "${degree}" 01 candidate 1
  compare_degree "${degree}"
done

CURRENT_LANE_DIR="${RESULT_ROOT}"
python3 - "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "schema_version": "minwm-s3-post-nsys-matrix/v1",
    "comparisons": {
        f"sp{degree}": json.loads(
            (root / "comparisons" / f"sp{degree}" / "comparison.json").read_text()
        )
        for degree in (2, 4)
    },
}
(root / "matrix-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({key: "complete" for key in summary["comparisons"]}, indent=2))
PY

CURRENT_LANE_DIR=""
date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/S3_POST_NSYS_COMPLETE"
