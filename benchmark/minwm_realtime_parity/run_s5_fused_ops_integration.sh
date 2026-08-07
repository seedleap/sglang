#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${SGLANG_GIT_REF:?set SGLANG_GIT_REF}"
: "${MINWM_GIT_REF:?set MINWM_GIT_REF}"
: "${MINWM_CONTAINER_IMAGE:?set MINWM_CONTAINER_IMAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SGLANG_SOURCE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
SUMMARY_ROOT="${RUN_ROOT}/s5-summary"
POSITION_DRIFT_PCT="${MINWM_S5_POSITION_DRIFT_PCT:-3.0}"
CURRENT_LANE_DIR="${RUN_ROOT}/preflight"

export MINWM_ALLOCATED_GPU_COUNT=8
export MINWM_S0_KV_CACHE_NUM_FRAMES=45
export MINWM_S0_OFF_WARMUP_CHUNKS=20
export MINWM_S0_OFF_MEASURED_CHUNKS=200
export MINWM_S0_PROFILE_PRECONDITION_CHUNKS=20
export MINWM_S0_PROFILE_DISCARD_CHUNKS=1
export MINWM_S0_PROFILE_MEASURED_CHUNKS=10
unset SGLANG_DIFFUSION_TORCH_PROFILER_DIR

mkdir -p "${CURRENT_LANE_DIR}" "${SUMMARY_ROOT}"

mark_invalid() {
  local status="$1" root="$2" reason="$3" timestamp
  mkdir -p "${root}"
  if compgen -G "${root}/invalid-marker-*.json" >/dev/null; then
    return
  fi
  timestamp="$(date --utc +%Y%m%dT%H%M%S%NZ)"
  python3 "${SCRIPT_DIR}/measurement_tool.py" mark-invalid \
    --root "${root}" \
    --reason "${reason} (status=${status}); artifacts preserved in place" \
    --marker "${root}/invalid-marker-${timestamp}.json" || true
}

on_exit() {
  local status="$?"
  trap - EXIT INT TERM
  if (( status != 0 )); then
    mark_invalid "${status}" "${CURRENT_LANE_DIR}" "S5 runner exited non-zero"
  fi
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assert_no_nsys() {
  if pgrep -x nsys >/dev/null; then
    echo "profiler-off gate found an active nsys process" >&2
    pgrep -a -x nsys >&2 || true
    exit 2
  fi
}

apply_config() {
  local config="$1"
  if ! [[ "${config}" =~ ^[01]{3}$ ]]; then
    echo "invalid S5 config ${config}" >&2
    exit 2
  fi
  export MINWM_HOIST_TIMESTEP_MODULATION="${config:0:1}"
  export MINWM_FUSED_POST_A2A_ROPE_CACHE="${config:1:1}"
  export MINWM_FUSED_QKV_PROJECTION="${config:2:1}"
}

assert_execution_profile() {
  local log_path="$1" config="$2"
  python3 - "${log_path}" "${config}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = sys.argv[2]
text = path.read_text(errors="replace")
expected = (
    f"hoist_timestep_modulation={config[0] == '1'}",
    f"fused_post_a2a_rope_cache={config[1] == '1'}",
    f"fused_qkv_requested={config[2] == '1'}",
)
profile_lines = [line for line in text.splitlines() if "MinWM execution profile:" in line]
if not profile_lines:
    raise SystemExit(f"execution-profile line missing from {path}")
if not any(all(item in line for item in expected) for line in profile_lines):
    raise SystemExit(f"execution-profile mismatch for {config}: {profile_lines}")
qkv_mode = (
    "single-gemm-fast-lane" if config[2] == "1" else "three-gemm-parity"
)
if f"MinWM QKV projection mode: {qkv_mode}" not in text:
    raise SystemExit(f"QKV mode {qkv_mode!r} missing from {path}")
if config[2] == "1" and "compatible three-projection fallback" in text:
    raise SystemExit(f"unexpected fused-QKV fallback in {path}")
PY
}

python3 - "${SOURCE_ROOT}" "${SCRIPT_DIR}/run_s0_measurement.sh" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
source_python = str(source_root / "python")
if source_python not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    raise SystemExit(f"source tree missing from PYTHONPATH: {source_python}")
import sglang.test.ci.ci_register  # noqa: F401, E402

required_ancestors = (
    "d5b25227d4487d113e62c86a0fb572a62d6bcc5b",
    "c5d7af2269f8c622a6da2dedbe3407ca9a478427",
    "0e30671cf8a00622fd138c71af3faa93353b5425",
    "f1c9082bb12ee58d610e6e83bb4db192d9ccf96b",
)
for revision in required_ancestors:
    subprocess.run(
        ["git", "-C", str(source_root), "merge-base", "--is-ancestor", revision, "HEAD"],
        check=True,
    )
runner = Path(sys.argv[2]).read_text()
for token in (
    "MINWM_S0_PROFILER_OFF_ONLY",
    "MINWM_S0_NSYS_ONLY",
    "MINWM_S0_OFF_REPEAT_COUNT",
    "MINWM_S0_RUN_BITWISE",
    "assert_no_nsys_processes",
    "--require-complete-stable-nsys",
):
    if token not in runner:
        raise SystemExit(f"S0 runner guard missing: {token}")
print(f"S5 source-registration gate passed: {source_python}")
PY

if [[ "$(nvidia-smi -L | wc -l | xargs)" != "8" ]]; then
  echo "S5 requires an isolated 8-GPU H200 node" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader \
  | tee "${CURRENT_LANE_DIR}/gpu-inventory.txt"
git -C "${SOURCE_ROOT}" rev-parse HEAD | tee "${CURRENT_LANE_DIR}/runner-sha.txt"

python3 -m pytest -q \
  "${SOURCE_ROOT}/test/registered/jit/diffusion/test_minwm_ulysses_fused.py"
python3 -m pytest -q \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/test/unit/test_minwm_qkv_projection.py"
python3 -m pytest -q \
  "${SOURCE_ROOT}/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py" \
  -k 'timestep_modulation or fused_post_a2a'
date --utc +%Y-%m-%dT%H:%M:%SZ > "${CURRENT_LANE_DIR}/complete.txt"

run_headline_lane() {
  local degree="$1" position="$2" config="$3" run_bitwise="$4"
  local label="headline-sp${degree}-${position}-${config}"
  apply_config "${config}"
  export MINWM_S0_SP_DEGREES="${degree}"
  export MINWM_S0_RUN_LABEL="${label}"
  export MINWM_S0_PROFILER_OFF_ONLY=1
  export MINWM_S0_NSYS_ONLY=0
  export MINWM_S0_OFF_REPEAT_COUNT=1
  export MINWM_S0_RUN_BITWISE="${run_bitwise}"
  export MINWM_S0_BITWISE_RESULTS_ROOT="${RUN_ROOT}/correctness/sp${degree}/results"
  export MINWM_S0_PARITY_DUMP_ALL_BLOCKS=1
  if [[ "${run_bitwise}" == "1" ]]; then
    export MINWM_S0_PARITY_DUMP_DIR="${RUN_ROOT}/correctness/sp${degree}/dumps/${config}"
    if [[ "${config}" == "000" ]]; then
      export MINWM_S0_BITWISE_OUTPUT_PREFIX=baseline
    else
      export MINWM_S0_BITWISE_OUTPUT_PREFIX=sglang
    fi
  else
    unset MINWM_S0_PARITY_DUMP_DIR MINWM_S0_BITWISE_OUTPUT_PREFIX
  fi
  CURRENT_LANE_DIR="${RUN_ROOT}/${label}"
  assert_no_nsys
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  assert_no_nsys
  local lane="${RUN_ROOT}/${label}/sp${degree}"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${lane}/profiler-off-repeat1.json"
  assert_execution_profile "${lane}/profiler-off-server.log" "${config}"
  if [[ "${run_bitwise}" == "1" ]]; then
    assert_execution_profile "${lane}/correctness-server.log" "${config}"
  fi
  CURRENT_LANE_DIR="${RUN_ROOT}"
}

aggregate_variant() {
  local degree="$1" config="$2"
  local paths=(
    "${RUN_ROOT}/headline-sp${degree}-a1-${config}/sp${degree}/profiler-off-repeat1.json"
    "${RUN_ROOT}/headline-sp${degree}-a2-${config}/sp${degree}/profiler-off-repeat1.json"
  )
  local adaptive="${RUN_ROOT}/headline-sp${degree}-adaptive-${config}/sp${degree}/profiler-off-repeat1.json"
  local args=()
  if [[ -f "${adaptive}" ]]; then
    paths+=("${adaptive}")
    args=(--noise-explanation "Adaptive third independent-server lane collected after CV or position-drift trigger.")
  fi
  python3 "${SCRIPT_DIR}/measurement_tool.py" aggregate \
    "${paths[@]}" "${args[@]}" \
    --output "${SUMMARY_ROOT}/headline-sp${degree}-${config}.json"
}

write_headline_summary() {
  local degree="$1"
  python3 - "${RUN_ROOT}" "${SUMMARY_ROOT}" "${degree}" "${POSITION_DRIFT_PCT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_root = Path(sys.argv[2])
degree = int(sys.argv[3])
drift_limit = float(sys.argv[4])
metric_names = (
    "client_fps",
    "scheduler_fps",
    "scheduler_chunk_wall_ms",
    "dit_wall_ms",
    "vae_wall_ms",
)
configs = ("111", "000")
positions = {"111": ("a1", "a2"), "000": ("a1", "a2")}
records = {}
for config in configs:
    records[config] = []
    for position in positions[config]:
        path = root / f"headline-sp{degree}-{position}-{config}" / f"sp{degree}" / "profiler-off-repeat1.json"
        records[config].append(json.loads(path.read_text()))

aggregates = {
    config: json.loads((summary_root / f"headline-sp{degree}-{config}.json").read_text())
    for config in configs
}
triggers = {}
for config in configs:
    drift = {}
    for name in metric_names:
        values = aggregates[config]["metrics"][name]["values"][:2]
        mean = sum(values) / len(values)
        drift[name] = abs(values[1] - values[0]) / mean * 100.0 if mean else 0.0
    reasons = []
    if not aggregates[config]["acceptance"]["passes_cv_target"]:
        reasons.append("necessary headline CV gate failed")
    drifted = {name: value for name, value in drift.items() if value > drift_limit}
    if drifted:
        reasons.append(f"position-pair drift exceeded {drift_limit}%: {drifted}")
    if reasons and len(aggregates[config]["run_ids"]) == 2:
        triggers[config] = reasons

base = aggregates["000"]
candidate = aggregates["111"]
comparisons = {}
for name in metric_names:
    before = base["metrics"][name]["mean"]
    after = candidate["metrics"][name]["mean"]
    delta = (after / before - 1.0) * 100.0 if before else None
    comparisons[name] = {
        "000": before,
        "111": after,
        "111_minus_000_pct": delta,
        "improvement_pct": delta if name.endswith("fps") else -delta,
    }
summary = {
    "schema_version": "minwm-s5-fused-ops-headline/v1",
    "comparison_contract": {
        "sp_degree": degree,
        "allocated_gpu_count": 8,
        "kv_cache_num_frames": 45,
        "warmup_chunks": 20,
        "measured_chunks": 200,
        "order": ["111", "000", "000", "111"],
        "independent_server_per_position": True,
        "position_drift_trigger_abs_pct": drift_limit,
    },
    "aggregates": aggregates,
    "comparison": comparisons,
    "adaptive_triggers": triggers,
}
(summary_root / f"headline-sp{degree}.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
(summary_root / f"headline-sp{degree}-adaptive-trigger.json").write_text(
    json.dumps(triggers, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(triggers, sort_keys=True))
PY
}

compare_correctness() {
  local degree="$1"
  local root="${RUN_ROOT}/correctness/sp${degree}"
  python3 "${SCRIPT_DIR}/compare_results.py" \
    --cases "${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}" \
    --case "${MINWM_CASE_ID:-00_forward_080_pottery_720p}" \
    --results "${root}/results" \
    --profile bf16_backend_candidate
  python3 - "${root}" "${degree}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
degree = int(sys.argv[2])
case_id = "00_forward_080_pottery_720p"
case_root = root / "results" / "cases" / case_id
baseline = np.load(case_root / "baseline.npy", allow_pickle=False)
candidate = np.load(case_root / "sglang.npy", allow_pickle=False)
if not np.array_equal(baseline, candidate):
    raise AssertionError("000 and 111 final videos are not bitwise equal")

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

latent_files = []
for rank in range(degree):
    base_dir = root / "dumps" / "000" / "sglang" / f"sp_{degree:02d}_rank_{rank:02d}"
    candidate_dir = root / "dumps" / "111" / "sglang" / f"sp_{degree:02d}_rank_{rank:02d}"
    base_paths = sorted(base_dir.glob("chunk_*_latents.pt"))
    if len(base_paths) != 8:
        raise AssertionError(f"rank {rank}: expected 8 baseline latent chunks")
    for base_path in base_paths:
        candidate_path = candidate_dir / base_path.name
        if not candidate_path.exists():
            raise AssertionError(f"missing candidate latent {candidate_path}")
        base_tensor = torch.load(base_path, map_location="cpu", weights_only=True)
        candidate_tensor = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if not torch.equal(base_tensor, candidate_tensor):
            raise AssertionError(f"latent mismatch: rank={rank} file={base_path.name}")
        latent_files.append(
            {
                "rank": rank,
                "name": base_path.name,
                "shape": list(base_tensor.shape),
                "dtype": str(base_tensor.dtype),
                "bitwise_equal": True,
                "max_abs": 0.0,
                "000_sha256": file_sha(base_path),
                "111_sha256": file_sha(candidate_path),
            }
        )
summary = {
    "schema_version": "minwm-s5-fused-ops-correctness/v1",
    "sp_degree": degree,
    "precision": "bf16",
    "dmd_forwards_per_chunk": 4,
    "clean_cache_forwards_per_chunk": 1,
    "chunk_count": 8,
    "kv_cache_num_frames": 45,
    "final_video": {
        "shape": list(baseline.shape),
        "dtype": str(baseline.dtype),
        "bitwise_equal": True,
        "max_abs": 0.0,
        "000_sha256": file_sha(case_root / "baseline.npy"),
        "111_sha256": file_sha(case_root / "sglang.npy"),
    },
    "latent_files": latent_files,
    "fallback_policy": {
        "qkv": "server log must report single-gemm-fast-lane for 111 and no compatible fallback",
        "post_a2a": "Nsight config with middle bit 1 must contain fused_rope_cache_update",
    },
}
(root / "correctness-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary["final_video"], indent=2, sort_keys=True))
PY
}

for degree in 2 4; do
  run_headline_lane "${degree}" a1 111 1
  run_headline_lane "${degree}" a1 000 1
  run_headline_lane "${degree}" a2 000 0
  run_headline_lane "${degree}" a2 111 0
  CURRENT_LANE_DIR="${SUMMARY_ROOT}/headline-sp${degree}-analysis"
  mkdir -p "${CURRENT_LANE_DIR}"
  aggregate_variant "${degree}" 111
  aggregate_variant "${degree}" 000
  write_headline_summary "${degree}"
  mapfile -t adaptive_configs < <(
    python3 - "${SUMMARY_ROOT}/headline-sp${degree}-adaptive-trigger.json" <<'PY'
import json, sys
for config in json.load(open(sys.argv[1])):
    print(config)
PY
  )
  for config in "${adaptive_configs[@]}"; do
    run_headline_lane "${degree}" adaptive "${config}" 0
    aggregate_variant "${degree}" "${config}"
  done
  write_headline_summary "${degree}"
  CURRENT_LANE_DIR="${RUN_ROOT}/correctness/sp${degree}"
  compare_correctness "${degree}"
done

run_nsys_lane() {
  local degree="$1" config="$2"
  local label="nsys-sp${degree}-${config}"
  apply_config "${config}"
  export MINWM_S0_SP_DEGREES="${degree}"
  export MINWM_S0_RUN_LABEL="${label}"
  export MINWM_S0_PROFILER_OFF_ONLY=0
  export MINWM_S0_NSYS_ONLY=1
  export MINWM_S0_OFF_REPEAT_COUNT=2
  export MINWM_S0_RUN_BITWISE=0
  unset MINWM_S0_PARITY_DUMP_DIR MINWM_S0_BITWISE_OUTPUT_PREFIX
  CURRENT_LANE_DIR="${RUN_ROOT}/${label}"
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
  local lane="${RUN_ROOT}/${label}/sp${degree}/profiler-on"
  python3 "${SCRIPT_DIR}/measurement_tool.py" validate \
    "${lane}/measurement.json" --require-complete-stable-nsys
  assert_execution_profile "${lane}/server.log" "${config}"
  CURRENT_LANE_DIR="${RUN_ROOT}"
}

for degree in 2 4; do
  if [[ "${degree}" == "2" ]]; then
    primary_order=(000 100 010 001 111)
  else
    primary_order=(111 001 010 100 000)
  fi
  printf '%s\n' "${primary_order[@]}" > "${SUMMARY_ROOT}/nsys-sp${degree}-order.txt"
  for config in "${primary_order[@]}"; do
    run_nsys_lane "${degree}" "${config}"
  done
  CURRENT_LANE_DIR="${SUMMARY_ROOT}/nsys-sp${degree}-analysis"
  mkdir -p "${CURRENT_LANE_DIR}"
  python3 "${SCRIPT_DIR}/compare_s5_fused_ops.py" \
    --root "${RUN_ROOT}" \
    --degree "${degree}" \
    --output "${SUMMARY_ROOT}/nsys-sp${degree}.json"
  pairwise_required="$(python3 - "${SUMMARY_ROOT}/nsys-sp${degree}.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["pairwise_required"]))
PY
)"
  if [[ "${pairwise_required}" == "1" ]]; then
    for config in 110 101 011; do
      run_nsys_lane "${degree}" "${config}"
    done
    CURRENT_LANE_DIR="${SUMMARY_ROOT}/nsys-sp${degree}-analysis"
    python3 "${SCRIPT_DIR}/compare_s5_fused_ops.py" \
      --root "${RUN_ROOT}" \
      --degree "${degree}" \
      --output "${SUMMARY_ROOT}/nsys-sp${degree}.json"
  fi
done

CURRENT_LANE_DIR="${SUMMARY_ROOT}"
python3 - "${RUN_ROOT}" "${SUMMARY_ROOT}/artifact-manifest.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.resolve() == output:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    files.append(
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "recoverable": True,
        }
    )
record = {
    "schema_version": "minwm-s5-artifact-manifest/v1",
    "root": str(root),
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "recoverability": "preserved_on_task_scoped_pvc",
    "files": files,
    "total_size_bytes": sum(item["size_bytes"] for item in files),
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps({"file_count": len(files), "total_size_bytes": record["total_size_bytes"]}))
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${SUMMARY_ROOT}/complete.txt"
echo "MINWM_S5_FUSED_OPS_COMPLETE results=${RUN_ROOT}"
