#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
BITWISE_CASE="00_forward_080_pottery_720p"
BITWISE_ROOT="${RESULT_ROOT}/temb-hoist-bitwise/cases/${BITWISE_CASE}"

export MINWM_S0_KV_CACHE_NUM_FRAMES=45
export MINWM_S0_RUN_BITWISE=1

python3 -m pytest -q \
  "${SCRIPT_DIR}/../../python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py" \
  -k 'hoisted_timestep_modulation_matches_compiled_cuda_index'

for lane in legacy candidate; do
  if [[ "${lane}" == "legacy" ]]; then
    export MINWM_HOIST_TIMESTEP_MODULATION=0
  else
    export MINWM_HOIST_TIMESTEP_MODULATION=1
  fi
  export MINWM_S0_RUN_LABEL="temb-hoist-${lane}"
  bash "${SCRIPT_DIR}/run_s0_measurement.sh"
done

python3 - "${BITWISE_ROOT}" "${RESULT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

bitwise_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
paths = {
    lane: bitwise_root / f"temb-hoist-{lane}.npy"
    for lane in ("legacy", "candidate")
}

def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()

legacy = np.load(paths["legacy"], mmap_mode="r")
candidate = np.load(paths["candidate"], mmap_mode="r")
equal = legacy.shape == candidate.shape
max_abs_diff = 0
if equal:
    for frame_index in range(legacy.shape[0]):
        lhs = legacy[frame_index]
        rhs = candidate[frame_index]
        if not np.array_equal(lhs, rhs):
            equal = False
            max_abs_diff = max(
                max_abs_diff,
                int(np.abs(lhs.astype(np.int16) - rhs.astype(np.int16)).max()),
            )

record = {
    "schema_version": "minwm-temb-hoist-bitwise/v1",
    "case": bitwise_root.name,
    "kv_cache_num_frames": 45,
    "legacy": {"path": str(paths["legacy"]), "sha256": digest(paths["legacy"])},
    "candidate": {
        "path": str(paths["candidate"]),
        "sha256": digest(paths["candidate"]),
    },
    "shape": list(legacy.shape),
    "array_equal": equal,
    "max_abs_diff_uint8": max_abs_diff,
}
output = result_root / "temb-hoist-bitwise-summary.json"
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
if not equal:
    raise SystemExit("legacy/candidate frame arrays are not bitwise equal")
PY

date --utc +%Y-%m-%dT%H:%M:%SZ | tee "${RESULT_ROOT}/temb-hoist-ab-complete.txt"
