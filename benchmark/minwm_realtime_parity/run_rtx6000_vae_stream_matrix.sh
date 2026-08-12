#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_MATRIX_ID:?set MINWM_MATRIX_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MATRIX_ID="${MINWM_MATRIX_ID}"

run_resolution() {
  local label="$1" cases="$2" case_id="$3"
  MINWM_MATRIX_ID="${BASE_MATRIX_ID}-${label}" \
  MINWM_THROUGHPUT_CASES_PATH="${SCRIPT_DIR}/${cases}" \
  MINWM_THROUGHPUT_CASE="${case_id}" \
    bash "${SCRIPT_DIR}/run_unified_exact_vae_spot_matrix.sh"
}

run_resolution 480p cases_480p_compile_smoke.json 00_forward_080_pottery_480p
run_resolution 704p cases_720p_compile_smoke.json 00_forward_080_pottery_720p
