#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO" >&2
  echo "restores the complete video as resumable 120-second sparse SBS segments" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="$1"
OUTPUT_PATH="$2"

[[ -f "$INPUT_PATH" ]] || {
  echo "error: input video not found: $INPUT_PATH" >&2
  exit 1
}

export JASNA_TEST_SECONDS=full
export JASNA_SEGMENT_SECONDS="${JASNA_SEGMENT_SECONDS:-120}"
export JASNA_MODEL_BATCH="${JASNA_MODEL_BATCH:-1}"
export JASNA_DIRECT_SBS_OUTPUT="${JASNA_DIRECT_SBS_OUTPUT:-1}"
export JASNA_VR_PROJECTION="${JASNA_VR_PROJECTION:-fisheye}"

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" \
  "$INPUT_PATH" "$OUTPUT_PATH" 0
