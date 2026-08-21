#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "runs a 30-second isolated eye-by-eye performance and quality test" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JASNA_TEST_SECONDS=30
export JASNA_DIRECT_SBS_OUTPUT=0
export JASNA_EYE_JOB_PROCESS_ISOLATION=1
export JASNA_EYE_PAIR_SEGMENTS=1
export JASNA_SEGMENT_SECONDS=30
export JASNA_MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
export JASNA_DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
export JASNA_ADAPTIVE_DETECT=0
export JASNA_FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
export JASNA_LOG_PEAK_MEMORY=1

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" "$1" "$2" "${3:-0}"
