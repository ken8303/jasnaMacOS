#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO left|right OUTPUT_EYE_VIDEO" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -x "$ROOT_DIR/.venv-mosaic/bin/python" ]] || {
  echo "error: first run $ROOT_DIR/script/setup_mosaic_detector.sh" >&2
  exit 1
}

JASNA_SPARSE_MOSAIC=1 \
JASNA_VR_PROJECTION="${JASNA_VR_PROJECTION:-fisheye}" \
  "$ROOT_DIR/script/restore_vr_eye_segments.sh" "$@"
