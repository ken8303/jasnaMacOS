#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [MOSAIC_RANGES]" >&2
  echo "restores one 120-second 4K eye segment per Metal process, then rebuilds SBS" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JASNA_DIRECT_SBS_OUTPUT=0
export JASNA_EYE_JOB_PROCESS_ISOLATION=1
export JASNA_EYE_PAIR_SEGMENTS=1
export JASNA_SEGMENT_SECONDS=120
export JASNA_MODEL_BATCH="${JASNA_MODEL_BATCH:-1}"
export JASNA_DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
export JASNA_ADAPTIVE_DETECT=0
export JASNA_FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
export JASNA_LOG_PEAK_MEMORY=1

echo "Jasna isolated eye-by-eye profile"
echo "Restoration: one 120-second 4096-pixel eye segment per Metal process"
echo "Output: validated eye segments -> two-minute 8K SBS segments -> final video"

if [[ $# -eq 3 ]]; then
  export JASNA_MOSAIC_RANGES="$3"
fi

exec "$ROOT_DIR/script/restore_vr_sparse_sbs.sh" "$1" "$2"
