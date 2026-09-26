#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_VENV="$ROOT_DIR/.venv-rfdetr-ane"
BASE_SITE_PACKAGES="$ROOT_DIR/.venv-coreai/lib/python3.13/site-packages"
CHECKPOINT="$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt"
PACKAGE="${JASNA_RFDETR_ANE_PACKAGE:-$ROOT_DIR/.test-results/rfdetr-ane/rfdetr-vr-large-768-batch2-float16.mlpackage}"
COMPUTE_UNITS="${JASNA_RFDETR_COREML_COMPUTE:-cpu-ne}"

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: $0 SBS_VIDEO [FRAME_INDEX]" >&2
  exit 2
}
SBS_VIDEO="$1"
FRAME_INDEX="${2:-0}"

echo "Jasna experimental RF-DETR ANE parity test"
echo "Video:   $SBS_VIDEO"
echo "Frame:   $FRAME_INDEX"
echo "Package: $PACKAGE"
echo "Policy:  $COMPUTE_UNITS"

PYTHONPATH="$BASE_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
  "$EXPERIMENT_VENV/bin/python" "$ROOT_DIR/tools/compare_rfdetr_coreml.py" \
  "$CHECKPOINT" "$PACKAGE" "$SBS_VIDEO" --frame "$FRAME_INDEX" \
  --compute-units "$COMPUTE_UNITS"
