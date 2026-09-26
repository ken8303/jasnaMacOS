#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_VENV="$ROOT_DIR/.venv-rfdetr-ane"
BASE_SITE_PACKAGES="$ROOT_DIR/.venv-coreai/lib/python3.13/site-packages"
CHECKPOINT="${1:-$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt}"
OUTPUT_DIRECTORY="${2:-$ROOT_DIR/.test-results/rfdetr-ane}"
BATCH_SIZE="${JASNA_RFDETR_ANE_BATCH:-2}"

[[ -x "$EXPERIMENT_VENV/bin/python" ]] || {
  echo "error: isolated RF-DETR ANE environment is missing: $EXPERIMENT_VENV" >&2
  exit 1
}
[[ -d "$BASE_SITE_PACKAGES" ]] || {
  echo "error: isolated Torch base environment is missing: $BASE_SITE_PACKAGES" >&2
  exit 1
}
[[ -f "$CHECKPOINT" ]] || {
  echo "error: RF-DETR checkpoint not found: $CHECKPOINT" >&2
  exit 1
}
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_RFDETR_ANE_BATCH must be a positive integer" >&2
  exit 1
}

echo "Jasna experimental RF-DETR Core ML export"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIRECTORY"
echo "Batch:      $BATCH_SIZE"
echo "Compute:    CPU + Neural Engine (GPU excluded during inference test)"

PYTHONPATH="$BASE_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
  "$EXPERIMENT_VENV/bin/python" "$ROOT_DIR/tools/export_rfdetr_coreml.py" \
  "$CHECKPOINT" "$OUTPUT_DIRECTORY" \
  --batch-size "$BATCH_SIZE" --resolution 768 --variant large --precision float16
