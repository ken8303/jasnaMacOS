#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 FINETUNE_DATASET_DIRECTORY" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="$ROOT_DIR/.venv-rfdetr/bin/python"
MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt"

[[ -x "$PYTHON_PATH" ]] || {
  echo "error: RF-DETR environment is unavailable: $PYTHON_PATH" >&2
  exit 1
}
[[ -f "$MODEL_PATH" ]] || {
  echo "error: RF-DETR VR checkpoint is unavailable: $MODEL_PATH" >&2
  exit 1
}

QUARANTINE_ARG=""
case "${JASNA_FINETUNE_QUARANTINE:-0}" in
  1|true|TRUE|yes|YES) QUARANTINE_ARG="--quarantine" ;;
esac

exec "$PYTHON_PATH" "$ROOT_DIR/tools/audit_finetune_dataset.py" \
  --dataset "$1" \
  --model "$MODEL_PATH" \
  --threshold "${JASNA_FINETUNE_AUDIT_CONFIDENCE:-0.10}" \
  --frame-stride "${JASNA_FINETUNE_AUDIT_STRIDE:-5}" \
  --batch-size "${JASNA_DETECT_BATCH_SIZE:-2}" \
  --minimum-skin-fraction "${JASNA_FINETUNE_MINIMUM_SKIN_FRACTION:-0.08}" \
  --minimum-luma "${JASNA_FINETUNE_MINIMUM_LUMA:-0.03}" \
  --minimum-peak-detail "${JASNA_FINETUNE_MINIMUM_PEAK_DETAIL:-80}" \
  --device "${JASNA_DETECT_DEVICE:-auto}" \
  ${QUARANTINE_ARG}
