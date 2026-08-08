#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: $0 INPUT_30FPS_EYE_VIDEO OUTPUT_MANIFEST.json" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="$ROOT_DIR/.venv-mosaic/bin/python"
MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/lada_vr_mosaic_detection_model_v2_fast.pt"
DETECT_BATCH_SIZE="${JASNA_DETECT_BATCH_SIZE:-2}"
DETECT_DECODE_MODE="${JASNA_DETECT_DECODE_MODE:-sequential}"
REGION_DURATION="${JASNA_REGION_DURATION:-1.0}"

[[ "$DETECT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_DETECT_BATCH_SIZE must be a positive integer" >&2
  exit 1
}
[[ "$DETECT_DECODE_MODE" == "sequential" || "$DETECT_DECODE_MODE" == "seek" ]] || {
  echo "error: JASNA_DETECT_DECODE_MODE must be sequential or seek" >&2
  exit 1
}

[[ -x "$PYTHON_PATH" && -s "$MODEL_PATH" ]] || {
  echo "error: mosaic detector is not set up" >&2
  echo "run: $ROOT_DIR/script/setup_mosaic_detector.sh" >&2
  exit 1
}

"$PYTHON_PATH" "$ROOT_DIR/tools/scan_mosaic_regions.py" \
  "$1" "$2" --model "$MODEL_PATH" \
  --batch-size "$DETECT_BATCH_SIZE" \
  --decode-mode "$DETECT_DECODE_MODE" \
  --region-duration "$REGION_DURATION"
