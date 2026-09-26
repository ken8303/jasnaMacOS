#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 || $# -eq 3 ]] || {
  echo "usage: $0 INPUT_30FPS_VIDEO OUTPUT_MANIFEST.json [RIGHT_MANIFEST.json]" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
case "$DETECTOR" in
  yolo-v2-fast)
    PYTHON_PATH="$ROOT_DIR/.venv-mosaic/bin/python"
    MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/lada_vr_mosaic_detection_model_v2_fast.pt"
    DETECTOR_BACKEND="yolo"
    RFDETR_VARIANT="large"
    DEFAULT_DETECT_CONFIDENCE="0.15"
    ;;
  rfdetr-v6)
    PYTHON_PATH="$ROOT_DIR/.venv-rfdetr/bin/python"
    MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/rfdetr-v6.pt"
    DETECTOR_BACKEND="rfdetr"
    RFDETR_VARIANT="medium"
    DEFAULT_DETECT_CONFIDENCE="0.35"
    ;;
  rfdetr-vr-v1)
    PYTHON_PATH="$ROOT_DIR/.venv-rfdetr/bin/python"
    MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt"
    DETECTOR_BACKEND="rfdetr"
    RFDETR_VARIANT="large"
    DEFAULT_DETECT_CONFIDENCE="0.15"
    ;;
  *)
    echo "error: JASNA_DETECTOR must be yolo-v2-fast, rfdetr-v6, or rfdetr-vr-v1" >&2
    exit 1
    ;;
esac
DETECT_BATCH_SIZE="${JASNA_DETECT_BATCH_SIZE:-1}"
DETECT_DEVICE="${JASNA_DETECT_DEVICE:-auto}"
DETECT_DECODE_MODE="${JASNA_DETECT_DECODE_MODE:-sequential}"
DETECT_SAMPLE_STRIDE="${JASNA_DETECT_SAMPLE_STRIDE:-0.1}"
STEREO_SAMPLE_MODE="${JASNA_STEREO_SAMPLE_MODE:-paired}"
DETECT_COARSE_STRIDE="${JASNA_DETECT_COARSE_STRIDE:-1.0}"
DETECT_COARSE_CONFIDENCE="${JASNA_DETECT_COARSE_CONFIDENCE:-0.05}"
DETECT_REFINE_PADDING="${JASNA_DETECT_REFINE_PADDING:-1.0}"
ADAPTIVE_DETECT="${JASNA_ADAPTIVE_DETECT:-0}"
REGION_DURATION="${JASNA_REGION_DURATION:-1.0}"
DETECT_CONFIDENCE="${JASNA_DETECT_CONFIDENCE:-$DEFAULT_DETECT_CONFIDENCE}"
TEMPORAL_PADDING="${JASNA_TEMPORAL_PADDING:-1.0}"
REGION_NMS_IOU="${JASNA_REGION_NMS_IOU:-0.45}"
MASK_EXPANSION="${JASNA_MASK_EXPANSION:-0.10}"
MASK_SIZE="${JASNA_MASK_SIZE:-128}"
RFDETR_MAX_DETECTIONS="${JASNA_RFDETR_MAX_DETECTIONS:-64}"
if [[ $# -eq 3 ]]; then
  CROP_EYE=both
else
  CROP_EYE="${JASNA_DETECT_EYE:-none}"
fi

[[ "$DETECT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_DETECT_BATCH_SIZE must be a positive integer" >&2
  exit 1
}
[[ "$DETECT_DEVICE" == "auto" || "$DETECT_DEVICE" == "mps" \
  || "$DETECT_DEVICE" == "cpu" || "$DETECT_DEVICE" == "mlx" ]] || {
  echo "error: JASNA_DETECT_DEVICE must be auto, mps, cpu, or mlx" >&2
  exit 1
}
[[ "$DETECT_DECODE_MODE" == "sequential" || "$DETECT_DECODE_MODE" == "seek" ]] || {
  echo "error: JASNA_DETECT_DECODE_MODE must be sequential or seek" >&2
  exit 1
}
[[ "$ADAPTIVE_DETECT" == "0" || "$ADAPTIVE_DETECT" == "1" ]] || {
  echo "error: JASNA_ADAPTIVE_DETECT must be 0 or 1" >&2
  exit 1
}
[[ "$STEREO_SAMPLE_MODE" == "paired" || "$STEREO_SAMPLE_MODE" == "alternating" ]] || {
  echo "error: JASNA_STEREO_SAMPLE_MODE must be paired or alternating" >&2
  exit 1
}
if [[ "$STEREO_SAMPLE_MODE" == "alternating" && "$CROP_EYE" != "both" ]]; then
  echo "error: JASNA_STEREO_SAMPLE_MODE=alternating requires shared stereo detection" >&2
  exit 1
fi
if [[ "$STEREO_SAMPLE_MODE" == "alternating" && "$ADAPTIVE_DETECT" == "1" ]]; then
  echo "error: alternating stereo sampling cannot be combined with adaptive detection" >&2
  exit 1
fi
for setting in \
  "JASNA_DETECT_SAMPLE_STRIDE:$DETECT_SAMPLE_STRIDE" \
  "JASNA_DETECT_COARSE_STRIDE:$DETECT_COARSE_STRIDE"; do
  name="${setting%%:*}"
  value="${setting#*:}"
  /usr/bin/awk -v value="$value" \
    'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' || {
    echo "error: $name must be greater than zero" >&2
    exit 1
  }
done
/usr/bin/awk -v value="$DETECT_COARSE_CONFIDENCE" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 1) }' || {
  echo "error: JASNA_DETECT_COARSE_CONFIDENCE must be greater than 0 and at most 1" >&2
  exit 1
}
/usr/bin/awk -v value="$DETECT_REFINE_PADDING" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0) }' || {
  echo "error: JASNA_DETECT_REFINE_PADDING must be zero or greater" >&2
  exit 1
}
/usr/bin/awk -v value="$DETECT_CONFIDENCE" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 1) }' || {
  echo "error: JASNA_DETECT_CONFIDENCE must be greater than 0 and at most 1" >&2
  exit 1
}
/usr/bin/awk -v value="$TEMPORAL_PADDING" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0) }' || {
  echo "error: JASNA_TEMPORAL_PADDING must be zero or greater" >&2
  exit 1
}
/usr/bin/awk -v value="$REGION_NMS_IOU" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 1) }' || {
  echo "error: JASNA_REGION_NMS_IOU must be greater than 0 and at most 1" >&2
  exit 1
}
/usr/bin/awk -v value="$MASK_EXPANSION" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 0.25) }' || {
  echo "error: JASNA_MASK_EXPANSION must be greater than 0 and at most 0.25" >&2
  exit 1
}
[[ "$MASK_SIZE" =~ ^[0-9]+$ ]] \
  && (( MASK_SIZE >= 32 && MASK_SIZE <= 256 )) \
  && (( (MASK_SIZE & (MASK_SIZE - 1)) == 0 )) || {
  echo "error: JASNA_MASK_SIZE must be a power of two from 32 through 256" >&2
  exit 1
}
[[ "$RFDETR_MAX_DETECTIONS" =~ ^[0-9]+$ ]] \
  && (( RFDETR_MAX_DETECTIONS >= 1 && RFDETR_MAX_DETECTIONS <= 200 )) || {
  echo "error: JASNA_RFDETR_MAX_DETECTIONS must be an integer from 1 through 200" >&2
  exit 1
}
[[ "$CROP_EYE" == "none" || "$CROP_EYE" == "left" \
  || "$CROP_EYE" == "right" || "$CROP_EYE" == "both" ]] || {
  echo "error: JASNA_DETECT_EYE must be none, left, right, or both" >&2
  exit 1
}

[[ -x "$PYTHON_PATH" && -s "$MODEL_PATH" ]] || {
  echo "error: $DETECTOR mosaic detector is not set up" >&2
  if [[ "$DETECTOR" == "yolo-v2-fast" ]]; then
    echo "run: $ROOT_DIR/script/setup_mosaic_detector.sh" >&2
  else
    echo "expected environment: $ROOT_DIR/.venv-rfdetr" >&2
    echo "expected model: $MODEL_PATH" >&2
  fi
  exit 1
}

export PYTORCH_ENABLE_MPS_FALLBACK=1
ADAPTIVE_ARGUMENTS=(
  --coarse-stride "$DETECT_COARSE_STRIDE"
  --coarse-confidence "$DETECT_COARSE_CONFIDENCE"
  --refine-padding "$DETECT_REFINE_PADDING"
)
if [[ -n "${JASNA_DETECT_ACTIVE_RANGES+x}" ]]; then
  ADAPTIVE_ARGUMENTS+=(--active-ranges "$JASNA_DETECT_ACTIVE_RANGES")
fi
if [[ "$ADAPTIVE_DETECT" == "1" ]]; then
  ADAPTIVE_ARGUMENTS+=(--adaptive-scan)
fi

# Build one non-empty command array. macOS's Bash 3.2 reports an empty
# "${array[@]}" expansion as unbound under `set -u`, which broke physical
# single-eye scans because they intentionally have no stereo manifest.
DETECT_COMMAND=(
  "$PYTHON_PATH" "$ROOT_DIR/tools/scan_mosaic_regions.py"
  "$1" "$2"
  --model "$MODEL_PATH"
  --backend "$DETECTOR_BACKEND"
  --rfdetr-variant "$RFDETR_VARIANT"
  --crop-eye "$CROP_EYE"
  --stereo-sample-mode "$STEREO_SAMPLE_MODE"
)
if [[ $# -eq 3 ]]; then
  DETECT_COMMAND+=(--stereo-right-manifest "$3")
fi
DETECT_COMMAND+=(
  --device "$DETECT_DEVICE"
  --batch-size "$DETECT_BATCH_SIZE"
  --max-detections "$RFDETR_MAX_DETECTIONS"
  --decode-mode "$DETECT_DECODE_MODE"
  --sample-stride "$DETECT_SAMPLE_STRIDE"
  --region-duration "$REGION_DURATION"
  --confidence "$DETECT_CONFIDENCE"
  --temporal-padding "$TEMPORAL_PADDING"
  --region-nms-iou "$REGION_NMS_IOU"
  --mask-expansion "$MASK_EXPANSION"
  --mask-size "$MASK_SIZE"
)
DETECT_COMMAND+=("${ADAPTIVE_ARGUMENTS[@]}")
"${DETECT_COMMAND[@]}"
