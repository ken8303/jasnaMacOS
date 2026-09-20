#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 3 && $# -le 4 ]] || {
  echo "usage: $0 JASNA_SOURCE AUDITED_DATASET OUTPUT_DIRECTORY [TARGET_STEPS]" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JASNA_SOURCE="$1"
DATASET="$2"
OUTPUT="$3"
TARGET_STEPS="${4:-1000}"
PYTHON_PATH="${JASNA_TRAIN_PYTHON:-python3}"
WEIGHTS="${JASNA_FINETUNE_WEIGHTS:-$ROOT_DIR/Models/SourceWeights/lada_mosaic_restoration_model_generic_v1.2.pth}"
RESUME="${JASNA_FINETUNE_RESUME:-}"
DEVICE="${JASNA_FINETUNE_DEVICE:-cuda}"

[[ -d "$JASNA_SOURCE" ]] || {
  echo "error: Jasna source directory does not exist: $JASNA_SOURCE" >&2
  exit 1
}
[[ -f "$DATASET/mosaic-audit.json" ]] || {
  echo "error: dataset has no mosaic-audit.json: $DATASET" >&2
  exit 1
}
[[ -f "$WEIGHTS" ]] || {
  echo "error: source restoration checkpoint does not exist: $WEIGHTS" >&2
  exit 1
}

mkdir -p "$OUTPUT"
ARGUMENTS=(
  --jasna-source "$JASNA_SOURCE"
  --weights "$WEIGHTS"
  --dataset "$DATASET"
  --output "$OUTPUT"
  --steps "$TARGET_STEPS"
  --frames "${JASNA_FINETUNE_FRAMES:-10}"
  --batch-size "${JASNA_FINETUNE_BATCH_SIZE:-1}"
  --learning-rate "${JASNA_FINETUNE_LEARNING_RATE:-0.00001}"
  --temporal-weight "${JASNA_FINETUNE_TEMPORAL_WEIGHT:-0.10}"
  --background-weight "${JASNA_FINETUNE_BACKGROUND_WEIGHT:-0.05}"
  --gradient-weight "${JASNA_FINETUNE_GRADIENT_WEIGHT:-0.0}"
  --ema-decay "${JASNA_FINETUNE_EMA_DECAY:-0.999}"
  --gradient-accumulation "${JASNA_FINETUNE_GRADIENT_ACCUMULATION:-1}"
  --validate-every "${JASNA_FINETUNE_VALIDATE_EVERY:-250}"
  --save-every "${JASNA_FINETUNE_SAVE_EVERY:-500}"
  --validation-clips "${JASNA_FINETUNE_VALIDATION_CLIPS:-8}"
  --minimum-psnr-gain "${JASNA_FINETUNE_MINIMUM_PSNR_GAIN:-0.10}"
  --maximum-temporal-regression "${JASNA_FINETUNE_MAXIMUM_TEMPORAL_REGRESSION:-1.02}"
  --corruption-profile "${JASNA_FINETUNE_CORRUPTION_PROFILE:-legacy}"
  --device "$DEVICE"
)
case "$DEVICE" in
  cuda) ;;
  cpu|mps) ARGUMENTS+=(--allow-experimental-non-cuda) ;;
  *)
    echo "error: JASNA_FINETUNE_DEVICE must be cuda, cpu, or mps" >&2
    exit 2
    ;;
esac
if [[ -n "$RESUME" ]]; then
  ARGUMENTS+=(--resume "$RESUME")
fi
if [[ "${JASNA_FINETUNE_RESET_BASELINE:-0}" == "1" ]]; then
  ARGUMENTS+=(--reset-quality-baseline)
fi

echo "Jasna BasicVSR++ fine-tuning"
echo "Dataset: $DATASET"
echo "Output:  $OUTPUT"
echo "Target:  $TARGET_STEPS steps; ${JASNA_FINETUNE_FRAMES:-10} frames/sample"
echo "Mosaic:  ${JASNA_FINETUNE_CORRUPTION_PROFILE:-legacy}"
echo "Gradient loss: ${JASNA_FINETUNE_GRADIENT_WEIGHT:-0.0}"
echo "EMA:     ${JASNA_FINETUNE_EMA_DECAY:-0.999}"
echo "Device:  $DEVICE"
if [[ "$DEVICE" != "cuda" ]]; then
  echo "WARNING: non-CUDA training is diagnostic and substantially slower"
fi
if [[ -n "$RESUME" ]]; then
  echo "Resume:  $RESUME"
fi

"$PYTHON_PATH" "$ROOT_DIR/tools/finetune_basicvsrpp.py" "${ARGUMENTS[@]}" \
  2>&1 | tee -a "$OUTPUT/training.log"
