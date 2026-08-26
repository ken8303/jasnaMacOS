#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 || $# -eq 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 fine-tuned-test.mov 00:25:57" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINE_TUNED_MODELS="${JASNA_FINETUNED_MODELS_DIR:-$ROOT_DIR/Models/MetalMLFineTuneMac1000}"

[[ -d "$FINE_TUNED_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: fine-tuned Metal packages are unavailable: $FINE_TUNED_MODELS" >&2
  exit 1
}

export JASNA_MODELS_DIR="$FINE_TUNED_MODELS"
export JASNA_MODEL_BATCH=1
export JASNA_TEST_SECONDS="${JASNA_TEST_SECONDS:-30}"
export JASNA_DETECT_CONFIDENCE="${JASNA_DETECT_CONFIDENCE:-0.10}"
export JASNA_MASK_EXPANSION="${JASNA_MASK_EXPANSION:-0.18}"
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS="${JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS:-1}"
export JASNA_TEMPORAL_WARMUP_FRAMES="${JASNA_TEMPORAL_WARMUP_FRAMES:-5}"

echo "Fine-tuned restoration A/B candidate: $(basename "$FINE_TUNED_MODELS")"
echo "Models: $JASNA_MODELS_DIR"
echo "Coverage: detector confidence $JASNA_DETECT_CONFIDENCE; mask expansion $JASNA_MASK_EXPANSION"
echo "Ordinary mask-hole recovery: $JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS"
echo "Temporal crop warm-up: $JASNA_TEMPORAL_WARMUP_FRAMES frame(s)"
exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" "$@"
