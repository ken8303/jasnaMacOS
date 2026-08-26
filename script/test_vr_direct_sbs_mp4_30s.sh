#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 || $# -eq 3 ]] || {
  echo "usage: $0 INPUT_SBS.mp4 OUTPUT_SBS.mp4 [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-direct-sbs.mp4 00:25:57" >&2
  exit 2
}

case "$1" in
  *.[mM][pP]4) ;;
  *)
    echo "error: this test requires an MP4 source" >&2
    exit 1
    ;;
esac
case "$2" in
  *.[mM][pP]4) ;;
  *)
    echo "error: output must use the .mp4 extension" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATED_MODELS="${JASNA_MODELS_DIR:-$ROOT_DIR/Models/MetalMLFineTuneMac1000}"
VALIDATED_BATCH2_MODELS="${JASNA_BATCH2_MODELS_DIR:-${VALIDATED_MODELS%/}Batch2}"

[[ -d "$VALIDATED_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: validated restoration models are unavailable: $VALIDATED_MODELS" >&2
  exit 1
}
[[ -d "$VALIDATED_BATCH2_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: matching batch-2 restoration models are unavailable: $VALIDATED_BATCH2_MODELS" >&2
  exit 1
}

# One SBS MP4 is decoded once. Detector and restoration crop both eyes in
# memory; the left/right job paths are only links and metadata namespaces.
export JASNA_TEST_SECONDS=30
export JASNA_WORK_CONTAINER=mp4
export JASNA_DIRECT_SBS_OUTPUT=1
export JASNA_SHARED_SBS_SOURCE=1
export JASNA_STEREO_DETECT=1
export JASNA_STEREO_SAMPLE_MODE=paired
export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_SAMPLE_STRIDE=0.1
export JASNA_ADAPTIVE_DETECT=0
export JASNA_MODELS_DIR="$VALIDATED_MODELS"
export JASNA_MODEL_BATCH=2
export JASNA_BATCH2_MODELS_DIR="$VALIDATED_BATCH2_MODELS"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1
export JASNA_TEMPORAL_WARMUP_FRAMES=5
export JASNA_LOG_PEAK_MEMORY=1

echo "Jasna direct-SBS MP4 30-second test"
echo "Source/output: MP4; paired RF-DETR: 10 Hz"
echo "Physical eye videos: disabled; eye crops and area restoration: in memory"
echo "Models: $JASNA_MODELS_DIR; batch 2: $JASNA_BATCH2_MODELS_DIR"

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" "$@"
