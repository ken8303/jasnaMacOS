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
BASELINE_MODELS="$ROOT_DIR/Models/MetalML"
BASELINE_BATCH2_MODELS="$ROOT_DIR/Models/MetalMLBatch2"

[[ -d "$BASELINE_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: baseline restoration models are unavailable: $BASELINE_MODELS" >&2
  exit 1
}
[[ -d "$BASELINE_BATCH2_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: baseline batch-2 restoration models are unavailable: $BASELINE_BATCH2_MODELS" >&2
  exit 1
}
if [[ -n "${JASNA_MODELS_DIR:-}" && "$JASNA_MODELS_DIR" != "$BASELINE_MODELS" ]]; then
  echo "WARNING: ignoring JASNA_MODELS_DIR for baseline runtime test: $JASNA_MODELS_DIR" >&2
  echo "Use script/test_vr_finetuned_30s.sh or script/test_vr_restore_ab.sh for candidate weights" >&2
fi

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
export JASNA_MODELS_DIR="$BASELINE_MODELS"
export JASNA_MODEL_BATCH=2
export JASNA_BATCH2_MODELS_DIR="$BASELINE_BATCH2_MODELS"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1
export JASNA_TEMPORAL_WARMUP_FRAMES=5
export JASNA_LOG_PEAK_MEMORY=1

echo "Jasna direct-SBS MP4 30-second test"
echo "Model track: baseline (fine-tuned candidates are test-only)"
echo "Source/output: MP4; paired RF-DETR: 10 Hz"
echo "Physical eye videos: disabled; eye crops and area restoration: in memory"
echo "Models: $JASNA_MODELS_DIR; batch 2: $JASNA_BATCH2_MODELS_DIR"

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" "$@"
