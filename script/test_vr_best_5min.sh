#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_MP4 [START_TIME] [MOSAIC_RANGES]" >&2
  echo "example: $0 input.mp4 restored-5min.mp4 00:12:00" >&2
  echo "known ranges example: $0 input.mp4 restored-5min.mp4 00:12:00 00:12:20-00:14:10" >&2
  echo "optional: JASNA_TEST_SECONDS=30 (defaults to 300)" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="$1"
OUTPUT_PATH="$2"
START_TIME="${3:-0}"

case "${OUTPUT_PATH##*.}" in
  mp4|MP4) ;;
  *)
    echo "error: the recommended five-minute test output must end in .mp4" >&2
    exit 1
    ;;
esac

# Quality-first, bounded-memory settings validated for 8K SBS VR on Apple M4.
export JASNA_TEST_SECONDS="${JASNA_TEST_SECONDS:-300}"
export JASNA_WORK_CONTAINER=mp4
export JASNA_DIRECT_SBS_OUTPUT=1
export JASNA_SHARED_SBS_SOURCE=1
export JASNA_STEREO_DETECT=1
export JASNA_STEREO_SAMPLE_MODE=paired
export JASNA_STEREO_MANIFEST_RECONCILE=1

# RF-DETR VR at paired 10 Hz gives the best validated moving-mosaic coverage.
export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_DEVICE=auto
export JASNA_DETECT_SAMPLE_STRIDE=0.1
export JASNA_DETECT_BATCH_SIZE=2
export JASNA_DETECT_DECODE_MODE=sequential
export JASNA_ADAPTIVE_DETECT=0

# Keep the full BasicVSR++ graph on the measured faster M4 profile. An
# equal-settings 30-second A/B found eight windows/process 4.3% faster than
# four while using slightly less peak memory.
export JASNA_MODEL_BATCH=2
export JASNA_BATCH2_MODELS_DIR="$ROOT_DIR/Models/MetalMLBatch2"
export JASNA_METAL_WINDOWS_PER_PROCESS=8
export JASNA_REGION_PREPARE_DEPTH=2
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
# Direct stereo lookahead regressed on M4 because Metal and VideoToolbox share
# memory bandwidth. Keep the measured faster serial compositor.
export JASNA_COMPOSITE_CONCURRENCY=1
export JASNA_TEMPORAL_WARMUP_FRAMES=5
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1

# Keep temporary data beside the output and remove it only after validation PASS.
export JASNA_RUNTIME_SCRATCH_ON_OUTPUT=1
export JASNA_FAST_ENCODE=1
export JASNA_LOG_PEAK_MEMORY=1

if [[ $# -eq 4 ]]; then
  export JASNA_MOSAIC_RANGES="$4"
else
  unset JASNA_MOSAIC_RANGES
fi

echo "Jasna best-settings ${JASNA_TEST_SECONDS}-second test"
echo "Quality: RF-DETR VR paired 10 Hz + full BasicVSR++"
echo "Performance: detector/model batch 2, eight Metal windows/process, 512 MiB crop cache"
echo "Output: direct 8K SBS MP4 with VideoToolbox encoding"

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" \
  "$INPUT_PATH" "$OUTPUT_PATH" "$START_TIME"
