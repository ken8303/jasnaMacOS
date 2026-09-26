#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_5MIN_WORK_DIR [COMPARISON_OUTPUT_DIR]" >&2
  echo "Runs detector-only RF-DETR batch-2 against the existing 120/120/60s SBS sources." >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="${1%/}"
OUTPUT_DIR="${2:-${REFERENCE_WORK_DIR}.detector-batch2}"
SOURCE_DIR="$REFERENCE_WORK_DIR/left-restored.left-segments-work/source"
BASELINE_LOG="${REFERENCE_WORK_DIR%-work}.log"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
[[ -f "$REFERENCE_WORK_DIR/source/test-sbs-30fps.mp4" ]] || {
  echo "error: prepared five-minute SBS MP4 is unavailable in the reference work directory" >&2
  exit 1
}

shopt -s nullglob
SOURCE_SEGMENTS=("$SOURCE_DIR"/left-*.mov)
shopt -u nullglob
[[ ${#SOURCE_SEGMENTS[@]} -eq 3 ]] || {
  echo "error: expected exactly three prepared SBS segments, found ${#SOURCE_SEGMENTS[@]}" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_PATH="$OUTPUT_DIR/detector-batch2.log"
RUNTIME_SCRATCH="$OUTPUT_DIR/runtime-scratch"
mkdir -p "$RUNTIME_SCRATCH/tmp" "$RUNTIME_SCRATCH/cache"
export TMPDIR="$RUNTIME_SCRATCH/tmp/"
export XDG_CACHE_HOME="$RUNTIME_SCRATCH/cache"

for index in 0 1 2; do
  LEFT_MANIFEST="$OUTPUT_DIR/$(printf 'left-%05d-mosaic-regions.json' "$index")"
  RIGHT_MANIFEST="$OUTPUT_DIR/$(printf 'right-%05d-mosaic-regions.json' "$index")"
  SEGMENT_LOG="$OUTPUT_DIR/$(printf 'segment-%05d.log' "$index")"
  [[ ! -e "$LEFT_MANIFEST" && ! -e "$RIGHT_MANIFEST" && ! -e "$SEGMENT_LOG" ]] || {
    echo "error: comparison manifests or segment logs already exist in $OUTPUT_DIR" >&2
    echo "use a new comparison output directory to keep measurements independent" >&2
    exit 1
  }
done

exec > >(/usr/bin/tee "$LOG_PATH") 2>&1

echo "===== Jasna detector-only batch-2 comparison $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Reference work: $REFERENCE_WORK_DIR"
echo "Prepared source: $REFERENCE_WORK_DIR/source/test-sbs-30fps.mp4"
echo "Source segments: ${#SOURCE_SEGMENTS[@]} (120/120/60 seconds)"
echo "Output: $OUTPUT_DIR"
echo "Detector: rfdetr-vr-v1, MPS auto, paired 10 Hz, sequential decode, batch 2"
echo "Restoration and final encoding: disabled"

export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_DEVICE=auto
export JASNA_DETECT_SAMPLE_STRIDE=0.1
export JASNA_DETECT_BATCH_SIZE=2
export JASNA_DETECT_DECODE_MODE=sequential
export JASNA_STEREO_SAMPLE_MODE=paired
export JASNA_ADAPTIVE_DETECT=0
export JASNA_DETECT_CONFIDENCE=0.15
export JASNA_TEMPORAL_PADDING=1.0
export JASNA_REGION_NMS_IOU=0.45
export JASNA_MASK_EXPANSION=0.10
export JASNA_MASK_SIZE=128
export JASNA_RFDETR_MAX_DETECTIONS=64

RUN_STARTED_SECONDS=$SECONDS
for index in 0 1 2; do
  SOURCE_SEGMENT="${SOURCE_SEGMENTS[$index]}"
  LEFT_MANIFEST="$OUTPUT_DIR/$(printf 'left-%05d-mosaic-regions.json' "$index")"
  RIGHT_MANIFEST="$OUTPUT_DIR/$(printf 'right-%05d-mosaic-regions.json' "$index")"
  SEGMENT_LOG="$OUTPUT_DIR/$(printf 'segment-%05d.log' "$index")"
  echo "Detector batch-2 segment $((index + 1))/3: $SOURCE_SEGMENT"
  "$ROOT_DIR/script/scan_mosaic_regions.sh" \
    "$SOURCE_SEGMENT" "$LEFT_MANIFEST" "$RIGHT_MANIFEST" \
    2>&1 | /usr/bin/tee "$SEGMENT_LOG"
done

RUN_WALL_SECONDS=$((SECONDS - RUN_STARTED_SECONDS))
echo "Detector batch-2 total wall: ${RUN_WALL_SECONDS}s"
BATCH2_SCAN_SECONDS="$('/usr/bin/awk' '
  /Detector scan:/ {
    value = $3
    sub(/s,$/, "", value)
    total += value
    count += 1
  }
  END { if (count > 0) printf "%.3f", total }
' "$OUTPUT_DIR"/segment-*.log)"
if [[ -n "$BATCH2_SCAN_SECONDS" ]]; then
  echo "Detector batch-2 scan total: ${BATCH2_SCAN_SECONDS}s"
fi

if [[ -s "$BASELINE_LOG" ]]; then
  BASELINE_SCAN_SECONDS="$('/usr/bin/awk' '
    /Detector scan:/ {
      value = $3
      sub(/s,$/, "", value)
      total += value
      count += 1
    }
    END { if (count > 0) printf "%.3f", total }
  ' "$BASELINE_LOG")"
  if [[ -n "$BASELINE_SCAN_SECONDS" ]]; then
    echo "Recorded detector batch-1 scan total: ${BASELINE_SCAN_SECONDS}s"
    /usr/bin/awk -v old="$BASELINE_SCAN_SECONDS" -v new="$BATCH2_SCAN_SECONDS" '
      BEGIN {
        if (old > 0 && new > 0) {
          printf "Batch-2 scan change versus recorded batch 1: %.2f%%\n", (new - old) * 100 / old
        }
      }
    '
  fi
fi

echo "Detector-only comparison: PASS"
echo "Manifests and log preserved at: $OUTPUT_DIR"
