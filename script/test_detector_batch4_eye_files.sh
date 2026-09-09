#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 PHYSICAL_EYE_DIR [DETECTOR_OUTPUT_DIR]" >&2
  echo "Runs detector-only RF-DETR batch 4 on five one-minute files per eye." >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EYE_DIR="${1%/}"
OUTPUT_DIR="${2:-${EYE_DIR}.detector-batch4}"
LEFT_DIR="$EYE_DIR/left"
RIGHT_DIR="$EYE_DIR/right"

[[ -d "$LEFT_DIR" && -d "$RIGHT_DIR" ]] || {
  echo "error: physical left/right eye directories are unavailable: $EYE_DIR" >&2
  exit 1
}

for eye in left right; do
  for index in 0 1 2 3 4; do
    source_path="$EYE_DIR/$eye/$(printf '%s-%05d.mp4' "$eye" "$index")"
    [[ -s "$source_path" ]] || {
      echo "error: missing physical eye file: $source_path" >&2
      exit 1
    }
  done
done

if [[ -d "$OUTPUT_DIR" && -n "$(/usr/bin/find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  existing_log="$OUTPUT_DIR/detector-batch4.log"
  if [[ -f "$existing_log" ]] \
    && /usr/bin/grep -q "Detector-only physical-eye batch-4 test: PASS" "$existing_log"; then
    echo "error: a completed comparison already exists in $OUTPUT_DIR" >&2
    echo "use a new detector output directory to keep measurements independent" >&2
    exit 1
  fi

  failed_output="${OUTPUT_DIR}.failed-$(date '+%Y%m%d-%H%M%S')"
  if [[ -e "$failed_output" ]]; then
    failed_output="${failed_output}-$$"
  fi
  /bin/mv "$OUTPUT_DIR" "$failed_output"
  echo "Archived incomplete detector attempt at: $failed_output"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_PATH="$OUTPUT_DIR/detector-batch4.log"
RUNTIME_SCRATCH="$OUTPUT_DIR/runtime-scratch"
mkdir -p "$RUNTIME_SCRATCH/tmp" "$RUNTIME_SCRATCH/cache"
export TMPDIR="$RUNTIME_SCRATCH/tmp/"
export XDG_CACHE_HOME="$RUNTIME_SCRATCH/cache"

exec > >(/usr/bin/tee "$LOG_PATH") 2>&1

echo "===== Jasna physical-eye detector batch-4 test $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Physical eyes: $EYE_DIR"
echo "Files: five left + five right, one minute/file"
echo "Detector: rfdetr-vr-v1, MPS auto, 10 Hz, sequential decode, batch 4"
echo "Restoration and final encoding: disabled"

export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_DEVICE=auto
export JASNA_DETECT_SAMPLE_STRIDE=0.1
export JASNA_DETECT_BATCH_SIZE=4
export JASNA_DETECT_DECODE_MODE=sequential
export JASNA_ADAPTIVE_DETECT=0
export JASNA_DETECT_CONFIDENCE=0.15
export JASNA_TEMPORAL_PADDING=1.0
export JASNA_REGION_NMS_IOU=0.45
export JASNA_MASK_EXPANSION=0.10
export JASNA_MASK_SIZE=128
export JASNA_RFDETR_MAX_DETECTIONS=64

RUN_STARTED_SECONDS=$SECONDS
for eye in left right; do
  for index in 0 1 2 3 4; do
    source_path="$EYE_DIR/$eye/$(printf '%s-%05d.mp4' "$eye" "$index")"
    manifest="$OUTPUT_DIR/$(printf '%s-%05d-mosaic-regions.json' "$eye" "$index")"
    segment_log="$OUTPUT_DIR/$(printf '%s-%05d.log' "$eye" "$index")"
    echo "Detector batch-4 $eye segment $((index + 1))/5: $source_path"
    JASNA_DETECT_EYE=none "$ROOT_DIR/script/scan_mosaic_regions.sh" \
      "$source_path" "$manifest" 2>&1 | /usr/bin/tee "$segment_log"
  done
done

RUN_WALL_SECONDS=$((SECONDS - RUN_STARTED_SECONDS))
SCAN_SECONDS="$('/usr/bin/awk' '
  /Detector scan:/ {
    value = $3
    sub(/s,$/, "", value)
    total += value
    count += 1
  }
  END { if (count > 0) printf "%.3f", total }
' "$OUTPUT_DIR"/left-*.log "$OUTPUT_DIR"/right-*.log)"

echo "Physical-eye detector batch-4 scan total: ${SCAN_SECONDS}s"
echo "Physical-eye detector batch-4 total wall: ${RUN_WALL_SECONDS}s"
echo "Detector-only physical-eye batch-4 test: PASS"
echo "Manifests and log preserved at: $OUTPUT_DIR"
