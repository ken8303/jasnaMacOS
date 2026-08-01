#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: $0 INPUT_SBS_30FPS_VIDEO OUTPUT_LEFT_EYE.mov" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="$1"
OUTPUT_PATH="$2"
TEST_SECONDS="${JASNA_TEST_SECONDS:-30}"
BITRATE="${JASNA_EYE_BITRATE:-20000000}"
PYTHON_PATH="$ROOT_DIR/.venv-mosaic/bin/python"
DEEP_SOURCE="${JASNA_DEEPMOSAICS_SOURCE:-$ROOT_DIR/../../work/DeepMosaics}"
WEIGHTS="${JASNA_DEEPMOSAICS_WEIGHTS:-/Users/kenlo/Downloads/clean_youknow_video.pth}"
FFMPEG_PATH="$(command -v ffmpeg)"
FFPROBE_PATH="$(command -v ffprobe)"

[[ -f "$INPUT_PATH" && -x "$PYTHON_PATH" && -d "$DEEP_SOURCE" && -s "$WEIGHTS" ]] || {
  echo "error: input, Python environment, DeepMosaics source, or weights are missing" >&2
  exit 1
}
[[ "$TEST_SECONDS" =~ ^[0-9]+$ ]] && (( TEST_SECONDS >= 1 && TEST_SECONDS <= 30 )) || {
  echo "error: JASNA_TEST_SECONDS must be from 1 to 30" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT_PATH")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.deepmosaics-work"
SOURCE_DIR="$WORK_DIR/source"
WINDOW_WORK="$WORK_DIR/restoration"
SOURCE_PATH="$SOURCE_DIR/left-${TEST_SECONDS}s.mov"
MANIFEST_PATH="$WORK_DIR/mosaic-regions.json"
CONCAT_PATH="$WORK_DIR/windows.txt"
TEMP_OUTPUT="$WORK_DIR/.joining.mov"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.deepmosaics.log"
FRAME_COUNT=$((TEST_SECONDS * 30))

mkdir -p "$SOURCE_DIR" "$WINDOW_WORK"
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== DeepMosaics left-eye test $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:    $INPUT_PATH"
echo "Output:   $OUTPUT_PATH"
echo "Work dir: $WORK_DIR"
echo "Log:      $LOG_PATH"

valid_frames() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -s "$path" ]] || return 1
  actual="$("$FFPROBE_PATH" -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 \
    "$path" 2>/dev/null)" || return 1
  [[ "$actual" == "$expected" ]]
}

if ! valid_frames "$SOURCE_PATH" "$FRAME_COUNT"; then
  TEMP_SOURCE="$SOURCE_DIR/.left-${TEST_SECONDS}s.encoding.mov"
  [[ ! -e "$TEMP_SOURCE" ]] || mv "$TEMP_SOURCE" \
    "$SOURCE_DIR/left-${TEST_SECONDS}s.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
  echo "Stage 1/4: creating persistent 4096x4096 left-eye source"
  "$FFMPEG_PATH" -hide_banner -i "$INPUT_PATH" -map '0:v:0' \
    -vf 'crop=iw/2:ih:0:0,fps=30' -frames:v "$FRAME_COUNT" -an \
    -c:v hevc_videotoolbox -allow_sw 1 -pix_fmt yuv420p -b:v "$BITRATE" \
    -maxrate "$((BITRATE * 3 / 2))" -bufsize "$((BITRATE * 3))" \
    -g 30 -bf 0 -tag:v hvc1 -movflags +faststart "$TEMP_SOURCE"
  valid_frames "$TEMP_SOURCE" "$FRAME_COUNT" || {
    echo "error: left-eye source validation failed" >&2
    exit 1
  }
  mv "$TEMP_SOURCE" "$SOURCE_PATH"
else
  echo "Stage 1/4: reusing persistent left-eye source"
fi

if [[ ! -s "$MANIFEST_PATH" ]]; then
  echo "Stage 2/4: scanning only for mosaic regions"
  "$ROOT_DIR/script/scan_mosaic_regions.sh" "$SOURCE_PATH" "$MANIFEST_PATH"
else
  echo "Stage 2/4: reusing mosaic manifest"
fi

echo "Stage 3/4: restoring resumable one-second windows"
"$PYTHON_PATH" "$ROOT_DIR/tools/restore_deepmosaics_windows.py" \
  --video "$SOURCE_PATH" --manifest "$MANIFEST_PATH" \
  --deepmosaics-source "$DEEP_SOURCE" --weights "$WEIGHTS" \
  --work-dir "$WINDOW_WORK" --ffmpeg "$FFMPEG_PATH" --ffprobe "$FFPROBE_PATH" \
  --bitrate "$BITRATE"

echo "Stage 4/4: joining validated windows without re-encoding"
: > "$CONCAT_PATH"
for WINDOW in "$WINDOW_WORK"/windows/window-*.mov; do
  ESCAPED_WINDOW="${WINDOW//\'/\'\\\'\'}"
  printf "file '%s'\n" "$ESCAPED_WINDOW" >> "$CONCAT_PATH"
done
[[ ! -e "$TEMP_OUTPUT" ]] || mv "$TEMP_OUTPUT" \
  "$WORK_DIR/joining.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
"$FFMPEG_PATH" -hide_banner -f concat -safe 0 -i "$CONCAT_PATH" \
  -map '0:v:0' -c copy -movflags +faststart "$TEMP_OUTPUT"
valid_frames "$TEMP_OUTPUT" "$FRAME_COUNT" || {
  echo "error: joined output validation failed" >&2
  exit 1
}
if [[ -e "$OUTPUT_PATH" ]]; then
  mv "$OUTPUT_PATH" "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').mov"
fi
mv "$TEMP_OUTPUT" "$OUTPUT_PATH"
echo "DeepMosaics left-eye test: PASS"
echo "Output: $OUTPUT_PATH"
echo "Persistent work: $WORK_DIR"
echo "Log: $LOG_PATH"
