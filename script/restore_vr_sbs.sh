#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO" >&2
  echo "optional: JASNA_LEFT_WORK_DIR=/path/to/existing/cache $0 INPUT OUTPUT" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

INPUT_PATH="$1"
OUTPUT_PATH="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$INPUT_PATH" ]] || {
  echo "error: input video not found: $INPUT_PATH" >&2
  exit 1
}
[[ "$INPUT_PATH" != *[[:space:]] ]] || {
  echo "error: input path ends with whitespace: '$INPUT_PATH'" >&2
  exit 1
}
[[ "$OUTPUT_PATH" != *[[:space:]] ]] || {
  echo "error: output path ends with whitespace: '$OUTPUT_PATH'" >&2
  exit 1
}

if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_PATH="$(command -v ffmpeg)"
elif [[ -x /opt/homebrew/bin/ffmpeg ]]; then
  FFMPEG_PATH="/opt/homebrew/bin/ffmpeg"
else
  echo "error: ffmpeg is not installed" >&2
  exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
  FFPROBE_PATH="$(command -v ffprobe)"
elif [[ -x /opt/homebrew/bin/ffprobe ]]; then
  FFPROBE_PATH="/opt/homebrew/bin/ffprobe"
else
  echo "error: ffprobe is not installed" >&2
  exit 1
fi

OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.vr-work"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.vr.log"
LEFT_OUTPUT="$WORK_DIR/left-restored.mov"
RIGHT_OUTPUT="$WORK_DIR/right-restored.mov"
LEFT_DONE="$WORK_DIR/left.done"
RIGHT_DONE="$WORK_DIR/right.done"
LEFT_WORK_DIR="${JASNA_LEFT_WORK_DIR:-$WORK_DIR/left.jasna-work}"
RIGHT_WORK_DIR="${JASNA_RIGHT_WORK_DIR:-$WORK_DIR/right.jasna-work}"
VIDEO_BITRATE="${JASNA_VR_BITRATE:-40000000}"

mkdir -p "$WORK_DIR" "$LEFT_WORK_DIR" "$RIGHT_WORK_DIR"
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== Jasna sequential-eye VR restoration $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:      $INPUT_PATH"
echo "Output:     $OUTPUT_PATH"
echo "Work dir:   $WORK_DIR"
echo "Left cache: $LEFT_WORK_DIR"
echo "Right cache:$RIGHT_WORK_DIR"

IFS=, read -r SOURCE_WIDTH SOURCE_HEIGHT < <(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$INPUT_PATH"
)
SOURCE_DURATION="$("$FFPROBE_PATH" -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "$INPUT_PATH")"
[[ "$SOURCE_WIDTH" =~ ^[0-9]+$ && "$SOURCE_HEIGHT" =~ ^[0-9]+$ ]] || {
  echo "error: unable to read input dimensions" >&2
  exit 1
}
(( SOURCE_WIDTH % 2 == 0 )) || {
  echo "error: SBS input width must be even: $SOURCE_WIDTH" >&2
  exit 1
}
echo "SBS canvas: ${SOURCE_WIDTH}x${SOURCE_HEIGHT}; each eye: $((SOURCE_WIDTH / 2))x${SOURCE_HEIGHT}"

video_is_complete() {
  local candidate="$1"
  [[ -s "$candidate" ]] || return 1
  local candidate_duration
  candidate_duration="$("$FFPROBE_PATH" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$candidate" 2>/dev/null)" || return 1
  /usr/bin/awk -v source="$SOURCE_DURATION" -v candidate="$candidate_duration" \
    'BEGIN { delta = source - candidate; if (delta < 0) delta = -delta; exit !(delta <= 0.05) }'
}

if [[ ! -f "$LEFT_DONE" ]] && video_is_complete "$LEFT_OUTPUT"; then
  echo "Recovered completed left-eye stage marker"
  /usr/bin/touch "$LEFT_DONE"
fi
if [[ ! -f "$RIGHT_DONE" ]] && video_is_complete "$RIGHT_OUTPUT"; then
  echo "Recovered completed right-eye stage marker"
  /usr/bin/touch "$RIGHT_DONE"
fi

if [[ ! -f "$LEFT_DONE" ]]; then
  echo "Stage 1/3: restoring left eye"
  JASNA_WORK_DIR="$LEFT_WORK_DIR" \
    "$ROOT_DIR/script/build_and_run.sh" --restore-sbs-eye \
      "$INPUT_PATH" left "$LEFT_OUTPUT"
  [[ -s "$LEFT_OUTPUT" ]] || {
    echo "error: left-eye restoration did not produce a video" >&2
    exit 1
  }
  /usr/bin/touch "$LEFT_DONE"
else
  [[ -s "$LEFT_OUTPUT" ]] || {
    echo "error: left.done exists but left-eye video is missing" >&2
    exit 1
  }
  echo "Stage 1/3: left eye already complete"
fi

if [[ ! -f "$RIGHT_DONE" ]]; then
  echo "Stage 2/3: restoring right eye"
  JASNA_WORK_DIR="$RIGHT_WORK_DIR" \
    "$ROOT_DIR/script/build_and_run.sh" --restore-sbs-eye \
      "$INPUT_PATH" right "$RIGHT_OUTPUT"
  [[ -s "$RIGHT_OUTPUT" ]] || {
    echo "error: right-eye restoration did not produce a video" >&2
    exit 1
  }
  /usr/bin/touch "$RIGHT_DONE"
else
  [[ -s "$RIGHT_OUTPUT" ]] || {
    echo "error: right.done exists but right-eye video is missing" >&2
    exit 1
  }
  echo "Stage 2/3: right eye already complete"
fi

if [[ -e "$OUTPUT_PATH" ]]; then
  ARCHIVE_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.interrupted-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
  mv "$OUTPUT_PATH" "$ARCHIVE_PATH"
  echo "Archived previous final output: $ARCHIVE_PATH"
fi

echo "Stage 3/3: combining restored eyes and copying original audio"
"$FFMPEG_PATH" \
  -hide_banner \
  -i "$LEFT_OUTPUT" \
  -i "$RIGHT_OUTPUT" \
  -i "$INPUT_PATH" \
  -filter_complex '[0:v:0][1:v:0]hstack=inputs=2[v]' \
  -map '[v]' \
  -map '2:a?' \
  -map_metadata 2 \
  -map_chapters 2 \
  -c:v hevc_videotoolbox \
  -pix_fmt yuv420p \
  -b:v "$VIDEO_BITRATE" \
  -maxrate "$((VIDEO_BITRATE * 3 / 2))" \
  -bufsize "$((VIDEO_BITRATE * 3))" \
  -tag:v hvc1 \
  -c:a copy \
  -r 30 \
  -movflags +faststart \
  -shortest \
  -n \
  "$OUTPUT_PATH"

FINAL_INFO="$($FFPROBE_PATH \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  "$OUTPUT_PATH")"

echo "Sequential-eye VR restoration: PASS"
echo "$FINAL_INFO"
echo "Output: $OUTPUT_PATH"
echo "Note: audio and ordinary container metadata were copied from the source."
echo "Note: Spherical Video st3d/sv3d atoms still require a dedicated metadata pass."
