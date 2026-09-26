#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_5MIN_WORK_DIR [OUTPUT_DIR]" >&2
  echo "Exports five one-minute 4096x4096, 30 fps, HEVC Main 8 MP4 files per eye." >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

REFERENCE_WORK_DIR="${1%/}"
OUTPUT_DIR="${2:-${REFERENCE_WORK_DIR}.eyes-4k-main8}"
INPUT_PATH="$REFERENCE_WORK_DIR/source/test-sbs-30fps.mp4"

[[ -f "$INPUT_PATH" ]] || {
  echo "error: prepared five-minute SBS MP4 not found: $INPUT_PATH" >&2
  exit 1
}

if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_PATH="$(command -v ffmpeg)"
elif [[ -x /opt/homebrew/bin/ffmpeg ]]; then
  FFMPEG_PATH=/opt/homebrew/bin/ffmpeg
else
  echo "error: ffmpeg is not installed" >&2
  exit 1
fi
if command -v ffprobe >/dev/null 2>&1; then
  FFPROBE_PATH="$(command -v ffprobe)"
elif [[ -x /opt/homebrew/bin/ffprobe ]]; then
  FFPROBE_PATH=/opt/homebrew/bin/ffprobe
else
  echo "error: ffprobe is not installed" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LEFT_DIR="$OUTPUT_DIR/left"
RIGHT_DIR="$OUTPUT_DIR/right"
LEFT_TEMP_DIR="$OUTPUT_DIR/.left-writing"
RIGHT_TEMP_DIR="$OUTPUT_DIR/.right-writing"
LOG_PATH="$OUTPUT_DIR/prepare-eyes.log"

video_value() {
  local path="$1"
  local entry="$2"
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries "stream=$entry" \
    -of default=noprint_wrappers=1:nokey=1 "$path" 2>/dev/null
}

valid_eye_segment() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  [[ "$(video_value "$path" codec_name)" == "hevc" ]] || return 1
  [[ "$(video_value "$path" profile)" == "Main" ]] || return 1
  [[ "$(video_value "$path" pix_fmt)" == "yuv420p" ]] || return 1
  [[ "$(video_value "$path" width)" == "4096" ]] || return 1
  [[ "$(video_value "$path" height)" == "4096" ]] || return 1
  [[ "$(video_value "$path" avg_frame_rate)" == "30/1" ]] || return 1
  [[ "$(video_value "$path" nb_frames)" == "1800" ]] || return 1
  "$FFMPEG_PATH" -v error -i "$path" -map 0:v:0 -frames:v 1 \
    -f null - >/dev/null 2>&1
}

valid_all_eye_segments() {
  local eye_dir="$1"
  local eye_name="$2"
  local index path
  for index in 0 1 2 3 4; do
    path="$eye_dir/$(printf '%s-%05d.mp4' "$eye_name" "$index")"
    valid_eye_segment "$path" || return 1
  done
  return 0
}

if valid_all_eye_segments "$LEFT_DIR" left \
  && valid_all_eye_segments "$RIGHT_DIR" right; then
  echo "Reusing ten validated physical 4K one-minute eye videos"
  echo "Directory: $OUTPUT_DIR"
  exit 0
fi

if [[ -e "$LEFT_DIR" || -e "$RIGHT_DIR" ]]; then
  echo "error: existing final eye segments are incomplete or invalid in $OUTPUT_DIR" >&2
  echo "use a new output directory so existing data is not overwritten" >&2
  exit 1
fi
if [[ -e "$LEFT_TEMP_DIR" || -e "$RIGHT_TEMP_DIR" ]]; then
  echo "error: interrupted temporary eye segments exist in $OUTPUT_DIR" >&2
  echo "use a new output directory, or archive the interrupted folders first" >&2
  exit 1
fi

mkdir -p "$LEFT_TEMP_DIR" "$RIGHT_TEMP_DIR"
exec > >(/usr/bin/tee "$LOG_PATH") 2>&1

echo "===== Jasna one-minute physical eye export $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_DIR"
echo "Files:  five left + five right, one minute/file"
echo "Format: 4096x4096, 30 fps, 1,800 frames, HEVC Main 8, yuv420p MP4"
echo "Decode: one shared SBS decode; encode: two VideoToolbox segment outputs"
echo "Detector and restoration: disabled"

RUN_STARTED_SECONDS=$SECONDS
"$FFMPEG_PATH" -hide_banner -i "$INPUT_PATH" \
  -filter_complex \
    "[0:v:0]split=2[leftbase][rightbase];[leftbase]crop=4096:4096:0:0,fps=30,format=yuv420p[left];[rightbase]crop=4096:4096:4096:0,fps=30,format=yuv420p[right]" \
  -map '[left]' -an -c:v hevc_videotoolbox -profile:v main \
  -realtime 1 -prio_speed 1 -pix_fmt yuv420p \
  -b:v 20000000 -maxrate 30000000 -bufsize 60000000 \
  -g 30 -force_key_frames 'expr:gte(t,n_forced*60)' -tag:v hvc1 \
  -f segment -segment_format mp4 -segment_time 60 -segment_time_delta 0.016667 \
  -reset_timestamps 1 "$LEFT_TEMP_DIR/left-%05d.mp4" \
  -map '[right]' -an -c:v hevc_videotoolbox -profile:v main \
  -realtime 1 -prio_speed 1 -pix_fmt yuv420p \
  -b:v 20000000 -maxrate 30000000 -bufsize 60000000 \
  -g 30 -force_key_frames 'expr:gte(t,n_forced*60)' -tag:v hvc1 \
  -f segment -segment_format mp4 -segment_time 60 -segment_time_delta 0.016667 \
  -reset_timestamps 1 "$RIGHT_TEMP_DIR/right-%05d.mp4"

valid_all_eye_segments "$LEFT_TEMP_DIR" left || {
  echo "error: left one-minute eye files failed codec, dimensions, frame-count, or decode validation" >&2
  exit 1
}
valid_all_eye_segments "$RIGHT_TEMP_DIR" right || {
  echo "error: right one-minute eye files failed codec, dimensions, frame-count, or decode validation" >&2
  exit 1
}

mv "$LEFT_TEMP_DIR" "$LEFT_DIR"
mv "$RIGHT_TEMP_DIR" "$RIGHT_DIR"
RUN_WALL_SECONDS=$((SECONDS - RUN_STARTED_SECONDS))

echo "Physical one-minute eye export: PASS"
echo "Files validated: 10"
echo "Frames per file: 1800"
echo "Frames per eye: 9000"
echo "Export wall: ${RUN_WALL_SECONDS}s"
echo "Left:  $LEFT_DIR"
echo "Right: $RIGHT_DIR"
echo "Next phase: detector batch 4 has not been started"
