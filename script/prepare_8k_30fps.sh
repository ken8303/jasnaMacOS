#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT OUTPUT" >&2
  echo "example: $0 input.mp4 input_30fps.mp4" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

INPUT_PATH="$1"
OUTPUT_PATH="$2"

[[ "$INPUT_PATH" != *[[:space:]] ]] || {
  echo "error: input path ends with whitespace: '$INPUT_PATH'" >&2
  exit 1
}
[[ "$OUTPUT_PATH" != *[[:space:]] ]] || {
  echo "error: output path ends with whitespace: '$OUTPUT_PATH'" >&2
  exit 1
}
[[ -f "$INPUT_PATH" ]] || {
  echo "error: input video not found: $INPUT_PATH" >&2
  exit 1
}
[[ ! -e "$OUTPUT_PATH" ]] || {
  echo "error: output already exists: $OUTPUT_PATH" >&2
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

echo "Preparing an AVFoundation-compatible 8K/30 Main 10 file"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Video decode uses FFmpeg software HEVC; encode uses Apple VideoToolbox."

"$FFMPEG_PATH" \
  -hide_banner \
  -i "$INPUT_PATH" \
  -map 0:v:0 \
  -map '0:a?' \
  -map_metadata 0 \
  -map_chapters 0 \
  -vf fps=30 \
  -c:v hevc_videotoolbox \
  -profile:v main10 \
  -pix_fmt p010le \
  -b:v 40M \
  -maxrate 60M \
  -bufsize 120M \
  -tag:v hvc1 \
  -c:a copy \
  -movflags +faststart \
  -n \
  "$OUTPUT_PATH"

OUTPUT_INFO="$($FFPROBE_PATH \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate \
  -of default=noprint_wrappers=1 \
  "$OUTPUT_PATH")"

echo "Prepared video:"
echo "$OUTPUT_INFO"
