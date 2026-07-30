#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT OUTPUT [START [END]]" >&2
  echo "example: $0 input.mov clip.mov 12:00 13:00" >&2
  exit 2
}

time_to_seconds() {
  local value="$1"
  local first second third
  IFS=: read -r first second third <<<"$value"
  if [[ -z "${second:-}" ]]; then
    [[ "$first" =~ ^[0-9]+$ ]] || return 1
    echo "$((10#$first))"
  elif [[ -z "${third:-}" ]]; then
    [[ "$first" =~ ^[0-9]+$ && "$second" =~ ^[0-9]+$ ]] || return 1
    echo "$((10#$first * 60 + 10#$second))"
  else
    [[ "$first" =~ ^[0-9]+$ && "$second" =~ ^[0-9]+$ && "$third" =~ ^[0-9]+$ ]] || return 1
    echo "$((10#$first * 3600 + 10#$second * 60 + 10#$third))"
  fi
}

[[ $# -ge 2 && $# -le 4 ]] || usage

INPUT_PATH="$1"
OUTPUT_PATH="$2"
START_TIME="${3:-12:00}"
END_TIME="${4:-13:00}"

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

START_SECONDS="$(time_to_seconds "$START_TIME")" || {
  echo "error: invalid start time: $START_TIME" >&2
  exit 1
}
END_SECONDS="$(time_to_seconds "$END_TIME")" || {
  echo "error: invalid end time: $END_TIME" >&2
  exit 1
}
DURATION_SECONDS="$((END_SECONDS - START_SECONDS))"
((DURATION_SECONDS > 0)) || {
  echo "error: end time must be after start time" >&2
  exit 1
}

echo "Fast cutting $INPUT_PATH"
echo "Requested range: $START_TIME → $END_TIME ($DURATION_SECONDS seconds)"
echo "Output: $OUTPUT_PATH"

"$FFMPEG_PATH" \
  -hide_banner \
  -ss "$START_SECONDS" \
  -i "$INPUT_PATH" \
  -t "$DURATION_SECONDS" \
  -map 0 \
  -map_metadata 0 \
  -map_chapters 0 \
  -c copy \
  -avoid_negative_ts make_zero \
  -n \
  "$OUTPUT_PATH"

echo "Done. Video streams were copied without re-encoding."
echo "Note: fast cuts align to nearby keyframes, so the result may start slightly early."

if command -v ffprobe >/dev/null 2>&1; then
  FFPROBE_PATH="$(command -v ffprobe)"
elif [[ -x /opt/homebrew/bin/ffprobe ]]; then
  FFPROBE_PATH="/opt/homebrew/bin/ffprobe"
else
  FFPROBE_PATH=""
fi

if [[ -n "$FFPROBE_PATH" ]]; then
  ACTUAL_DURATION="$($FFPROBE_PATH \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$OUTPUT_PATH")"
  echo "Actual output duration: ${ACTUAL_DURATION}s"
fi
