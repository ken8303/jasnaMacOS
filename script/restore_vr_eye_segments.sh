#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO left|right OUTPUT_EYE_VIDEO" >&2
  echo "optional: JASNA_SEGMENT_SECONDS=60 JASNA_EYE_BITRATE=20000000" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage

INPUT_PATH="$1"
EYE="$2"
OUTPUT_PATH="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEGMENT_SECONDS="${JASNA_SEGMENT_SECONDS:-60}"
EYE_BITRATE="${JASNA_EYE_BITRATE:-20000000}"

[[ "$EYE" == "left" || "$EYE" == "right" ]] || usage
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
[[ "$SEGMENT_SECONDS" =~ ^[0-9]+$ ]] && (( SEGMENT_SECONDS >= 30 && SEGMENT_SECONDS <= 120 )) || {
  echo "error: JASNA_SEGMENT_SECONDS must be an integer from 30 to 120" >&2
  exit 1
}
[[ "$EYE_BITRATE" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_EYE_BITRATE must be an integer bit rate" >&2
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
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.${EYE}-segments-work"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.${EYE}-segments.log"
SOURCE_DIR="$WORK_DIR/source"
RESTORED_DIR="$WORK_DIR/restored"
CACHE_DIR="$WORK_DIR/cache"
SOURCE_DONE="$WORK_DIR/source.done"
MANIFEST_PATH="$WORK_DIR/restored-concat.txt"
TEMP_OUTPUT="$WORK_DIR/${EYE}-joined.mov"

mkdir -p "$WORK_DIR" "$RESTORED_DIR" "$CACHE_DIR"
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== Jasna segmented $EYE-eye restoration $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:            $INPUT_PATH"
echo "Output:           $OUTPUT_PATH"
echo "Work dir:         $WORK_DIR"
echo "Log:              $LOG_PATH"
echo "Segment duration: $SEGMENT_SECONDS seconds"

IFS=, read -r SOURCE_WIDTH SOURCE_HEIGHT < <(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$INPUT_PATH"
)
[[ "$SOURCE_WIDTH" =~ ^[0-9]+$ && "$SOURCE_HEIGHT" =~ ^[0-9]+$ ]] || {
  echo "error: unable to read input dimensions" >&2
  exit 1
}
(( SOURCE_WIDTH % 2 == 0 )) || {
  echo "error: SBS input width must be even: $SOURCE_WIDTH" >&2
  exit 1
}

EYE_WIDTH=$((SOURCE_WIDTH / 2))
if [[ "$EYE" == "left" ]]; then
  CROP_X=0
else
  CROP_X="$EYE_WIDTH"
fi
echo "SBS canvas: ${SOURCE_WIDTH}x${SOURCE_HEIGHT}; selected eye: ${EYE_WIDTH}x${SOURCE_HEIGHT}"

video_duration() {
  local candidate="$1"
  local duration
  duration="$("$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 \
    "$candidate" 2>/dev/null)" || return 1
  if [[ -z "$duration" || "$duration" == "N/A" ]]; then
    duration="$("$FFPROBE_PATH" -v error -show_entries format=duration \
      -of default=noprint_wrappers=1:nokey=1 "$candidate" 2>/dev/null)" || return 1
  fi
  [[ "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  echo "$duration"
}

SOURCE_DURATION="$(video_duration "$INPUT_PATH")" || {
  echo "error: unable to read input video duration" >&2
  exit 1
}

video_duration_matches() {
  local candidate="$1"
  local expected="$2"
  [[ -s "$candidate" ]] || return 1
  local candidate_duration
  candidate_duration="$(video_duration "$candidate")" || return 1
  /usr/bin/awk -v expected="$expected" -v candidate="$candidate_duration" \
    'BEGIN { delta = expected - candidate; if (delta < 0) delta = -delta; exit !(delta <= 0.01) }'
}

completed_eye_output() {
  local candidate="$1"
  video_duration_matches "$candidate" "$SOURCE_DURATION" || return 1
  local candidate_width candidate_height
  IFS=, read -r candidate_width candidate_height < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=width,height -of csv=p=0 "$candidate"
  )
  [[ "$candidate_width" == "$EYE_WIDTH" && "$candidate_height" == "$SOURCE_HEIGHT" ]]
}

if [[ ! -f "$SOURCE_DONE" ]]; then
  if [[ -d "$SOURCE_DIR" ]]; then
    ARCHIVED_SOURCE_DIR="$WORK_DIR/source.interrupted-$(date '+%Y%m%d-%H%M%S')"
    mv "$SOURCE_DIR" "$ARCHIVED_SOURCE_DIR"
    echo "Archived incomplete source split: $ARCHIVED_SOURCE_DIR"
  fi
  mkdir -p "$SOURCE_DIR"
  echo "Stage 1/3: decoding, cropping, and writing physical $EYE-eye source segments"
  "$FFMPEG_PATH" \
    -hide_banner \
    -i "$INPUT_PATH" \
    -map '0:v:0' \
    -vf "crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:${CROP_X}:0,fps=30" \
    -an \
    -c:v hevc_videotoolbox \
    -pix_fmt yuv420p \
    -b:v "$EYE_BITRATE" \
    -maxrate "$((EYE_BITRATE * 3 / 2))" \
    -bufsize "$((EYE_BITRATE * 3))" \
    -g 30 \
    -force_key_frames "expr:gte(t,n_forced*${SEGMENT_SECONDS})" \
    -tag:v hvc1 \
    -f segment \
    -segment_format mov \
    -segment_time "$SEGMENT_SECONDS" \
    -segment_time_delta 0.016667 \
    -reset_timestamps 1 \
    "$SOURCE_DIR/${EYE}-%05d.mov"
  /usr/bin/touch "$SOURCE_DONE"
else
  echo "Stage 1/3: physical $EYE-eye source segments already complete"
fi

SOURCE_SEGMENTS=("$SOURCE_DIR"/"$EYE"-*.mov)
[[ -e "${SOURCE_SEGMENTS[0]}" ]] || {
  echo "error: no $EYE-eye source segments were produced" >&2
  exit 1
}

echo "Stage 2/3: restoring ${#SOURCE_SEGMENTS[@]} $EYE-eye segment(s)"
RESTORED_SEGMENTS=()
for SOURCE_SEGMENT in "${SOURCE_SEGMENTS[@]}"; do
  SEGMENT_NAME="$(basename "$SOURCE_SEGMENT")"
  SEGMENT_STEM="${SEGMENT_NAME%.*}"
  RESTORED_SEGMENT="$RESTORED_DIR/${SEGMENT_STEM}-restored.mov"
  SEGMENT_DONE="$RESTORED_DIR/${SEGMENT_STEM}.done"
  SEGMENT_CACHE="$CACHE_DIR/$SEGMENT_STEM.jasna-work"
  SEGMENT_WINDOWS="$RESTORED_DIR/${SEGMENT_STEM}.windows"
  SEGMENT_MANIFEST="$RESTORED_DIR/${SEGMENT_STEM}-windows.txt"
  TEMP_RESTORED_SEGMENT="$RESTORED_DIR/.${SEGMENT_STEM}-joining.mov"
  SEGMENT_DURATION="$(video_duration "$SOURCE_SEGMENT")"

  if [[ ! -f "$SEGMENT_DONE" ]] && video_duration_matches "$RESTORED_SEGMENT" "$SEGMENT_DURATION"; then
    echo "Recovered completed marker for $SEGMENT_NAME"
    /usr/bin/touch "$SEGMENT_DONE"
  fi

  if [[ ! -f "$SEGMENT_DONE" ]]; then
    echo "Restoring $SEGMENT_NAME (${SEGMENT_DURATION}s)"
    mkdir -p "$SEGMENT_CACHE" "$SEGMENT_WINDOWS"
    JASNA_WORK_DIR="$SEGMENT_CACHE" \
      "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows \
        "$SOURCE_SEGMENT" "$SEGMENT_WINDOWS"

    WINDOW_OUTPUTS=("$SEGMENT_WINDOWS"/window-[0-9][0-9][0-9][0-9][0-9].mov)
    [[ -e "${WINDOW_OUTPUTS[0]}" ]] || {
      echo "error: no restored windows were produced for $SEGMENT_NAME" >&2
      exit 1
    }
    : > "$SEGMENT_MANIFEST"
    for WINDOW_OUTPUT in "${WINDOW_OUTPUTS[@]}"; do
      ESCAPED_WINDOW="${WINDOW_OUTPUT//\'/\'\\\'\'}"
      printf "file '%s'\n" "$ESCAPED_WINDOW" >> "$SEGMENT_MANIFEST"
    done
    if [[ -e "$TEMP_RESTORED_SEGMENT" ]]; then
      TEMP_SEGMENT_ARCHIVE="$RESTORED_DIR/${SEGMENT_STEM}-joining.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
      mv "$TEMP_RESTORED_SEGMENT" "$TEMP_SEGMENT_ARCHIVE"
    fi
    "$FFMPEG_PATH" \
      -hide_banner \
      -f concat \
      -safe 0 \
      -i "$SEGMENT_MANIFEST" \
      -map '0:v:0' \
      -c copy \
      -movflags +faststart \
      -n \
      "$TEMP_RESTORED_SEGMENT"
    video_duration_matches "$TEMP_RESTORED_SEGMENT" "$SEGMENT_DURATION" || {
      echo "error: restored duration does not match $SEGMENT_NAME" >&2
      exit 1
    }
    if [[ -e "$RESTORED_SEGMENT" ]]; then
      PREVIOUS_SEGMENT="$RESTORED_DIR/${SEGMENT_STEM}-restored.previous-$(date '+%Y%m%d-%H%M%S').mov"
      mv "$RESTORED_SEGMENT" "$PREVIOUS_SEGMENT"
      echo "Archived previous restored segment: $PREVIOUS_SEGMENT"
    fi
    mv "$TEMP_RESTORED_SEGMENT" "$RESTORED_SEGMENT"
    /usr/bin/touch "$SEGMENT_DONE"
  else
    echo "Skipping completed segment: $SEGMENT_NAME"
  fi
  RESTORED_SEGMENTS+=("$RESTORED_SEGMENT")
done

if completed_eye_output "$OUTPUT_PATH"; then
  echo "Stage 3/3: joined $EYE-eye output already complete"
  echo "Segmented $EYE-eye restoration: PASS"
  echo "Output: $OUTPUT_PATH"
  echo "Persistent work and every source/restored segment: $WORK_DIR"
  exit 0
fi

: > "$MANIFEST_PATH"
for RESTORED_SEGMENT in "${RESTORED_SEGMENTS[@]}"; do
  ESCAPED_SEGMENT="${RESTORED_SEGMENT//\'/\'\\\'\'}"
  printf "file '%s'\n" "$ESCAPED_SEGMENT" >> "$MANIFEST_PATH"
done

if [[ -e "$TEMP_OUTPUT" ]]; then
  TEMP_ARCHIVE="$WORK_DIR/${EYE}-joined.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
  mv "$TEMP_OUTPUT" "$TEMP_ARCHIVE"
  echo "Archived incomplete joined eye video: $TEMP_ARCHIVE"
fi

echo "Stage 3/3: joining restored $EYE-eye segments without re-encoding"
"$FFMPEG_PATH" \
  -hide_banner \
  -f concat \
  -safe 0 \
  -i "$MANIFEST_PATH" \
  -map '0:v:0' \
  -c copy \
  -movflags +faststart \
  -n \
  "$TEMP_OUTPUT"

video_duration_matches "$TEMP_OUTPUT" "$SOURCE_DURATION" || {
  echo "error: joined eye duration does not match the SBS source" >&2
  exit 1
}

if [[ -e "$OUTPUT_PATH" ]]; then
  OUTPUT_ARCHIVE="$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').mov"
  mv "$OUTPUT_PATH" "$OUTPUT_ARCHIVE"
  echo "Archived previous output: $OUTPUT_ARCHIVE"
fi
mv "$TEMP_OUTPUT" "$OUTPUT_PATH"

FINAL_INFO="$("$FFPROBE_PATH" \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  "$OUTPUT_PATH")"

echo "Segmented $EYE-eye restoration: PASS"
echo "$FINAL_INFO"
echo "Output: $OUTPUT_PATH"
echo "Persistent work and every source/restored segment: $WORK_DIR"
