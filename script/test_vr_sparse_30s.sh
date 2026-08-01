#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-test.mov 00:12:00" >&2
  echo "optional: JASNA_TEST_SECONDS=30 JASNA_EYE_BITRATE=20000000 JASNA_VR_BITRATE=40000000" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

INPUT_PATH="$1"
OUTPUT_PATH="$2"
START_TIME="${3:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_SECONDS="${JASNA_TEST_SECONDS:-30}"
EYE_BITRATE="${JASNA_EYE_BITRATE:-20000000}"
VR_BITRATE="${JASNA_VR_BITRATE:-40000000}"
FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
FAST_SOURCE_COPY="${JASNA_FAST_SOURCE_COPY:-auto}"

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
[[ "$TEST_SECONDS" =~ ^[0-9]+$ ]] && (( TEST_SECONDS >= 1 && TEST_SECONDS <= 120 )) || {
  echo "error: JASNA_TEST_SECONDS must be an integer from 1 to 120" >&2
  exit 1
}
[[ "$EYE_BITRATE" =~ ^[0-9]+$ && "$VR_BITRATE" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_EYE_BITRATE and JASNA_VR_BITRATE must be integer bit rates" >&2
  exit 1
}
[[ "$FAST_ENCODE" == "0" || "$FAST_ENCODE" == "1" ]] || {
  echo "error: JASNA_FAST_ENCODE must be 0 or 1" >&2
  exit 1
}
[[ "$FAST_SOURCE_COPY" == "auto" || "$FAST_SOURCE_COPY" == "0" || "$FAST_SOURCE_COPY" == "1" ]] || {
  echo "error: JASNA_FAST_SOURCE_COPY must be auto, 0, or 1" >&2
  exit 1
}

ENCODER_SPEED_ARGS=()
if [[ "$FAST_ENCODE" == "1" ]]; then
  ENCODER_SPEED_ARGS=(-realtime 1 -prio_speed 1)
fi

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

mkdir -p "$(dirname "$OUTPUT_PATH")"
INPUT_PATH="$(cd "$(dirname "$INPUT_PATH")" && pwd)/$(basename "$INPUT_PATH")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.jasna-vr30-work"
SOURCE_DIR="$WORK_DIR/source"
RUN_CONFIG_PATH="$WORK_DIR/run-config.txt"
TEST_INPUT="$SOURCE_DIR/test-sbs-30fps.mov"
TEST_INPUT_TEMP="$SOURCE_DIR/.test-sbs-30fps-writing.mov"
TEST_INPUT_DONE="$SOURCE_DIR/test-sbs-30fps.done"
LEFT_OUTPUT="$WORK_DIR/left-restored.mov"
RIGHT_OUTPUT="$WORK_DIR/right-restored.mov"
FINAL_TEMP="$WORK_DIR/.joined-sbs-writing.${OUTPUT_NAME##*.}"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.jasna-vr30.log"
SHARED_BATCH_PATH="$WORK_DIR/pending-eye-restorations.tsv"

mkdir -p "$SOURCE_DIR"
RUN_CONFIG="input=$INPUT_PATH
start=$START_TIME
seconds=$TEST_SECONDS
eye_bitrate=$EYE_BITRATE
vr_bitrate=$VR_BITRATE
fast_encode=$FAST_ENCODE
fast_source_copy=$FAST_SOURCE_COPY
projection=fisheye"
if [[ -s "$RUN_CONFIG_PATH" && "$(<"$RUN_CONFIG_PATH")" != "$RUN_CONFIG" ]]; then
  echo "error: this output path belongs to a different test configuration" >&2
  echo "use a new output filename, or restore the original input/start/settings" >&2
  exit 1
fi
if [[ ! -s "$RUN_CONFIG_PATH" ]]; then
  printf '%s\n' "$RUN_CONFIG" > "$RUN_CONFIG_PATH"
fi
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== Jasna sparse VR test $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:       $INPUT_PATH"
echo "Start:       $START_TIME"
echo "Test length: $TEST_SECONDS seconds"
echo "Output:      $OUTPUT_PATH"
echo "Work dir:    $WORK_DIR"
echo "Log:         $LOG_PATH"
echo "Projection:  fisheye"
echo "Fast encode: $FAST_ENCODE"

video_duration() {
  "$FFPROBE_PATH" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

duration_matches() {
  local candidate="$1"
  local expected="$2"
  [[ -s "$candidate" ]] || return 1
  local candidate_duration
  candidate_duration="$(video_duration "$candidate")" || return 1
  [[ "$candidate_duration" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  /usr/bin/awk -v expected="$expected" -v candidate="$candidate_duration" \
    'BEGIN { delta = expected - candidate; if (delta < 0) delta = -delta; exit !(delta <= 0.05) }'
}

IFS=, read -r SOURCE_WIDTH SOURCE_HEIGHT < <(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$INPUT_PATH"
)
[[ "$SOURCE_WIDTH" =~ ^[0-9]+$ && "$SOURCE_HEIGHT" =~ ^[0-9]+$ ]] || {
  echo "error: unable to read source dimensions" >&2
  exit 1
}
(( SOURCE_WIDTH % 2 == 0 )) || {
  echo "error: SBS input width must be even: $SOURCE_WIDTH" >&2
  exit 1
}
echo "SBS canvas: ${SOURCE_WIDTH}x${SOURCE_HEIGHT}; each eye: $((SOURCE_WIDTH / 2))x${SOURCE_HEIGHT}"

SOURCE_FRAME_RATE="$("$FFPROBE_PATH" -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT_PATH")"
SOURCE_IS_30_FPS=0
if /usr/bin/awk -F/ '
  NF == 2 && $2 != 0 { rate = $1 / $2 }
  NF == 1 { rate = $1 }
  END { exit !(rate >= 29.95 && rate <= 30.05) }
' <<< "$SOURCE_FRAME_RATE"; then
  SOURCE_IS_30_FPS=1
fi
START_IS_ZERO=0
if [[ "$START_TIME" =~ ^(0+([.]0+)?|00:00:00([.]0+)?)$ ]]; then
  START_IS_ZERO=1
fi
USE_FAST_SOURCE_COPY=0
if [[ "$FAST_SOURCE_COPY" == "1" ]] \
  || [[ "$FAST_SOURCE_COPY" == "auto" && "$SOURCE_IS_30_FPS" == "1" && "$START_IS_ZERO" == "1" ]]; then
  USE_FAST_SOURCE_COPY=1
fi

if [[ ! -f "$TEST_INPUT_DONE" ]]; then
  if [[ -e "$TEST_INPUT_TEMP" ]]; then
    mv "$TEST_INPUT_TEMP" "$SOURCE_DIR/test-sbs.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
  fi
  if [[ -e "$TEST_INPUT" ]]; then
    mv "$TEST_INPUT" "$SOURCE_DIR/test-sbs.previous-$(date '+%Y%m%d-%H%M%S').mov"
  fi

  if [[ "$USE_FAST_SOURCE_COPY" == "1" ]]; then
    echo "Stage 1/4: copying the existing 30 fps SBS packets without re-encoding"
    "$FFMPEG_PATH" \
      -hide_banner \
      -ss "$START_TIME" \
      -i "$INPUT_PATH" \
      -t "$TEST_SECONDS" \
      -map '0:v:0' \
      -map '0:a?' \
      -c copy \
      -movflags +faststart \
      -n \
      "$TEST_INPUT_TEMP"
  else
    echo "Stage 1/4: preparing a hardware-encoded 30 fps SBS test clip"
    "$FFMPEG_PATH" \
      -hide_banner \
      -ss "$START_TIME" \
      -i "$INPUT_PATH" \
      -t "$TEST_SECONDS" \
      -map '0:v:0' \
      -map '0:a?' \
      -vf fps=30 \
      -c:v hevc_videotoolbox \
      "${ENCODER_SPEED_ARGS[@]}" \
      -pix_fmt yuv420p \
      -b:v "$VR_BITRATE" \
      -maxrate "$((VR_BITRATE * 3 / 2))" \
      -bufsize "$((VR_BITRATE * 3))" \
      -g 30 \
      -tag:v hvc1 \
      -c:a aac \
      -b:a 256k \
      -movflags +faststart \
      -n \
      "$TEST_INPUT_TEMP"
  fi

  TEST_DURATION="$(video_duration "$TEST_INPUT_TEMP")"
  [[ "$TEST_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "error: unable to validate prepared test clip" >&2
    exit 1
  }
  /usr/bin/awk -v duration="$TEST_DURATION" \
    'BEGIN { exit !(duration >= 0.5) }' || {
      echo "error: the selected range did not contain enough video" >&2
      exit 1
    }
  mv "$TEST_INPUT_TEMP" "$TEST_INPUT"
  /usr/bin/touch "$TEST_INPUT_DONE"
else
  [[ -s "$TEST_INPUT" ]] || {
    echo "error: test source marker exists but the test clip is missing" >&2
    exit 1
  }
  echo "Stage 1/4: prepared SBS test clip already complete"
fi

TEST_DURATION="$(video_duration "$TEST_INPUT")"

if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
  for CANDIDATE in \
    "$ROOT_DIR/.build/out/Products/Release/JasnaMetalPoC" \
    "$ROOT_DIR/.build/release/JasnaMetalPoC" \
    "$ROOT_DIR/.build/arm64-apple-macosx/release/JasnaMetalPoC" \
    "$ROOT_DIR/.build/x86_64-apple-macosx/release/JasnaMetalPoC"
  do
    [[ -x "$CANDIDATE" ]] || continue
    BINARY_IS_STALE=0
    if [[ "$ROOT_DIR/Package.swift" -nt "$CANDIDATE" ]]; then
      BINARY_IS_STALE=1
    fi
    while IFS= read -r SOURCE_FILE; do
      if [[ "$SOURCE_FILE" -nt "$CANDIDATE" ]]; then
        BINARY_IS_STALE=1
        break
      fi
    done < <(find "$ROOT_DIR/Sources" -type f -print)
    if [[ "$BINARY_IS_STALE" == "0" ]]; then
      JASNA_APP_BINARY="$CANDIDATE"
      export JASNA_APP_BINARY
      echo "Reusing fresh optimized Swift executable: $JASNA_APP_BINARY"
      break
    fi
  done
fi

if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
  echo "Building one shared optimized Swift executable for both eyes"
  mkdir -p "$ROOT_DIR/.build/ModuleCache"
  export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
  export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
  (
    cd "$ROOT_DIR"
    swift build --disable-sandbox -c release
  )
  JASNA_APP_BINARY="$(
    cd "$ROOT_DIR"
    swift build --disable-sandbox -c release --show-bin-path
  )/JasnaMetalPoC"
  [[ -x "$JASNA_APP_BINARY" ]] || {
    echo "error: optimized JasnaMetalPoC executable was not produced" >&2
    exit 1
  }
  export JASNA_APP_BINARY
fi

echo "Stage 2/4: preparing mosaic regions for both eyes"
: > "$SHARED_BATCH_PATH"
JASNA_SPARSE_BATCH_MODE=prepare \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS=30 \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" left "$LEFT_OUTPUT"

JASNA_SPARSE_BATCH_MODE=prepare \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS=30 \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" right "$RIGHT_OUTPUT"

SHARED_BATCH_ARGS=()
while IFS=$'\t' read -r JOB_INPUT JOB_WINDOWS JOB_MANIFEST JOB_CACHE JOB_EXTRA; do
  [[ -n "$JOB_INPUT" ]] || continue
  [[ -n "$JOB_WINDOWS" && -n "$JOB_MANIFEST" && -n "$JOB_CACHE" && -z "$JOB_EXTRA" ]] || {
    echo "error: malformed coordinated restoration entry" >&2
    exit 1
  }
  SHARED_BATCH_ARGS+=("$JOB_INPUT" "$JOB_WINDOWS" "$JOB_MANIFEST" "$JOB_CACHE")
done < "$SHARED_BATCH_PATH"

if (( ${#SHARED_BATCH_ARGS[@]} > 0 )); then
  echo "Restoring $((${#SHARED_BATCH_ARGS[@]} / 4)) left/right segment job(s) with one retained Metal ML graph"
  JASNA_VR_PROJECTION=fisheye \
    "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows-sparse-batch \
      "${SHARED_BATCH_ARGS[@]}"
else
  echo "All left/right restoration windows are already complete"
fi

echo "Stage 3/4: finalizing independently restartable left and right eyes"
JASNA_SPARSE_BATCH_MODE=finalize \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS=30 \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" left "$LEFT_OUTPUT"

JASNA_SPARSE_BATCH_MODE=finalize \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS=30 \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" right "$RIGHT_OUTPUT"

if duration_matches "$OUTPUT_PATH" "$TEST_DURATION" \
    && [[ "$OUTPUT_PATH" -nt "$LEFT_OUTPUT" && "$OUTPUT_PATH" -nt "$RIGHT_OUTPUT" ]]; then
  echo "Stage 4/4: combined SBS output already complete"
else
  if [[ -e "$FINAL_TEMP" ]]; then
    mv "$FINAL_TEMP" "$WORK_DIR/joined-sbs.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
  fi
  if [[ -e "$OUTPUT_PATH" ]]; then
    mv "$OUTPUT_PATH" "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
  fi

  echo "Stage 4/4: rebuilding the side-by-side VR preview and copying audio"
  "$FFMPEG_PATH" \
    -hide_banner \
    -i "$LEFT_OUTPUT" \
    -i "$RIGHT_OUTPUT" \
    -i "$TEST_INPUT" \
    -filter_complex '[0:v:0][1:v:0]hstack=inputs=2[v]' \
    -map '[v]' \
    -map '2:a?' \
    -map_metadata 2 \
    -map_chapters 2 \
    -c:v hevc_videotoolbox \
    "${ENCODER_SPEED_ARGS[@]}" \
    -pix_fmt yuv420p \
    -b:v "$VR_BITRATE" \
    -maxrate "$((VR_BITRATE * 3 / 2))" \
    -bufsize "$((VR_BITRATE * 3))" \
    -g 30 \
    -tag:v hvc1 \
    -c:a copy \
    -r 30 \
    -movflags +faststart \
    -shortest \
    -n \
    "$FINAL_TEMP"

  duration_matches "$FINAL_TEMP" "$TEST_DURATION" || {
    echo "error: combined output duration does not match the test source" >&2
    exit 1
  }
  mv "$FINAL_TEMP" "$OUTPUT_PATH"
fi

FINAL_INFO="$("$FFPROBE_PATH" \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  "$OUTPUT_PATH")"

echo "Sparse 30-second VR test: PASS"
echo "$FINAL_INFO"
echo "Output:   $OUTPUT_PATH"
echo "Left eye: $LEFT_OUTPUT"
echo "Right eye:$RIGHT_OUTPUT"
echo "Log:      $LOG_PATH"
echo "All source clips, manifests, windows, and caches remain under: $WORK_DIR"
