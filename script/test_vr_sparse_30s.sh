#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-test.mov 00:12:00" >&2
  echo "optional: JASNA_TEST_SECONDS=30 (1-300, or full)" >&2
  echo "          JASNA_ENCODER_WINDOWS_PER_SEGMENT=4 (bounded eye-by-eye disk use)" >&2
  echo "          JASNA_METAL_WINDOWS_PER_PROCESS=4 (fast; set 2 balanced or 1 minimum memory)" >&2
  echo "          JASNA_EYE_BITRATE=20000000 JASNA_VR_BITRATE=40000000" >&2
  echo "          JASNA_DIRECT_SBS_OUTPUT=1 (set 0 for lower-memory eye-by-eye output)" >&2
  echo "          JASNA_CLEAN_WORK_ON_SUCCESS=0 (set 1 to remove restart data after PASS)" >&2
  echo "          JASNA_LARGE_REGION_MAX_BLEND=768 JASNA_LARGE_REGION_OVERLAP=96" >&2
  echo "          JASNA_LARGE_REGION_MASK_GROWTH=0.05 JASNA_LARGE_REGION_MASK_FEATHER=0.025" >&2
  echo "          JASNA_LARGE_REGION_BLOCK_GROWTH=0.04 JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS=1" >&2
  echo "          JASNA_LARGE_REGION_DETAIL_CROPS=1 JASNA_LARGE_REGION_DETAIL_DIMENSION=576" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

INPUT_PATH="$1"
OUTPUT_PATH="$2"
START_TIME="${3:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/script/restoration_identity.sh"
TEST_SECONDS="${JASNA_TEST_SECONDS:-30}"
EYE_BITRATE="${JASNA_EYE_BITRATE:-20000000}"
VR_BITRATE="${JASNA_VR_BITRATE:-40000000}"
FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
FAST_SOURCE_COPY="${JASNA_FAST_SOURCE_COPY:-auto}"
DIRECT_SBS_OUTPUT="${JASNA_DIRECT_SBS_OUTPUT:-1}"
METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-4}"
MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
CLEAN_WORK_ON_SUCCESS="${JASNA_CLEAN_WORK_ON_SUCCESS:-0}"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"

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
if [[ "$TEST_SECONDS" == "full" ]]; then
  TEST_SEGMENT_SECONDS=120
  DURATION_ARGS=()
  RUN_DESCRIPTION="full source"
  ARTIFACT_TAG="jasna-vr-full-v15"
else
  [[ "$TEST_SECONDS" =~ ^[0-9]+$ ]] \
    && (( TEST_SECONDS >= 1 && TEST_SECONDS <= 300 )) || {
      echo "error: JASNA_TEST_SECONDS must be an integer from 1 to 300, or full" >&2
      exit 1
    }
  TEST_SEGMENT_SECONDS="$TEST_SECONDS"
  (( TEST_SEGMENT_SECONDS < 30 )) && TEST_SEGMENT_SECONDS=30
  (( TEST_SEGMENT_SECONDS > 120 )) && TEST_SEGMENT_SECONDS=120
  DURATION_ARGS=(-t "$TEST_SECONDS")
  RUN_DESCRIPTION="$TEST_SECONDS seconds"
  ARTIFACT_TAG="jasna-vr30-v15"
fi
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
[[ "$DIRECT_SBS_OUTPUT" == "0" || "$DIRECT_SBS_OUTPUT" == "1" ]] || {
  echo "error: JASNA_DIRECT_SBS_OUTPUT must be 0 or 1" >&2
  exit 1
}
[[ "$METAL_WINDOWS_PER_PROCESS" =~ ^[0-9]+$ ]] \
  && (( METAL_WINDOWS_PER_PROCESS >= 1 && METAL_WINDOWS_PER_PROCESS <= 30 )) || {
    echo "error: JASNA_METAL_WINDOWS_PER_PROCESS must be an integer from 1 to 30" >&2
    exit 1
  }
[[ "$MODEL_BATCH" == "1" || "$MODEL_BATCH" == "2" ]] || {
  echo "error: JASNA_MODEL_BATCH must be 1 or 2" >&2
  exit 1
}
[[ "$CLEAN_WORK_ON_SUCCESS" == "0" || "$CLEAN_WORK_ON_SUCCESS" == "1" ]] || {
  echo "error: JASNA_CLEAN_WORK_ON_SUCCESS must be 0 or 1" >&2
  exit 1
}
export JASNA_MODEL_BATCH="$MODEL_BATCH"

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
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.${ARTIFACT_TAG}-work"
SOURCE_DIR="$WORK_DIR/source"
RUN_CONFIG_PATH="$WORK_DIR/run-config.txt"
TEST_INPUT="$SOURCE_DIR/test-sbs-30fps.mov"
TEST_INPUT_TEMP="$SOURCE_DIR/.test-sbs-30fps-writing.mov"
TEST_INPUT_DONE="$SOURCE_DIR/test-sbs-30fps.done"
LEFT_OUTPUT="$WORK_DIR/left-restored.mov"
RIGHT_OUTPUT="$WORK_DIR/right-restored.mov"
LEFT_EYE_WORK_DIR="$WORK_DIR/left-restored.left-segments-work"
RIGHT_EYE_WORK_DIR="$WORK_DIR/right-restored.right-segments-work"
FINAL_TEMP="$WORK_DIR/.joined-sbs-writing.${OUTPUT_NAME##*.}"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.${ARTIFACT_TAG}.log"
SHARED_BATCH_PATH="$WORK_DIR/pending-eye-restorations.tsv"
DIRECT_SEGMENT_DIR="$WORK_DIR/direct-sbs-segments"
DIRECT_CONCAT_PATH="$WORK_DIR/direct-sbs-concat.txt"
WORKFLOW_LOCK="$WORK_DIR/.jasna-workflow-lock"

mkdir -p "$WORK_DIR"
if ! mkdir "$WORKFLOW_LOCK" 2>/dev/null; then
  EXISTING_WORKFLOW_PID="$(/bin/cat "$WORKFLOW_LOCK/pid" 2>/dev/null || true)"
  if [[ "$EXISTING_WORKFLOW_PID" =~ ^[0-9]+$ ]] \
    && kill -0 "$EXISTING_WORKFLOW_PID" 2>/dev/null; then
    echo "error: this restoration workflow is already active (PID $EXISTING_WORKFLOW_PID)" >&2
    echo "work dir: $WORK_DIR" >&2
    exit 1
  fi
  if [[ ! "$EXISTING_WORKFLOW_PID" =~ ^[0-9]+$ ]]; then
    echo "error: workflow lock exists without a valid owner PID" >&2
    echo "lock: $WORKFLOW_LOCK" >&2
    exit 1
  fi
  STALE_WORKFLOW_LOCK="$WORK_DIR/.jasna-workflow-lock.stale-$(date '+%Y%m%d-%H%M%S')-$$"
  mv "$WORKFLOW_LOCK" "$STALE_WORKFLOW_LOCK"
  mkdir "$WORKFLOW_LOCK"
fi
printf '%s\n' "$$" > "$WORKFLOW_LOCK/pid"
cleanup_workflow_lock() {
  [[ -d "$WORKFLOW_LOCK" ]] || return 0
  local owner_pid
  owner_pid="$(/bin/cat "$WORKFLOW_LOCK/pid" 2>/dev/null || true)"
  [[ "$owner_pid" == "$$" ]] || return 0
  rm -f "$WORKFLOW_LOCK/pid"
  rmdir "$WORKFLOW_LOCK" 2>/dev/null || true
}
trap cleanup_workflow_lock EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

cleanup_successful_work() {
  if [[ "$CLEAN_WORK_ON_SUCCESS" == "0" ]]; then
    echo "Persistent segments and caches: $WORK_DIR"
    return
  fi
  if [[ ! -d "$WORK_DIR" || "$WORK_DIR" == "$OUTPUT_DIR" \
    || "$WORK_DIR" != "$OUTPUT_DIR/"* || "$WORK_DIR" != *.jasna-vr*-work ]]; then
    echo "error: refusing unsafe successful-run cleanup path: $WORK_DIR" >&2
    return 1
  fi
  echo "Final output passed validation; removing restart data: $WORK_DIR"
  /bin/rm -rf -- "$WORK_DIR"
  echo "Successful-run restart data removed"
}

mkdir -p "$SOURCE_DIR"
SOURCE_FINGERPRINT="$(jasna_source_fingerprint "$INPUT_PATH")"
IMPLEMENTATION_FINGERPRINT="$(jasna_implementation_fingerprint "$ROOT_DIR")"
MODEL_FINGERPRINT="$(jasna_model_fingerprint "$ROOT_DIR" "$DETECTOR")"
RUN_CONFIG="input=$INPUT_PATH
source_fingerprint=$SOURCE_FINGERPRINT
implementation_fingerprint=$IMPLEMENTATION_FINGERPRINT
model_fingerprint=$MODEL_FINGERPRINT
start=$START_TIME
seconds=$TEST_SECONDS
eye_bitrate=$EYE_BITRATE
vr_bitrate=$VR_BITRATE
fast_encode=$FAST_ENCODE
fast_source_copy=$FAST_SOURCE_COPY
direct_sbs_output=$DIRECT_SBS_OUTPUT
model_batch=$MODEL_BATCH
metal_windows_per_process=$METAL_WINDOWS_PER_PROCESS
diagnostic_full_region_blend=${JASNA_DIAGNOSTIC_FULL_REGION_BLEND:-0}
metal_texture_compositor=${JASNA_METAL_TEXTURE_COMPOSITOR:-1}
metal_compositor=${JASNA_METAL_COMPOSITOR:-1}
quality_profile=lower-detail-crop-v15
detect_batch_size=${JASNA_DETECT_BATCH_SIZE:-2}
detect_decode_mode=${JASNA_DETECT_DECODE_MODE:-sequential}
rfdetr_max_detections=${JASNA_RFDETR_MAX_DETECTIONS:-64}
detect_confidence=${JASNA_DETECT_CONFIDENCE:-0.15}
temporal_padding=${JASNA_TEMPORAL_PADDING:-1.0}
region_nms_iou=${JASNA_REGION_NMS_IOU:-0.45}
mask_expansion=${JASNA_MASK_EXPANSION:-0.10}
mask_size=${JASNA_MASK_SIZE:-128}
region_duration=${JASNA_REGION_DURATION:-1.0}
large_region_max_blend=${JASNA_LARGE_REGION_MAX_BLEND:-768}
large_region_overlap=${JASNA_LARGE_REGION_OVERLAP:-96}
large_region_split_limit=${JASNA_LARGE_REGION_SPLIT_LIMIT:-1}
large_region_max_axis_crops=${JASNA_LARGE_REGION_MAX_AXIS_CROPS:-3}
large_region_mask_growth=${JASNA_LARGE_REGION_MASK_GROWTH:-0.05}
large_region_mask_feather=${JASNA_LARGE_REGION_MASK_FEATHER:-0.025}
large_region_block_growth=${JASNA_LARGE_REGION_BLOCK_GROWTH:-0.04}
large_region_mask_temporal_radius=${JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS:-1}
large_region_detail_crops=${JASNA_LARGE_REGION_DETAIL_CROPS:-1}
large_region_detail_dimension=${JASNA_LARGE_REGION_DETAIL_DIMENSION:-576}
projection=fisheye
detector=$DETECTOR"
if [[ -s "$RUN_CONFIG_PATH" ]]; then
  EXISTING_STABLE_CONFIG="$(
    /usr/bin/sed '/^metal_windows_per_process=/d' "$RUN_CONFIG_PATH"
  )"
  if ! /usr/bin/grep -q '^detector=' "$RUN_CONFIG_PATH"; then
    EXISTING_STABLE_CONFIG="$EXISTING_STABLE_CONFIG
detector=yolo-v2-fast"
  fi
  CURRENT_STABLE_CONFIG="$(
    printf '%s\n' "$RUN_CONFIG" | /usr/bin/sed '/^metal_windows_per_process=/d'
  )"
  if [[ "$EXISTING_STABLE_CONFIG" != "$CURRENT_STABLE_CONFIG" ]]; then
    echo "error: this output path belongs to a different test configuration" >&2
    echo "use a new output filename, or restore the original input/start/settings" >&2
    exit 1
  fi
fi
RUN_CONFIG_TEMP="$WORK_DIR/.run-config-writing-$$"
printf '%s\n' "$RUN_CONFIG" > "$RUN_CONFIG_TEMP"
mv "$RUN_CONFIG_TEMP" "$RUN_CONFIG_PATH"
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== Jasna sparse VR test $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:       $INPUT_PATH"
echo "Start:       $START_TIME"
echo "Selected range: $RUN_DESCRIPTION"
echo "Output:      $OUTPUT_PATH"
echo "Work dir:    $WORK_DIR"
echo "Log:         $LOG_PATH"
echo "Projection:  fisheye"
echo "Fast encode: $FAST_ENCODE"
echo "Direct SBS:  $DIRECT_SBS_OUTPUT"
echo "Model batch: $MODEL_BATCH"
echo "Detector:    $DETECTOR"

video_duration() {
  "$FFPROBE_PATH" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

video_stream_duration() {
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=duration \
    -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

video_frame_count() {
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=nb_frames \
    -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

duration_matches() {
  local candidate="$1"
  local expected="$2"
  [[ -s "$candidate" ]] || return 1
  local candidate_duration
  candidate_duration="$(video_stream_duration "$candidate")" || return 1
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
EYE_WIDTH=$((SOURCE_WIDTH / 2))
echo "SBS canvas: ${SOURCE_WIDTH}x${SOURCE_HEIGHT}; each eye: ${EYE_WIDTH}x${SOURCE_HEIGHT}"

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
      ${DURATION_ARGS[@]+"${DURATION_ARGS[@]}"} \
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
      ${DURATION_ARGS[@]+"${DURATION_ARGS[@]}"} \
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
TEST_VIDEO_DURATION="$(video_stream_duration "$TEST_INPUT")"
[[ "$TEST_VIDEO_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "error: unable to read prepared source video duration" >&2
  exit 1
}
EXPECTED_FRAME_COUNT="$(video_frame_count "$TEST_INPUT")"
if [[ ! "$EXPECTED_FRAME_COUNT" =~ ^[0-9]+$ ]]; then
  EXPECTED_FRAME_COUNT="$(
    /usr/bin/awk -v duration="$TEST_VIDEO_DURATION" \
      'BEGIN { printf "%d\n", int(duration * 30 + 0.5) }'
  )"
fi

completed_sbs_output() {
  local candidate="$1"
  duration_matches "$candidate" "$TEST_VIDEO_DURATION" || return 1
  local codec width height frame_rate frame_count extra
  IFS=, read -r codec width height frame_rate frame_count extra < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$candidate" 2>/dev/null
  )
  [[ -z "$extra" \
    && "$codec" == "hevc" \
    && "$width" == "$SOURCE_WIDTH" \
    && "$height" == "$SOURCE_HEIGHT" \
    && "$frame_count" =~ ^[0-9]+$ ]] || return 1
  /usr/bin/awk -F/ '
    NF == 2 && $2 != 0 { rate = $1 / $2 }
    NF == 1 { rate = $1 }
    END { exit !(rate >= 29.99 && rate <= 30.01) }
  ' <<< "$frame_rate" || return 1
  /usr/bin/awk -v actual="$frame_count" -v expected="$EXPECTED_FRAME_COUNT" '
    BEGIN { delta = actual - expected; if (delta < 0) delta = -delta; exit !(delta <= 1) }
  ' || return 1
  "$FFMPEG_PATH" -v error -i "$candidate" -map '0:v:0' \
    -frames:v 1 -f null - </dev/null >/dev/null 2>&1
}

valid_direct_segment() {
  local candidate="$1"
  local expected_frames="$2"
  [[ -s "$candidate" ]] || return 1
  local codec width height frame_rate frame_count extra
  IFS=, read -r codec width height frame_rate frame_count extra < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$candidate" 2>/dev/null
  )
  [[ -z "$extra" \
    && "$codec" == "hevc" \
    && "$width" == "$SOURCE_WIDTH" \
    && "$height" == "$SOURCE_HEIGHT" \
    && "$frame_count" =~ ^[0-9]+$ ]] || return 1
  /usr/bin/awk -F/ '
    NF == 2 && $2 != 0 { rate = $1 / $2 }
    NF == 1 { rate = $1 }
    END { exit !(rate >= 29.99 && rate <= 30.01) }
  ' <<< "$frame_rate" || return 1
  /usr/bin/awk -v actual="$frame_count" -v expected="$expected_frames" '
    BEGIN { delta = actual - expected; if (delta < 0) delta = -delta; exit !(delta <= 1) }
  ' || return 1
  "$FFMPEG_PATH" -v error -i "$candidate" -map '0:v:0' \
    -frames:v 1 -f null - </dev/null >/dev/null 2>&1
}

reusable_direct_segment() {
  local job_index="$1"
  local window_start="$2"
  local job_window_count="$3"
  local job_frame_count="$4"
  local prefix candidate filename end expected_frames
  local best_end=0
  local best_path=""
  prefix="$(printf 'segment-%05d-windows-%05d-' "$job_index" "$window_start")"
  for candidate in "$DIRECT_SEGMENT_DIR"/"$prefix"*.mov; do
    [[ -e "$candidate" ]] || continue
    filename="$(basename "$candidate")"
    [[ "$filename" =~ ^${prefix}([0-9]{5})[.]mov$ ]] || continue
    end=$((10#${BASH_REMATCH[1]}))
    (( end > window_start && end <= job_window_count )) || continue
    expected_frames=$((end * 30))
    (( expected_frames > job_frame_count )) && expected_frames="$job_frame_count"
    expected_frames=$((expected_frames - window_start * 30))
    if (( end > best_end )) && valid_direct_segment "$candidate" "$expected_frames"; then
      best_end="$end"
      best_path="$candidate"
    fi
  done
  [[ -n "$best_path" ]] || return 1
  printf '%s\t%s\n' "$best_end" "$best_path"
}

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
LEFT_SOURCE_DIR="$LEFT_EYE_WORK_DIR/source"
RIGHT_SOURCE_DIR="$RIGHT_EYE_WORK_DIR/source"
LEFT_SOURCE_DONE="$LEFT_EYE_WORK_DIR/source.done"
RIGHT_SOURCE_DONE="$RIGHT_EYE_WORK_DIR/source.done"
if [[ "$DIRECT_SBS_OUTPUT" == "1" \
  && ! -f "$LEFT_SOURCE_DONE" && ! -f "$RIGHT_SOURCE_DONE" ]]; then
  if [[ -d "$LEFT_SOURCE_DIR" ]]; then
    mv "$LEFT_SOURCE_DIR" \
      "$LEFT_EYE_WORK_DIR/source.interrupted-$(date '+%Y%m%d-%H%M%S')"
  fi
  if [[ -d "$RIGHT_SOURCE_DIR" ]]; then
    mv "$RIGHT_SOURCE_DIR" \
      "$RIGHT_EYE_WORK_DIR/source.interrupted-$(date '+%Y%m%d-%H%M%S')"
  fi
  mkdir -p "$LEFT_SOURCE_DIR" "$RIGHT_SOURCE_DIR"
  echo "Preparing left/right 4K segments with one shared 8K decode"
  "$FFMPEG_PATH" \
    -hide_banner \
    -i "$TEST_INPUT" \
    -filter_complex \
      "[0:v:0]split=2[leftbase][rightbase];[leftbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:0:0[left];[rightbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:${EYE_WIDTH}:0[right]" \
    -map '[left]' \
    -an \
    -c:v hevc_videotoolbox \
    "${ENCODER_SPEED_ARGS[@]}" \
    -pix_fmt yuv420p \
    -b:v "$EYE_BITRATE" \
    -maxrate "$((EYE_BITRATE * 3 / 2))" \
    -bufsize "$((EYE_BITRATE * 3))" \
    -g 30 \
    -force_key_frames "expr:gte(t,n_forced*${TEST_SEGMENT_SECONDS})" \
    -tag:v hvc1 \
    -f segment \
    -segment_format mov \
    -segment_time "$TEST_SEGMENT_SECONDS" \
    -segment_time_delta 0.016667 \
    -reset_timestamps 1 \
    "$LEFT_SOURCE_DIR/left-%05d.mov" \
    -map '[right]' \
    -an \
    -c:v hevc_videotoolbox \
    "${ENCODER_SPEED_ARGS[@]}" \
    -pix_fmt yuv420p \
    -b:v "$EYE_BITRATE" \
    -maxrate "$((EYE_BITRATE * 3 / 2))" \
    -bufsize "$((EYE_BITRATE * 3))" \
    -g 30 \
    -force_key_frames "expr:gte(t,n_forced*${TEST_SEGMENT_SECONDS})" \
    -tag:v hvc1 \
    -f segment \
    -segment_format mov \
    -segment_time "$TEST_SEGMENT_SECONDS" \
    -segment_time_delta 0.016667 \
    -reset_timestamps 1 \
    "$RIGHT_SOURCE_DIR/right-%05d.mov"
  [[ -s "$LEFT_SOURCE_DIR/left-00000.mov" \
    && -s "$RIGHT_SOURCE_DIR/right-00000.mov" ]] || {
      echo "error: shared stereo source preparation did not produce both eyes" >&2
      exit 1
    }
  /usr/bin/touch "$LEFT_SOURCE_DONE" "$RIGHT_SOURCE_DONE"
  echo "Shared stereo source preparation complete"
fi
JASNA_SPARSE_BATCH_MODE=prepare \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" left "$LEFT_OUTPUT"

JASNA_SPARSE_BATCH_MODE=prepare \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" right "$RIGHT_OUTPUT"

SHARED_BATCH_ARGS=()
LEFT_JOB_INPUTS=()
LEFT_JOB_MANIFESTS=()
LEFT_JOB_CACHES=()
RIGHT_JOB_INPUTS=()
RIGHT_JOB_MANIFESTS=()
RIGHT_JOB_CACHES=()
while IFS=$'\t' read -r JOB_INPUT JOB_WINDOWS JOB_MANIFEST JOB_CACHE JOB_EXTRA; do
  [[ -n "$JOB_INPUT" ]] || continue
  [[ -n "$JOB_WINDOWS" && -n "$JOB_MANIFEST" && -n "$JOB_CACHE" && -z "$JOB_EXTRA" ]] || {
    echo "error: malformed coordinated restoration entry" >&2
    exit 1
  }
  SHARED_BATCH_ARGS+=("$JOB_INPUT" "$JOB_WINDOWS" "$JOB_MANIFEST" "$JOB_CACHE")
  case "$(basename "$JOB_INPUT")" in
    left-*)
      LEFT_JOB_INPUTS+=("$JOB_INPUT")
      LEFT_JOB_MANIFESTS+=("$JOB_MANIFEST")
      LEFT_JOB_CACHES+=("$JOB_CACHE")
      ;;
    right-*)
      RIGHT_JOB_INPUTS+=("$JOB_INPUT")
      RIGHT_JOB_MANIFESTS+=("$JOB_MANIFEST")
      RIGHT_JOB_CACHES+=("$JOB_CACHE")
      ;;
    *)
      echo "error: unable to identify eye for coordinated input: $JOB_INPUT" >&2
      exit 1
      ;;
  esac
done < "$SHARED_BATCH_PATH"

if [[ "$DIRECT_SBS_OUTPUT" == "1" ]]; then
  (( ${#LEFT_JOB_INPUTS[@]} == ${#RIGHT_JOB_INPUTS[@]} )) || {
    echo "error: direct SBS restoration requires matching left/right segments" >&2
    exit 1
  }
  (( ${#LEFT_JOB_INPUTS[@]} > 0 )) || {
    echo "error: no paired eye segments were prepared for direct SBS restoration" >&2
    exit 1
  }
  mkdir -p "$DIRECT_SEGMENT_DIR"
  DIRECT_SEGMENTS=()
  echo "Metal process isolation: at most $METAL_WINDOWS_PER_PROCESS temporal windows/process"
  for ((JOB_INDEX = 0; JOB_INDEX < ${#LEFT_JOB_INPUTS[@]}; JOB_INDEX++)); do
    JOB_FRAME_COUNT="$(
      "$FFPROBE_PATH" -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \
        "${LEFT_JOB_INPUTS[$JOB_INDEX]}"
    )"
    if [[ ! "$JOB_FRAME_COUNT" =~ ^[0-9]+$ ]]; then
      JOB_DURATION="$(video_duration "${LEFT_JOB_INPUTS[$JOB_INDEX]}")"
      JOB_FRAME_COUNT="$(
        /usr/bin/awk -v duration="$JOB_DURATION" \
          'BEGIN { printf "%d\n", int(duration * 30 + 0.5) }'
      )"
    fi
    JOB_WINDOW_COUNT=$(( (JOB_FRAME_COUNT + 29) / 30 ))
    BATCH_SEGMENT="$DIRECT_SEGMENT_DIR/$(
      printf 'segment-%05d-windows-%05d-%05d.mov' \
        "$JOB_INDEX" 0 "$JOB_WINDOW_COUNT"
    )"
    if valid_direct_segment "$BATCH_SEGMENT" "$JOB_FRAME_COUNT"; then
      DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
      echo "Reusing validated two-minute SBS batch $((JOB_INDEX + 1))/${#LEFT_JOB_INPUTS[@]}"
      continue
    fi
    JOB_DIRECT_SEGMENTS=()
    WINDOW_START=0
    while (( WINDOW_START < JOB_WINDOW_COUNT )); do
      REUSABLE_SEGMENT="$(
        reusable_direct_segment \
          "$JOB_INDEX" "$WINDOW_START" "$JOB_WINDOW_COUNT" "$JOB_FRAME_COUNT" \
          || true
      )"
      if [[ -n "$REUSABLE_SEGMENT" ]]; then
        IFS=$'\t' read -r REUSABLE_END DIRECT_SEGMENT <<< "$REUSABLE_SEGMENT"
        JOB_DIRECT_SEGMENTS+=("$DIRECT_SEGMENT")
        echo "Reusing validated direct SBS windows $((WINDOW_START + 1))-$REUSABLE_END/$JOB_WINDOW_COUNT"
        WINDOW_START="$REUSABLE_END"
        continue
      fi
      WINDOW_COUNT=$((JOB_WINDOW_COUNT - WINDOW_START))
      (( WINDOW_COUNT > METAL_WINDOWS_PER_PROCESS )) \
        && WINDOW_COUNT="$METAL_WINDOWS_PER_PROCESS"
      WINDOW_END=$((WINDOW_START + WINDOW_COUNT))
      DIRECT_SEGMENT="$DIRECT_SEGMENT_DIR/$(
        printf 'segment-%05d-windows-%05d-%05d.mov' \
          "$JOB_INDEX" "$WINDOW_START" "$WINDOW_END"
      )"
      JOB_DIRECT_SEGMENTS+=("$DIRECT_SEGMENT")
      echo "Restoring source segment $((JOB_INDEX + 1))/${#LEFT_JOB_INPUTS[@]}, windows $((WINDOW_START + 1))-$WINDOW_END/$JOB_WINDOW_COUNT"
      JASNA_WINDOW_START="$WINDOW_START" \
      JASNA_WINDOW_COUNT="$WINDOW_COUNT" \
      JASNA_VIDEO_BITRATE="$VR_BITRATE" \
      JASNA_VR_PROJECTION=fisheye \
        "$ROOT_DIR/script/build_and_run.sh" --restore-stereo-sparse-batch \
          "${LEFT_JOB_INPUTS[$JOB_INDEX]}" \
          "${RIGHT_JOB_INPUTS[$JOB_INDEX]}" \
          "$DIRECT_SEGMENT" \
          "${LEFT_JOB_MANIFESTS[$JOB_INDEX]}" \
          "${RIGHT_JOB_MANIFESTS[$JOB_INDEX]}" \
          "${LEFT_JOB_CACHES[$JOB_INDEX]}" \
          "${RIGHT_JOB_CACHES[$JOB_INDEX]}"
      WINDOW_START="$WINDOW_END"
    done
    if (( ${#JOB_DIRECT_SEGMENTS[@]} == 1 )); then
      BATCH_SEGMENT="${JOB_DIRECT_SEGMENTS[0]}"
    else
      BATCH_CONCAT="$DIRECT_SEGMENT_DIR/$(
        printf '.segment-%05d.concat.txt' "$JOB_INDEX"
      )"
      BATCH_TEMP="$DIRECT_SEGMENT_DIR/$(
        printf '.segment-%05d.two-minute-writing.mov' "$JOB_INDEX"
      )"
      : > "$BATCH_CONCAT"
      for DIRECT_SEGMENT in "${JOB_DIRECT_SEGMENTS[@]}"; do
        ESCAPED_SEGMENT="${DIRECT_SEGMENT//\'/\'\\\'\'}"
        printf "file '%s'\n" "$ESCAPED_SEGMENT" >> "$BATCH_CONCAT"
      done
      if [[ -e "$BATCH_TEMP" ]]; then
        mv "$BATCH_TEMP" \
          "$DIRECT_SEGMENT_DIR/$(printf 'segment-%05d.interrupted-%s.mov' \
            "$JOB_INDEX" "$(date '+%Y%m%d-%H%M%S')")"
      fi
      echo "Joining source segment $((JOB_INDEX + 1))/${#LEFT_JOB_INPUTS[@]} into one two-minute SBS batch"
      "$FFMPEG_PATH" -hide_banner -loglevel error \
        -f concat -safe 0 -i "$BATCH_CONCAT" -map '0:v:0' -c copy \
        -movflags +faststart "$BATCH_TEMP"
      valid_direct_segment "$BATCH_TEMP" "$JOB_FRAME_COUNT" || {
        echo "error: joined two-minute SBS batch failed validation: $BATCH_TEMP" >&2
        exit 1
      }
      if [[ -e "$BATCH_SEGMENT" ]]; then
        mv "$BATCH_SEGMENT" \
          "$DIRECT_SEGMENT_DIR/$(printf 'segment-%05d.invalid-%s.mov' \
            "$JOB_INDEX" "$(date '+%Y%m%d-%H%M%S')")"
      fi
      mv "$BATCH_TEMP" "$BATCH_SEGMENT"
      for DIRECT_SEGMENT in "${JOB_DIRECT_SEGMENTS[@]}"; do
        [[ "$DIRECT_SEGMENT" == "$BATCH_SEGMENT" ]] || rm -f "$DIRECT_SEGMENT"
      done
      rm -f "$BATCH_CONCAT"
      echo "Validated two-minute SBS batch: $BATCH_SEGMENT"
    fi
    DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
  done
  echo "Restored ${#DIRECT_SEGMENTS[@]} grouped 8K SBS batch(es)"

  : > "$DIRECT_CONCAT_PATH"
  for DIRECT_SEGMENT in "${DIRECT_SEGMENTS[@]}"; do
    [[ -s "$DIRECT_SEGMENT" ]] || {
      echo "error: direct SBS segment is missing: $DIRECT_SEGMENT" >&2
      exit 1
    }
    ESCAPED_SEGMENT="${DIRECT_SEGMENT//\'/\'\\\'\'}"
    printf "file '%s'\n" "$ESCAPED_SEGMENT" >> "$DIRECT_CONCAT_PATH"
  done
  if completed_sbs_output "$OUTPUT_PATH"; then
    echo "Final direct SBS output already complete"
  elif completed_sbs_output "$FINAL_TEMP"; then
    echo "Promoting previously joined and validated direct SBS output"
    if [[ -e "$OUTPUT_PATH" ]]; then
      mv "$OUTPUT_PATH" \
        "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
    fi
    mv "$FINAL_TEMP" "$OUTPUT_PATH"
  else
    if [[ -e "$FINAL_TEMP" ]]; then
      mv "$FINAL_TEMP" "$WORK_DIR/direct-sbs.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
    fi
    if [[ -e "$OUTPUT_PATH" ]]; then
      mv "$OUTPUT_PATH" \
        "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
    fi
    echo "Joining direct SBS segments and copying source audio without video re-encoding"
    "$FFMPEG_PATH" \
      -hide_banner \
      -f concat -safe 0 -i "$DIRECT_CONCAT_PATH" \
      -i "$TEST_INPUT" \
      -map '0:v:0' \
      -map '1:a?' \
      -map_metadata 1 \
      -map_chapters 1 \
      -c copy \
      -movflags +faststart \
      -shortest \
      -n \
      "$FINAL_TEMP"
    completed_sbs_output "$FINAL_TEMP" || {
      echo "error: direct SBS output failed codec, dimensions, frame-count, or decode validation" >&2
      exit 1
    }
    mv "$FINAL_TEMP" "$OUTPUT_PATH"
  fi
  FINAL_INFO="$("$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT_PATH")"
  echo "Sparse SBS VR restoration ($RUN_DESCRIPTION): PASS"
  echo "$FINAL_INFO"
  echo "Output: $OUTPUT_PATH"
  echo "Log:    $LOG_PATH"
  cleanup_successful_work
  exit 0
elif (( ${#SHARED_BATCH_ARGS[@]} > 0 )); then
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
JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" left "$LEFT_OUTPUT"

JASNA_SPARSE_BATCH_MODE=finalize \
JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
JASNA_EYE_BITRATE="$EYE_BITRATE" \
JASNA_VR_PROJECTION=fisheye \
  "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
    "$TEST_INPUT" right "$RIGHT_OUTPUT"

if completed_sbs_output "$OUTPUT_PATH" \
    && [[ "$OUTPUT_PATH" -nt "$LEFT_OUTPUT" && "$OUTPUT_PATH" -nt "$RIGHT_OUTPUT" ]]; then
  echo "Stage 4/4: combined SBS output already complete"
elif completed_sbs_output "$FINAL_TEMP"; then
  echo "Stage 4/4: promoting previously joined and validated SBS output"
  if [[ -e "$OUTPUT_PATH" ]]; then
    mv "$OUTPUT_PATH" \
      "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
  fi
  mv "$FINAL_TEMP" "$OUTPUT_PATH"
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

  completed_sbs_output "$FINAL_TEMP" || {
    echo "error: combined output failed codec, dimensions, frame-count, or decode validation" >&2
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

echo "Sparse VR restoration ($RUN_DESCRIPTION): PASS"
echo "$FINAL_INFO"
echo "Output:   $OUTPUT_PATH"
echo "Left eye: $LEFT_OUTPUT"
echo "Right eye:$RIGHT_OUTPUT"
echo "Log:      $LOG_PATH"
cleanup_successful_work
