#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO left|right OUTPUT_EYE_VIDEO" >&2
  echo "optional: JASNA_SEGMENT_SECONDS=120 JASNA_EYE_BITRATE=20000000" >&2
  echo "          JASNA_ALLOW_IMPLEMENTATION_RESUME=1 (one-time compatible resume)" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage

INPUT_PATH="$1"
EYE="$2"
OUTPUT_PATH="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/script/restoration_identity.sh"
SEGMENT_SECONDS="${JASNA_SEGMENT_SECONDS:-120}"
EYE_BITRATE="${JASNA_EYE_BITRATE:-20000000}"
SPARSE_MOSAIC="${JASNA_SPARSE_MOSAIC:-0}"
VR_PROJECTION="${JASNA_VR_PROJECTION:-raw}"
FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
DETECT_BATCH_SIZE="${JASNA_DETECT_BATCH_SIZE:-2}"
DETECT_DECODE_MODE="${JASNA_DETECT_DECODE_MODE:-sequential}"
REGION_DURATION="${JASNA_REGION_DURATION:-1.0}"
SPARSE_BATCH_MODE="${JASNA_SPARSE_BATCH_MODE:-run}"
SPARSE_BATCH_FILE="${JASNA_SPARSE_BATCH_FILE:-}"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
ALLOW_IMPLEMENTATION_RESUME="${JASNA_ALLOW_IMPLEMENTATION_RESUME:-0}"
MANUAL_RANGE_MODE=0
MOSAIC_RANGES_RELATIVE=""
if [[ -n "${JASNA_MOSAIC_RANGES_RELATIVE+x}" ]]; then
  MANUAL_RANGE_MODE=1
  MOSAIC_RANGES_RELATIVE="$JASNA_MOSAIC_RANGES_RELATIVE"
fi

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
[[ "$VR_PROJECTION" == "raw" || "$VR_PROJECTION" == "fisheye" ]] || {
  echo "error: JASNA_VR_PROJECTION must be raw or fisheye" >&2
  exit 1
}
[[ "$FAST_ENCODE" == "0" || "$FAST_ENCODE" == "1" ]] || {
  echo "error: JASNA_FAST_ENCODE must be 0 or 1" >&2
  exit 1
}
[[ "$ALLOW_IMPLEMENTATION_RESUME" == "0" || "$ALLOW_IMPLEMENTATION_RESUME" == "1" ]] || {
  echo "error: JASNA_ALLOW_IMPLEMENTATION_RESUME must be 0 or 1" >&2
  exit 1
}
[[ "$SPARSE_BATCH_MODE" == "run" || "$SPARSE_BATCH_MODE" == "prepare" \
  || "$SPARSE_BATCH_MODE" == "finalize" ]] || {
  echo "error: JASNA_SPARSE_BATCH_MODE must be run, prepare, or finalize" >&2
  exit 1
}
if [[ "$SPARSE_BATCH_MODE" != "run" ]]; then
  [[ "$SPARSE_MOSAIC" == "1" && -n "$SPARSE_BATCH_FILE" ]] || {
    echo "error: coordinated sparse mode requires JASNA_SPARSE_BATCH_FILE" >&2
    exit 1
  }
fi
if [[ "$SPARSE_MOSAIC" == "1" \
  && "$DETECTOR" != "rfdetr-vr-v1" \
  && "$DETECTOR" != "yolo-v2-fast" ]]; then
  echo "error: JASNA_DETECTOR must be rfdetr-vr-v1 or yolo-v2-fast" >&2
  exit 1
fi

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

OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.${EYE}-segments-work"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.${EYE}-segments.log"
SOURCE_DIR="$WORK_DIR/source"
RESTORED_DIR="$WORK_DIR/restored"
CACHE_DIR="$WORK_DIR/cache"
if [[ "$SPARSE_MOSAIC" == "1" ]]; then
  # v15 adds a focused lower-edge detail crop to each oversized masked region.
  RESTORED_DIR="$WORK_DIR/restored-sparse-crop-v15-lower-detail-$VR_PROJECTION-$DETECTOR"
  CACHE_DIR="$WORK_DIR/cache-sparse-crop-v15-lower-detail-$VR_PROJECTION-$DETECTOR"
fi
OUTPUT_DONE="$RESTORED_DIR/${EYE}-joined.done"
SOURCE_DONE="$WORK_DIR/source.done"
MANIFEST_PATH="$WORK_DIR/restored-concat.txt"
TEMP_OUTPUT="$WORK_DIR/${EYE}-joined.mov"
RUN_CONFIG_PATH="$WORK_DIR/run-config.txt"
WORKFLOW_LOCK="$WORK_DIR/.jasna-eye-workflow-lock"

mkdir -p "$WORK_DIR" "$RESTORED_DIR" "$CACHE_DIR"
if ! mkdir "$WORKFLOW_LOCK" 2>/dev/null; then
  EXISTING_WORKFLOW_PID="$(/bin/cat "$WORKFLOW_LOCK/pid" 2>/dev/null || true)"
  if [[ "$EXISTING_WORKFLOW_PID" =~ ^[0-9]+$ ]] \
    && kill -0 "$EXISTING_WORKFLOW_PID" 2>/dev/null; then
    echo "error: this $EYE-eye restoration workflow is already active (PID $EXISTING_WORKFLOW_PID)" >&2
    echo "work dir: $WORK_DIR" >&2
    exit 1
  fi
  if [[ ! "$EXISTING_WORKFLOW_PID" =~ ^[0-9]+$ ]]; then
    echo "error: eye workflow lock exists without a valid owner PID: $WORKFLOW_LOCK" >&2
    exit 1
  fi
  STALE_WORKFLOW_LOCK="$WORK_DIR/.jasna-eye-workflow-lock.stale-$(date '+%Y%m%d-%H%M%S')-$$"
  mv "$WORKFLOW_LOCK" "$STALE_WORKFLOW_LOCK"
  mkdir "$WORKFLOW_LOCK"
fi
printf '%s\n' "$$" > "$WORKFLOW_LOCK/pid"
cleanup_eye_workflow_lock() {
  [[ -d "$WORKFLOW_LOCK" ]] || return 0
  local owner_pid
  owner_pid="$(/bin/cat "$WORKFLOW_LOCK/pid" 2>/dev/null || true)"
  [[ "$owner_pid" == "$$" ]] || return 0
  rm -f "$WORKFLOW_LOCK/pid"
  rmdir "$WORKFLOW_LOCK" 2>/dev/null || true
}
trap cleanup_eye_workflow_lock EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

SOURCE_FINGERPRINT="$(jasna_source_fingerprint "$INPUT_PATH")"
IMPLEMENTATION_FINGERPRINT="$(jasna_implementation_fingerprint "$ROOT_DIR")"
MODEL_FINGERPRINT="$(jasna_model_fingerprint "$ROOT_DIR" "$DETECTOR")"
RUN_CONFIG="input=$INPUT_PATH
source_fingerprint=$SOURCE_FINGERPRINT
implementation_fingerprint=$IMPLEMENTATION_FINGERPRINT
model_fingerprint=$MODEL_FINGERPRINT
eye=$EYE
segment_seconds=$SEGMENT_SECONDS
eye_bitrate=$EYE_BITRATE
sparse_mosaic=$SPARSE_MOSAIC
projection=$VR_PROJECTION
fast_encode=$FAST_ENCODE
detector=$DETECTOR
detect_batch_size=$DETECT_BATCH_SIZE
detect_decode_mode=$DETECT_DECODE_MODE
adaptive_detect=${JASNA_ADAPTIVE_DETECT:-0}
detect_sample_stride=${JASNA_DETECT_SAMPLE_STRIDE:-0.1}
detect_coarse_stride=${JASNA_DETECT_COARSE_STRIDE:-1.0}
detect_coarse_confidence=${JASNA_DETECT_COARSE_CONFIDENCE:-0.05}
detect_refine_padding=${JASNA_DETECT_REFINE_PADDING:-1.0}
manual_range_mode=$MANUAL_RANGE_MODE
manual_mosaic_ranges_relative=$MOSAIC_RANGES_RELATIVE
rfdetr_max_detections=${JASNA_RFDETR_MAX_DETECTIONS:-64}
region_duration=$REGION_DURATION
detect_confidence=${JASNA_DETECT_CONFIDENCE:-0.15}
temporal_padding=${JASNA_TEMPORAL_PADDING:-1.0}
region_nms_iou=${JASNA_REGION_NMS_IOU:-0.45}
mask_expansion=${JASNA_MASK_EXPANSION:-0.10}
mask_size=${JASNA_MASK_SIZE:-128}
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
model_batch=${JASNA_MODEL_BATCH:-1}
diagnostic_full_region_blend=${JASNA_DIAGNOSTIC_FULL_REGION_BLEND:-0}
metal_texture_compositor=${JASNA_METAL_TEXTURE_COMPOSITOR:-1}
metal_compositor=${JASNA_METAL_COMPOSITOR:-1}"
if [[ -s "$RUN_CONFIG_PATH" ]]; then
  EXISTING_RUN_CONFIG="$(/bin/cat "$RUN_CONFIG_PATH")"
  if [[ "$EXISTING_RUN_CONFIG" != "$RUN_CONFIG" ]]; then
    EXISTING_WITHOUT_IMPLEMENTATION="$(
      printf '%s\n' "$EXISTING_RUN_CONFIG" \
        | /usr/bin/sed '/^implementation_fingerprint=/d'
    )"
    CURRENT_WITHOUT_IMPLEMENTATION="$(
      printf '%s\n' "$RUN_CONFIG" \
        | /usr/bin/sed '/^implementation_fingerprint=/d'
    )"
    if [[ "$ALLOW_IMPLEMENTATION_RESUME" == "1" \
      && "$EXISTING_WITHOUT_IMPLEMENTATION" == "$CURRENT_WITHOUT_IMPLEMENTATION" ]]; then
      echo "WARNING: accepting one-time $EYE-eye implementation-only resume; source, model, ranges, and quality settings match"
    else
      echo "error: this eye output path belongs to a different source or restoration configuration" >&2
      if [[ "$EXISTING_WITHOUT_IMPLEMENTATION" == "$CURRENT_WITHOUT_IMPLEMENTATION" ]]; then
        echo "only the implementation changed; set JASNA_ALLOW_IMPLEMENTATION_RESUME=1 once" >&2
      else
        echo "use a new output filename, or restore the original settings" >&2
      fi
      exit 1
    fi
  fi
elif [[ "$SPARSE_BATCH_MODE" != "prepare" \
  && ( -f "$SOURCE_DONE" || -d "$SOURCE_DIR" ) ]]; then
  echo "error: legacy eye resume data has no configuration identity" >&2
  echo "use a new output filename so stale manifests cannot be reused" >&2
  exit 1
fi
RUN_CONFIG_TEMP="$WORK_DIR/.run-config-writing-$$"
printf '%s\n' "$RUN_CONFIG" > "$RUN_CONFIG_TEMP"
mv "$RUN_CONFIG_TEMP" "$RUN_CONFIG_PATH"
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

echo
echo "===== Jasna segmented $EYE-eye restoration $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input:            $INPUT_PATH"
echo "Output:           $OUTPUT_PATH"
echo "Work dir:         $WORK_DIR"
echo "Log:              $LOG_PATH"
echo "Segment duration: $SEGMENT_SECONDS seconds"
echo "Sparse mosaic:    $SPARSE_MOSAIC"
echo "VR projection:    $VR_PROJECTION"
echo "Fast encoding:    $FAST_ENCODE"
if [[ "$SPARSE_MOSAIC" == "1" ]]; then
  echo "Detector:         $DETECTOR; $DETECT_DECODE_MODE decode, batch $DETECT_BATCH_SIZE"
  echo "Temporal clips:   $REGION_DURATION seconds"
  echo "Detection quality: confidence ${JASNA_DETECT_CONFIDENCE:-0.15}, padding ${JASNA_TEMPORAL_PADDING:-1.0}s"
  echo "Region overlap:    suppress at IoU ${JASNA_REGION_NMS_IOU:-0.45}"
  echo "Mask expansion:    ${JASNA_MASK_EXPANSION:-0.10} of mask resolution"
  echo "Mask resolution:   ${JASNA_MASK_SIZE:-128}x${JASNA_MASK_SIZE:-128}"
  echo "Mask timing:       interpolated per output frame"
  echo "Large crop grid:   max ${JASNA_LARGE_REGION_MAX_BLEND:-768}px, overlap ${JASNA_LARGE_REGION_OVERLAP:-96}px"
  echo "Large split limit: ${JASNA_LARGE_REGION_SPLIT_LIMIT:-1} region(s)/window, max ${JASNA_LARGE_REGION_MAX_AXIS_CROPS:-3}/axis"
  echo "Large crop blend:  normalized Metal delta accumulation"
  echo "Block mask halo:    ${JASNA_LARGE_REGION_BLOCK_GROWTH:-0.04}; temporal radius ${JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS:-1}"
  echo "Lower detail crop:  ${JASNA_LARGE_REGION_DETAIL_CROPS:-1} at ${JASNA_LARGE_REGION_DETAIL_DIMENSION:-576}px"
fi

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
  # Prefer the container timeline. A constant-frame-rate conversion can have
  # one final video frame beyond the track's reported stream duration while
  # still matching the MP4/MOV presentation duration exactly.
  duration="$("$FFPROBE_PATH" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$candidate" 2>/dev/null)" || return 1
  if [[ -z "$duration" || "$duration" == "N/A" ]]; then
    duration="$("$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 \
      "$candidate" 2>/dev/null)" || return 1
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
    'BEGIN { delta = expected - candidate; if (delta < 0) delta = -delta; exit !(delta <= 0.05) }'
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
    "${ENCODER_SPEED_ARGS[@]}" \
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
SPARSE_BATCH_ARGS=()
if [[ "$SPARSE_MOSAIC" == "1" && "$SPARSE_BATCH_MODE" != "finalize" ]]; then
  for SOURCE_SEGMENT in "${SOURCE_SEGMENTS[@]}"; do
    SEGMENT_NAME="$(basename "$SOURCE_SEGMENT")"
    SEGMENT_STEM="${SEGMENT_NAME%.*}"
    RESTORED_SEGMENT="$RESTORED_DIR/${SEGMENT_STEM}-restored.mov"
    SEGMENT_DONE="$RESTORED_DIR/${SEGMENT_STEM}.done"
    SEGMENT_CACHE="$CACHE_DIR/$SEGMENT_STEM.jasna-work"
    SEGMENT_WINDOWS="$RESTORED_DIR/${SEGMENT_STEM}.windows"
    MOSAIC_MANIFEST="$RESTORED_DIR/${SEGMENT_STEM}-mosaic-regions.json"
    SEGMENT_DURATION="$(video_duration "$SOURCE_SEGMENT")"
    SEGMENT_RANGE_ARGUMENTS=()
    if [[ "$MANUAL_RANGE_MODE" == "1" ]]; then
      SEGMENT_NUMBER="${SEGMENT_STEM##*-}"
      SEGMENT_OFFSET=$((10#$SEGMENT_NUMBER * SEGMENT_SECONDS))
      SEGMENT_ACTIVE_RANGES="$(
        /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" segment \
          "$MOSAIC_RANGES_RELATIVE" "$SEGMENT_OFFSET" "$SEGMENT_DURATION"
      )"
      SEGMENT_RANGE_ARGUMENTS=(JASNA_DETECT_ACTIVE_RANGES="$SEGMENT_ACTIVE_RANGES")
      if [[ -n "$SEGMENT_ACTIVE_RANGES" ]]; then
        echo "Manual mosaic range for $SEGMENT_NAME: $SEGMENT_ACTIVE_RANGES"
      else
        echo "Manual mosaic range for $SEGMENT_NAME: clean segment"
      fi
    fi

    if [[ ! -f "$SEGMENT_DONE" ]] && video_duration_matches "$RESTORED_SEGMENT" "$SEGMENT_DURATION"; then
      echo "Recovered completed marker for $SEGMENT_NAME"
      /usr/bin/touch "$SEGMENT_DONE"
    fi
    if [[ ! -f "$SEGMENT_DONE" ]]; then
      mkdir -p "$SEGMENT_CACHE" "$SEGMENT_WINDOWS"
      if [[ ! -s "$MOSAIC_MANIFEST" ]]; then
        if [[ "$MANUAL_RANGE_MODE" == "1" ]]; then
          env "${SEGMENT_RANGE_ARGUMENTS[@]}" \
            "$ROOT_DIR/script/scan_mosaic_regions.sh" \
              "$SOURCE_SEGMENT" "$MOSAIC_MANIFEST"
        else
          "$ROOT_DIR/script/scan_mosaic_regions.sh" "$SOURCE_SEGMENT" "$MOSAIC_MANIFEST"
        fi
      else
        echo "Reusing mosaic-region manifest: $MOSAIC_MANIFEST"
      fi
      SPARSE_BATCH_ARGS+=(
        "$SOURCE_SEGMENT" "$SEGMENT_WINDOWS" "$MOSAIC_MANIFEST" "$SEGMENT_CACHE"
      )
    fi
  done
  if [[ "$SPARSE_BATCH_MODE" == "prepare" ]]; then
    for ((ARG_INDEX = 0; ARG_INDEX < ${#SPARSE_BATCH_ARGS[@]}; ARG_INDEX += 4)); do
      for ((FIELD_INDEX = ARG_INDEX; FIELD_INDEX < ARG_INDEX + 4; FIELD_INDEX++)); do
        [[ "${SPARSE_BATCH_ARGS[$FIELD_INDEX]}" != *$'\t'* \
          && "${SPARSE_BATCH_ARGS[$FIELD_INDEX]}" != *$'\n'* ]] || {
          echo "error: restoration paths cannot contain tabs or newlines" >&2
          exit 1
        }
      done
      printf '%s\t%s\t%s\t%s\n' \
        "${SPARSE_BATCH_ARGS[$ARG_INDEX]}" \
        "${SPARSE_BATCH_ARGS[$((ARG_INDEX + 1))]}" \
        "${SPARSE_BATCH_ARGS[$((ARG_INDEX + 2))]}" \
        "${SPARSE_BATCH_ARGS[$((ARG_INDEX + 3))]}" \
        >> "$SPARSE_BATCH_FILE"
    done
    echo "Prepared $((${#SPARSE_BATCH_ARGS[@]} / 4)) pending segment(s) for shared restoration"
    exit 0
  elif (( ${#SPARSE_BATCH_ARGS[@]} > 0 )); then
    echo "Restoring $((${#SPARSE_BATCH_ARGS[@]} / 4)) pending segment(s) with one retained Metal ML graph"
    "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows-sparse-batch \
      "${SPARSE_BATCH_ARGS[@]}"
  fi
fi

for SOURCE_SEGMENT in "${SOURCE_SEGMENTS[@]}"; do
  SEGMENT_NAME="$(basename "$SOURCE_SEGMENT")"
  SEGMENT_STEM="${SEGMENT_NAME%.*}"
  RESTORED_SEGMENT="$RESTORED_DIR/${SEGMENT_STEM}-restored.mov"
  SEGMENT_DONE="$RESTORED_DIR/${SEGMENT_STEM}.done"
  SEGMENT_CACHE="$CACHE_DIR/$SEGMENT_STEM.jasna-work"
  SEGMENT_WINDOWS="$RESTORED_DIR/${SEGMENT_STEM}.windows"
  SEGMENT_MANIFEST="$RESTORED_DIR/${SEGMENT_STEM}-windows.txt"
  MOSAIC_MANIFEST="$RESTORED_DIR/${SEGMENT_STEM}-mosaic-regions.json"
  TEMP_RESTORED_SEGMENT="$RESTORED_DIR/.${SEGMENT_STEM}-joining.mov"
  SEGMENT_DURATION="$(video_duration "$SOURCE_SEGMENT")"

  if [[ ! -f "$SEGMENT_DONE" ]] && video_duration_matches "$RESTORED_SEGMENT" "$SEGMENT_DURATION"; then
    echo "Recovered completed marker for $SEGMENT_NAME"
    /usr/bin/touch "$SEGMENT_DONE"
  fi

  if [[ ! -f "$SEGMENT_DONE" ]]; then
    echo "Restoring $SEGMENT_NAME (${SEGMENT_DURATION}s)"
    mkdir -p "$SEGMENT_CACHE" "$SEGMENT_WINDOWS"
    if [[ "$SPARSE_MOSAIC" == "1" ]]; then
      echo "Using sparse windows completed by the retained-graph batch"
    else
      JASNA_WORK_DIR="$SEGMENT_CACHE" \
        "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows \
          "$SOURCE_SEGMENT" "$SEGMENT_WINDOWS"
    fi

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

if [[ -f "$OUTPUT_DONE" ]] && completed_eye_output "$OUTPUT_PATH"; then
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
/usr/bin/touch "$OUTPUT_DONE"

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
