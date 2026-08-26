#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-test.mov 00:12:00" >&2
  echo "optional: JASNA_TEST_SECONDS=30 (1-300, or full)" >&2
  echo "          JASNA_MOSAIC_RANGES=00:12:00-00:14:00,00:20:30-00:22:00" >&2
  echo "          JASNA_ENCODER_WINDOWS_PER_SEGMENT=4 (bounded eye-by-eye disk use)" >&2
  echo "          JASNA_METAL_WINDOWS_PER_PROCESS=2 (validated; set 1 minimum memory or 4 experimental fast)" >&2
  echo "          JASNA_GPU_TIMEOUT_RETRIES=2 (fresh-process checkpoint retries)" >&2
  echo "          JASNA_DETECT_DEVICE=auto (MPS with automatic CPU fallback; or force cpu)" >&2
  echo "          JASNA_STEREO_DETECT=1 (one RF-DETR load/decode for both SBS eyes)" >&2
  echo "          JASNA_STEREO_SAMPLE_MODE=paired (experimental: alternating)" >&2
  echo "          JASNA_EYE_BITRATE=20000000 JASNA_VR_BITRATE=40000000" >&2
  echo "          JASNA_DIRECT_SBS_OUTPUT=1 (set 0 for lower-memory eye-by-eye output)" >&2
  echo "          JASNA_EYE_JOB_PROCESS_ISOLATION=1 (fresh process per 30-120 second 4K eye job)" >&2
  echo "          JASNA_EYE_PAIR_SEGMENTS=1 (combine each eye pair before the final join)" >&2
  echo "          JASNA_ALLOW_IMPLEMENTATION_RESUME=1 (one-time reuse after a script update)" >&2
  echo "          JASNA_CLEAN_WORK_ON_SUCCESS=0 (set 1 to remove restart data after PASS)" >&2
  echo "          JASNA_LARGE_REGION_MAX_BLEND=768 JASNA_LARGE_REGION_OVERLAP=96" >&2
  echo "          JASNA_LARGE_REGION_MASK_GROWTH=0.05 JASNA_LARGE_REGION_MASK_FEATHER=0.025" >&2
  echo "          JASNA_LARGE_REGION_BLOCK_GROWTH=0.04 JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS=1" >&2
  echo "          JASNA_LARGE_REGION_DETAIL_CROPS=1 JASNA_LARGE_REGION_DETAIL_DIMENSION=576" >&2
  echo "          JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=0 (experimental: 1)" >&2
  echo "          JASNA_TEMPORAL_WARMUP_FRAMES=5 (set 0 to disable)" >&2
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
WORK_CONTAINER="${JASNA_WORK_CONTAINER:-mov}"
DIRECT_SBS_OUTPUT="${JASNA_DIRECT_SBS_OUTPUT:-1}"
SHARED_SBS_SOURCE="${JASNA_SHARED_SBS_SOURCE:-$DIRECT_SBS_OUTPUT}"
STEREO_DETECT="${JASNA_STEREO_DETECT:-1}"
STEREO_SAMPLE_MODE="${JASNA_STEREO_SAMPLE_MODE:-paired}"
EYE_JOB_PROCESS_ISOLATION="${JASNA_EYE_JOB_PROCESS_ISOLATION:-1}"
EYE_PAIR_SEGMENTS="${JASNA_EYE_PAIR_SEGMENTS:-0}"
METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
CLEAN_WORK_ON_SUCCESS="${JASNA_CLEAN_WORK_ON_SUCCESS:-0}"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
DETECT_DEVICE="${JASNA_DETECT_DEVICE:-auto}"
if [[ "$DETECTOR" == "rfdetr-v6" ]]; then
  DEFAULT_DETECT_CONFIDENCE="0.35"
else
  DEFAULT_DETECT_CONFIDENCE="0.15"
fi
DETECT_CONFIDENCE="${JASNA_DETECT_CONFIDENCE:-$DEFAULT_DETECT_CONFIDENCE}"
IN_MEMORY_CROP_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-1}"
IN_MEMORY_CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-512}"
TEMPORAL_WARMUP_FRAMES="${JASNA_TEMPORAL_WARMUP_FRAMES:-5}"
MOSAIC_RANGES="${JASNA_MOSAIC_RANGES:-}"
ALLOW_IMPLEMENTATION_RESUME="${JASNA_ALLOW_IMPLEMENTATION_RESUME:-0}"
RUN_WALL_STARTED_SECONDS=$SECONDS

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
  ARTIFACT_TAG="jasna-vr-full-v22"
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
  ARTIFACT_TAG="jasna-vr30-v22"
fi
if [[ -n "$MOSAIC_RANGES" ]]; then
  MOSAIC_RANGES_RELATIVE="$(
    /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" normalize \
      "$MOSAIC_RANGES" "$START_TIME" "$TEST_SECONDS"
  )" || {
    echo "error: invalid JASNA_MOSAIC_RANGES: $MOSAIC_RANGES" >&2
    exit 1
  }
  [[ -n "$MOSAIC_RANGES_RELATIVE" ]] || {
    echo "error: the manual mosaic ranges do not intersect the selected video" >&2
    exit 1
  }
  export JASNA_MOSAIC_RANGES_RELATIVE="$MOSAIC_RANGES_RELATIVE"
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
[[ "$WORK_CONTAINER" == "mov" || "$WORK_CONTAINER" == "mp4" ]] || {
  echo "error: JASNA_WORK_CONTAINER must be mov or mp4" >&2
  exit 1
}
[[ "$DIRECT_SBS_OUTPUT" == "0" || "$DIRECT_SBS_OUTPUT" == "1" ]] || {
  echo "error: JASNA_DIRECT_SBS_OUTPUT must be 0 or 1" >&2
  exit 1
}
[[ "$SHARED_SBS_SOURCE" == "0" || "$SHARED_SBS_SOURCE" == "1" ]] || {
  echo "error: JASNA_SHARED_SBS_SOURCE must be 0 or 1" >&2
  exit 1
}
if [[ "$SHARED_SBS_SOURCE" == "1" && "$DIRECT_SBS_OUTPUT" != "1" ]]; then
  echo "error: JASNA_SHARED_SBS_SOURCE=1 requires JASNA_DIRECT_SBS_OUTPUT=1" >&2
  exit 1
fi
[[ "$IN_MEMORY_CROP_CACHE" == "0" || "$IN_MEMORY_CROP_CACHE" == "1" ]] || {
  echo "error: JASNA_IN_MEMORY_CROP_CACHE must be 0 or 1" >&2
  exit 1
}
[[ "$IN_MEMORY_CACHE_LIMIT_MB" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_IN_MEMORY_CACHE_LIMIT_MB must be a non-negative integer" >&2
  exit 1
}
[[ "$DETECT_DEVICE" == "auto" || "$DETECT_DEVICE" == "mps" \
  || "$DETECT_DEVICE" == "cpu" ]] || {
  echo "error: JASNA_DETECT_DEVICE must be auto, mps, or cpu" >&2
  exit 1
}
[[ "$TEMPORAL_WARMUP_FRAMES" =~ ^[0-9]+$ ]] \
  && (( TEMPORAL_WARMUP_FRAMES <= 5 )) || {
    echo "error: JASNA_TEMPORAL_WARMUP_FRAMES must be an integer from 0 to 5" >&2
    exit 1
  }
export JASNA_DETECT_DEVICE="$DETECT_DEVICE"
export JASNA_SHARED_SBS_SOURCE="$SHARED_SBS_SOURCE"
export JASNA_IN_MEMORY_CROP_CACHE="$IN_MEMORY_CROP_CACHE"
export JASNA_IN_MEMORY_CACHE_LIMIT_MB="$IN_MEMORY_CACHE_LIMIT_MB"
export JASNA_TEMPORAL_WARMUP_FRAMES="$TEMPORAL_WARMUP_FRAMES"
[[ "$EYE_JOB_PROCESS_ISOLATION" == "0" || "$EYE_JOB_PROCESS_ISOLATION" == "1" ]] || {
  echo "error: JASNA_EYE_JOB_PROCESS_ISOLATION must be 0 or 1" >&2
  exit 1
}
[[ "$EYE_PAIR_SEGMENTS" == "0" || "$EYE_PAIR_SEGMENTS" == "1" ]] || {
  echo "error: JASNA_EYE_PAIR_SEGMENTS must be 0 or 1" >&2
  exit 1
}
if [[ "$DIRECT_SBS_OUTPUT" == "1" && "$EYE_PAIR_SEGMENTS" == "1" ]]; then
  echo "error: JASNA_EYE_PAIR_SEGMENTS=1 requires JASNA_DIRECT_SBS_OUTPUT=0" >&2
  exit 1
fi
[[ "$ALLOW_IMPLEMENTATION_RESUME" == "0" || "$ALLOW_IMPLEMENTATION_RESUME" == "1" ]] || {
  echo "error: JASNA_ALLOW_IMPLEMENTATION_RESUME must be 0 or 1" >&2
  exit 1
}
[[ "$STEREO_DETECT" == "0" || "$STEREO_DETECT" == "1" ]] || {
  echo "error: JASNA_STEREO_DETECT must be 0 or 1" >&2
  exit 1
}
[[ "$STEREO_SAMPLE_MODE" == "paired" || "$STEREO_SAMPLE_MODE" == "alternating" ]] || {
  echo "error: JASNA_STEREO_SAMPLE_MODE must be paired or alternating" >&2
  exit 1
}
if [[ "$STEREO_SAMPLE_MODE" == "alternating" && "$STEREO_DETECT" != "1" ]]; then
  echo "error: alternating stereo sampling requires JASNA_STEREO_DETECT=1" >&2
  exit 1
fi
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
TEST_INPUT="$SOURCE_DIR/test-sbs-30fps.$WORK_CONTAINER"
TEST_INPUT_TEMP="$SOURCE_DIR/.test-sbs-30fps-writing.$WORK_CONTAINER"
TEST_INPUT_DONE="$SOURCE_DIR/test-sbs-30fps.done"
LEFT_OUTPUT="$WORK_DIR/left-restored.mov"
RIGHT_OUTPUT="$WORK_DIR/right-restored.mov"
LEFT_EYE_WORK_DIR="$WORK_DIR/left-restored.left-segments-work"
RIGHT_EYE_WORK_DIR="$WORK_DIR/right-restored.right-segments-work"
FINAL_TEMP="$WORK_DIR/.joined-sbs-writing.${OUTPUT_NAME##*.}"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.${ARTIFACT_TAG}.log"
SHARED_BATCH_PATH="$WORK_DIR/pending-eye-restorations.tsv"
DIRECT_SEGMENT_DIR="$WORK_DIR/direct-sbs-segments"
DIRECT_NORMALIZED_DIR="$WORK_DIR/direct-sbs-timescale-600"
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
work_container=$WORK_CONTAINER
direct_sbs_output=$DIRECT_SBS_OUTPUT
shared_sbs_source=$SHARED_SBS_SOURCE
stereo_detect=$STEREO_DETECT
stereo_sample_mode=$STEREO_SAMPLE_MODE
in_memory_crop_cache=$IN_MEMORY_CROP_CACHE
in_memory_cache_limit_mb=$IN_MEMORY_CACHE_LIMIT_MB
eye_job_process_isolation=$EYE_JOB_PROCESS_ISOLATION
eye_pair_segments=$EYE_PAIR_SEGMENTS
model_batch=$MODEL_BATCH
metal_windows_per_process=$METAL_WINDOWS_PER_PROCESS
diagnostic_full_region_blend=${JASNA_DIAGNOSTIC_FULL_REGION_BLEND:-0}
metal_texture_compositor=${JASNA_METAL_TEXTURE_COMPOSITOR:-1}
metal_compositor=${JASNA_METAL_COMPOSITOR:-1}
quality_profile=stable-balanced-v22
detect_batch_size=${JASNA_DETECT_BATCH_SIZE:-2}
detect_decode_mode=${JASNA_DETECT_DECODE_MODE:-sequential}
adaptive_detect=${JASNA_ADAPTIVE_DETECT:-0}
detect_sample_stride=${JASNA_DETECT_SAMPLE_STRIDE:-0.1}
detect_coarse_stride=${JASNA_DETECT_COARSE_STRIDE:-1.0}
detect_coarse_confidence=${JASNA_DETECT_COARSE_CONFIDENCE:-0.05}
detect_refine_padding=${JASNA_DETECT_REFINE_PADDING:-1.0}
stereo_manifest_reconcile=${JASNA_STEREO_MANIFEST_RECONCILE:-1}
manual_mosaic_ranges=$MOSAIC_RANGES
manual_mosaic_ranges_relative=${MOSAIC_RANGES_RELATIVE:-}
rfdetr_max_detections=${JASNA_RFDETR_MAX_DETECTIONS:-64}
detect_confidence=$DETECT_CONFIDENCE
temporal_padding=${JASNA_TEMPORAL_PADDING:-1.0}
region_nms_iou=${JASNA_REGION_NMS_IOU:-0.45}
mask_expansion=${JASNA_MASK_EXPANSION:-0.10}
mask_size=${JASNA_MASK_SIZE:-128}
region_duration=${JASNA_REGION_DURATION:-1.0}
large_region_max_blend=${JASNA_LARGE_REGION_MAX_BLEND:-768}
large_region_overlap=${JASNA_LARGE_REGION_OVERLAP:-96}
large_region_split_limit=${JASNA_LARGE_REGION_SPLIT_LIMIT:-1}
large_region_max_axis_crops=${JASNA_LARGE_REGION_MAX_AXIS_CROPS:-4}
large_region_mask_growth=${JASNA_LARGE_REGION_MASK_GROWTH:-0.05}
large_region_mask_feather=${JASNA_LARGE_REGION_MASK_FEATHER:-0.025}
large_region_block_growth=${JASNA_LARGE_REGION_BLOCK_GROWTH:-0.04}
large_region_mask_temporal_radius=${JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS:-1}
large_region_detail_crops=${JASNA_LARGE_REGION_DETAIL_CROPS:-1}
large_region_detail_dimension=${JASNA_LARGE_REGION_DETAIL_DIMENSION:-576}
temporal_crop_frames=${JASNA_TEMPORAL_CROP_FRAMES:-0}
temporal_crop_padding=${JASNA_TEMPORAL_CROP_PADDING:-128}
temporal_crop_min_dimension=${JASNA_TEMPORAL_CROP_MIN_DIMENSION:-1024}
temporal_crop_motion=${JASNA_TEMPORAL_CROP_MOTION:-0.08}
mosaic_detail_residual_limit=${JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT:-0.03}
mosaic_mask_recovery_threshold=${JASNA_MOSAIC_MASK_RECOVERY_THRESHOLD:-0.025}
mosaic_mask_recovery_all_regions=${JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS:-0}
temporal_warmup_frames=$TEMPORAL_WARMUP_FRAMES
projection=fisheye
detector=$DETECTOR"
RUN_CONFIG="$RUN_CONFIG
detect_device=$DETECT_DEVICE"
if [[ -s "$RUN_CONFIG_PATH" ]]; then
  EXISTING_STABLE_CONFIG="$(
    /usr/bin/sed \
      -e '/^metal_windows_per_process=/d' \
      -e '/^eye_job_process_isolation=/d' \
      -e '/^eye_pair_segments=/d' \
      "$RUN_CONFIG_PATH"
  )"
  if ! /usr/bin/grep -q '^detector=' "$RUN_CONFIG_PATH"; then
    EXISTING_STABLE_CONFIG="$EXISTING_STABLE_CONFIG
detector=yolo-v2-fast"
  fi
  CURRENT_STABLE_CONFIG="$(
    printf '%s\n' "$RUN_CONFIG" | /usr/bin/sed \
      -e '/^metal_windows_per_process=/d' \
      -e '/^eye_job_process_isolation=/d' \
      -e '/^eye_pair_segments=/d'
  )"
  if [[ "$EXISTING_STABLE_CONFIG" != "$CURRENT_STABLE_CONFIG" ]]; then
    EXISTING_WITHOUT_IMPLEMENTATION="$(
      printf '%s\n' "$EXISTING_STABLE_CONFIG" \
        | /usr/bin/sed '/^implementation_fingerprint=/d'
    )"
    CURRENT_WITHOUT_IMPLEMENTATION="$(
      printf '%s\n' "$CURRENT_STABLE_CONFIG" \
        | /usr/bin/sed '/^implementation_fingerprint=/d'
    )"
    if [[ "$ALLOW_IMPLEMENTATION_RESUME" == "1" \
      && "$EXISTING_WITHOUT_IMPLEMENTATION" == "$CURRENT_WITHOUT_IMPLEMENTATION" ]]; then
      echo "WARNING: accepting a one-time implementation-only resume; input, models, ranges, and quality settings match"
    else
      echo "error: this output path belongs to a different test configuration" >&2
      if [[ "$EXISTING_WITHOUT_IMPLEMENTATION" == "$CURRENT_WITHOUT_IMPLEMENTATION" ]]; then
        echo "only the implementation changed; set JASNA_ALLOW_IMPLEMENTATION_RESUME=1 once to reuse compatible work" >&2
      else
        echo "use a new output filename, or restore the original input/start/settings" >&2
      fi
      exit 1
    fi
  fi
fi
RUN_CONFIG_TEMP="$WORK_DIR/.run-config-writing-$$"
printf '%s\n' "$RUN_CONFIG" > "$RUN_CONFIG_TEMP"
mv "$RUN_CONFIG_TEMP" "$RUN_CONFIG_PATH"
LOG_SESSION_START_LINE=0
if [[ -f "$LOG_PATH" ]]; then
  LOG_SESSION_START_LINE="$(/usr/bin/wc -l < "$LOG_PATH")"
fi
exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1

report_compositor_fallback_summary() {
  local summary
  summary="$(/usr/bin/awk -v start="$LOG_SESSION_START_LINE" '
    NR > start && /WARNING: Fused Metal stereo composite failed/ { fused += 1 }
    NR > start && /WARNING: Metal stereo copy failed/ { cpu += 1 }
    END { printf "%d %d", fused + 0, cpu + 0 }
  ' "$LOG_PATH")"
  local fused_count="${summary%% *}"
  local cpu_count="${summary##* }"
  if (( fused_count == 0 && cpu_count == 0 )); then
    echo "Stereo compositor fallback summary: PASS, fused 0, CPU 0 (this session)"
  else
    echo "Stereo compositor fallback summary: WARNING, fused $fused_count, CPU $cpu_count (this session)"
  fi
}

report_total_wall_time() {
  local elapsed_seconds=$((SECONDS - RUN_WALL_STARTED_SECONDS))
  /usr/bin/awk -v seconds="$elapsed_seconds" \
    'BEGIN { printf "Total wall time: %d seconds (%.2f minutes)\n", seconds, seconds / 60 }'
}

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
echo "Shared SBS source: $SHARED_SBS_SOURCE"
echo "Stereo detector sampling: $STEREO_SAMPLE_MODE"
echo "SBS working container: $WORK_CONTAINER"
echo "Crop handoff: $([[ "$IN_MEMORY_CROP_CACHE" == "1" ]] && echo memory-up-to-${IN_MEMORY_CACHE_LIMIT_MB}MiB || echo disk)"
if [[ "$DIRECT_SBS_OUTPUT" == "0" ]]; then
  echo "Eye workers:  $([[ "$EYE_JOB_PROCESS_ISOLATION" == "1" ]] && echo isolated-${TEST_SEGMENT_SECONDS}s || echo retained-graph)"
  echo "Pair output:  $([[ "$EYE_PAIR_SEGMENTS" == "1" ]] && echo ${TEST_SEGMENT_SECONDS}s-sbs || echo full-eye-join)"
fi
echo "Model batch: $MODEL_BATCH"
echo "Detector:    $DETECTOR"
echo "Detect device: $DETECT_DEVICE (auto prefers MPS and falls back to CPU)"
if [[ -n "$MOSAIC_RANGES" ]]; then
  echo "Manual mosaic ranges: $MOSAIC_RANGES"
  echo "Only these source-timeline ranges will be detected/restored"
fi

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
    mv "$TEST_INPUT_TEMP" \
      "$SOURCE_DIR/test-sbs.interrupted-$(date '+%Y%m%d-%H%M%S').$WORK_CONTAINER"
  fi
  if [[ -e "$TEST_INPUT" ]]; then
    mv "$TEST_INPUT" \
      "$SOURCE_DIR/test-sbs.previous-$(date '+%Y%m%d-%H%M%S').$WORK_CONTAINER"
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

valid_eye_source_segment() {
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
    && "$width" == "$EYE_WIDTH" \
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

valid_shared_sbs_source_segment() {
  local candidate="$1"
  local expected_frames="$2"
  [[ -s "$candidate" ]] || return 1
  local codec width height frame_rate frame_count extra
  IFS=, read -r codec width height frame_rate frame_count extra < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$candidate" 2>/dev/null
  )
  [[ -z "$extra" && "$codec" == "hevc" \
    && "$width" == "$SOURCE_WIDTH" && "$height" == "$SOURCE_HEIGHT" \
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
SHARED_SOURCE_DIR="$WORK_DIR/shared-sbs-source"
LEFT_SOURCE_DONE="$LEFT_EYE_WORK_DIR/source.done"
RIGHT_SOURCE_DONE="$RIGHT_EYE_WORK_DIR/source.done"
stereo_source_segments_valid() {
  local left_segments=("$LEFT_SOURCE_DIR"/left-*.mov)
  local right_segments=("$RIGHT_SOURCE_DIR"/right-*.mov)
  [[ -e "${left_segments[0]}" \
    && -e "${right_segments[0]}" \
    && ${#left_segments[@]} -eq ${#right_segments[@]} ]] || return 1
  local left_segment segment_name segment_index right_segment expected_frames
  for left_segment in "${left_segments[@]}"; do
    segment_name="$(basename "$left_segment" .mov)"
    [[ "$segment_name" =~ ^left-([0-9]{5})$ ]] || return 1
    segment_index=$((10#${BASH_REMATCH[1]}))
    right_segment="$RIGHT_SOURCE_DIR/$(printf 'right-%05d.mov' "$segment_index")"
    expected_frames=$((EXPECTED_FRAME_COUNT - segment_index * TEST_SEGMENT_SECONDS * 30))
    (( expected_frames > TEST_SEGMENT_SECONDS * 30 )) \
      && expected_frames=$((TEST_SEGMENT_SECONDS * 30))
    (( expected_frames > 0 )) || return 1
    if [[ "$SHARED_SBS_SOURCE" == "1" ]]; then
      [[ "$left_segment" -ef "$right_segment" ]] || return 1
      valid_shared_sbs_source_segment "$left_segment" "$expected_frames" || return 1
    else
      valid_eye_source_segment "$left_segment" "$expected_frames" || return 1
      valid_eye_source_segment "$right_segment" "$expected_frames" || return 1
    fi
  done
}

link_shared_sbs_segments() {
  local shared_segment segment_name segment_index
  local shared_segments=("$SHARED_SOURCE_DIR"/shared-*.mov)
  [[ -e "${shared_segments[0]}" ]] || return 1
  for shared_segment in "${shared_segments[@]}"; do
    segment_name="$(basename "$shared_segment" .mov)"
    [[ "$segment_name" =~ ^shared-([0-9]{5})$ ]] || return 1
    segment_index="${BASH_REMATCH[1]}"
    /bin/ln -s "$shared_segment" "$LEFT_SOURCE_DIR/left-${segment_index}.mov"
    /bin/ln -s "$shared_segment" "$RIGHT_SOURCE_DIR/right-${segment_index}.mov"
  done
}

prepare_reencoded_shared_sbs_segments() {
  if [[ -n "$MOSAIC_RANGES" ]]; then
    local active_source_segments=()
    read -r -a active_source_segments <<< "$(
      /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" indices \
        "$MOSAIC_RANGES_RELATIVE" "$TEST_VIDEO_DURATION" "$TEST_SEGMENT_SECONDS"
    )"
    (( ${#active_source_segments[@]} > 0 )) || return 1
    local segment_index segment_offset segment_duration shared_segment
    for segment_index in "${active_source_segments[@]}"; do
      segment_offset=$((segment_index * TEST_SEGMENT_SECONDS))
      segment_duration="$(/usr/bin/awk \
        -v offset="$segment_offset" \
        -v duration="$TEST_VIDEO_DURATION" \
        -v maximum="$TEST_SEGMENT_SECONDS" \
        'BEGIN { remaining = duration - offset; if (remaining > maximum) remaining = maximum; printf "%.6f\n", remaining }'
      )"
      shared_segment="$SHARED_SOURCE_DIR/$(printf 'shared-%05d.mov' "$segment_index")"
      "$FFMPEG_PATH" -hide_banner -ss "$segment_offset" -i "$TEST_INPUT" \
        -map '0:v:0' -t "$segment_duration" -an -vf fps=30 \
        -c:v hevc_videotoolbox "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p \
        -b:v "$VR_BITRATE" -maxrate "$((VR_BITRATE * 3 / 2))" \
        -bufsize "$((VR_BITRATE * 3))" -g 30 -tag:v hvc1 \
        -movflags +faststart "$shared_segment"
    done
  else
    "$FFMPEG_PATH" -hide_banner -i "$TEST_INPUT" -map '0:v:0' -an -vf fps=30 \
      -c:v hevc_videotoolbox "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p \
      -b:v "$VR_BITRATE" -maxrate "$((VR_BITRATE * 3 / 2))" \
      -bufsize "$((VR_BITRATE * 3))" -g 30 \
      -force_key_frames "expr:gte(t,n_forced*${TEST_SEGMENT_SECONDS})" \
      -tag:v hvc1 -f segment -segment_format mov \
      -segment_time "$TEST_SEGMENT_SECONDS" -segment_time_delta 0.016667 \
      -reset_timestamps 1 "$SHARED_SOURCE_DIR/shared-%05d.mov"
  fi
  link_shared_sbs_segments
}

if [[ "$DIRECT_SBS_OUTPUT" == "1" ]] && stereo_source_segments_valid; then
  if [[ ! -f "$LEFT_SOURCE_DONE" || ! -f "$RIGHT_SOURCE_DONE" ]]; then
    /usr/bin/touch "$LEFT_SOURCE_DONE" "$RIGHT_SOURCE_DONE"
    echo "Recovered paired shared-source completion markers from validated segments"
  fi
fi

if [[ "$DIRECT_SBS_OUTPUT" == "1" \
  && ( -f "$LEFT_SOURCE_DONE" || -f "$RIGHT_SOURCE_DONE" ) ]] \
  && ! stereo_source_segments_valid; then
  INVALID_SOURCE_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
  echo "WARNING: cached left/right source segments are incomplete or have mismatched timelines"
  if [[ -d "$LEFT_EYE_WORK_DIR" ]]; then
    mv "$LEFT_EYE_WORK_DIR" \
      "$LEFT_EYE_WORK_DIR.invalid-source-$INVALID_SOURCE_TIMESTAMP"
  fi
  if [[ -d "$RIGHT_EYE_WORK_DIR" ]]; then
    mv "$RIGHT_EYE_WORK_DIR" \
      "$RIGHT_EYE_WORK_DIR.invalid-source-$INVALID_SOURCE_TIMESTAMP"
  fi
  if [[ -d "$SHARED_SOURCE_DIR" ]]; then
    mv "$SHARED_SOURCE_DIR" \
      "$SHARED_SOURCE_DIR.invalid-source-$INVALID_SOURCE_TIMESTAMP"
  fi
  mkdir -p "$LEFT_EYE_WORK_DIR" "$RIGHT_EYE_WORK_DIR"
  echo "Archived incompatible eye caches; regenerating paired source segments"
fi
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
  if [[ "$SHARED_SBS_SOURCE" == "1" && -d "$SHARED_SOURCE_DIR" ]]; then
    mv "$SHARED_SOURCE_DIR" \
      "$SHARED_SOURCE_DIR.interrupted-$(date '+%Y%m%d-%H%M%S')"
  fi
  mkdir -p "$LEFT_SOURCE_DIR" "$RIGHT_SOURCE_DIR"
  if [[ "$SHARED_SBS_SOURCE" == "1" ]]; then
    mkdir -p "$SHARED_SOURCE_DIR"
    if [[ -n "$MOSAIC_RANGES" ]]; then
      read -r -a ACTIVE_SOURCE_SEGMENTS <<< "$(
        /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" indices \
          "$MOSAIC_RANGES_RELATIVE" "$TEST_VIDEO_DURATION" "$TEST_SEGMENT_SECONDS"
      )"
      (( ${#ACTIVE_SOURCE_SEGMENTS[@]} > 0 )) || {
        echo "error: the manual mosaic ranges do not intersect the selected video" >&2
        exit 1
      }
      echo "Packet-copying SBS video only for ${#ACTIVE_SOURCE_SEGMENTS[@]} active ${TEST_SEGMENT_SECONDS}-second segment(s)"
      for SEGMENT_INDEX in "${ACTIVE_SOURCE_SEGMENTS[@]}"; do
        SEGMENT_OFFSET=$((SEGMENT_INDEX * TEST_SEGMENT_SECONDS))
        SEGMENT_DURATION="$(/usr/bin/awk \
          -v offset="$SEGMENT_OFFSET" \
          -v duration="$TEST_VIDEO_DURATION" \
          -v maximum="$TEST_SEGMENT_SECONDS" \
          'BEGIN { remaining = duration - offset; if (remaining > maximum) remaining = maximum; printf "%.6f\n", remaining }'
        )"
        SHARED_SEGMENT="$SHARED_SOURCE_DIR/$(printf 'shared-%05d.mov' "$SEGMENT_INDEX")"
        LEFT_SEGMENT="$LEFT_SOURCE_DIR/$(printf 'left-%05d.mov' "$SEGMENT_INDEX")"
        RIGHT_SEGMENT="$RIGHT_SOURCE_DIR/$(printf 'right-%05d.mov' "$SEGMENT_INDEX")"
        echo "Active shared SBS segment $(printf '%05d' "$SEGMENT_INDEX"): timeline ${SEGMENT_OFFSET}s-$((SEGMENT_OFFSET + TEST_SEGMENT_SECONDS))s"
        "$FFMPEG_PATH" -hide_banner -ss "$SEGMENT_OFFSET" -i "$TEST_INPUT" \
          -map '0:v:0' -t "$SEGMENT_DURATION" -an -c copy \
          -avoid_negative_ts make_zero -movflags +faststart "$SHARED_SEGMENT"
        /bin/ln -s "$SHARED_SEGMENT" "$LEFT_SEGMENT"
        /bin/ln -s "$SHARED_SEGMENT" "$RIGHT_SEGMENT"
      done
    elif /usr/bin/awk -v duration="$TEST_VIDEO_DURATION" -v maximum="$TEST_SEGMENT_SECONDS" \
      'BEGIN { exit !(duration <= maximum + 0.05) }'; then
      echo "Reusing the prepared SBS test clip directly for both eyes"
      /bin/ln -s "$TEST_INPUT" "$LEFT_SOURCE_DIR/left-00000.mov"
      /bin/ln -s "$TEST_INPUT" "$RIGHT_SOURCE_DIR/right-00000.mov"
    else
      echo "Packet-copying SBS source into ${TEST_SEGMENT_SECONDS}-second shared segments"
      "$FFMPEG_PATH" -hide_banner -i "$TEST_INPUT" -map '0:v:0' -an -c copy \
        -f segment -segment_format mov -segment_time "$TEST_SEGMENT_SECONDS" \
        -segment_time_delta 0.016667 -reset_timestamps 1 \
        "$SHARED_SOURCE_DIR/shared-%05d.mov"
      link_shared_sbs_segments || {
        echo "error: shared SBS source preparation produced no valid segments" >&2
        exit 1
      }
    fi
  elif [[ -n "$MOSAIC_RANGES" ]]; then
    read -r -a ACTIVE_SOURCE_SEGMENTS <<< "$(
      /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" indices \
        "$MOSAIC_RANGES_RELATIVE" "$TEST_VIDEO_DURATION" "$TEST_SEGMENT_SECONDS"
    )"
    (( ${#ACTIVE_SOURCE_SEGMENTS[@]} > 0 )) || {
      echo "error: the manual mosaic ranges do not intersect the selected video" >&2
      exit 1
    }
    echo "Preparing left/right 4K video only for ${#ACTIVE_SOURCE_SEGMENTS[@]} active ${TEST_SEGMENT_SECONDS}-second segment(s)"
    for SEGMENT_INDEX in "${ACTIVE_SOURCE_SEGMENTS[@]}"; do
      SEGMENT_OFFSET=$((SEGMENT_INDEX * TEST_SEGMENT_SECONDS))
      SEGMENT_DURATION="$(/usr/bin/awk \
        -v offset="$SEGMENT_OFFSET" \
        -v duration="$TEST_VIDEO_DURATION" \
        -v maximum="$TEST_SEGMENT_SECONDS" \
        'BEGIN { remaining = duration - offset; if (remaining > maximum) remaining = maximum; printf "%.6f\n", remaining }'
      )"
      LEFT_SEGMENT="$LEFT_SOURCE_DIR/$(printf 'left-%05d.mov' "$SEGMENT_INDEX")"
      RIGHT_SEGMENT="$RIGHT_SOURCE_DIR/$(printf 'right-%05d.mov' "$SEGMENT_INDEX")"
      echo "Active source segment $(printf '%05d' "$SEGMENT_INDEX"): timeline ${SEGMENT_OFFSET}s-$((SEGMENT_OFFSET + TEST_SEGMENT_SECONDS))s"
      "$FFMPEG_PATH" \
        -hide_banner \
        -ss "$SEGMENT_OFFSET" \
        -i "$TEST_INPUT" \
        -filter_complex \
          "[0:v:0]split=2[leftbase][rightbase];[leftbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:0:0[left];[rightbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:${EYE_WIDTH}:0[right]" \
        -map '[left]' -t "$SEGMENT_DURATION" -an -c:v hevc_videotoolbox \
        "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p -b:v "$EYE_BITRATE" \
        -maxrate "$((EYE_BITRATE * 3 / 2))" -bufsize "$((EYE_BITRATE * 3))" \
        -g 30 -tag:v hvc1 -movflags +faststart "$LEFT_SEGMENT" \
        -map '[right]' -t "$SEGMENT_DURATION" -an -c:v hevc_videotoolbox \
        "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p -b:v "$EYE_BITRATE" \
        -maxrate "$((EYE_BITRATE * 3 / 2))" -bufsize "$((EYE_BITRATE * 3))" \
        -g 30 -tag:v hvc1 -movflags +faststart "$RIGHT_SEGMENT"
    done
  else
    echo "Preparing left/right 4K segments with one shared 8K decode"
    "$FFMPEG_PATH" \
      -hide_banner \
      -i "$TEST_INPUT" \
      -filter_complex \
        "[0:v:0]split=2[leftbase][rightbase];[leftbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:0:0[left];[rightbase]crop=${EYE_WIDTH}:${SOURCE_HEIGHT}:${EYE_WIDTH}:0[right]" \
      -map '[left]' -an -c:v hevc_videotoolbox \
      "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p -b:v "$EYE_BITRATE" \
      -maxrate "$((EYE_BITRATE * 3 / 2))" -bufsize "$((EYE_BITRATE * 3))" \
      -g 30 -force_key_frames "expr:gte(t,n_forced*${TEST_SEGMENT_SECONDS})" \
      -tag:v hvc1 -f segment -segment_format mov \
      -segment_time "$TEST_SEGMENT_SECONDS" -segment_time_delta 0.016667 \
      -reset_timestamps 1 "$LEFT_SOURCE_DIR/left-%05d.mov" \
      -map '[right]' -an -c:v hevc_videotoolbox \
      "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p -b:v "$EYE_BITRATE" \
      -maxrate "$((EYE_BITRATE * 3 / 2))" -bufsize "$((EYE_BITRATE * 3))" \
      -g 30 -force_key_frames "expr:gte(t,n_forced*${TEST_SEGMENT_SECONDS})" \
      -tag:v hvc1 -f segment -segment_format mov \
      -segment_time "$TEST_SEGMENT_SECONDS" -segment_time_delta 0.016667 \
      -reset_timestamps 1 "$RIGHT_SOURCE_DIR/right-%05d.mov"
  fi
  PREPARED_LEFT_SEGMENTS=("$LEFT_SOURCE_DIR"/left-*.mov)
  PREPARED_RIGHT_SEGMENTS=("$RIGHT_SOURCE_DIR"/right-*.mov)
  [[ -e "${PREPARED_LEFT_SEGMENTS[0]}" \
    && -e "${PREPARED_RIGHT_SEGMENTS[0]}" \
    && ${#PREPARED_LEFT_SEGMENTS[@]} -eq ${#PREPARED_RIGHT_SEGMENTS[@]} ]] || {
      echo "error: shared stereo source preparation did not produce both eyes" >&2
      exit 1
  }
  if ! stereo_source_segments_valid; then
    if [[ "$SHARED_SBS_SOURCE" != "1" ]]; then
      echo "error: shared stereo source preparation produced invalid eye timelines" >&2
      exit 1
    fi
    FALLBACK_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
    echo "WARNING: packet-copied SBS segments were not frame-exact or decodable"
    echo "Falling back to one hardware SBS encode with exact segment keyframes"
    mv "$LEFT_SOURCE_DIR" \
      "$LEFT_EYE_WORK_DIR/source.packet-copy-invalid-$FALLBACK_TIMESTAMP"
    mv "$RIGHT_SOURCE_DIR" \
      "$RIGHT_EYE_WORK_DIR/source.packet-copy-invalid-$FALLBACK_TIMESTAMP"
    mv "$SHARED_SOURCE_DIR" \
      "$SHARED_SOURCE_DIR.packet-copy-invalid-$FALLBACK_TIMESTAMP"
    mkdir -p "$LEFT_SOURCE_DIR" "$RIGHT_SOURCE_DIR" "$SHARED_SOURCE_DIR"
    prepare_reencoded_shared_sbs_segments || {
      echo "error: hardware fallback did not produce shared SBS segments" >&2
      exit 1
    }
    stereo_source_segments_valid || {
      echo "error: hardware fallback produced invalid shared SBS timelines" >&2
      exit 1
    }
  fi
  /usr/bin/touch "$LEFT_SOURCE_DONE" "$RIGHT_SOURCE_DONE"
  if [[ "$SHARED_SBS_SOURCE" == "1" ]]; then
    echo "Shared SBS source preparation complete (no temporary 4K eye encodes)"
  else
    echo "Shared stereo source preparation complete"
  fi
fi

if [[ "$STEREO_DETECT" == "1" && "$SHARED_SBS_SOURCE" == "1" \
  && ( "$DETECTOR" == "rfdetr-vr-v1" || "$DETECTOR" == "rfdetr-v6" ) ]]; then
  LEFT_DETECT_DIR="$LEFT_EYE_WORK_DIR/restored-sparse-crop-v22-stable-balanced-fisheye-$DETECTOR"
  RIGHT_DETECT_DIR="$RIGHT_EYE_WORK_DIR/restored-sparse-crop-v22-stable-balanced-fisheye-$DETECTOR"
  mkdir -p "$LEFT_DETECT_DIR" "$RIGHT_DETECT_DIR"
  echo "Preparing paired eye manifests with one shared RF-DETR model and SBS decode"
  for LEFT_SOURCE_SEGMENT in "$LEFT_SOURCE_DIR"/left-*.mov; do
    [[ -e "$LEFT_SOURCE_SEGMENT" ]] || continue
    SEGMENT_STEM="$(basename "$LEFT_SOURCE_SEGMENT" .mov)"
    SEGMENT_INDEX="$((10#${SEGMENT_STEM##*-}))"
    RIGHT_SOURCE_SEGMENT="$RIGHT_SOURCE_DIR/$(printf 'right-%05d.mov' "$SEGMENT_INDEX")"
    [[ -e "$RIGHT_SOURCE_SEGMENT" ]] || {
      echo "error: missing right-eye source segment for $SEGMENT_STEM" >&2
      exit 1
    }
    LEFT_STEREO_MANIFEST="$LEFT_DETECT_DIR/$(printf 'left-%05d-mosaic-regions.json' "$SEGMENT_INDEX")"
    RIGHT_STEREO_MANIFEST="$RIGHT_DETECT_DIR/$(printf 'right-%05d-mosaic-regions.json' "$SEGMENT_INDEX")"
    if [[ -s "$LEFT_STEREO_MANIFEST" && -s "$RIGHT_STEREO_MANIFEST" ]]; then
      echo "Reusing paired mosaic-region manifests for segment $(printf '%05d' "$SEGMENT_INDEX")"
      continue
    fi
    if [[ -n "$MOSAIC_RANGES" ]]; then
      SEGMENT_DURATION="$(video_duration "$LEFT_SOURCE_SEGMENT")"
      SEGMENT_OFFSET=$((SEGMENT_INDEX * TEST_SEGMENT_SECONDS))
      SEGMENT_ACTIVE_RANGES="$(
        /usr/bin/python3 "$ROOT_DIR/tools/mosaic_time_ranges.py" segment \
          "$MOSAIC_RANGES_RELATIVE" "$SEGMENT_OFFSET" "$SEGMENT_DURATION"
      )"
      JASNA_DETECT_ACTIVE_RANGES="$SEGMENT_ACTIVE_RANGES" \
        "$ROOT_DIR/script/scan_mosaic_regions.sh" \
          "$LEFT_SOURCE_SEGMENT" "$LEFT_STEREO_MANIFEST" "$RIGHT_STEREO_MANIFEST"
    else
      "$ROOT_DIR/script/scan_mosaic_regions.sh" \
        "$LEFT_SOURCE_SEGMENT" "$LEFT_STEREO_MANIFEST" "$RIGHT_STEREO_MANIFEST"
    fi
  done
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
LEFT_JOB_INDICES=()
RIGHT_JOB_INPUTS=()
RIGHT_JOB_MANIFESTS=()
RIGHT_JOB_CACHES=()
RIGHT_JOB_INDICES=()
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
      JOB_STEM="$(basename "$JOB_INPUT" .mov)"
      LEFT_JOB_INDICES+=("$((10#${JOB_STEM##*-}))")
      ;;
    right-*)
      RIGHT_JOB_INPUTS+=("$JOB_INPUT")
      RIGHT_JOB_MANIFESTS+=("$JOB_MANIFEST")
      RIGHT_JOB_CACHES+=("$JOB_CACHE")
      JOB_STEM="$(basename "$JOB_INPUT" .mov)"
      RIGHT_JOB_INDICES+=("$((10#${JOB_STEM##*-}))")
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
  for ((JOB_ARRAY_INDEX = 0; JOB_ARRAY_INDEX < ${#LEFT_JOB_INDICES[@]}; JOB_ARRAY_INDEX++)); do
    [[ "${LEFT_JOB_INDICES[$JOB_ARRAY_INDEX]}" == "${RIGHT_JOB_INDICES[$JOB_ARRAY_INDEX]}" ]] || {
      echo "error: direct SBS restoration has mismatched left/right timeline segments" >&2
      exit 1
    }
  done
  (( ${#LEFT_JOB_INPUTS[@]} > 0 )) || {
    echo "error: no paired eye segments were prepared for direct SBS restoration" >&2
    exit 1
  }
  STEREO_MANIFEST_RECONCILE="${JASNA_STEREO_MANIFEST_RECONCILE:-1}"
  [[ "$STEREO_MANIFEST_RECONCILE" == "0" \
    || "$STEREO_MANIFEST_RECONCILE" == "1" ]] || {
    echo "error: JASNA_STEREO_MANIFEST_RECONCILE must be 0 or 1" >&2
    exit 1
  }
  if [[ "$STEREO_MANIFEST_RECONCILE" == "1" ]]; then
    RECONCILED_MANIFEST_DIR="$WORK_DIR/stereo-reconciled-manifests"
    mkdir -p "$RECONCILED_MANIFEST_DIR"
    for ((JOB_INDEX = 0; JOB_INDEX < ${#LEFT_JOB_INPUTS[@]}; JOB_INDEX++)); do
      RECONCILED_LEFT="$RECONCILED_MANIFEST_DIR/$(printf 'left-%05d.json' "$JOB_INDEX")"
      RECONCILED_RIGHT="$RECONCILED_MANIFEST_DIR/$(printf 'right-%05d.json' "$JOB_INDEX")"
      /usr/bin/python3 "$ROOT_DIR/tools/reconcile_stereo_manifests.py" \
        "${LEFT_JOB_MANIFESTS[$JOB_INDEX]}" \
        "${RIGHT_JOB_MANIFESTS[$JOB_INDEX]}" \
        "$RECONCILED_LEFT" "$RECONCILED_RIGHT"
      LEFT_JOB_MANIFESTS[$JOB_INDEX]="$RECONCILED_LEFT"
      RIGHT_JOB_MANIFESTS[$JOB_INDEX]="$RECONCILED_RIGHT"
    done
  fi
  mkdir -p "$DIRECT_SEGMENT_DIR"
  DIRECT_SEGMENTS=()
  BYPASSED_WINDOW_COUNT=0
  RESTORED_WINDOW_COUNT=0
  TOTAL_TIMELINE_SEGMENTS=$((
    (EXPECTED_FRAME_COUNT + TEST_SEGMENT_SECONDS * 30 - 1) \
      / (TEST_SEGMENT_SECONDS * 30)
  ))
  ACTIVE_JOB_CURSOR=0
  echo "Metal process isolation: at most $METAL_WINDOWS_PER_PROCESS temporal windows/process"
  for ((JOB_INDEX = 0; JOB_INDEX < TOTAL_TIMELINE_SEGMENTS; JOB_INDEX++)); do
    SEGMENT_START_FRAME=$((JOB_INDEX * TEST_SEGMENT_SECONDS * 30))
    JOB_FRAME_COUNT=$((EXPECTED_FRAME_COUNT - SEGMENT_START_FRAME))
    (( JOB_FRAME_COUNT > TEST_SEGMENT_SECONDS * 30 )) \
      && JOB_FRAME_COUNT=$((TEST_SEGMENT_SECONDS * 30))
    JOB_WINDOW_COUNT=$(( (JOB_FRAME_COUNT + 29) / 30 ))
    BATCH_SEGMENT="$DIRECT_SEGMENT_DIR/$(
      printf 'segment-%05d-windows-%05d-%05d.mov' \
        "$JOB_INDEX" 0 "$JOB_WINDOW_COUNT"
    )"
    if (( ACTIVE_JOB_CURSOR >= ${#LEFT_JOB_INDICES[@]} )) \
      || (( LEFT_JOB_INDICES[ACTIVE_JOB_CURSOR] != JOB_INDEX )); then
      if valid_direct_segment "$BATCH_SEGMENT" "$JOB_FRAME_COUNT"; then
        DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
        echo "Reusing clean SBS timeline segment $((JOB_INDEX + 1))/$TOTAL_TIMELINE_SEGMENTS"
        continue
      fi
      CLEAN_TEMP="${BATCH_SEGMENT%.mov}.bypass-writing.mov"
      [[ ! -e "$CLEAN_TEMP" ]] || mv "$CLEAN_TEMP" \
        "${CLEAN_TEMP%.mov}.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
      SEGMENT_START_SECONDS=$((JOB_INDEX * TEST_SEGMENT_SECONDS))
      SEGMENT_DURATION="$(/usr/bin/awk \
        -v frames="$JOB_FRAME_COUNT" 'BEGIN { printf "%.6f\n", frames / 30 }'
      )"
      echo "Bypassing clean SBS timeline segment $((JOB_INDEX + 1))/$TOTAL_TIMELINE_SEGMENTS without eye conversion"
      "$FFMPEG_PATH" -hide_banner -loglevel error \
        -ss "$SEGMENT_START_SECONDS" -i "$TEST_INPUT" \
        -t "$SEGMENT_DURATION" -map '0:v:0' -an -c copy \
        -avoid_negative_ts make_zero -video_track_timescale 600 \
        -movflags +faststart "$CLEAN_TEMP"
      valid_direct_segment "$CLEAN_TEMP" "$JOB_FRAME_COUNT" || {
        echo "error: clean SBS timeline segment failed validation: $CLEAN_TEMP" >&2
        exit 1
      }
      mv "$CLEAN_TEMP" "$BATCH_SEGMENT"
      DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
      BYPASSED_WINDOW_COUNT=$((BYPASSED_WINDOW_COUNT + JOB_WINDOW_COUNT))
      continue
    fi
    JOB_ARRAY_INDEX="$ACTIVE_JOB_CURSOR"
    ACTIVE_JOB_CURSOR=$((ACTIVE_JOB_CURSOR + 1))
    JOB_LEFT_INPUT="${LEFT_JOB_INPUTS[$JOB_ARRAY_INDEX]}"
    JOB_RIGHT_INPUT="${RIGHT_JOB_INPUTS[$JOB_ARRAY_INDEX]}"
    JOB_LEFT_MANIFEST="${LEFT_JOB_MANIFESTS[$JOB_ARRAY_INDEX]}"
    JOB_RIGHT_MANIFEST="${RIGHT_JOB_MANIFESTS[$JOB_ARRAY_INDEX]}"
    JOB_LEFT_CACHE="${LEFT_JOB_CACHES[$JOB_ARRAY_INDEX]}"
    JOB_RIGHT_CACHE="${RIGHT_JOB_CACHES[$JOB_ARRAY_INDEX]}"
    JOB_FRAME_COUNT="$(
      "$FFPROBE_PATH" -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \
        "$JOB_LEFT_INPUT"
    )"
    if [[ ! "$JOB_FRAME_COUNT" =~ ^[0-9]+$ ]]; then
      JOB_DURATION="$(video_duration "$JOB_LEFT_INPUT")"
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
      echo "Reusing validated ${TEST_SEGMENT_SECONDS}-second SBS batch $((JOB_INDEX + 1))/${#LEFT_JOB_INPUTS[@]}"
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
      WINDOW_ACTIVITY="$(
        /usr/bin/python3 "$ROOT_DIR/tools/manifest_window_runs.py" \
          "$JOB_LEFT_MANIFEST" \
          "$JOB_RIGHT_MANIFEST" \
          "$WINDOW_START"
      )"
      IFS=$'\t' read -r WINDOW_MODE WINDOW_RUN_COUNT WINDOW_EXTRA <<< "$WINDOW_ACTIVITY"
      [[ ( "$WINDOW_MODE" == "active" || "$WINDOW_MODE" == "empty" ) \
        && "$WINDOW_RUN_COUNT" =~ ^[1-9][0-9]*$ && -z "$WINDOW_EXTRA" ]] || {
        echo "error: invalid manifest window activity: $WINDOW_ACTIVITY" >&2
        exit 1
      }
      WINDOW_COUNT="$WINDOW_RUN_COUNT"
      if [[ "$WINDOW_MODE" == "active" ]] \
        && (( WINDOW_COUNT > METAL_WINDOWS_PER_PROCESS )); then
        WINDOW_COUNT="$METAL_WINDOWS_PER_PROCESS"
      fi
      WINDOW_END=$((WINDOW_START + WINDOW_COUNT))
      DIRECT_SEGMENT="$DIRECT_SEGMENT_DIR/$(
        printf 'segment-%05d-windows-%05d-%05d.mov' \
          "$JOB_INDEX" "$WINDOW_START" "$WINDOW_END"
      )"
      JOB_DIRECT_SEGMENTS+=("$DIRECT_SEGMENT")
      BYPASS_SUCCEEDED=0
      if [[ "$WINDOW_MODE" == "empty" ]]; then
        BYPASS_TEMP="${DIRECT_SEGMENT%.mov}.bypass-writing.mov"
        if [[ -e "$BYPASS_TEMP" ]]; then
          mv "$BYPASS_TEMP" \
            "${BYPASS_TEMP%.mov}.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
        fi
        GLOBAL_START_SECONDS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + WINDOW_START))
        EXPECTED_BYPASS_FRAMES=$((WINDOW_COUNT * 30))
        echo "Bypassing empty source windows $((WINDOW_START + 1))-$WINDOW_END/$JOB_WINDOW_COUNT with SBS packet copy"
        if "$FFMPEG_PATH" -hide_banner -loglevel error \
            -ss "$GLOBAL_START_SECONDS" -i "$TEST_INPUT" \
            -t "$WINDOW_COUNT" -map '0:v:0' -an -c copy \
            -avoid_negative_ts make_zero -video_track_timescale 600 \
            -movflags +faststart "$BYPASS_TEMP" \
          && valid_direct_segment "$BYPASS_TEMP" "$EXPECTED_BYPASS_FRAMES"; then
          mv "$BYPASS_TEMP" "$DIRECT_SEGMENT"
          BYPASS_SUCCEEDED=1
          BYPASSED_WINDOW_COUNT=$((BYPASSED_WINDOW_COUNT + WINDOW_COUNT))
          echo "Bypassed and validated $WINDOW_COUNT empty window(s): $DIRECT_SEGMENT"
        else
          if [[ -e "$BYPASS_TEMP" ]]; then
            mv "$BYPASS_TEMP" \
              "${BYPASS_TEMP%.mov}.invalid-$(date '+%Y%m%d-%H%M%S').mov"
          fi
          echo "WARNING: empty-window packet copy was not frame-exact; using Metal writer fallback"
        fi
      fi
      if [[ "$BYPASS_SUCCEEDED" == "0" ]]; then
        echo "Restoring source segment $((JOB_INDEX + 1))/${#LEFT_JOB_INPUTS[@]}, windows $((WINDOW_START + 1))-$WINDOW_END/$JOB_WINDOW_COUNT"
        GPU_TIMEOUT_RETRIES="${JASNA_GPU_TIMEOUT_RETRIES:-2}"
        [[ "$GPU_TIMEOUT_RETRIES" =~ ^[0-9]+$ ]] || {
          echo "error: JASNA_GPU_TIMEOUT_RETRIES must be a non-negative integer" >&2
          exit 1
        }
        RESTORE_ATTEMPT=0
        while true; do
          RESTORE_STATUS=0
          RESTORE_LOG="${DIRECT_SEGMENT%.mov}.jasna.log"
          RESTORE_LOG_START_LINES=0
          if [[ -f "$RESTORE_LOG" ]]; then
            RESTORE_LOG_START_LINES="$(/usr/bin/wc -l < "$RESTORE_LOG")"
            RESTORE_LOG_START_LINES="${RESTORE_LOG_START_LINES//[[:space:]]/}"
          fi
          JASNA_WINDOW_START="$WINDOW_START" \
          JASNA_WINDOW_COUNT="$WINDOW_COUNT" \
          JASNA_VIDEO_BITRATE="$VR_BITRATE" \
          JASNA_VR_PROJECTION=fisheye \
            "$ROOT_DIR/script/build_and_run.sh" --restore-stereo-sparse-batch \
              "$JOB_LEFT_INPUT" \
              "$JOB_RIGHT_INPUT" \
              "$DIRECT_SEGMENT" \
              "$JOB_LEFT_MANIFEST" \
              "$JOB_RIGHT_MANIFEST" \
              "$JOB_LEFT_CACHE" \
              "$JOB_RIGHT_CACHE" || RESTORE_STATUS=$?
          (( RESTORE_STATUS == 0 )) && break
          if ! /usr/bin/tail -n "+$((RESTORE_LOG_START_LINES + 1))" \
              "$RESTORE_LOG" 2>/dev/null \
            | /usr/bin/grep 'GPU Timeout Error' >/dev/null; then
            echo "error: Metal restoration failed for a reason other than GPU timeout" >&2
            exit "$RESTORE_STATUS"
          fi
          if (( RESTORE_ATTEMPT >= GPU_TIMEOUT_RETRIES )); then
            echo "error: Metal restoration exhausted $GPU_TIMEOUT_RETRIES fresh-process GPU-timeout retries" >&2
            exit "$RESTORE_STATUS"
          fi
          RESTORE_ATTEMPT=$((RESTORE_ATTEMPT + 1))
          echo "WARNING: Metal GPU timeout; restarting the process and resuming its checkpoint ($RESTORE_ATTEMPT/$GPU_TIMEOUT_RETRIES)"
        done
        RESTORED_WINDOW_COUNT=$((RESTORED_WINDOW_COUNT + WINDOW_COUNT))
      fi
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
      echo "Joining source segment $((JOB_INDEX + 1))/$TOTAL_TIMELINE_SEGMENTS into one ${TEST_SEGMENT_SECONDS}-second SBS batch"
      "$FFMPEG_PATH" -hide_banner -loglevel error \
        -f concat -safe 0 -i "$BATCH_CONCAT" -map '0:v:0' -c copy \
        -movflags +faststart "$BATCH_TEMP"
      valid_direct_segment "$BATCH_TEMP" "$JOB_FRAME_COUNT" || {
        echo "error: joined ${TEST_SEGMENT_SECONDS}-second SBS batch failed validation: $BATCH_TEMP" >&2
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
      echo "Validated ${TEST_SEGMENT_SECONDS}-second SBS batch: $BATCH_SEGMENT"
    fi
    DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
  done
  (( ACTIVE_JOB_CURSOR == ${#LEFT_JOB_INPUTS[@]} )) || {
    echo "error: not all active eye segments were placed on the SBS timeline" >&2
    exit 1
  }
  echo "Prepared ${#DIRECT_SEGMENTS[@]} grouped 8K SBS batch(es); bypassed/restored windows $BYPASSED_WINDOW_COUNT/$RESTORED_WINDOW_COUNT"

  mkdir -p "$DIRECT_NORMALIZED_DIR"
  : > "$DIRECT_CONCAT_PATH"
  for DIRECT_SEGMENT in "${DIRECT_SEGMENTS[@]}"; do
    [[ -s "$DIRECT_SEGMENT" ]] || {
      echo "error: direct SBS segment is missing: $DIRECT_SEGMENT" >&2
      exit 1
    }
    CONCAT_SEGMENT="$DIRECT_SEGMENT"
    SEGMENT_TIME_BASE="$(
      "$FFPROBE_PATH" -v error -select_streams v:0 \
        -show_entries stream=time_base \
        -of default=noprint_wrappers=1:nokey=1 "$DIRECT_SEGMENT"
    )"
    if [[ "$SEGMENT_TIME_BASE" != "1/600" ]]; then
      SEGMENT_FRAME_COUNT="$(
        "$FFPROBE_PATH" -v error -select_streams v:0 \
          -show_entries stream=nb_frames \
          -of default=noprint_wrappers=1:nokey=1 "$DIRECT_SEGMENT"
      )"
      [[ "$SEGMENT_FRAME_COUNT" =~ ^[0-9]+$ ]] || {
        echo "error: unable to read direct SBS segment frame count: $DIRECT_SEGMENT" >&2
        exit 1
      }
      NORMALIZED_SEGMENT="$DIRECT_NORMALIZED_DIR/$(basename "${DIRECT_SEGMENT%.mov}").timescale-600.mov"
      NORMALIZED_TIME_BASE="$(
        "$FFPROBE_PATH" -v error -select_streams v:0 \
          -show_entries stream=time_base \
          -of default=noprint_wrappers=1:nokey=1 "$NORMALIZED_SEGMENT" \
          2>/dev/null || true
      )"
      if ! valid_direct_segment "$NORMALIZED_SEGMENT" "$SEGMENT_FRAME_COUNT" \
        || [[ "$NORMALIZED_TIME_BASE" != "1/600" ]]; then
        NORMALIZED_TEMP="${NORMALIZED_SEGMENT%.mov}.writing.mov"
        if [[ -e "$NORMALIZED_TEMP" ]]; then
          mv "$NORMALIZED_TEMP" \
            "${NORMALIZED_TEMP%.mov}.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
        fi
        echo "Normalizing restored SBS packet timescale without re-encoding: $(basename "$DIRECT_SEGMENT")"
        "$FFMPEG_PATH" -hide_banner -loglevel error \
          -i "$DIRECT_SEGMENT" -map '0:v:0' -an -c copy \
          -video_track_timescale 600 -movflags +faststart "$NORMALIZED_TEMP"
        valid_direct_segment "$NORMALIZED_TEMP" "$SEGMENT_FRAME_COUNT" || {
          echo "error: normalized direct SBS segment failed validation: $NORMALIZED_TEMP" >&2
          exit 1
        }
        NORMALIZED_TIME_BASE="$(
          "$FFPROBE_PATH" -v error -select_streams v:0 \
            -show_entries stream=time_base \
            -of default=noprint_wrappers=1:nokey=1 "$NORMALIZED_TEMP"
        )"
        [[ "$NORMALIZED_TIME_BASE" == "1/600" ]] || {
          echo "error: normalized segment has unexpected time base: $NORMALIZED_TIME_BASE" >&2
          exit 1
        }
        mv "$NORMALIZED_TEMP" "$NORMALIZED_SEGMENT"
      fi
      CONCAT_SEGMENT="$NORMALIZED_SEGMENT"
    fi
    ESCAPED_SEGMENT="${CONCAT_SEGMENT//\'/\'\\\'\'}"
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
      -video_track_timescale 600 \
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
  report_compositor_fallback_summary
  report_total_wall_time
  echo "Output: $OUTPUT_PATH"
  echo "Log:    $LOG_PATH"
  cleanup_successful_work
  exit 0
elif (( ${#SHARED_BATCH_ARGS[@]} > 0 )); then
  EYE_JOB_COUNT=$((${#SHARED_BATCH_ARGS[@]} / 4))
  if [[ "$EYE_JOB_PROCESS_ISOLATION" == "1" ]]; then
    echo "Restoring $EYE_JOB_COUNT ${TEST_SEGMENT_SECONDS}-second 4K eye job(s) in isolated Metal processes"
    for ((ARG_INDEX = 0; ARG_INDEX < ${#SHARED_BATCH_ARGS[@]}; ARG_INDEX += 4)); do
      JOB_NUMBER=$((ARG_INDEX / 4 + 1))
      JOB_INPUT="${SHARED_BATCH_ARGS[$ARG_INDEX]}"
      echo "Starting isolated 4K eye job $JOB_NUMBER/$EYE_JOB_COUNT: $(basename "$JOB_INPUT")"
      JASNA_VR_PROJECTION=fisheye \
        "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows-sparse-batch \
          "$JOB_INPUT" \
          "${SHARED_BATCH_ARGS[$((ARG_INDEX + 1))]}" \
          "${SHARED_BATCH_ARGS[$((ARG_INDEX + 2))]}" \
          "${SHARED_BATCH_ARGS[$((ARG_INDEX + 3))]}"
      echo "Finished isolated 4K eye job $JOB_NUMBER/$EYE_JOB_COUNT; Metal process released"
    done
  else
    echo "Restoring $EYE_JOB_COUNT left/right segment job(s) with one retained Metal ML graph"
    JASNA_VR_PROJECTION=fisheye \
      "$ROOT_DIR/script/build_and_run.sh" --restore-eye-windows-sparse-batch \
        "${SHARED_BATCH_ARGS[@]}"
  fi
else
  echo "All left/right restoration windows are already complete"
fi

if [[ "$EYE_PAIR_SEGMENTS" == "1" ]]; then
  LEFT_RESTORED_SEGMENT_FILE="$WORK_DIR/left-restored-segments.txt"
  RIGHT_RESTORED_SEGMENT_FILE="$WORK_DIR/right-restored-segments.txt"

  echo "Stage 3/4: finalizing ${TEST_SEGMENT_SECONDS}-second 4K eye segments without full-eye movies"
  JASNA_SPARSE_BATCH_MODE=finalize-segments \
  JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
  JASNA_RESTORED_SEGMENT_FILE="$LEFT_RESTORED_SEGMENT_FILE" \
  JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
  JASNA_EYE_BITRATE="$EYE_BITRATE" \
  JASNA_VR_PROJECTION=fisheye \
    "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
      "$TEST_INPUT" left "$LEFT_OUTPUT"

  JASNA_SPARSE_BATCH_MODE=finalize-segments \
  JASNA_SPARSE_BATCH_FILE="$SHARED_BATCH_PATH" \
  JASNA_RESTORED_SEGMENT_FILE="$RIGHT_RESTORED_SEGMENT_FILE" \
  JASNA_SEGMENT_SECONDS="$TEST_SEGMENT_SECONDS" \
  JASNA_EYE_BITRATE="$EYE_BITRATE" \
  JASNA_VR_PROJECTION=fisheye \
    "$ROOT_DIR/script/restore_vr_eye_sparse.sh" \
      "$TEST_INPUT" right "$RIGHT_OUTPUT"

  LEFT_RESTORED_SEGMENTS=()
  while IFS= read -r RESTORED_SEGMENT; do
    [[ -n "$RESTORED_SEGMENT" ]] && LEFT_RESTORED_SEGMENTS+=("$RESTORED_SEGMENT")
  done < "$LEFT_RESTORED_SEGMENT_FILE"
  RIGHT_RESTORED_SEGMENTS=()
  while IFS= read -r RESTORED_SEGMENT; do
    [[ -n "$RESTORED_SEGMENT" ]] && RIGHT_RESTORED_SEGMENTS+=("$RESTORED_SEGMENT")
  done < "$RIGHT_RESTORED_SEGMENT_FILE"
  (( ${#LEFT_RESTORED_SEGMENTS[@]} > 0 \
    && ${#LEFT_RESTORED_SEGMENTS[@]} == ${#RIGHT_RESTORED_SEGMENTS[@]} )) || {
      echo "error: restored left/right segment lists are empty or mismatched" >&2
      exit 1
    }

  PAIR_SEGMENT_DIR="$WORK_DIR/eye-pair-sbs-segments"
  PAIR_CONCAT_PATH="$WORK_DIR/eye-pair-sbs-concat.txt"
  mkdir -p "$PAIR_SEGMENT_DIR"
  : > "$PAIR_CONCAT_PATH"
  echo "Stage 4/4: combining ${#LEFT_RESTORED_SEGMENTS[@]} eye pair(s) into resumable SBS segments"
  for ((PAIR_INDEX = 0; PAIR_INDEX < ${#LEFT_RESTORED_SEGMENTS[@]}; PAIR_INDEX++)); do
    LEFT_SEGMENT="${LEFT_RESTORED_SEGMENTS[$PAIR_INDEX]}"
    RIGHT_SEGMENT="${RIGHT_RESTORED_SEGMENTS[$PAIR_INDEX]}"
    LEFT_NAME="$(basename "$LEFT_SEGMENT")"
    RIGHT_NAME="$(basename "$RIGHT_SEGMENT")"
    [[ "$LEFT_NAME" =~ ^left-([0-9]{5})-restored[.]mov$ ]] || {
      echo "error: invalid restored left-eye segment name: $LEFT_NAME" >&2
      exit 1
    }
    PAIR_NUMBER="${BASH_REMATCH[1]}"
    [[ "$RIGHT_NAME" == "right-${PAIR_NUMBER}-restored.mov" ]] || {
      echo "error: restored eye segment timelines do not match: $LEFT_NAME / $RIGHT_NAME" >&2
      exit 1
    }
    EXPECTED_PAIR_FRAMES="$(
      "$FFPROBE_PATH" -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \
        "$LEFT_SEGMENT"
    )"
    [[ "$EXPECTED_PAIR_FRAMES" =~ ^[0-9]+$ ]] || {
      echo "error: unable to read restored frame count: $LEFT_SEGMENT" >&2
      exit 1
    }
    valid_eye_source_segment "$LEFT_SEGMENT" "$EXPECTED_PAIR_FRAMES" || {
      echo "error: invalid restored left-eye segment: $LEFT_SEGMENT" >&2
      exit 1
    }
    valid_eye_source_segment "$RIGHT_SEGMENT" "$EXPECTED_PAIR_FRAMES" || {
      echo "error: invalid restored right-eye segment: $RIGHT_SEGMENT" >&2
      exit 1
    }

    PAIR_SEGMENT="$PAIR_SEGMENT_DIR/segment-${PAIR_NUMBER}.mov"
    PAIR_TEMP="$PAIR_SEGMENT_DIR/.segment-${PAIR_NUMBER}-writing.mov"
    if valid_direct_segment "$PAIR_SEGMENT" "$EXPECTED_PAIR_FRAMES"; then
      echo "Reusing validated SBS eye pair $((PAIR_INDEX + 1))/${#LEFT_RESTORED_SEGMENTS[@]}"
    else
      if [[ -e "$PAIR_TEMP" ]]; then
        mv "$PAIR_TEMP" \
          "$PAIR_SEGMENT_DIR/segment-${PAIR_NUMBER}.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
      fi
      echo "Encoding SBS eye pair $((PAIR_INDEX + 1))/${#LEFT_RESTORED_SEGMENTS[@]}: $PAIR_NUMBER"
      "$FFMPEG_PATH" \
        -hide_banner \
        -i "$LEFT_SEGMENT" \
        -i "$RIGHT_SEGMENT" \
        -filter_complex '[0:v:0][1:v:0]hstack=inputs=2[v]' \
        -map '[v]' \
        -an \
        -c:v hevc_videotoolbox \
        "${ENCODER_SPEED_ARGS[@]}" \
        -pix_fmt yuv420p \
        -b:v "$VR_BITRATE" \
        -maxrate "$((VR_BITRATE * 3 / 2))" \
        -bufsize "$((VR_BITRATE * 3))" \
        -g 30 \
        -tag:v hvc1 \
        -r 30 \
        -video_track_timescale 600 \
        -movflags +faststart \
        -shortest \
        -n \
        "$PAIR_TEMP"
      valid_direct_segment "$PAIR_TEMP" "$EXPECTED_PAIR_FRAMES" || {
        echo "error: combined SBS eye pair failed validation: $PAIR_TEMP" >&2
        exit 1
      }
      if [[ -e "$PAIR_SEGMENT" ]]; then
        mv "$PAIR_SEGMENT" \
          "$PAIR_SEGMENT_DIR/segment-${PAIR_NUMBER}.invalid-$(date '+%Y%m%d-%H%M%S').mov"
      fi
      mv "$PAIR_TEMP" "$PAIR_SEGMENT"
      echo "Validated SBS eye pair: $PAIR_SEGMENT"
    fi
    ESCAPED_PAIR_SEGMENT="${PAIR_SEGMENT//\'/\'\\\'\'}"
    printf "file '%s'\n" "$ESCAPED_PAIR_SEGMENT" >> "$PAIR_CONCAT_PATH"
  done

  if completed_sbs_output "$OUTPUT_PATH"; then
    echo "Final eye-pair SBS output already complete"
  else
    if [[ -e "$FINAL_TEMP" ]]; then
      mv "$FINAL_TEMP" \
        "$WORK_DIR/eye-pair-sbs.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
    fi
    if [[ -e "$OUTPUT_PATH" ]]; then
      mv "$OUTPUT_PATH" \
        "$OUTPUT_DIR/${OUTPUT_STEM}.previous-$(date '+%Y%m%d-%H%M%S').${OUTPUT_NAME##*.}"
    fi
    echo "Joining validated SBS pair segments and copying source audio without re-encoding"
    "$FFMPEG_PATH" \
      -hide_banner \
      -f concat -safe 0 -i "$PAIR_CONCAT_PATH" \
      -i "$TEST_INPUT" \
      -map '0:v:0' \
      -map '1:a?' \
      -map_metadata 1 \
      -map_chapters 1 \
      -c copy \
      -video_track_timescale 600 \
      -movflags +faststart \
      -shortest \
      -n \
      "$FINAL_TEMP"
    completed_sbs_output "$FINAL_TEMP" || {
      echo "error: final eye-pair SBS output failed validation" >&2
      exit 1
    }
    mv "$FINAL_TEMP" "$OUTPUT_PATH"
  fi

  FINAL_INFO="$("$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT_PATH")"
  echo "Sparse eye-pair VR restoration ($RUN_DESCRIPTION): PASS"
  echo "$FINAL_INFO"
  report_compositor_fallback_summary
  report_total_wall_time
  echo "Output:       $OUTPUT_PATH"
  echo "SBS segments: $PAIR_SEGMENT_DIR"
  echo "Log:          $LOG_PATH"
  cleanup_successful_work
  exit 0
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
report_compositor_fallback_summary
report_total_wall_time
echo "Output:   $OUTPUT_PATH"
echo "Left eye: $LEFT_OUTPUT"
echo "Right eye:$RIGHT_OUTPUT"
echo "Log:      $LOG_PATH"
cleanup_successful_work
