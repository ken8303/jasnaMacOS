#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-test.mov 00:12:00" >&2
  echo "optional: JASNA_TEST_SECONDS=30 (1-300, or full)" >&2
  echo "          JASNA_MOSAIC_RANGES=00:12:00-00:14:00,00:20:30-00:22:00" >&2
  echo "          JASNA_ENCODER_WINDOWS_PER_SEGMENT=4 (bounded eye-by-eye disk use)" >&2
  echo "          JASNA_METAL_WINDOWS_PER_PROCESS=2 (balanced default; set 1 for minimum memory or 8 for validated fast mode)" >&2
  echo "          JASNA_GPU_TIMEOUT_RETRIES=2 (fresh-process checkpoint retries)" >&2
  echo "          JASNA_DETECT_DEVICE=auto (MPS with automatic CPU fallback; or force cpu)" >&2
  echo "          JASNA_STEREO_DETECT=1 (one RF-DETR load/decode for both SBS eyes)" >&2
  echo "          JASNA_STEREO_SAMPLE_MODE=paired (experimental: alternating)" >&2
  echo "          JASNA_EYE_BITRATE=20000000 JASNA_VR_BITRATE=40000000" >&2
  echo "          JASNA_DIRECT_SBS_OUTPUT=1 (set 0 for lower-memory eye-by-eye output)" >&2
  echo "          JASNA_EYE_JOB_PROCESS_ISOLATION=1 (fresh process per 30-120 second 4K eye job)" >&2
  echo "          JASNA_EYE_PAIR_SEGMENTS=1 (combine each eye pair before the final join)" >&2
  echo "          JASNA_ALLOW_IMPLEMENTATION_RESUME=1 (one-time reuse after a script update)" >&2
  echo "          JASNA_LARGE_REGION_MAX_BLEND=768 JASNA_LARGE_REGION_OVERLAP=96" >&2
  echo "          JASNA_LARGE_REGION_MASK_GROWTH=0.05 JASNA_LARGE_REGION_MASK_FEATHER=0.025" >&2
  echo "          JASNA_LARGE_REGION_BLOCK_GROWTH=0.04 JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS=1" >&2
  echo "          JASNA_LARGE_REGION_MASK_TEMPORAL_STRENGTH=0.5 (experimental: 1.0)" >&2
  echo "          JASNA_LARGE_REGION_DETAIL_CROPS=1 JASNA_LARGE_REGION_DETAIL_DIMENSION=576" >&2
  echo "          JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=0 (experimental: 1)" >&2
  echo "          JASNA_TEMPORAL_WARMUP_FRAMES=5 (set 0 to disable)" >&2
  echo "          JASNA_STEREO_WRITER_DEPTH=2 (set 1 for minimum output-buffer memory)" >&2
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
REGION_PREPARE_DEPTH="${JASNA_REGION_PREPARE_DEPTH:-1}"
MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
DETECT_DEVICE="${JASNA_DETECT_DEVICE:-auto}"
if [[ "$DETECTOR" == "rfdetr-v6" ]]; then
  DEFAULT_DETECT_CONFIDENCE="0.35"
else
  DEFAULT_DETECT_CONFIDENCE="0.15"
fi
DETECT_CONFIDENCE="${JASNA_DETECT_CONFIDENCE:-$DEFAULT_DETECT_CONFIDENCE}"
IN_MEMORY_CROP_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-0}"
IN_MEMORY_CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-128}"
COMPOSITE_CONCURRENCY="${JASNA_COMPOSITE_CONCURRENCY:-1}"
STEREO_WRITER_DEPTH="${JASNA_STEREO_WRITER_DEPTH:-2}"
RUNTIME_SCRATCH_ON_OUTPUT="${JASNA_RUNTIME_SCRATCH_ON_OUTPUT:-1}"
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
[[ "$COMPOSITE_CONCURRENCY" == "1" || "$COMPOSITE_CONCURRENCY" == "2" ]] || {
  echo "error: JASNA_COMPOSITE_CONCURRENCY must be 1 or 2" >&2
  exit 1
}
[[ "$STEREO_WRITER_DEPTH" == "1" || "$STEREO_WRITER_DEPTH" == "2" ]] || {
  echo "error: JASNA_STEREO_WRITER_DEPTH must be 1 or 2" >&2
  exit 1
}
[[ "$RUNTIME_SCRATCH_ON_OUTPUT" == "0" || "$RUNTIME_SCRATCH_ON_OUTPUT" == "1" ]] || {
  echo "error: JASNA_RUNTIME_SCRATCH_ON_OUTPUT must be 0 or 1" >&2
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
export JASNA_COMPOSITE_CONCURRENCY="$COMPOSITE_CONCURRENCY"
export JASNA_STEREO_WRITER_DEPTH="$STEREO_WRITER_DEPTH"
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
export JASNA_MODEL_BATCH="$MODEL_BATCH"
[[ "$REGION_PREPARE_DEPTH" == "1" || "$REGION_PREPARE_DEPTH" == "2" ]] || {
  echo "error: JASNA_REGION_PREPARE_DEPTH must be 1 or 2" >&2
  exit 1
}
export JASNA_REGION_PREPARE_DEPTH="$REGION_PREPARE_DEPTH"

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
RUNTIME_SCRATCH_DIR=""
cleanup_runtime_scratch() {
  [[ -n "$RUNTIME_SCRATCH_DIR" && -d "$RUNTIME_SCRATCH_DIR" ]] || return 0
  if [[ "$RUNTIME_SCRATCH_DIR" != "$WORK_DIR/runtime-scratch" \
    || ! -f "$RUNTIME_SCRATCH_DIR/.jasna-runtime-scratch" ]]; then
    echo "WARNING: refusing unsafe runtime scratch cleanup: $RUNTIME_SCRATCH_DIR" >&2
    return 0
  fi
  /bin/rm -rf -- "$RUNTIME_SCRATCH_DIR"
}
cleanup_workflow() {
  cleanup_runtime_scratch
  cleanup_workflow_lock
}
prepare_runtime_scratch() {
  [[ "$RUNTIME_SCRATCH_ON_OUTPUT" == "1" ]] || return 0
  RUNTIME_SCRATCH_DIR="$WORK_DIR/runtime-scratch"
  if [[ -e "$RUNTIME_SCRATCH_DIR" ]]; then
    [[ -f "$RUNTIME_SCRATCH_DIR/.jasna-runtime-scratch" ]] || {
      echo "error: refusing to replace unrecognized runtime scratch: $RUNTIME_SCRATCH_DIR" >&2
      exit 1
    }
    /bin/rm -rf -- "$RUNTIME_SCRATCH_DIR"
  fi
  mkdir -p "$RUNTIME_SCRATCH_DIR/tmp" \
    "$RUNTIME_SCRATCH_DIR/cache" \
    "$RUNTIME_SCRATCH_DIR/torchinductor" \
    "$RUNTIME_SCRATCH_DIR/pycache"
  /usr/bin/touch "$RUNTIME_SCRATCH_DIR/.jasna-runtime-scratch"
  export TMPDIR="$RUNTIME_SCRATCH_DIR/tmp/"
  export XDG_CACHE_HOME="$RUNTIME_SCRATCH_DIR/cache"
  export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_SCRATCH_DIR/torchinductor"
  export PYTHONPYCACHEPREFIX="$RUNTIME_SCRATCH_DIR/pycache"
}
trap cleanup_workflow EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM
prepare_runtime_scratch

report_persistent_work() {
  echo "Persistent segments and caches: $WORK_DIR"
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
region_prepare_depth=$REGION_PREPARE_DEPTH
allow_passthrough=${JASNA_ALLOW_PASSTHROUGH:-0}
diagnostic_full_region_blend=${JASNA_DIAGNOSTIC_FULL_REGION_BLEND:-0}
metal_texture_compositor=${JASNA_METAL_TEXTURE_COMPOSITOR:-1}
metal_compositor=${JASNA_METAL_COMPOSITOR:-1}
composite_concurrency=$COMPOSITE_CONCURRENCY
stereo_writer_depth=$STEREO_WRITER_DEPTH
quality_profile=stable-balanced-v22
detect_batch_size=${JASNA_DETECT_BATCH_SIZE:-1}
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
large_region_mask_temporal_strength=${JASNA_LARGE_REGION_MASK_TEMPORAL_STRENGTH:-0.5}
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
      -e '/^region_prepare_depth=/d' \
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
      -e '/^region_prepare_depth=/d' \
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
echo "Frame compositing concurrency: $COMPOSITE_CONCURRENCY"
echo "Direct SBS writer pipeline depth: $STEREO_WRITER_DEPTH"
if [[ "$RUNTIME_SCRATCH_ON_OUTPUT" == "1" ]]; then
  echo "Runtime scratch: $RUNTIME_SCRATCH_DIR (removed when this workflow exits)"
else
  echo "Runtime scratch: macOS system temporary/cache folders"
fi
if [[ "$DIRECT_SBS_OUTPUT" == "0" ]]; then
  echo "Eye workers:  $([[ "$EYE_JOB_PROCESS_ISOLATION" == "1" ]] && echo isolated-${TEST_SEGMENT_SECONDS}s || echo retained-graph)"
  echo "Pair output:  $([[ "$EYE_PAIR_SEGMENTS" == "1" ]] && echo ${TEST_SEGMENT_SECONDS}s-sbs || echo full-eye-join)"
fi
echo "Model batch: $MODEL_BATCH"
echo "Crop preparation pipeline depth: $REGION_PREPARE_DEPTH"
echo "Detector:    $DETECTOR"
echo "Detect device: $DETECT_DEVICE (auto prefers MPS and falls back to CPU)"
if [[ -n "$MOSAIC_RANGES" ]]; then
  echo "Manual mosaic ranges: $MOSAIC_RANGES"
  echo "Only these source-timeline ranges will be detected/restored"
fi

source "$ROOT_DIR/script/lib/vr_sparse_media.sh"
source "$ROOT_DIR/script/lib/vr_sparse_direct.sh"
source "$ROOT_DIR/script/lib/vr_sparse_eye_pair.sh"

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
IFS=, read -r SOURCE_CODEC SOURCE_PROFILE SOURCE_PIXEL_FORMAT < <(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,profile,pix_fmt -of csv=p=0 "$INPUT_PATH"
)
SOURCE_IS_30_FPS=0
if /usr/bin/awk -F/ '
  NF == 2 && $2 != 0 { rate = $1 / $2 }
  NF == 1 { rate = $1 }
  END { exit !(rate >= 29.95 && rate <= 30.05) }
' <<< "$SOURCE_FRAME_RATE"; then
  SOURCE_IS_30_FPS=1
fi
SOURCE_IS_HEVC_MAIN8=0
if [[ "$SOURCE_CODEC" == "hevc" && "$SOURCE_PROFILE" == "Main" \
  && "$SOURCE_PIXEL_FORMAT" == "yuv420p" ]]; then
  SOURCE_IS_HEVC_MAIN8=1
fi
START_IS_ZERO=0
if [[ "$START_TIME" =~ ^(0+([.]0+)?|00:00:00([.]0+)?)$ ]]; then
  START_IS_ZERO=1
fi
USE_FAST_SOURCE_COPY=0
if [[ "$SOURCE_IS_30_FPS" == "1" && "$SOURCE_IS_HEVC_MAIN8" == "1" ]] \
  && { [[ "$FAST_SOURCE_COPY" == "1" ]] \
    || [[ "$FAST_SOURCE_COPY" == "auto" && "$START_IS_ZERO" == "1" ]]; }; then
  USE_FAST_SOURCE_COPY=1
fi
if [[ "$FAST_SOURCE_COPY" == "1" \
  && ( "$SOURCE_IS_30_FPS" != "1" || "$SOURCE_IS_HEVC_MAIN8" != "1" ) ]]; then
  echo "Source packet copy requested, but the input is not 30 fps HEVC Main 8 yuv420p"
  echo "Normalizing to 30 fps HEVC Main 8 so every restored/bypassed segment matches"
fi

echo "JASNA_PROGRESS|1|0|1|Preparing source video"
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
echo "JASNA_PROGRESS|1|1|1|Source video ready"

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
PROGRESS_TOTAL_SEGMENTS="$(
  /usr/bin/awk -v frames="$EXPECTED_FRAME_COUNT" -v seconds="$TEST_SEGMENT_SECONDS" \
    'BEGIN { print int((frames + seconds * 30 - 1) / (seconds * 30)) }'
)"
PROGRESS_DETECT_COMPLETED=0
echo "JASNA_PROGRESS|2|0|$PROGRESS_TOTAL_SEGMENTS|Detecting mosaic regions"
: > "$SHARED_BATCH_PATH"
LEFT_SOURCE_DIR="$LEFT_EYE_WORK_DIR/source"
RIGHT_SOURCE_DIR="$RIGHT_EYE_WORK_DIR/source"
SHARED_SOURCE_DIR="$WORK_DIR/shared-sbs-source"
LEFT_SOURCE_DONE="$LEFT_EYE_WORK_DIR/source.done"
RIGHT_SOURCE_DONE="$RIGHT_EYE_WORK_DIR/source.done"
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
          -avoid_negative_ts make_zero "$SHARED_SEGMENT"
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
        -g 30 -tag:v hvc1 "$LEFT_SEGMENT" \
        -map '[right]' -t "$SEGMENT_DURATION" -an -c:v hevc_videotoolbox \
        "${ENCODER_SPEED_ARGS[@]}" -pix_fmt yuv420p -b:v "$EYE_BITRATE" \
        -maxrate "$((EYE_BITRATE * 3 / 2))" -bufsize "$((EYE_BITRATE * 3))" \
        -g 30 -tag:v hvc1 "$RIGHT_SEGMENT"
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
      PROGRESS_DETECT_COMPLETED=$((PROGRESS_DETECT_COMPLETED + 1))
      echo "JASNA_PROGRESS|2|$PROGRESS_DETECT_COMPLETED|$PROGRESS_TOTAL_SEGMENTS|Detecting mosaic regions"
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
    PROGRESS_DETECT_COMPLETED=$((PROGRESS_DETECT_COMPLETED + 1))
    echo "JASNA_PROGRESS|2|$PROGRESS_DETECT_COMPLETED|$PROGRESS_TOTAL_SEGMENTS|Detecting mosaic regions"
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
  run_direct_sbs_pipeline
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
  run_eye_pair_pipeline
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
report_persistent_work
