#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_SBS_VIDEO" >&2
  echo "example: $0 previous-test.jasna-vr30-v22-work restored-model-b.mov" >&2
  echo "" >&2
  echo "Reuses the prepared SBS source and first-segment detector manifests from" >&2
  echo "a previous test. Only restoration and final packet joining are repeated." >&2
  echo "" >&2
  echo "optional: JASNA_MODELS_DIR=/path/to/MetalMLModels" >&2
  echo "          JASNA_RESTORE_ONLY_START_SECOND=0 (relative to selected segment)" >&2
  echo "          JASNA_RESTORE_ONLY_SECONDS=remaining (1 through remaining seconds)" >&2
  echo "          JASNA_RESTORE_ONLY_AUDIO_START_SECOND=segment-relative start" >&2
  echo "          JASNA_METAL_WINDOWS_PER_PROCESS=1" >&2
  echo "          JASNA_MODEL_BATCH=1" >&2
  echo "          JASNA_REGION_PREPARE_DEPTH=2 (overlap CPU crop preparation with Metal)" >&2
  echo "          JASNA_VIDEO_BITRATE=40000000" >&2
  echo "          JASNA_TEMPORAL_WARMUP_FRAMES=5 (set 0 to disable)" >&2
  echo "          JASNA_STEREO_WRITER_DEPTH=2 (set 1 for an A/B control)" >&2
  echo "          JASNA_RESTORE_ONLY_VALIDATE=1 (check reused assets, then exit)" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

REFERENCE_WORK_DIR="$1"
OUTPUT_PATH="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
REFERENCE_WORK_DIR="$(cd "$REFERENCE_WORK_DIR" && pwd)"
SOURCE_CONFIG="$REFERENCE_WORK_DIR/run-config.txt"
source "$ROOT_DIR/script/lib/run_configuration.sh"

# Restore all settings that can change crop scheduling, model execution, or
# final compositing. An explicit environment value remains an intentional A/B
# override and is never replaced by the recorded value.
RECORDED_RESTORATION_SETTINGS=(
  "model_batch:JASNA_MODEL_BATCH"
  "metal_windows_per_process:JASNA_METAL_WINDOWS_PER_PROCESS"
  "region_prepare_depth:JASNA_REGION_PREPARE_DEPTH"
  "large_region_max_blend:JASNA_LARGE_REGION_MAX_BLEND"
  "large_region_overlap:JASNA_LARGE_REGION_OVERLAP"
  "large_region_split_limit:JASNA_LARGE_REGION_SPLIT_LIMIT"
  "large_region_max_axis_crops:JASNA_LARGE_REGION_MAX_AXIS_CROPS"
  "large_region_mask_growth:JASNA_LARGE_REGION_MASK_GROWTH"
  "large_region_mask_feather:JASNA_LARGE_REGION_MASK_FEATHER"
  "large_region_block_growth:JASNA_LARGE_REGION_BLOCK_GROWTH"
  "large_region_mask_temporal_radius:JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS"
  "large_region_mask_temporal_strength:JASNA_LARGE_REGION_MASK_TEMPORAL_STRENGTH"
  "large_region_detail_crops:JASNA_LARGE_REGION_DETAIL_CROPS"
  "large_region_detail_dimension:JASNA_LARGE_REGION_DETAIL_DIMENSION"
  "temporal_crop_frames:JASNA_TEMPORAL_CROP_FRAMES"
  "temporal_crop_padding:JASNA_TEMPORAL_CROP_PADDING"
  "temporal_crop_min_dimension:JASNA_TEMPORAL_CROP_MIN_DIMENSION"
  "temporal_crop_motion:JASNA_TEMPORAL_CROP_MOTION"
  "mosaic_detail_residual_limit:JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT"
  "mosaic_mask_recovery_threshold:JASNA_MOSAIC_MASK_RECOVERY_THRESHOLD"
  "mosaic_mask_recovery_all_regions:JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS"
  "temporal_warmup_frames:JASNA_TEMPORAL_WARMUP_FRAMES"
  "allow_passthrough:JASNA_ALLOW_PASSTHROUGH"
  "diagnostic_full_region_blend:JASNA_DIAGNOSTIC_FULL_REGION_BLEND"
  "metal_texture_compositor:JASNA_METAL_TEXTURE_COMPOSITOR"
  "metal_compositor:JASNA_METAL_COMPOSITOR"
  "composite_concurrency:JASNA_COMPOSITE_CONCURRENCY"
  "stereo_writer_depth:JASNA_STEREO_WRITER_DEPTH"
  "in_memory_crop_cache:JASNA_IN_MEMORY_CROP_CACHE"
  "in_memory_cache_limit_mb:JASNA_IN_MEMORY_CACHE_LIMIT_MB"
)
for recorded_setting in "${RECORDED_RESTORATION_SETTINGS[@]}"; do
  IFS=: read -r recorded_key environment_name <<< "$recorded_setting"
  jasna_adopt_recorded_environment \
    "$SOURCE_CONFIG" "$recorded_key" "$environment_name"
done

WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-1}"
VIDEO_BITRATE="${JASNA_VIDEO_BITRATE:-40000000}"
VALIDATE_ONLY="${JASNA_RESTORE_ONLY_VALIDATE:-0}"
GPU_TIMEOUT_RETRIES="${JASNA_GPU_TIMEOUT_RETRIES:-2}"
START_WINDOW="${JASNA_RESTORE_ONLY_START_SECOND:-0}"
SELECTED_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-}"
AUDIO_START_SECOND="${JASNA_RESTORE_ONLY_AUDIO_START_SECOND:-$START_WINDOW}"
TEMPORAL_WARMUP_FRAMES="${JASNA_TEMPORAL_WARMUP_FRAMES:-5}"
STEREO_WRITER_DEPTH="${JASNA_STEREO_WRITER_DEPTH:-2}"
RUN_STARTED_SECONDS=$SECONDS

[[ "$OUTPUT_PATH" != *[[:space:]] ]] || {
  echo "error: output path ends with whitespace: '$OUTPUT_PATH'" >&2
  exit 1
}
[[ "$OUTPUT_PATH" == *.mov ]] || {
  echo "error: restoration-only output must use a .mov filename" >&2
  exit 1
}
[[ "$WINDOWS_PER_PROCESS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_METAL_WINDOWS_PER_PROCESS must be a positive integer" >&2
  exit 1
}
[[ "$VIDEO_BITRATE" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_VIDEO_BITRATE must be a positive integer" >&2
  exit 1
}
[[ "$GPU_TIMEOUT_RETRIES" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_GPU_TIMEOUT_RETRIES must be a non-negative integer" >&2
  exit 1
}
[[ "$TEMPORAL_WARMUP_FRAMES" =~ ^[0-9]+$ ]] \
  && (( TEMPORAL_WARMUP_FRAMES <= 5 )) || {
  echo "error: JASNA_TEMPORAL_WARMUP_FRAMES must be an integer from 0 to 5" >&2
  exit 1
}
export JASNA_TEMPORAL_WARMUP_FRAMES="$TEMPORAL_WARMUP_FRAMES"
[[ "$STEREO_WRITER_DEPTH" == "1" || "$STEREO_WRITER_DEPTH" == "2" ]] || {
  echo "error: JASNA_STEREO_WRITER_DEPTH must be 1 or 2" >&2
  exit 1
}
export JASNA_STEREO_WRITER_DEPTH="$STEREO_WRITER_DEPTH"
[[ "$START_WINDOW" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_RESTORE_ONLY_START_SECOND must be a non-negative integer" >&2
  exit 1
}
[[ -z "$SELECTED_SECONDS" || "$SELECTED_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_RESTORE_ONLY_SECONDS must be a positive integer" >&2
  exit 1
}
[[ "$AUDIO_START_SECOND" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_RESTORE_ONLY_AUDIO_START_SECOND must be a non-negative integer" >&2
  exit 1
}
[[ "$VALIDATE_ONLY" == "0" || "$VALIDATE_ONLY" == "1" ]] || {
  echo "error: JASNA_RESTORE_ONLY_VALIDATE must be 0 or 1" >&2
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

RECORDED_WORK_CONTAINER=""
if [[ -s "$SOURCE_CONFIG" ]]; then
  RECORDED_WORK_CONTAINER="$(
    /usr/bin/awk -F= '$1 == "work_container" { print $2; exit }' "$SOURCE_CONFIG"
  )"
fi
case "$RECORDED_WORK_CONTAINER" in
  mov|mp4)
    SOURCE_SBS="$REFERENCE_WORK_DIR/source/test-sbs-30fps.$RECORDED_WORK_CONTAINER"
    ;;
  "")
    MOV_SOURCE="$REFERENCE_WORK_DIR/source/test-sbs-30fps.mov"
    MP4_SOURCE="$REFERENCE_WORK_DIR/source/test-sbs-30fps.mp4"
    if [[ -s "$MOV_SOURCE" && ! -s "$MP4_SOURCE" ]]; then
      SOURCE_SBS="$MOV_SOURCE"
    elif [[ -s "$MP4_SOURCE" && ! -s "$MOV_SOURCE" ]]; then
      SOURCE_SBS="$MP4_SOURCE"
    elif [[ -s "$MOV_SOURCE" && -s "$MP4_SOURCE" ]]; then
      echo "error: reference work contains both MOV and MP4 prepared sources" >&2
      echo "record work_container=mov or work_container=mp4 in $SOURCE_CONFIG" >&2
      exit 1
    else
      SOURCE_SBS="$MOV_SOURCE"
    fi
    ;;
  *)
    echo "error: unsupported recorded work container: $RECORDED_WORK_CONTAINER" >&2
    exit 1
    ;;
esac
PHYSICAL_LEFT_INPUT="${JASNA_RESTORE_ONLY_LEFT_INPUT:-$REFERENCE_WORK_DIR/left-restored.left-segments-work/source/left-00000.mov}"
PHYSICAL_RIGHT_INPUT="${JASNA_RESTORE_ONLY_RIGHT_INPUT:-$REFERENCE_WORK_DIR/right-restored.right-segments-work/source/right-00000.mov}"
if [[ -n "${JASNA_RESTORE_ONLY_LEFT_INPUT:-}" \
  || -n "${JASNA_RESTORE_ONLY_RIGHT_INPUT:-}" ]]; then
  [[ -n "${JASNA_RESTORE_ONLY_LEFT_INPUT:-}" \
    && -n "${JASNA_RESTORE_ONLY_RIGHT_INPUT:-}" ]] || {
    echo "error: both restoration-only eye input overrides are required" >&2
    exit 1
  }
  [[ -s "$PHYSICAL_LEFT_INPUT" && -s "$PHYSICAL_RIGHT_INPUT" ]] || {
    echo "error: an explicitly selected eye/shared input is missing or empty" >&2
    echo "refusing to substitute a different source segment" >&2
    exit 1
  }
fi
if [[ -s "$PHYSICAL_LEFT_INPUT" && -s "$PHYSICAL_RIGHT_INPUT" ]]; then
  LEFT_INPUT="$PHYSICAL_LEFT_INPUT"
  RIGHT_INPUT="$PHYSICAL_RIGHT_INPUT"
  if [[ "$PHYSICAL_LEFT_INPUT" -ef "$PHYSICAL_RIGHT_INPUT" ]]; then
    RESTORE_INPUT_KIND="one shared 8K SBS decode"
  else
    RESTORE_INPUT_KIND="physical 4K eye files"
  fi
else
  RECORDED_SHARED_SBS="$(
    /usr/bin/awk -F= '$1 == "shared_sbs_source" { print $2; exit }' "$SOURCE_CONFIG"
  )"
  [[ "$RECORDED_SHARED_SBS" == "1" ]] || {
    echo "error: reusable physical eye inputs are missing and shared SBS was not recorded" >&2
    exit 1
  }
  SHARED_SEGMENT_INPUT=""
  for candidate in \
      "$REFERENCE_WORK_DIR/shared-sbs-source/shared-00000.mp4" \
      "$REFERENCE_WORK_DIR/shared-sbs-source/shared-00000.mov"; do
    if [[ -s "$candidate" ]]; then
      SHARED_SEGMENT_INPUT="$candidate"
      break
    fi
  done
  [[ -n "$SHARED_SEGMENT_INPUT" ]] || SHARED_SEGMENT_INPUT="$SOURCE_SBS"
  # The direct stereo command recognizes identical 8K left/right paths and
  # performs one shared decode with in-memory eye crops. Newer production work
  # directories intentionally omit the old physical 4K intermediates. Prefer
  # the first bounded shared segment because its timeline matches manifest 0.
  LEFT_INPUT="$SHARED_SEGMENT_INPUT"
  RIGHT_INPUT="$SHARED_SEGMENT_INPUT"
  RESTORE_INPUT_KIND="one shared 8K SBS decode"
fi

find_one_manifest() {
  local eye="$1"
  local eye_work="$REFERENCE_WORK_DIR/$eye-restored.$eye-segments-work"
  local found
  found="$(find "$eye_work" -type f -name "$eye-00000-mosaic-regions.json" -print -quit 2>/dev/null)"
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

if [[ -n "${JASNA_RESTORE_ONLY_LEFT_MANIFEST:-}" \
  || -n "${JASNA_RESTORE_ONLY_RIGHT_MANIFEST:-}" ]]; then
  [[ -n "${JASNA_RESTORE_ONLY_LEFT_MANIFEST:-}" \
    && -n "${JASNA_RESTORE_ONLY_RIGHT_MANIFEST:-}" ]] || {
    echo "error: both restoration-only manifest overrides are required" >&2
    exit 1
  }
  LEFT_MANIFEST="$JASNA_RESTORE_ONLY_LEFT_MANIFEST"
  RIGHT_MANIFEST="$JASNA_RESTORE_ONLY_RIGHT_MANIFEST"
  MANIFEST_KIND="selected reconciled stereo"
elif [[ -s "$REFERENCE_WORK_DIR/stereo-reconciled-manifests/left-00000.json" \
  && -s "$REFERENCE_WORK_DIR/stereo-reconciled-manifests/right-00000.json" ]]; then
  LEFT_MANIFEST="$REFERENCE_WORK_DIR/stereo-reconciled-manifests/left-00000.json"
  RIGHT_MANIFEST="$REFERENCE_WORK_DIR/stereo-reconciled-manifests/right-00000.json"
  MANIFEST_KIND="reconciled stereo"
else
  LEFT_MANIFEST="$(find_one_manifest left)" || {
    echo "error: no reusable left-eye detector manifest found" >&2
    exit 1
  }
  RIGHT_MANIFEST="$(find_one_manifest right)" || {
    echo "error: no reusable right-eye detector manifest found" >&2
    exit 1
  }
  MANIFEST_KIND="original detector"
fi

for required_path in "$SOURCE_SBS" "$LEFT_INPUT" "$RIGHT_INPUT" \
    "$LEFT_MANIFEST" "$RIGHT_MANIFEST"; do
  [[ -s "$required_path" ]] || {
    echo "error: reusable asset is missing or empty: $required_path" >&2
    exit 1
  }
done

SOURCE_INFO="$(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -of csv=p=0 "$SOURCE_SBS"
)"
IFS=, read -r SOURCE_CODEC SOURCE_WIDTH SOURCE_HEIGHT SOURCE_RATE \
  SOURCE_FRAME_COUNT SOURCE_EXTRA <<< "$SOURCE_INFO"
[[ -z "$SOURCE_EXTRA" && "$SOURCE_WIDTH" == "8192" \
  && "$SOURCE_HEIGHT" == "4096" && "$SOURCE_FRAME_COUNT" =~ ^[0-9]+$ ]] || {
  echo "error: reference source must be a frame-counted 8192x4096 video" >&2
  echo "found: $SOURCE_INFO" >&2
  exit 1
}
/usr/bin/awk -F/ '
  NF == 2 && $2 != 0 { rate = $1 / $2 }
  NF == 1 { rate = $1 }
  END { exit !(rate >= 29.99 && rate <= 30.01) }
' <<< "$SOURCE_RATE" || {
  echo "error: reference source must be 30 fps" >&2
  exit 1
}

LEFT_INPUT_FRAME_COUNT="$(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=nb_frames -of default=nw=1:nk=1 "$LEFT_INPUT"
)"
RIGHT_INPUT_FRAME_COUNT="$(
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=nb_frames -of default=nw=1:nk=1 "$RIGHT_INPUT"
)"
[[ "$LEFT_INPUT_FRAME_COUNT" =~ ^[0-9]+$ \
  && "$RIGHT_INPUT_FRAME_COUNT" == "$LEFT_INPUT_FRAME_COUNT" \
  && "$LEFT_INPUT_FRAME_COUNT" -ge 30 ]] || {
  echo "error: reusable eye inputs must have matching frame counts" >&2
  exit 1
}
INPUT_SECONDS=$((LEFT_INPUT_FRAME_COUNT / 30))
[[ "$START_WINDOW" -lt "$INPUT_SECONDS" ]] || {
  echo "error: selected segment contains only $INPUT_SECONDS second(s)" >&2
  exit 1
}
[[ -n "$SELECTED_SECONDS" ]] || SELECTED_SECONDS=$((INPUT_SECONDS - START_WINDOW))
END_WINDOW=$((START_WINDOW + SELECTED_SECONDS))
[[ "$END_WINDOW" -le "$INPUT_SECONDS" ]] || {
  echo "error: restoration range $START_WINDOW-$END_WINDOW exceeds the selected segment" >&2
  exit 1
}
[[ "$SOURCE_FRAME_COUNT" -ge $(((AUDIO_START_SECOND + SELECTED_SECONDS) * 30)) ]] || {
  echo "error: prepared SBS source does not contain the selected audio timeline" >&2
  exit 1
}

/usr/bin/python3 - "$LEFT_MANIFEST" "$RIGHT_MANIFEST" "$LEFT_INPUT_FRAME_COUNT" <<'PY'
import json
import sys

expected_frames = int(sys.argv[3])
for eye, path in zip(("left", "right"), sys.argv[1:3]):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual = (
        manifest.get("width"), manifest.get("height"),
        manifest.get("frameCount"), manifest.get("framesPerSecond"),
    )
    if actual != (4096, 4096, expected_frames, 30):
        raise SystemExit(
            f"error: {eye} manifest does not match the reusable 4096x4096, "
            f"{expected_frames}-frame, 30-fps input: {actual}"
        )
PY

echo "Restoration-only selected-range test"
echo "Reference:  $REFERENCE_WORK_DIR"
echo "Source:     $SOURCE_SBS"
echo "Container:  ${SOURCE_SBS##*.} (reused without conversion)"
echo "Manifests:  $MANIFEST_KIND (detector will not run)"
echo "Restore input: $RESTORE_INPUT_KIND"
echo "Frames:     $SOURCE_FRAME_COUNT at $SOURCE_RATE"
echo "Selected:   relative seconds $START_WINDOW-$END_WINDOW ($SELECTED_SECONDS second(s))"
echo "Source timeline: seconds $AUDIO_START_SECOND-$((AUDIO_START_SECOND + SELECTED_SECONDS))"
echo "Output:     $OUTPUT_PATH"
echo "Windows/process: $WINDOWS_PER_PROCESS"
echo "Models:     ${JASNA_MODELS_DIR:-$ROOT_DIR/Models/MetalML}"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "Reusable source and manifests: PASS"
  echo "No video was decoded, detected, restored, or written."
  exit 0
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"
OUTPUT_STEM="${OUTPUT_NAME%.*}"
WORK_DIR="$OUTPUT_DIR/${OUTPUT_STEM}.jasna-restore-only-work"
LOG_PATH="$OUTPUT_DIR/${OUTPUT_STEM}.jasna-restore-only.log"
SEGMENT_DIR="$WORK_DIR/segments"
LEFT_CACHE="$WORK_DIR/left-cache"
RIGHT_CACHE="$WORK_DIR/right-cache"
PROCESS_WORK="$WORK_DIR/process"
CONCAT_PATH="$WORK_DIR/segments.txt"
FINAL_TEMP="$WORK_DIR/.final-writing.mov"
LOCK_DIR="$WORK_DIR/.workflow-lock"

mkdir -p "$WORK_DIR" "$SEGMENT_DIR" "$LEFT_CACHE" "$RIGHT_CACHE" "$PROCESS_WORK"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  EXISTING_PID="$(/bin/cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "error: this restoration-only test is already active (PID $EXISTING_PID)" >&2
    exit 1
  fi
  mv "$LOCK_DIR" "$WORK_DIR/.workflow-lock.stale-$(date '+%Y%m%d-%H%M%S')-$$"
  mkdir "$LOCK_DIR"
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"
cleanup_lock() {
  [[ -d "$LOCK_DIR" ]] || return 0
  local owner_pid
  owner_pid="$(/bin/cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  [[ "$owner_pid" == "$$" ]] || return 0
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

exec > >(/usr/bin/tee -a "$LOG_PATH") 2>&1
echo
echo "===== Jasna restoration-only session $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Decode preparation: REUSED"
echo "Mosaic detection:   REUSED"
echo "Reference work:     $REFERENCE_WORK_DIR"
echo "Output work:        $WORK_DIR"
echo "Log:                $LOG_PATH"
echo "Ordinary mask-hole recovery: ${JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS:-0}"
echo "Temporal crop warm-up:       $TEMPORAL_WARMUP_FRAMES frame(s)"

if [[ -e "$OUTPUT_PATH" ]]; then
  echo "error: output already exists; use a new filename: $OUTPUT_PATH" >&2
  exit 1
fi

if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
  echo "Building one optimized Swift executable"
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
  export JASNA_APP_BINARY
fi
[[ -x "$JASNA_APP_BINARY" ]] || {
  echo "error: optimized JasnaMetalPoC executable was not produced" >&2
  exit 1
}

valid_segment() {
  local candidate="$1"
  local expected_frames="$2"
  [[ -s "$candidate" ]] || return 1
  local codec width height rate frames extra
  IFS=, read -r codec width height rate frames extra < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$candidate" 2>/dev/null
  )
  [[ -z "$extra" && "$codec" == "hevc" && "$width" == "8192" \
    && "$height" == "4096" && "$frames" =~ ^[0-9]+$ ]] || return 1
  /usr/bin/awk -v actual="$frames" -v expected="$expected_frames" '
    BEGIN { delta = actual - expected; if (delta < 0) delta = -delta; exit !(delta <= 1) }
  ' || return 1
  "$FFMPEG_PATH" -v error -i "$candidate" -map '0:v:0' \
    -frames:v 1 -f null - </dev/null >/dev/null 2>&1
}

: > "$CONCAT_PATH"
WINDOW_START="$START_WINDOW"
RESTORED_WINDOWS=0
BYPASSED_WINDOWS=0
while (( WINDOW_START < END_WINDOW )); do
  ACTIVITY="$(
    /usr/bin/python3 "$ROOT_DIR/tools/manifest_window_runs.py" \
      "$LEFT_MANIFEST" "$RIGHT_MANIFEST" "$WINDOW_START"
  )"
  IFS=$'\t' read -r WINDOW_MODE RUN_COUNT ACTIVITY_EXTRA <<< "$ACTIVITY"
  [[ ( "$WINDOW_MODE" == "active" || "$WINDOW_MODE" == "empty" ) \
    && "$RUN_COUNT" =~ ^[1-9][0-9]*$ && -z "$ACTIVITY_EXTRA" ]] || {
    echo "error: invalid manifest window activity: $ACTIVITY" >&2
    exit 1
  }
  WINDOW_COUNT="$RUN_COUNT"
  if (( WINDOW_START + WINDOW_COUNT > END_WINDOW )); then
    WINDOW_COUNT=$((END_WINDOW - WINDOW_START))
  fi
  if [[ "$WINDOW_MODE" == "active" && "$WINDOW_COUNT" -gt "$WINDOWS_PER_PROCESS" ]]; then
    WINDOW_COUNT="$WINDOWS_PER_PROCESS"
  fi
  WINDOW_END=$((WINDOW_START + WINDOW_COUNT))
  SEGMENT="$SEGMENT_DIR/$(printf 'windows-%05d-%05d.mov' "$WINDOW_START" "$WINDOW_END")"
  EXPECTED_SEGMENT_FRAMES=$((WINDOW_COUNT * 30))

  if valid_segment "$SEGMENT" "$EXPECTED_SEGMENT_FRAMES"; then
    echo "Reusing restored windows $((WINDOW_START + 1))-$WINDOW_END/30"
  else
    BYPASS_OK=0
    if [[ "$WINDOW_MODE" == "empty" ]]; then
      BYPASS_TEMP="${SEGMENT%.mov}.bypass-writing.mov"
      [[ ! -e "$BYPASS_TEMP" ]] || mv "$BYPASS_TEMP" \
        "${BYPASS_TEMP%.mov}.interrupted-$(date '+%Y%m%d-%H%M%S').mov"
      echo "Bypassing detector-confirmed clean windows $((WINDOW_START + 1))-$WINDOW_END/30"
      BYPASS_SOURCE_SECOND=$((AUDIO_START_SECOND + WINDOW_START - START_WINDOW))
      if "$FFMPEG_PATH" -hide_banner -loglevel error \
          -ss "$BYPASS_SOURCE_SECOND" -i "$SOURCE_SBS" -t "$WINDOW_COUNT" \
          -map '0:v:0' -an -c copy -avoid_negative_ts make_zero \
          -video_track_timescale 600 -movflags +faststart "$BYPASS_TEMP" \
        && valid_segment "$BYPASS_TEMP" "$EXPECTED_SEGMENT_FRAMES"; then
        mv "$BYPASS_TEMP" "$SEGMENT"
        BYPASS_OK=1
        BYPASSED_WINDOWS=$((BYPASSED_WINDOWS + WINDOW_COUNT))
      fi
    fi
    if [[ "$BYPASS_OK" == "0" ]]; then
      echo "Restoring windows $((WINDOW_START + 1))-$WINDOW_END/30"
      ATTEMPT=0
      while true; do
        STATUS=0
        RESTORE_LOG="${SEGMENT%.mov}.jasna.log"
        RESTORE_LOG_START_LINES=0
        if [[ -f "$RESTORE_LOG" ]]; then
          RESTORE_LOG_START_LINES="$(/usr/bin/wc -l < "$RESTORE_LOG")"
          RESTORE_LOG_START_LINES="${RESTORE_LOG_START_LINES//[[:space:]]/}"
        fi
        JASNA_WINDOW_START="$WINDOW_START" \
        JASNA_WINDOW_COUNT="$WINDOW_COUNT" \
        JASNA_VIDEO_BITRATE="$VIDEO_BITRATE" \
        JASNA_VR_PROJECTION=fisheye \
        JASNA_WORK_DIR="$PROCESS_WORK" \
          "$ROOT_DIR/script/build_and_run.sh" --restore-stereo-sparse-batch \
            "$LEFT_INPUT" "$RIGHT_INPUT" "$SEGMENT" \
            "$LEFT_MANIFEST" "$RIGHT_MANIFEST" \
            "$LEFT_CACHE" "$RIGHT_CACHE" || STATUS=$?
        (( STATUS == 0 )) && break
        if ! /usr/bin/tail -n "+$((RESTORE_LOG_START_LINES + 1))" \
            "$RESTORE_LOG" 2>/dev/null \
          | /usr/bin/grep 'GPU Timeout Error' >/dev/null; then
          echo "error: Metal restoration failed for a reason other than GPU timeout" >&2
          exit "$STATUS"
        fi
        if (( ATTEMPT >= GPU_TIMEOUT_RETRIES )); then
          echo "error: restoration exhausted $GPU_TIMEOUT_RETRIES GPU-timeout retries" >&2
          exit "$STATUS"
        fi
        ATTEMPT=$((ATTEMPT + 1))
        echo "WARNING: Metal GPU timeout; resuming its checkpoint in a fresh process ($ATTEMPT/$GPU_TIMEOUT_RETRIES)"
      done
      valid_segment "$SEGMENT" "$EXPECTED_SEGMENT_FRAMES" || {
        echo "error: restored segment failed validation: $SEGMENT" >&2
        exit 1
      }
      RESTORED_WINDOWS=$((RESTORED_WINDOWS + WINDOW_COUNT))
    fi
  fi
  ESCAPED_SEGMENT="${SEGMENT//\'/\'\\\'\'}"
  printf "file '%s'\n" "$ESCAPED_SEGMENT" >> "$CONCAT_PATH"
  WINDOW_START="$WINDOW_END"
done

if [[ -e "$FINAL_TEMP" ]]; then
  mv "$FINAL_TEMP" "$WORK_DIR/.final-interrupted-$(date '+%Y%m%d-%H%M%S').mov"
fi
echo "Joining restored video and copying source audio"
"$FFMPEG_PATH" -hide_banner \
  -f concat -safe 0 -i "$CONCAT_PATH" \
  -ss "$AUDIO_START_SECOND" -t "$SELECTED_SECONDS" -i "$SOURCE_SBS" \
  -map '0:v:0' -map '1:a?' -map_metadata 1 -map_chapters 1 \
  -c copy -video_track_timescale 600 -movflags +faststart -shortest \
  "$FINAL_TEMP"
valid_segment "$FINAL_TEMP" "$((SELECTED_SECONDS * 30))" || {
  echo "error: final restoration-only output failed validation" >&2
  exit 1
}
mv "$FINAL_TEMP" "$OUTPUT_PATH"

ELAPSED_SECONDS=$((SECONDS - RUN_STARTED_SECONDS))
echo "Restoration-only selected-range test: PASS"
echo "Preparation/detection time: 0 seconds (reused)"
echo "Selected source-relative seconds: $START_WINDOW-$END_WINDOW"
echo "Selected prepared-SBS timeline: $AUDIO_START_SECOND-$((AUDIO_START_SECOND + SELECTED_SECONDS))"
echo "Restored/bypassed windows: $RESTORED_WINDOWS/$BYPASSED_WINDOWS"
echo "Wall time: ${ELAPSED_SECONDS}s"
echo "Output: $OUTPUT_PATH"
echo "Log:    $LOG_PATH"
