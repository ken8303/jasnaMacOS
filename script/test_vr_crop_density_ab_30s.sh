#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: ./script/test_vr_crop_density_ab_30s.sh REFERENCE_WORK_DIR OUTPUT_PREFIX [START_TIME]

Runs restoration only against the same prepared SBS source and detector
manifests. START_TIME is on the prepared source timeline, for example
00:19:50. A range crossing a cached 120-second boundary is restored in pieces
and joined without repeating source conversion or mosaic detection.

The baseline keeps the current 768px subdivision grid. The candidate uses a
1024px grid, reducing BasicVSR++ crops for large regions.

Outputs:
  OUTPUT_PREFIX-grid768.mov
  OUTPUT_PREFIX-grid1024.mov

Optional environment variables:
  JASNA_RESTORE_ONLY_SECONDS=30
  JASNA_AB_BASELINE_MAX_BLEND=768
  JASNA_AB_CANDIDATE_MAX_BLEND=1024
  JASNA_REFERENCE_SEGMENT_SECONDS=120
  JASNA_METAL_WINDOWS_PER_PROCESS=4
  JASNA_AB_ORDER=candidate-first (or baseline-first, the default)
  JASNA_GRAPH_PHASE_TELEMETRY=1 (enabled for both variants by default)
  JASNA_GRAPH_TRACE=1 (immediate phase markers; set 0 to reduce logging)

Use a fresh output prefix for every comparison. Interrupted measurements are
preserved, but are not reused as independent timing samples.
EOF
  exit 2
}

[[ $# -eq 2 || $# -eq 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="${1%/}"
OUTPUT_PREFIX="${2%.mov}"
START_TIME="${3:-00:00:00}"
RUN_CONFIG="$REFERENCE_WORK_DIR/run-config.txt"
BASELINE_BLEND="${JASNA_AB_BASELINE_MAX_BLEND:-768}"
CANDIDATE_BLEND="${JASNA_AB_CANDIDATE_MAX_BLEND:-1024}"
SELECTED_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-30}"
REFERENCE_SEGMENT_SECONDS="${JASNA_REFERENCE_SEGMENT_SECONDS:-120}"
VALIDATE_ONLY="${JASNA_RESTORE_ONLY_VALIDATE:-0}"
AB_ORDER="${JASNA_AB_ORDER:-baseline-first}"
COMPARISON_KIND="${JASNA_AB_COMPARISON:-crop-density}"
export JASNA_GRAPH_PHASE_TELEMETRY="${JASNA_GRAPH_PHASE_TELEMETRY:-1}"
export JASNA_GRAPH_TRACE="${JASNA_GRAPH_TRACE:-1}"
# Resumed timeout attempts must not be counted as independent timing samples.
export JASNA_GPU_TIMEOUT_RETRIES=0

case "$COMPARISON_KIND" in
  crop-density) ;;
  crop-handoff)
    BASELINE_BLEND="${JASNA_AB_HANDOFF_MAX_BLEND:-1024}"
    CANDIDATE_BLEND="$BASELINE_BLEND"
    ;;
  *) echo "error: unsupported A/B comparison: $COMPARISON_KIND" >&2; exit 1 ;;
esac

[[ -d "$REFERENCE_WORK_DIR" && -s "$RUN_CONFIG" ]] || {
  echo "error: reference work directory or run-config.txt is unavailable" >&2
  exit 1
}
[[ "$OUTPUT_PREFIX" != *[[:space:]] ]] || {
  echo "error: output prefix ends with whitespace: '$OUTPUT_PREFIX'" >&2
  exit 1
}
for value in "$BASELINE_BLEND" "$CANDIDATE_BLEND" \
    "$SELECTED_SECONDS" "$REFERENCE_SEGMENT_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: crop dimensions, duration, and segment length must be positive integers" >&2
    exit 1
  }
done
[[ "$COMPARISON_KIND" == "crop-handoff" ]] || (( CANDIDATE_BLEND > BASELINE_BLEND )) || {
  echo "error: candidate crop grid must be larger than the baseline" >&2
  exit 1
}
(( SELECTED_SECONDS <= 30 )) || {
  echo "error: JASNA_RESTORE_ONLY_SECONDS must be from 1 through 30" >&2
  exit 1
}
[[ "$VALIDATE_ONLY" == "0" || "$VALIDATE_ONLY" == "1" ]] || {
  echo "error: JASNA_RESTORE_ONLY_VALIDATE must be 0 or 1" >&2
  exit 1
}
case "$AB_ORDER" in
  baseline-first|candidate-first) ;;
  *)
    echo "error: JASNA_AB_ORDER must be baseline-first or candidate-first" >&2
    exit 1
    ;;
esac
for value in "$JASNA_GRAPH_PHASE_TELEMETRY" "$JASNA_GRAPH_TRACE"; do
  [[ "$value" == "0" || "$value" == "1" ]] || {
    echo "error: graph telemetry and trace flags must be 0 or 1" >&2
    exit 1
  }
done

parse_time() {
  local value="$1"
  local hours minutes seconds extra
  IFS=: read -r hours minutes seconds extra <<< "$value"
  if [[ -z "${seconds:-}" ]]; then
    seconds="$minutes"
    minutes="$hours"
    hours=0
  fi
  [[ -z "${extra:-}" && "$hours" =~ ^[0-9]+$ \
    && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ \
    && "$minutes" -lt 60 && "$seconds" -lt 60 ]] || return 1
  printf '%d\n' "$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

START_TOTAL_SECONDS="$(parse_time "$START_TIME")" || {
  echo "error: START_TIME must be MM:SS or HH:MM:SS" >&2
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

read_config() {
  local key="$1"
  local fallback="$2"
  local value
  value="$(/usr/bin/awk -F= -v key="$key" '$1 == key { print $2; exit }' "$RUN_CONFIG")"
  printf '%s\n' "${value:-$fallback}"
}

MODEL_BATCH="${JASNA_MODEL_BATCH:-$(read_config model_batch 2)}"
WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-$(read_config metal_windows_per_process 4)}"
BASELINE_NAME="grid${BASELINE_BLEND}"
CANDIDATE_NAME="grid${CANDIDATE_BLEND}"
BASELINE_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-$(read_config in_memory_crop_cache 0)}"
CANDIDATE_CACHE="$BASELINE_CACHE"
CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-$(read_config in_memory_cache_limit_mb 128)}"
if [[ "$COMPARISON_KIND" == "crop-handoff" ]]; then
  BASELINE_NAME=disk
  CANDIDATE_NAME=memory512
  BASELINE_CACHE=0
  CANDIDATE_CACHE=1
  # The limit is per eye/window, not a process RSS cap. Oversized windows keep
  # the existing automatic disk fallback rather than exceeding this bound.
  CACHE_LIMIT_MB=512
fi
BASELINE_OUTPUT="${OUTPUT_PREFIX}-${BASELINE_NAME}.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-${CANDIDATE_NAME}.mov"
WORK_DIR="${OUTPUT_PREFIX}.jasna-${COMPARISON_KIND}-ab-work"

if [[ "$VALIDATE_ONLY" != "1" ]]; then
  for output in "$BASELINE_OUTPUT" "$CANDIDATE_OUTPUT"; do
    [[ ! -e "$output" ]] || {
      echo "error: comparison output already exists: $output" >&2
      exit 1
    }
  done
  mkdir -p "$(dirname "$WORK_DIR")"
  mkdir "$WORK_DIR" 2>/dev/null || {
    echo "error: comparison work directory already exists or cannot be created: $WORK_DIR" >&2
    echo "use a fresh output prefix; old measurements will not be overwritten or reused" >&2
    exit 1
  }
  exec > >(/usr/bin/tee -a "$WORK_DIR/comparison.log") 2>&1
fi

if [[ "$VALIDATE_ONLY" != "1" && -z "${JASNA_APP_BINARY:-}" ]]; then
  DEFAULT_BINARY="$ROOT_DIR/.build/out/Products/Release/JasnaMetalPoC"
  NEED_BUILD=0
  [[ -x "$DEFAULT_BINARY" ]] || NEED_BUILD=1
  if [[ "$NEED_BUILD" == "0" ]] \
    && [[ -n "$(find "$ROOT_DIR/Sources" "$ROOT_DIR/Package.swift" \
      -newer "$DEFAULT_BINARY" -print -quit 2>/dev/null)" ]]; then
    NEED_BUILD=1
  fi
  if [[ "$NEED_BUILD" == "1" ]]; then
    echo "Building one optimized executable for both A/B runs"
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
  else
    JASNA_APP_BINARY="$DEFAULT_BINARY"
    echo "Reusing optimized executable: $JASNA_APP_BINARY"
  fi
  export JASNA_APP_BINARY
fi

segment_asset() {
  local kind="$1"
  local eye="$2"
  local index="$3"
  local formatted
  formatted="$(printf '%05d' "$index")"
  case "$kind" in
    input)
      local physical
      for physical in \
          "$REFERENCE_WORK_DIR/$eye-restored.$eye-segments-work/source/$eye-$formatted.mp4" \
          "$REFERENCE_WORK_DIR/$eye-restored.$eye-segments-work/source/$eye-$formatted.mov"; do
        if [[ -s "$physical" ]]; then
          printf '%s\n' "$physical"
          return 0
        fi
      done
      local shared
      for shared in \
          "$REFERENCE_WORK_DIR/shared-sbs-source/shared-$formatted.mp4" \
          "$REFERENCE_WORK_DIR/shared-sbs-source/shared-$formatted.mov"; do
        if [[ -s "$shared" ]]; then
          printf '%s\n' "$shared"
          return 0
        fi
      done
      ;;
    manifest)
      local reconciled="$REFERENCE_WORK_DIR/stereo-reconciled-manifests/$eye-$formatted.json"
      if [[ -s "$reconciled" ]]; then
        printf '%s\n' "$reconciled"
        return 0
      fi
      local detected
      detected="$(find "$REFERENCE_WORK_DIR/$eye-restored.$eye-segments-work" \
        -type f -name "$eye-$formatted-mosaic-regions.json" -print -quit 2>/dev/null)" || return 1
      if [[ -n "$detected" && -s "$detected" ]]; then
        printf '%s\n' "$detected"
        return 0
      fi
      ;;
  esac
  return 1
}

valid_joined_output() {
  local path="$1"
  local expected_frames="$2"
  [[ -s "$path" ]] || return 1
  local codec width height rate frames extra
  IFS=, read -r codec width height rate frames extra < <(
    "$FFPROBE_PATH" -v error -select_streams v:0 \
      -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
      -of csv=p=0 "$path" 2>/dev/null
  )
  [[ -z "$extra" && "$codec" == "hevc" && "$width" == "8192" \
    && "$height" == "4096" && "$frames" == "$expected_frames" ]] || return 1
  /usr/bin/awk -F/ '
    NF == 2 && $2 != 0 { rate = $1 / $2 }
    NF == 1 { rate = $1 }
    END { exit !(rate >= 29.99 && rate <= 30.01) }
  ' <<< "$rate"
}

run_variant() {
  local blend="$1"
  local final_output="$2"
  local variant_name="$3"
  local memory_cache="$4"
  local variant_dir="$WORK_DIR/$variant_name"
  local concat_path="$variant_dir/parts.txt"
  local absolute_second="$START_TOTAL_SECONDS"
  local remaining="$SELECTED_SECONDS"
  local part_index=0
  [[ "$VALIDATE_ONLY" == "1" ]] || {
    mkdir -p "$variant_dir"
    : > "$concat_path"
  }

  while (( remaining > 0 )); do
    local segment_index=$((absolute_second / REFERENCE_SEGMENT_SECONDS))
    local segment_start=$((segment_index * REFERENCE_SEGMENT_SECONDS))
    local local_start=$((absolute_second - segment_start))
    local available=$((REFERENCE_SEGMENT_SECONDS - local_start))
    local part_seconds="$remaining"
    (( part_seconds <= available )) || part_seconds="$available"
    local formatted_part
    formatted_part="$(printf '%02d' "$part_index")"
    local part_output="$variant_dir/part-$formatted_part.mov"
    local left_input right_input left_manifest right_manifest
    left_input="$(segment_asset input left "$segment_index")" || {
      echo "error: cached left/shared input segment $segment_index is unavailable" >&2
      exit 1
    }
    right_input="$(segment_asset input right "$segment_index")" || {
      echo "error: cached right/shared input segment $segment_index is unavailable" >&2
      exit 1
    }
    left_manifest="$(segment_asset manifest left "$segment_index")" || {
      echo "error: cached left manifest segment $segment_index is unavailable" >&2
      exit 1
    }
    right_manifest="$(segment_asset manifest right "$segment_index")" || {
      echo "error: cached right manifest segment $segment_index is unavailable" >&2
      exit 1
    }
    echo "Grid $blend: source second $absolute_second, segment $segment_index, " \
      "local second $local_start, duration ${part_seconds}s"
    echo "Variant $variant_name: in-memory crop cache=$memory_cache, per-eye limit=${CACHE_LIMIT_MB} MiB"
    JASNA_RESTORE_ONLY_START_SECOND="$local_start" \
    JASNA_RESTORE_ONLY_AUDIO_START_SECOND="$absolute_second" \
    JASNA_RESTORE_ONLY_SECONDS="$part_seconds" \
    JASNA_RESTORE_ONLY_LEFT_INPUT="$left_input" \
    JASNA_RESTORE_ONLY_RIGHT_INPUT="$right_input" \
    JASNA_RESTORE_ONLY_LEFT_MANIFEST="$left_manifest" \
    JASNA_RESTORE_ONLY_RIGHT_MANIFEST="$right_manifest" \
    JASNA_RESTORE_ONLY_VALIDATE="$VALIDATE_ONLY" \
    JASNA_MODEL_BATCH="$MODEL_BATCH" \
    JASNA_METAL_WINDOWS_PER_PROCESS="$WINDOWS_PER_PROCESS" \
    JASNA_IN_MEMORY_CROP_CACHE="$memory_cache" \
    JASNA_IN_MEMORY_CACHE_LIMIT_MB="$CACHE_LIMIT_MB" \
    JASNA_LARGE_REGION_MAX_BLEND="$blend" \
      "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
        "$REFERENCE_WORK_DIR" "$part_output"
    if [[ "$VALIDATE_ONLY" != "1" ]]; then
      local escaped_part="${part_output//\'/\'\\\'\'}"
      printf "file '%s'\n" "$escaped_part" >> "$concat_path"
    fi
    absolute_second=$((absolute_second + part_seconds))
    remaining=$((remaining - part_seconds))
    part_index=$((part_index + 1))
  done

  [[ "$VALIDATE_ONLY" != "1" ]] || return 0
  local writing_output="$variant_dir/.joined-writing.mov"
  [[ ! -e "$writing_output" ]] || {
    echo "error: interrupted join exists: $writing_output" >&2
    exit 1
  }
  "$FFMPEG_PATH" -hide_banner -loglevel error \
    -f concat -safe 0 -i "$concat_path" -c copy \
    -video_track_timescale 600 -movflags +faststart "$writing_output"
  valid_joined_output "$writing_output" "$((SELECTED_SECONDS * 30))" || {
    echo "error: joined crop-density output failed validation" >&2
    exit 1
  }
  mv "$writing_output" "$final_output"
}

echo "Recovery-only $COMPARISON_KIND A/B"
echo "Reference:   $REFERENCE_WORK_DIR"
echo "Source time: $START_TIME (${START_TOTAL_SECONDS}s), duration ${SELECTED_SECONDS}s"
echo "Detection:   reused cached manifests"
echo "Model batch: $MODEL_BATCH"
echo "Windows/process: $WINDOWS_PER_PROCESS"
echo "Baseline:    max blend ${BASELINE_BLEND}px"
echo "Candidate:   max blend ${CANDIDATE_BLEND}px"
echo "Handoff:     $BASELINE_NAME=$BASELINE_CACHE, $CANDIDATE_NAME=$CANDIDATE_CACHE; limit ${CACHE_LIMIT_MB} MiB per eye/window"
echo "Run order:   $AB_ORDER"
echo "Graph diagnostics: phases=$JASNA_GRAPH_PHASE_TELEMETRY, trace=$JASNA_GRAPH_TRACE"

if [[ "$AB_ORDER" == "candidate-first" ]]; then
  echo "A/B 1/2: $CANDIDATE_NAME"
  run_variant "$CANDIDATE_BLEND" "$CANDIDATE_OUTPUT" "$CANDIDATE_NAME" "$CANDIDATE_CACHE"
  echo "A/B 2/2: $BASELINE_NAME"
  run_variant "$BASELINE_BLEND" "$BASELINE_OUTPUT" "$BASELINE_NAME" "$BASELINE_CACHE"
else
  echo "A/B 1/2: $BASELINE_NAME"
  run_variant "$BASELINE_BLEND" "$BASELINE_OUTPUT" "$BASELINE_NAME" "$BASELINE_CACHE"
  echo "A/B 2/2: $CANDIDATE_NAME"
  run_variant "$CANDIDATE_BLEND" "$CANDIDATE_OUTPUT" "$CANDIDATE_NAME" "$CANDIDATE_CACHE"
fi

echo
if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "$COMPARISON_KIND source-time asset validation: PASS"
  echo "No restoration or output writing was performed."
else
  echo "$COMPARISON_KIND recovery-only A/B: PASS"
  echo "Compare mosaic cleanliness before considering the candidate for rollout."
  echo "Baseline:  $BASELINE_OUTPUT"
  echo "Candidate: $CANDIDATE_OUTPUT"
fi
