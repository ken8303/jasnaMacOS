#!/usr/bin/env bash

# Media inspection and restart validation helpers for test_vr_sparse_30s.sh.
# This file is sourced after the workflow has validated its configuration; it
# deliberately uses the caller's FFmpeg paths, dimensions, and work directories.

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

explain_sbs_validation_failure() {
  local candidate="$1"
  echo "SBS validation details for: $candidate" >&2
  if [[ ! -s "$candidate" ]]; then
    echo "  output is missing or empty" >&2
    return
  fi
  "$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,time_base \
    -show_entries format=duration,size \
    -of default=noprint_wrappers=1 "$candidate" >&2 || true
  echo "  expected: codec=hevc, ${SOURCE_WIDTH}x${SOURCE_HEIGHT}, 30 fps, frames=${EXPECTED_FRAME_COUNT}, video duration=${TEST_VIDEO_DURATION}s" >&2
  if ! "$FFMPEG_PATH" -v error -i "$candidate" -map '0:v:0' \
      -frames:v 1 -f null - </dev/null >/dev/null 2>&1; then
    echo "  first video frame could not be decoded" >&2
  fi
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
        "$shared_segment"
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
