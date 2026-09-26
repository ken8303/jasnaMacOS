#!/usr/bin/env bash

# Resumable eye-pair assembly stage for test_vr_sparse_30s.sh.
# Sourced by the main workflow and intentionally uses its validated globals.

run_eye_pair_pipeline() {
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

  echo "JASNA_PROGRESS|4|0|1|Assembling and verifying output"
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
      -n \
      "$FINAL_TEMP"
    completed_sbs_output "$FINAL_TEMP" || {
      explain_sbs_validation_failure "$FINAL_TEMP"
      echo "error: final eye-pair SBS output failed validation" >&2
      exit 1
    }
    mv "$FINAL_TEMP" "$OUTPUT_PATH"
  fi
  echo "JASNA_PROGRESS|4|1|1|Output verified"

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
  report_persistent_work
  exit 0
}
