#!/usr/bin/env bash

# Direct side-by-side restoration stage for test_vr_sparse_30s.sh.
# Sourced by the main workflow and intentionally uses its validated globals.

run_direct_sbs_pipeline() {
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
  SKIPPED_INTERMEDIATE_REMUX_COUNT=0
  TOTAL_TIMELINE_SEGMENTS=$((
    (EXPECTED_FRAME_COUNT + TEST_SEGMENT_SECONDS * 30 - 1) \
      / (TEST_SEGMENT_SECONDS * 30)
  ))
  TOTAL_RESTORE_WINDOWS=$(( (EXPECTED_FRAME_COUNT + 29) / 30 ))
  ACTIVE_JOB_CURSOR=0
  echo "Metal process isolation: at most $METAL_WINDOWS_PER_PROCESS temporal windows/process"
  echo "JASNA_PROGRESS|3|0|$TOTAL_RESTORE_WINDOWS|Restoring video"
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
        COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + JOB_WINDOW_COUNT))
        (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
        echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
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
        -avoid_negative_ts make_zero -video_track_timescale 600 "$CLEAN_TEMP"
      valid_direct_segment "$CLEAN_TEMP" "$JOB_FRAME_COUNT" || {
        echo "error: clean SBS timeline segment failed validation: $CLEAN_TEMP" >&2
        exit 1
      }
      mv "$CLEAN_TEMP" "$BATCH_SEGMENT"
      DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
      BYPASSED_WINDOW_COUNT=$((BYPASSED_WINDOW_COUNT + JOB_WINDOW_COUNT))
      COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + JOB_WINDOW_COUNT))
      (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
      echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
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
      COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + JOB_WINDOW_COUNT))
      (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
      echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
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
        COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + WINDOW_START))
        (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
        echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
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
        EXPECTED_BYPASS_FRAMES=$((JOB_FRAME_COUNT - WINDOW_START * 30))
        (( EXPECTED_BYPASS_FRAMES > WINDOW_COUNT * 30 )) \
          && EXPECTED_BYPASS_FRAMES=$((WINDOW_COUNT * 30))
        BYPASS_DURATION="$(/usr/bin/awk \
          -v frames="$EXPECTED_BYPASS_FRAMES" 'BEGIN { printf "%.6f\n", frames / 30 }'
        )"
        echo "Bypassing empty source windows $((WINDOW_START + 1))-$WINDOW_END/$JOB_WINDOW_COUNT with SBS packet copy"
        if "$FFMPEG_PATH" -hide_banner -loglevel error \
            -ss "$GLOBAL_START_SECONDS" -i "$TEST_INPUT" \
            -t "$BYPASS_DURATION" -map '0:v:0' -an -c copy \
            -avoid_negative_ts make_zero -video_track_timescale 600 "$BYPASS_TEMP" \
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
      COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + WINDOW_END))
      (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
      echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
    done
    if (( ${#JOB_DIRECT_SEGMENTS[@]} == 1 )); then
      BATCH_SEGMENT="${JOB_DIRECT_SEGMENTS[0]}"
      DIRECT_SEGMENTS+=("$BATCH_SEGMENT")
    else
      # Each window is already frame-count/decode validated and is the smallest useful
      # restart checkpoint. Keep those files and feed them directly to the final concat;
      # materializing a two-minute copy here made every restored byte traverse the output
      # drive once more without adding recovery safety.
      DIRECT_SEGMENTS+=("${JOB_DIRECT_SEGMENTS[@]}")
      SKIPPED_INTERMEDIATE_REMUX_COUNT=$((SKIPPED_INTERMEDIATE_REMUX_COUNT + 1))
      echo "Recorded ${#JOB_DIRECT_SEGMENTS[@]} validated window checkpoint(s) for source segment $((JOB_INDEX + 1))/$TOTAL_TIMELINE_SEGMENTS without an intermediate remux"
    fi
    COMPLETED_RESTORE_WINDOWS=$((JOB_INDEX * TEST_SEGMENT_SECONDS + JOB_WINDOW_COUNT))
    (( COMPLETED_RESTORE_WINDOWS > TOTAL_RESTORE_WINDOWS )) && COMPLETED_RESTORE_WINDOWS="$TOTAL_RESTORE_WINDOWS"
    echo "JASNA_PROGRESS|3|$COMPLETED_RESTORE_WINDOWS|$TOTAL_RESTORE_WINDOWS|Restoring video"
  done
  (( ACTIVE_JOB_CURSOR == ${#LEFT_JOB_INPUTS[@]} )) || {
    echo "error: not all active eye segments were placed on the SBS timeline" >&2
    exit 1
  }
  echo "Prepared ${#DIRECT_SEGMENTS[@]} validated 8K SBS timeline fragment(s); bypassed/restored windows $BYPASSED_WINDOW_COUNT/$RESTORED_WINDOW_COUNT"
  echo "Intermediate two-minute remuxes avoided: $SKIPPED_INTERMEDIATE_REMUX_COUNT"

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
          -video_track_timescale 600 "$NORMALIZED_TEMP"
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
  echo "JASNA_PROGRESS|4|0|1|Assembling and verifying output"
  FINAL_ASSEMBLY_STARTED_SECONDS=$SECONDS
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
      -n \
      "$FINAL_TEMP"
    completed_sbs_output "$FINAL_TEMP" || {
      explain_sbs_validation_failure "$FINAL_TEMP"
      echo "error: direct SBS output failed codec, dimensions, frame-count, or decode validation" >&2
      exit 1
    }
    mv "$FINAL_TEMP" "$OUTPUT_PATH"
  fi
  echo "JASNA_PROGRESS|4|1|1|Output verified"
  echo "Final assembly/validation wall: $((SECONDS - FINAL_ASSEMBLY_STARTED_SECONDS))s; streaming relocation disabled"
  FINAL_INFO="$("$FFPROBE_PATH" -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT_PATH")"
  echo "Sparse SBS VR restoration ($RUN_DESCRIPTION): PASS"
  echo "$FINAL_INFO"
  report_compositor_fallback_summary
  report_total_wall_time
  echo "Output: $OUTPUT_PATH"
  echo "Log:    $LOG_PATH"
  report_persistent_work
  exit 0
}
