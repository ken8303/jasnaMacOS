#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="JasnaMetalPoC"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$MODE" in
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window)
    [[ $# -ge 3 ]] || {
      echo "error: restore mode requires input and output video paths" >&2
      exit 2
    }
    RESTORE_OUTPUT_PATH="$3"
    ;;
  --restore-sbs-eye|restore-sbs-eye)
    [[ $# -ge 4 ]] || {
      echo "error: restore-eye mode requires input, left|right, and output paths" >&2
      exit 2
    }
    RESTORE_OUTPUT_PATH="$4"
    ;;
  --restore-eye-video|restore-eye-video)
    [[ $# -ge 3 ]] || {
      echo "error: restore-eye-video mode requires input and output paths" >&2
      exit 2
    }
    RESTORE_OUTPUT_PATH="$3"
    ;;
  --restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse)
    [[ $# -ge 3 ]] || {
      echo "error: restore-eye-windows mode requires input and output-directory paths" >&2
      exit 2
    }
    mkdir -p "$3"
    RESTORE_OUTPUT_PATH="$3/windowed-output.mov"
    ;;
  --restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch)
    [[ $# -ge 5 ]] || {
      echo "error: sparse batch mode requires at least one restoration job" >&2
      exit 2
    }
    mkdir -p "$3"
    RESTORE_OUTPUT_PATH="$3/batched-windowed-output.mov"
    ;;
  --restore-stereo-sparse-batch|restore-stereo-sparse-batch)
    [[ $# -ge 8 ]] || {
      echo "error: direct SBS batch mode requires at least one paired restoration job" >&2
      exit 2
    }
    mkdir -p "$(dirname "$4")"
    RESTORE_OUTPUT_PATH="$4"
    ;;
esac

case "$MODE" in
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window|--restore-sbs-eye|restore-sbs-eye|--restore-eye-video|restore-eye-video|--restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse|--restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch|--restore-stereo-sparse-batch|restore-stereo-sparse-batch)
    RESTORE_OUTPUT_DIR="$(cd "$(dirname "$RESTORE_OUTPUT_PATH")" && pwd)"
    RESTORE_OUTPUT_NAME="$(basename "$RESTORE_OUTPUT_PATH")"
    RESTORE_OUTPUT_STEM="${RESTORE_OUTPUT_NAME%.*}"
    export JASNA_WORK_DIR="${JASNA_WORK_DIR:-$RESTORE_OUTPUT_DIR/${RESTORE_OUTPUT_STEM}.jasna-work}"
    export JASNA_LOG_PEAK_MEMORY="${JASNA_LOG_PEAK_MEMORY:-1}"
    if [[ "${JASNA_MODEL_BATCH:-1}" == "2" \
          && -d "$ROOT_DIR/Models/MetalMLBatch2/feature_extract.mtlpackage" ]]; then
      export JASNA_BATCH2_MODELS_DIR="${JASNA_BATCH2_MODELS_DIR:-$ROOT_DIR/Models/MetalMLBatch2}"
    fi
    JASNA_LOG_PATH="$RESTORE_OUTPUT_DIR/${RESTORE_OUTPUT_STEM}.jasna.log"
    mkdir -p "$JASNA_WORK_DIR"
    JASNA_PROCESS_LOCK="$JASNA_WORK_DIR/.jasna-process-lock"
    if ! mkdir "$JASNA_PROCESS_LOCK" 2>/dev/null; then
      EXISTING_PID="$(/bin/cat "$JASNA_PROCESS_LOCK/pid" 2>/dev/null || true)"
      if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "error: this restoration work directory is already active (PID $EXISTING_PID)" >&2
        echo "work dir: $JASNA_WORK_DIR" >&2
        exit 1
      fi
      STALE_LOCK="$JASNA_WORK_DIR/.jasna-process-lock.stale-$(date '+%Y%m%d-%H%M%S')-$$"
      mv "$JASNA_PROCESS_LOCK" "$STALE_LOCK"
      mkdir "$JASNA_PROCESS_LOCK"
    fi
    printf '%s\n' "$$" > "$JASNA_PROCESS_LOCK/pid"
    cleanup_jasna_process_lock() {
      [[ -d "$JASNA_PROCESS_LOCK" ]] || return 0
      local owner_pid
      owner_pid="$(/bin/cat "$JASNA_PROCESS_LOCK/pid" 2>/dev/null || true)"
      [[ "$owner_pid" == "$$" ]] || return 0
      rm -f "$JASNA_PROCESS_LOCK/pid"
      rmdir "$JASNA_PROCESS_LOCK" 2>/dev/null || true
    }
    trap cleanup_jasna_process_lock EXIT
    trap 'exit 130' INT
    trap 'exit 143' HUP TERM
    exec > >(/usr/bin/tee -a "$JASNA_LOG_PATH") 2>&1
    echo
    echo "===== Jasna restoration session $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "Log:      $JASNA_LOG_PATH"
    echo "Work dir: $JASNA_WORK_DIR"
    echo "Output:   $RESTORE_OUTPUT_PATH"
    if [[ -n "${JASNA_BATCH2_MODELS_DIR:-}" ]]; then
      echo "Model batch: 2 ($JASNA_BATCH2_MODELS_DIR)"
    else
      echo "Model batch: 1"
    fi
    ;;
esac

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/.build/ModuleCache"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
BUILD_ARGUMENTS=(--disable-sandbox)
case "$MODE" in
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window|--restore-sbs-eye|restore-sbs-eye|--restore-eye-video|restore-eye-video|--restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse|--restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch|--restore-stereo-sparse-batch|restore-stereo-sparse-batch)
    BUILD_ARGUMENTS+=(-c release)
    echo "Building optimized restoration binary..."
    ;;
esac
if [[ -n "${JASNA_APP_BINARY:-}" ]]; then
  [[ -x "$JASNA_APP_BINARY" ]] || {
    echo "error: JASNA_APP_BINARY is not executable: $JASNA_APP_BINARY" >&2
    exit 1
  }
  APP_BINARY="$JASNA_APP_BINARY"
  echo "Using shared optimized restoration binary: $APP_BINARY"
else
  swift build "${BUILD_ARGUMENTS[@]}"
  APP_BINARY="$(swift build "${BUILD_ARGUMENTS[@]}" --show-bin-path)/$APP_NAME"
fi

case "$MODE" in
  run)
    "$APP_BINARY"
    ;;
  --debug|debug)
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    "$APP_BINARY" &
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    "$APP_BINARY" &
    /usr/bin/log stream --info --style compact --predicate 'subsystem == "com.jasna.metalpoc"'
    ;;
  --verify|verify)
    "$APP_BINARY" --self-test
    ;;
  --metal-ml-probe|metal-ml-probe)
    "$APP_BINARY" --metal-ml-probe "$ROOT_DIR/Models/MetalML/feature_extract.mtlpackage"
    ;;
  --metal-ml-benchmark|metal-ml-benchmark)
    "$APP_BINARY" --metal-ml-benchmark "$ROOT_DIR/Models/MetalML/feature_extract.mtlpackage"
    ;;
  --metal-ml-interop|metal-ml-interop)
    "$APP_BINARY" --metal-ml-interop "$ROOT_DIR/Models/MetalML/feature_extract.mtlpackage"
    ;;
  --core-ai-feature-extract|core-ai-feature-extract)
    "$APP_BINARY" --core-ai-feature-extract "$ROOT_DIR/Models/CoreAI/feature_extract.aimodel"
    ;;
  --propagation-smoke|propagation-smoke)
    "$APP_BINARY" --propagation-smoke "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --propagation-suite|propagation-suite)
    "$APP_BINARY" --propagation-suite "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --reconstruct-frame|reconstruct-frame)
    "$APP_BINARY" --reconstruct-frame "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame|zero-copy-frame)
    "$APP_BINARY" --zero-copy-frame "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-grouped|zero-copy-frame-grouped)
    "$APP_BINARY" --zero-copy-frame-grouped "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-staged|zero-copy-frame-staged)
    "$APP_BINARY" --zero-copy-frame-staged "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-fused|zero-copy-frame-fused)
    "$APP_BINARY" --zero-copy-frame-fused "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --spynet-pair|spynet-pair)
    "$APP_BINARY" --spynet-pair "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --frame-with-spynet|frame-with-spynet)
    "$APP_BINARY" --frame-with-spynet "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --temporal-inputs|temporal-inputs)
    "$APP_BINARY" --temporal-inputs "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-recurrence|three-frame-recurrence)
    "$APP_BINARY" --three-frame-recurrence "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-first-pass|three-frame-first-pass)
    "$APP_BINARY" --three-frame-first-pass "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-four-pass|three-frame-four-pass)
    "$APP_BINARY" --three-frame-four-pass "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle" "$ROOT_DIR/Models/FullModelOracle"
    ;;
  --variable-clip|variable-clip)
    FRAME_COUNT="${2:-5}"
    "$APP_BINARY" --variable-clip "$FRAME_COUNT" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle" "$ROOT_DIR/Models/FullModelOracle/$FRAME_COUNT"
    ;;
  --plan-sbs-video|plan-sbs-video)
    "$APP_BINARY" --plan-sbs-video "${2:-7680}" "${3:-4320}" "${4:-60}" "${5:-1}"
    ;;
  --inspect-sbs-video|inspect-sbs-video)
    "$APP_BINARY" --inspect-sbs-video "${2:?input video path required}"
    ;;
  --transcode-sbs-30|transcode-sbs-30)
    "$APP_BINARY" --transcode-sbs-30 "${2:?input video path required}" "${3:?output .mov path required}"
    ;;
  --transcode-sbs-30-tiled|transcode-sbs-30-tiled)
    "$APP_BINARY" --transcode-sbs-30-tiled "${2:?input video path required}" "${3:?output .mov path required}"
    ;;
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window)
    "$APP_BINARY" --restore-sbs-video "${2:?input video path required}" "${3:?output .mov path required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-sbs-eye|restore-sbs-eye)
    "$APP_BINARY" --restore-sbs-eye "${2:?input video path required}" "${3:?left or right required}" "${4:?output .mov path required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-video|restore-eye-video)
    "$APP_BINARY" --restore-eye-video "${2:?input video path required}" "${3:?output .mov path required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-windows|restore-eye-windows)
    "$APP_BINARY" --restore-eye-windows "${2:?input video path required}" "${3:?output directory required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-windows-sparse|restore-eye-windows-sparse)
    "$APP_BINARY" --restore-eye-windows-sparse "${2:?input video path required}" "${3:?output directory required}" "${4:?mosaic region manifest required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch)
    "$APP_BINARY" --restore-eye-windows-sparse-batch "${@:2}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --restore-stereo-sparse-batch|restore-stereo-sparse-batch)
    "$APP_BINARY" --restore-stereo-sparse-batch "${@:2}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --diagnose-sbs-tile|diagnose-sbs-tile)
    "$APP_BINARY" --diagnose-sbs-tile "${2:?input video path required}" "${3:?one-based tile number required}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --single-run-clip|single-run-clip)
    "$APP_BINARY" --single-run-clip "${2:-30}" "$ROOT_DIR/Models/MetalML" "$ROOT_DIR/Models/DeformConv"
    ;;
  --schedule|schedule)
    "$APP_BINARY" --schedule "${2:-5}"
    ;;
  --validate-package-graph|validate-package-graph)
    "$APP_BINARY" --validate-package-graph "$ROOT_DIR/Models/MetalML"
    ;;
  --allocate-frame-graph|allocate-frame-graph)
    "$APP_BINARY" --allocate-frame-graph "${2:-5}"
    ;;
  --validate-deform-weights|validate-deform-weights)
    "$APP_BINARY" --validate-deform-weights "$ROOT_DIR/Models/DeformConv"
    ;;
  --benchmark-real-weights|benchmark-real-weights)
    "$APP_BINARY" --benchmark-real-weights "$ROOT_DIR/Models/DeformConv"
    ;;
  --metal-ml-suite|metal-ml-suite)
    for package in "$ROOT_DIR"/Models/MetalML/*.mtlpackage; do
      case "$package" in
        */spynet.mtlpackage) continue ;;
      esac
      "$APP_BINARY" --metal-ml-benchmark "$package"
    done
    ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify|--metal-ml-probe|--metal-ml-benchmark|--metal-ml-interop|--core-ai-feature-extract|--propagation-smoke|--propagation-suite|--reconstruct-frame|--zero-copy-frame|--zero-copy-frame-grouped|--zero-copy-frame-staged|--zero-copy-frame-fused|--spynet-pair|--frame-with-spynet|--temporal-inputs|--three-frame-recurrence|--three-frame-first-pass|--three-frame-four-pass|--variable-clip [frames]|--single-run-clip [frames]|--plan-sbs-video [width height source-fps duration]|--inspect-sbs-video input|--transcode-sbs-30 input output.mov|--transcode-sbs-30-tiled input output.mov|--restore-sbs-video input output.mov|--restore-sbs-eye input left|right output.mov|--restore-eye-video input output.mov|--restore-eye-windows input output-directory|--restore-stereo-sparse-batch paired-job...|--diagnose-sbs-tile input tile-number|--metal-ml-suite|--schedule [frames]|--validate-package-graph|--allocate-frame-graph [frames]|--validate-deform-weights|--benchmark-real-weights]" >&2
    exit 2
    ;;
esac
