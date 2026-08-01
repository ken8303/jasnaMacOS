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
esac

case "$MODE" in
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window|--restore-sbs-eye|restore-sbs-eye|--restore-eye-video|restore-eye-video|--restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse)
    RESTORE_OUTPUT_DIR="$(cd "$(dirname "$RESTORE_OUTPUT_PATH")" && pwd)"
    RESTORE_OUTPUT_NAME="$(basename "$RESTORE_OUTPUT_PATH")"
    RESTORE_OUTPUT_STEM="${RESTORE_OUTPUT_NAME%.*}"
    export JASNA_WORK_DIR="${JASNA_WORK_DIR:-$RESTORE_OUTPUT_DIR/${RESTORE_OUTPUT_STEM}.jasna-work}"
    JASNA_LOG_PATH="$RESTORE_OUTPUT_DIR/${RESTORE_OUTPUT_STEM}.jasna.log"
    mkdir -p "$JASNA_WORK_DIR"
    exec > >(/usr/bin/tee -a "$JASNA_LOG_PATH") 2>&1
    echo
    echo "===== Jasna restoration session $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "Log:      $JASNA_LOG_PATH"
    echo "Work dir: $JASNA_WORK_DIR"
    echo "Output:   $RESTORE_OUTPUT_PATH"
    ;;
esac

cd "$ROOT_DIR"
pkill -x "$APP_NAME" >/dev/null 2>&1 || true
mkdir -p "$ROOT_DIR/.build/ModuleCache"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
BUILD_ARGUMENTS=(--disable-sandbox)
case "$MODE" in
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window|--restore-sbs-eye|restore-sbs-eye|--restore-eye-video|restore-eye-video|--restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse)
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
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify|--metal-ml-probe|--metal-ml-benchmark|--metal-ml-interop|--propagation-smoke|--propagation-suite|--reconstruct-frame|--zero-copy-frame|--zero-copy-frame-grouped|--zero-copy-frame-staged|--zero-copy-frame-fused|--spynet-pair|--frame-with-spynet|--temporal-inputs|--three-frame-recurrence|--three-frame-first-pass|--three-frame-four-pass|--variable-clip [frames]|--single-run-clip [frames]|--plan-sbs-video [width height source-fps duration]|--inspect-sbs-video input|--transcode-sbs-30 input output.mov|--transcode-sbs-30-tiled input output.mov|--restore-sbs-video input output.mov|--restore-sbs-eye input left|right output.mov|--restore-eye-video input output.mov|--restore-eye-windows input output-directory|--diagnose-sbs-tile input tile-number|--metal-ml-suite|--schedule [frames]|--validate-package-graph|--allocate-frame-graph [frames]|--validate-deform-weights|--benchmark-real-weights]" >&2
    exit 2
    ;;
esac
