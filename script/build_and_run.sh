#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="JasnaMetalPoC"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${JASNA_MODELS_DIR:-$ROOT_DIR/Models/MetalML}"

validate_metal_model_directory() {
  local directory="$1"
  local label="$2"
  local package
  local packages=(
    feature_extract
    spynet_level_0 spynet_level_1 spynet_level_2
    spynet_level_3 spynet_level_4 spynet_level_5
    offset_backward_1 offset_forward_1 offset_backward_2 offset_forward_2
    backbone_backward_1 backbone_forward_1 backbone_backward_2 backbone_forward_2
    upsample
  )
  [[ -d "$directory" ]] || {
    echo "error: $label directory does not exist: $directory" >&2
    return 1
  }
  for package in "${packages[@]}"; do
    [[ -d "$directory/$package.mtlpackage" ]] || {
      echo "error: $label directory is missing $package.mtlpackage: $directory" >&2
      return 1
    }
  done
}

[[ -d "$MODELS_DIR" ]] || {
  echo "error: Metal ML model directory does not exist: $MODELS_DIR" >&2
  exit 1
}
MODELS_DIR="$(cd "$MODELS_DIR" && pwd -P)"

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
    validate_metal_model_directory "$MODELS_DIR" "Metal ML model"
    if [[ "${JASNA_MODEL_BATCH:-1}" == "2" ]]; then
      if [[ -n "${JASNA_BATCH2_MODELS_DIR:-}" ]]; then
        validate_metal_model_directory "$JASNA_BATCH2_MODELS_DIR" "batch-2 Metal ML model"
        EXPECTED_BATCH2_MODELS_DIR="${MODELS_DIR%/}Batch2"
        [[ -d "$EXPECTED_BATCH2_MODELS_DIR" ]] || {
          echo "error: matching batch-2 model directory does not exist: $EXPECTED_BATCH2_MODELS_DIR" >&2
          exit 1
        }
        RESOLVED_BATCH2_MODELS_DIR="$(cd "$JASNA_BATCH2_MODELS_DIR" && pwd -P)"
        RESOLVED_EXPECTED_BATCH2_MODELS_DIR="$(cd "$EXPECTED_BATCH2_MODELS_DIR" && pwd -P)"
        [[ "$RESOLVED_BATCH2_MODELS_DIR" == "$RESOLVED_EXPECTED_BATCH2_MODELS_DIR" ]] || {
          echo "error: batch-2 packages do not match the selected batch-1 model set" >&2
          echo "selected: $MODELS_DIR" >&2
          echo "expected: $EXPECTED_BATCH2_MODELS_DIR" >&2
          echo "received: $JASNA_BATCH2_MODELS_DIR" >&2
          exit 1
        }
        if [[ -f "$MODELS_DIR/model-family.txt" \
              || -f "$JASNA_BATCH2_MODELS_DIR/model-family.txt" ]]; then
          [[ -f "$MODELS_DIR/model-family.txt" \
              && -f "$JASNA_BATCH2_MODELS_DIR/model-family.txt" ]] || {
            echo "error: both batch-1 and batch-2 model sets require model-family.txt" >&2
            exit 1
          }
          /usr/bin/cmp -s \
            "$MODELS_DIR/model-family.txt" \
            "$JASNA_BATCH2_MODELS_DIR/model-family.txt" || {
            echo "error: batch-1 and batch-2 model provenance does not match" >&2
            exit 1
          }
        fi
      elif [[ "$MODELS_DIR" == "$ROOT_DIR/Models/MetalML" \
              && -d "$ROOT_DIR/Models/MetalMLBatch2" ]]; then
        validate_metal_model_directory \
          "$ROOT_DIR/Models/MetalMLBatch2" "batch-2 Metal ML model"
        export JASNA_BATCH2_MODELS_DIR="$ROOT_DIR/Models/MetalMLBatch2"
      else
        echo "WARNING: no matching batch-2 packages for $MODELS_DIR; using batch 1"
        export JASNA_MODEL_BATCH=1
      fi
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
    echo "Models:   $MODELS_DIR"
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
  --restore-sbs-video|restore-sbs-video|--restore-sbs-window|restore-sbs-window|--restore-sbs-eye|restore-sbs-eye|--restore-eye-video|restore-eye-video|--restore-eye-windows|restore-eye-windows|--restore-eye-windows-sparse|restore-eye-windows-sparse|--restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch|--restore-stereo-sparse-batch|restore-stereo-sparse-batch|--core-ml-comparison|core-ml-comparison|--core-ml-spynet|core-ml-spynet)
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
    "$APP_BINARY" --metal-ml-probe "$MODELS_DIR/feature_extract.mtlpackage"
    ;;
  --metal-ml-benchmark|metal-ml-benchmark)
    "$APP_BINARY" --metal-ml-benchmark "$MODELS_DIR/feature_extract.mtlpackage"
    ;;
  --metal-ml-interop|metal-ml-interop)
    "$APP_BINARY" --metal-ml-interop "$MODELS_DIR/feature_extract.mtlpackage"
    ;;
  --core-ml-comparison|core-ml-comparison)
    COREML_MODELS_DIR="${JASNA_COREML_MODELS_DIR:-$ROOT_DIR/Models/CoreML}"
    [[ -d "$COREML_MODELS_DIR" ]] || {
      echo "error: Core ML model directory does not exist: $COREML_MODELS_DIR" >&2
      exit 1
    }
    "$APP_BINARY" --core-ml-comparison \
      "$COREML_MODELS_DIR" "$MODELS_DIR" "${JASNA_COREML_BENCHMARK_ITERATIONS:-10}"
    ;;
  --core-ml-spynet|core-ml-spynet)
    COREML_MODELS_DIR="${JASNA_COREML_MODELS_DIR:-$ROOT_DIR/Models/CoreML}"
    [[ -d "$COREML_MODELS_DIR" ]] || {
      echo "error: Core ML model directory does not exist: $COREML_MODELS_DIR" >&2
      exit 1
    }
    "$APP_BINARY" --core-ml-spynet \
      "$COREML_MODELS_DIR" "$MODELS_DIR" "$ROOT_DIR/Models/SPyNetOracle" \
      "${JASNA_COREML_SPYNET_ITERATIONS:-7}"
    ;;
  --core-ai-feature-extract|core-ai-feature-extract)
    "$APP_BINARY" --core-ai-feature-extract "$ROOT_DIR/Models/CoreAI/feature_extract.aimodel"
    ;;
  --propagation-smoke|propagation-smoke)
    "$APP_BINARY" --propagation-smoke "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --propagation-suite|propagation-suite)
    "$APP_BINARY" --propagation-suite "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --reconstruct-frame|reconstruct-frame)
    "$APP_BINARY" --reconstruct-frame "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame|zero-copy-frame)
    "$APP_BINARY" --zero-copy-frame "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-grouped|zero-copy-frame-grouped)
    "$APP_BINARY" --zero-copy-frame-grouped "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-staged|zero-copy-frame-staged)
    "$APP_BINARY" --zero-copy-frame-staged "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --zero-copy-frame-fused|zero-copy-frame-fused)
    "$APP_BINARY" --zero-copy-frame-fused "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --spynet-pair|spynet-pair)
    "$APP_BINARY" --spynet-pair "$MODELS_DIR" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --frame-with-spynet|frame-with-spynet)
    "$APP_BINARY" --frame-with-spynet "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --temporal-inputs|temporal-inputs)
    "$APP_BINARY" --temporal-inputs "$MODELS_DIR" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-recurrence|three-frame-recurrence)
    "$APP_BINARY" --three-frame-recurrence "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-first-pass|three-frame-first-pass)
    "$APP_BINARY" --three-frame-first-pass "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle"
    ;;
  --three-frame-four-pass|three-frame-four-pass)
    "$APP_BINARY" --three-frame-four-pass "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle" "$ROOT_DIR/Models/FullModelOracle"
    ;;
  --variable-clip|variable-clip)
    FRAME_COUNT="${2:-5}"
    "$APP_BINARY" --variable-clip "$FRAME_COUNT" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "$ROOT_DIR/Models/SPyNetOracle" "$ROOT_DIR/Models/FullModelOracle/$FRAME_COUNT"
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
    "$APP_BINARY" --restore-sbs-video "${2:?input video path required}" "${3:?output .mov path required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-sbs-eye|restore-sbs-eye)
    "$APP_BINARY" --restore-sbs-eye "${2:?input video path required}" "${3:?left or right required}" "${4:?output .mov path required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-video|restore-eye-video)
    "$APP_BINARY" --restore-eye-video "${2:?input video path required}" "${3:?output .mov path required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-windows|restore-eye-windows)
    "$APP_BINARY" --restore-eye-windows "${2:?input video path required}" "${3:?output directory required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --restore-eye-windows-sparse|restore-eye-windows-sparse)
    "$APP_BINARY" --restore-eye-windows-sparse "${2:?input video path required}" "${3:?output directory required}" "${4:?mosaic region manifest required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --restore-eye-windows-sparse-batch|restore-eye-windows-sparse-batch)
    "$APP_BINARY" --restore-eye-windows-sparse-batch "${@:2}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --restore-stereo-sparse-batch|restore-stereo-sparse-batch)
    "$APP_BINARY" --restore-stereo-sparse-batch "${@:2}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv" "${JASNA_VR_PROJECTION:-raw}"
    ;;
  --diagnose-sbs-tile|diagnose-sbs-tile)
    "$APP_BINARY" --diagnose-sbs-tile "${2:?input video path required}" "${3:?one-based tile number required}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --single-run-clip|single-run-clip)
    "$APP_BINARY" --single-run-clip "${2:-30}" "$MODELS_DIR" "$ROOT_DIR/Models/DeformConv"
    ;;
  --schedule|schedule)
    "$APP_BINARY" --schedule "${2:-5}"
    ;;
  --validate-package-graph|validate-package-graph)
    "$APP_BINARY" --validate-package-graph "$MODELS_DIR"
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
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify|--metal-ml-probe|--metal-ml-benchmark|--metal-ml-interop|--core-ml-comparison|--core-ml-spynet|--core-ai-feature-extract|--propagation-smoke|--propagation-suite|--reconstruct-frame|--zero-copy-frame|--zero-copy-frame-grouped|--zero-copy-frame-staged|--zero-copy-frame-fused|--spynet-pair|--frame-with-spynet|--temporal-inputs|--three-frame-recurrence|--three-frame-first-pass|--three-frame-four-pass|--variable-clip [frames]|--single-run-clip [frames]|--plan-sbs-video [width height source-fps duration]|--inspect-sbs-video input|--transcode-sbs-30 input output.mov|--transcode-sbs-30-tiled input output.mov|--restore-sbs-video input output.mov|--restore-sbs-eye input left|right output.mov|--restore-eye-video input output.mov|--restore-eye-windows input output-directory|--restore-stereo-sparse-batch paired-job...|--diagnose-sbs-tile input tile-number|--metal-ml-suite|--schedule [frames]|--validate-package-graph|--allocate-frame-graph [frames]|--validate-deform-weights|--benchmark-real-weights]" >&2
    exit 2
    ;;
esac
