#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="JasnaMetalPoC"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
pkill -x "$APP_NAME" >/dev/null 2>&1 || true
mkdir -p "$ROOT_DIR/.build/ModuleCache"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
swift build --disable-sandbox
APP_BINARY="$(swift build --disable-sandbox --show-bin-path)/$APP_NAME"

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
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify|--metal-ml-probe|--metal-ml-benchmark|--metal-ml-interop|--propagation-smoke|--propagation-suite|--reconstruct-frame|--zero-copy-frame|--zero-copy-frame-grouped|--zero-copy-frame-staged|--zero-copy-frame-fused|--spynet-pair|--frame-with-spynet|--temporal-inputs|--three-frame-recurrence|--three-frame-first-pass|--three-frame-four-pass|--variable-clip [frames]|--metal-ml-suite|--schedule [frames]|--validate-package-graph|--allocate-frame-graph [frames]|--validate-deform-weights|--benchmark-real-weights]" >&2
    exit 2
    ;;
esac
