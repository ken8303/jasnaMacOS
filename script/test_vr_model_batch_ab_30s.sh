#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_PREFIX" >&2
  echo "example: $0 previous.jasna-vr30-v22-work /path/to/model-batch-ab" >&2
  echo "Runs restoration only: batch 1 first, then batch 2, using identical assets." >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="$1"
OUTPUT_PREFIX="${2%.mov}"
BATCH2_MODELS="$ROOT_DIR/Models/MetalMLBatch2"

[[ -d "$BATCH2_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: batch-2 Metal packages are unavailable: $BATCH2_MODELS" >&2
  exit 1
}
[[ "$OUTPUT_PREFIX" != *[[:space:]] ]] || {
  echo "error: output prefix ends with whitespace: '$OUTPUT_PREFIX'" >&2
  exit 1
}

export JASNA_RESTORE_ONLY_START_SECOND="${JASNA_RESTORE_ONLY_START_SECOND:-0}"
export JASNA_RESTORE_ONLY_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-30}"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_TEMPORAL_WARMUP_FRAMES=5
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1
# A benchmark must fail on a timeout; retrying after a Metal watchdog reset
# produces misleading wall-clock and GPU measurements.
export JASNA_GPU_TIMEOUT_RETRIES=0
export JASNA_GRAPH_PHASE_TELEMETRY=1

BASELINE_OUTPUT="${OUTPUT_PREFIX}-batch1.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-batch2.mov"

if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
  echo "Building one shared optimized Swift executable before timing either candidate"
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

echo "Jasna ${JASNA_RESTORE_ONLY_SECONDS}-second restoration model-batch A/B"
echo "Prepared source and detector manifests: reused"
echo "Windows/process: $JASNA_METAL_WINDOWS_PER_PROCESS; temporal warm-up: 5; in-memory handoff: 512 MiB"

echo
echo "A/B 1/2: model batch 1"
JASNA_MODEL_BATCH=1 \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$BASELINE_OUTPUT"

echo
echo "A/B 2/2: model batch 2 with automatic batch-1 circuit breaker"
JASNA_MODEL_BATCH=2 \
JASNA_BATCH2_MODELS_DIR="$BATCH2_MODELS" \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$CANDIDATE_OUTPUT"

echo
echo "Restoration model-batch A/B: PASS"
echo "Batch 1: $BASELINE_OUTPUT"
echo "Batch 2: $CANDIDATE_OUTPUT"
