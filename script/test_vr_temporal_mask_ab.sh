#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: ./script/test_vr_temporal_mask_ab.sh REFERENCE_WORK_DIR OUTPUT_PREFIX

Runs restoration only against the same cached source and detector manifests.
The baseline keeps the current 50% adjacent-mask contribution; the candidate
uses full adjacent-mask coverage for fast motion. Detection and source
conversion are not repeated.

Optional environment variables:
  JASNA_RESTORE_ONLY_START_SECOND=0
  JASNA_RESTORE_ONLY_SECONDS=3
  JASNA_AB_TEMPORAL_BASELINE_STRENGTH=0.5
  JASNA_AB_TEMPORAL_CANDIDATE_STRENGTH=1.0
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="${1%/}"
OUTPUT_PREFIX="${2%.mov}"
BASELINE_STRENGTH="${JASNA_AB_TEMPORAL_BASELINE_STRENGTH:-0.5}"
CANDIDATE_STRENGTH="${JASNA_AB_TEMPORAL_CANDIDATE_STRENGTH:-1.0}"
RUN_CONFIG="$REFERENCE_WORK_DIR/run-config.txt"

[[ -d "$REFERENCE_WORK_DIR" && -s "$RUN_CONFIG" ]] || {
  echo "error: reference work directory or run-config.txt is unavailable" >&2
  exit 1
}
[[ "$OUTPUT_PREFIX" != *[[:space:]] ]] || {
  echo "error: output prefix ends with whitespace: '$OUTPUT_PREFIX'" >&2
  exit 1
}
for value in "$BASELINE_STRENGTH" "$CANDIDATE_STRENGTH"; do
  /usr/bin/awk -v value="$value" \
    'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1) }' || {
    echo "error: temporal mask strengths must be between 0 and 1" >&2
    exit 1
  }
done
[[ "$BASELINE_STRENGTH" != "$CANDIDATE_STRENGTH" ]] || {
  echo "error: baseline and candidate temporal strengths must differ" >&2
  exit 1
}

read_config() {
  local key="$1"
  local fallback="$2"
  local value
  value="$(/usr/bin/awk -F= -v key="$key" '$1 == key { print $2; exit }' "$RUN_CONFIG")"
  printf '%s\n' "${value:-$fallback}"
}

MODEL_BATCH="$(read_config model_batch 1)"
WINDOWS_PER_PROCESS="$(read_config metal_windows_per_process 1)"
MASK_TEMPORAL_RADIUS="$(read_config large_region_mask_temporal_radius 1)"
MASK_RECOVERY="$(read_config mosaic_mask_recovery_all_regions 1)"
BASELINE_OUTPUT="${OUTPUT_PREFIX}-baseline.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-temporal-union.mov"

if [[ "${JASNA_RESTORE_ONLY_VALIDATE:-0}" != "1" \
  && -z "${JASNA_APP_BINARY:-}" ]]; then
  DEFAULT_BINARY="$ROOT_DIR/.build/out/Products/Release/JasnaMetalPoC"
  NEED_BUILD=0
  [[ -x "$DEFAULT_BINARY" ]] || NEED_BUILD=1
  if [[ "$NEED_BUILD" == "0" ]] \
    && [[ -n "$(find "$ROOT_DIR/Sources" "$ROOT_DIR/Package.swift" \
      -newer "$DEFAULT_BINARY" -print -quit 2>/dev/null)" ]]; then
    NEED_BUILD=1
  fi
  if [[ "$NEED_BUILD" == "1" ]]; then
    echo "Building one optimized executable for both temporal-mask runs"
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

echo "Recovery-only temporal-mask A/B"
echo "Reference:  $REFERENCE_WORK_DIR"
echo "Selected:   relative second ${JASNA_RESTORE_ONLY_START_SECOND:-0}, duration ${JASNA_RESTORE_ONLY_SECONDS:-3}s"
echo "Model:      unchanged, batch $MODEL_BATCH"
echo "Detection:  reused cached manifests"
echo "Baseline:   adjacent-mask strength $BASELINE_STRENGTH"
echo "Candidate:  adjacent-mask strength $CANDIDATE_STRENGTH"

echo
echo "A/B 1/2: current temporal mask"
JASNA_RESTORE_ONLY_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-3}" \
JASNA_MODEL_BATCH="$MODEL_BATCH" \
JASNA_METAL_WINDOWS_PER_PROCESS="$WINDOWS_PER_PROCESS" \
JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS="$MASK_RECOVERY" \
JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS="$MASK_TEMPORAL_RADIUS" \
JASNA_LARGE_REGION_MASK_TEMPORAL_STRENGTH="$BASELINE_STRENGTH" \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$BASELINE_OUTPUT"

echo
echo "A/B 2/2: full temporal union"
JASNA_RESTORE_ONLY_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-3}" \
JASNA_MODEL_BATCH="$MODEL_BATCH" \
JASNA_METAL_WINDOWS_PER_PROCESS="$WINDOWS_PER_PROCESS" \
JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS="$MASK_RECOVERY" \
JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS="$MASK_TEMPORAL_RADIUS" \
JASNA_LARGE_REGION_MASK_TEMPORAL_STRENGTH="$CANDIDATE_STRENGTH" \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$CANDIDATE_OUTPUT"

echo
echo "Temporal-mask recovery-only A/B: PASS"
echo "Baseline:  $BASELINE_OUTPUT"
echo "Candidate: $CANDIDATE_OUTPUT"
