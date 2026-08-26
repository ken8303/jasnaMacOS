#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_PREFIX" >&2
  echo "example: $0 previous.jasna-vr30-v22-work /path/to/recovery-ab" >&2
  echo "" >&2
  echo "Runs baseline and candidate restoration against the same prepared source" >&2
  echo "and detector manifests. Outputs are OUTPUT_PREFIX-baseline.mov and" >&2
  echo "OUTPUT_PREFIX-candidate.mov." >&2
  echo "" >&2
  echo "optional: JASNA_RESTORE_ONLY_START_SECOND=0" >&2
  echo "          JASNA_RESTORE_ONLY_SECONDS=3" >&2
  echo "          JASNA_AB_BASELINE_MODELS_DIR=Models/MetalML" >&2
  echo "          JASNA_AB_CANDIDATE_MODELS_DIR=Models/MetalMLFineTuneMac1000" >&2
  echo "          JASNA_APP_BINARY=/path/to/JasnaMetalPoC" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

REFERENCE_WORK_DIR="$1"
OUTPUT_PREFIX="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_MODELS="${JASNA_AB_BASELINE_MODELS_DIR:-$ROOT_DIR/Models/MetalML}"
CANDIDATE_MODELS="${JASNA_AB_CANDIDATE_MODELS_DIR:-$ROOT_DIR/Models/MetalMLFineTuneMac1000}"
MODEL_BATCH="${JASNA_MODEL_BATCH:-1}"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
for model_dir in "$BASELINE_MODELS" "$CANDIDATE_MODELS"; do
  [[ -d "$model_dir" ]] || {
    echo "error: model directory not found: $model_dir" >&2
    exit 1
  }
done
[[ "$BASELINE_MODELS" != "$CANDIDATE_MODELS" ]] || {
  echo "error: baseline and candidate model directories must be different" >&2
  exit 1
}
[[ "$OUTPUT_PREFIX" != *.mov ]] || OUTPUT_PREFIX="${OUTPUT_PREFIX%.mov}"
[[ "$OUTPUT_PREFIX" != *[[:space:]] ]] || {
  echo "error: output prefix ends with whitespace: '$OUTPUT_PREFIX'" >&2
  exit 1
}

BASELINE_OUTPUT="${OUTPUT_PREFIX}-baseline.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-candidate.mov"

if [[ "${JASNA_RESTORE_ONLY_VALIDATE:-0}" != "1" ]]; then
  if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
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
  fi
  [[ -x "$JASNA_APP_BINARY" ]] || {
    echo "error: JasnaMetalPoC executable is not available: $JASNA_APP_BINARY" >&2
    exit 1
  }
  export JASNA_APP_BINARY
fi

echo
echo "Recovery-only A/B"
echo "Reference: $REFERENCE_WORK_DIR"
echo "Baseline:  $BASELINE_MODELS"
echo "Candidate: $CANDIDATE_MODELS"
echo "Selected:  relative second ${JASNA_RESTORE_ONLY_START_SECOND:-0}, duration ${JASNA_RESTORE_ONLY_SECONDS:-remaining}"

echo
echo "A/B 1/2: baseline"
JASNA_MODELS_DIR="$BASELINE_MODELS" \
JASNA_MODEL_BATCH="$MODEL_BATCH" \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$BASELINE_OUTPUT"

echo
echo "A/B 2/2: candidate"
JASNA_MODELS_DIR="$CANDIDATE_MODELS" \
JASNA_MODEL_BATCH="$MODEL_BATCH" \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
    "$REFERENCE_WORK_DIR" "$CANDIDATE_OUTPUT"

echo
echo "Recovery-only A/B: PASS"
echo "Baseline:  $BASELINE_OUTPUT"
echo "Candidate: $CANDIDATE_OUTPUT"
