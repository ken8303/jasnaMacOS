#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: ./script/test_vr_region_prepare_ab.sh REFERENCE_WORK_DIR OUTPUT_PREFIX

Runs restoration only twice against the same prepared video and manifests.
Depth 1 is the synchronous control; depth 2 extracts the following crop batch
on CPU while the current batch runs on Metal. Detection and source conversion
are not repeated.

Optional environment variables:
  JASNA_RESTORE_ONLY_START_SECOND=0
  JASNA_RESTORE_ONLY_SECONDS=8
  JASNA_MODEL_BATCH=2
  JASNA_METAL_WINDOWS_PER_PROCESS=8
  JASNA_AB_ORDER=control-first (or candidate-first)
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="${1%/}"
OUTPUT_PREFIX="${2%.mov}"
SELECTED_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-8}"
ORDER="${JASNA_AB_ORDER:-control-first}"
CONTROL_OUTPUT="${OUTPUT_PREFIX}-prepare1.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-prepare2.mov"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
[[ "$SELECTED_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_RESTORE_ONLY_SECONDS must be a positive integer" >&2
  exit 1
}
case "$ORDER" in
  control-first|candidate-first) ;;
  *)
    echo "error: JASNA_AB_ORDER must be control-first or candidate-first" >&2
    exit 1
    ;;
esac
for output in "$CONTROL_OUTPUT" "$CANDIDATE_OUTPUT"; do
  [[ ! -e "$output" ]] || {
    echo "error: output already exists: $output" >&2
    echo "use a fresh output prefix for an independent measurement" >&2
    exit 1
  }
done

export JASNA_RESTORE_ONLY_SECONDS="$SELECTED_SECONDS"
export JASNA_MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-8}"
export JASNA_GPU_TIMEOUT_RETRIES=0
export JASNA_GRAPH_TRACE="${JASNA_GRAPH_TRACE:-0}"

run_variant() {
  local depth="$1"
  local output="$2"
  echo
  echo "Crop-preparation depth $depth: $output"
  JASNA_REGION_PREPARE_DEPTH="$depth" \
    "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
      "$REFERENCE_WORK_DIR" "$output"
}

echo "Jasna bounded crop-preparation A/B"
echo "Reference:       $REFERENCE_WORK_DIR"
echo "Selected range:  second ${JASNA_RESTORE_ONLY_START_SECOND:-0}, ${SELECTED_SECONDS}s"
echo "Model batch:     $JASNA_MODEL_BATCH"
echo "Windows/process: $JASNA_METAL_WINDOWS_PER_PROCESS"
echo "Control:         preparation depth 1"
echo "Candidate:       preparation depth 2"
echo "Run order:       $ORDER"

if [[ "$ORDER" == "candidate-first" ]]; then
  run_variant 2 "$CANDIDATE_OUTPUT"
  run_variant 1 "$CONTROL_OUTPUT"
else
  run_variant 1 "$CONTROL_OUTPUT"
  run_variant 2 "$CANDIDATE_OUTPUT"
fi

echo
echo "Crop-preparation A/B: COMPLETE"
echo "Compare Wall time and sparse hot-path foreground wait in:"
echo "  ${CONTROL_OUTPUT%.mov}.jasna-restore-only.log"
echo "  ${CANDIDATE_OUTPUT%.mov}.jasna-restore-only.log"
