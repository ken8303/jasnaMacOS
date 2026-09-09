#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_MOV" >&2
  echo "example: $0 previous-test.jasna-vr30-v22-work windows8-test.mov" >&2
  echo "" >&2
  echo "Runs restoration only for 30 seconds using the validated fast settings," >&2
  echo "but increases Metal process isolation from four to eight windows/process." >&2
  echo "Prepared video and detector manifests are reused from REFERENCE_WORK_DIR." >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

REFERENCE_WORK_DIR="$1"
OUTPUT_PATH="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
[[ "$OUTPUT_PATH" == *.mov ]] || {
  echo "error: output must use a new .mov filename" >&2
  exit 1
}
[[ ! -e "$OUTPUT_PATH" ]] || {
  echo "error: output already exists: $OUTPUT_PATH" >&2
  echo "use a new filename so this measurement remains independent" >&2
  exit 1
}

echo "Jasna restoration-only eight-window test"
echo "Prepared source/detection: reused"
echo "Selected duration:          30 seconds"
echo "Metal windows/process:      8"
echo "Model batch:                2"
echo "Crop handoff:               memory, bounded to 512 MiB"
echo "Writer pipeline depth:      2"
echo "Reference:                  $REFERENCE_WORK_DIR"
echo "Output:                     $OUTPUT_PATH"

exec env \
  JASNA_METAL_WINDOWS_PER_PROCESS=8 \
  JASNA_REGION_PREPARE_DEPTH=2 \
  JASNA_MODEL_BATCH=2 \
  JASNA_IN_MEMORY_CROP_CACHE=1 \
  JASNA_IN_MEMORY_CACHE_LIMIT_MB=512 \
  JASNA_STEREO_WRITER_DEPTH=2 \
  JASNA_RESTORE_ONLY_SECONDS=30 \
  "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
  "$REFERENCE_WORK_DIR" \
  "$OUTPUT_PATH"
