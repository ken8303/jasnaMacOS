#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  echo "Loads eight propagation packages twice; no video or model inference." >&2
  echo "Default packages: Models/MetalMLBatch2; override with JASNA_MODELS_DIR." >&2
  exit 2
}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIRECTORY="${1%/}"
[[ -n "$OUTPUT_DIRECTORY" && "$OUTPUT_DIRECTORY" != *[[:space:]] ]] || {
  echo "error: output directory is empty or ends with whitespace" >&2
  exit 1
}
export JASNA_MODELS_DIR="${JASNA_MODELS_DIR:-$ROOT_DIR/Models/MetalMLBatch2}"
[[ -d "$JASNA_MODELS_DIR" ]] || {
  echo "error: model directory is unavailable: $JASNA_MODELS_DIR" >&2
  exit 1
}
mkdir -p "$(dirname "$OUTPUT_DIRECTORY")"
mkdir "$OUTPUT_DIRECTORY" 2>/dev/null || {
  echo "error: use a new diagnostic output directory: $OUTPUT_DIRECTORY" >&2
  exit 1
}
OUTPUT_DIRECTORY="$(cd "$OUTPUT_DIRECTORY" && pwd)"
mkdir -p "$OUTPUT_DIRECTORY/runtime/tmp" "$OUTPUT_DIRECTORY/runtime/cache"
export TMPDIR="$OUTPUT_DIRECTORY/runtime/tmp/"
export XDG_CACHE_HOME="$OUTPUT_DIRECTORY/runtime/cache"
export JASNA_WORK_DIR="$OUTPUT_DIRECTORY/load-only.jasna-work"
export JASNA_LOG_PEAK_MEMORY=1
echo "Load-only diagnostic: $OUTPUT_DIRECTORY/load-only.jasna.log"
echo "First load plus in-process cache reuse; Apple persistent caches are not cleared."
exec "$ROOT_DIR/script/build_and_run.sh" --metal-ml-load-only "$OUTPUT_DIRECTORY"
