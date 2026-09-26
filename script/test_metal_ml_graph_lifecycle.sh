#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  echo "Runs generated full/partial/warm-up graph lifecycles; no video or detector." >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIRECTORY="${1%/}"
[[ -n "$OUTPUT_DIRECTORY" && "$OUTPUT_DIRECTORY" != *[[:space:]] ]] || {
  echo "error: output directory is empty or ends with whitespace" >&2
  exit 1
}
mkdir -p "$(dirname "$OUTPUT_DIRECTORY")"
mkdir "$OUTPUT_DIRECTORY" 2>/dev/null || {
  echo "error: use a new diagnostic output directory: $OUTPUT_DIRECTORY" >&2
  exit 1
}
OUTPUT_DIRECTORY="$(cd "$OUTPUT_DIRECTORY" && pwd -P)"

export JASNA_MODELS_DIR="${JASNA_MODELS_DIR:-$ROOT_DIR/Models/MetalMLBatch2}"
export JASNA_RETAINED_GRAPH=1
export JASNA_GRAPH_TRACE=1
export JASNA_GRAPH_PHASE_TELEMETRY=1
export JASNA_LOG_PEAK_MEMORY=1
export JASNA_GRAPH_LIFECYCLE_REPEATS="${JASNA_GRAPH_LIFECYCLE_REPEATS:-8}"
export JASNA_WORK_DIR="$OUTPUT_DIRECTORY/graph-lifecycle.jasna-work"

echo "Metal ML graph-lifecycle diagnostic"
echo "Generated input only; no video decode, detection, compositor, or encoder."
echo "Plan: 30-frame, partial 17-frame, then 35-frame build/reuse, batch 2."
echo "Retained 35-frame measurements: $JASNA_GRAPH_LIFECYCLE_REPEATS"
echo "This may pause for several minutes while macOS specializes the first graph."
exec "$ROOT_DIR/script/build_and_run.sh" --metal-ml-graph-lifecycle "$OUTPUT_DIRECTORY"
