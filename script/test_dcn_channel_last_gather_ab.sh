#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  echo "A/B tests production DCNv2 gather against channel-last staging." >&2
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

export JASNA_GRAPH_LIFECYCLE_REPEATS="${JASNA_GRAPH_LIFECYCLE_REPEATS:-4}"

echo "DCNv2 channel-last gather A/B"
echo "Generated input only; normal restoration remains unchanged."
echo "Candidate timing includes NCHW-to-NHWC staging on every alignment."

echo "Running production buffer control..."
JASNA_DCN_CHANNEL_LAST_GATHER=0 \
  "$ROOT_DIR/script/test_basicvsr_package_profile.sh" "$OUTPUT_DIRECTORY/control"

echo "Running channel-last candidate..."
JASNA_DCN_CHANNEL_LAST_GATHER=1 \
  "$ROOT_DIR/script/test_basicvsr_package_profile.sh" "$OUTPUT_DIRECTORY/channel-last"

echo "A/B complete: $OUTPUT_DIRECTORY"
