#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  echo "Profiles learned DCNv2 offset locality with generated inputs." >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JASNA_DCN_OFFSET_LOCALITY=1
export JASNA_GRAPH_LIFECYCLE_REPEATS="${JASNA_GRAPH_LIFECYCLE_REPEATS:-4}"

echo "DCNv2 learned-offset locality diagnostic"
echo "Generated tensors only; no video, detector, compositor, or encoder."
echo "Reports displacement, out-of-bounds sampling, and spatial variation by branch."

exec "$ROOT_DIR/script/test_basicvsr_package_profile.sh" "$1"
