#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  echo "Profiles BasicVSR++ package families with generated inputs only." >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JASNA_GRAPH_COMPONENT_TELEMETRY=1
export JASNA_GRAPH_PROPAGATION_TELEMETRY=1
export JASNA_GRAPH_LIFECYCLE_REPEATS="${JASNA_GRAPH_LIFECYCLE_REPEATS:-8}"

echo "BasicVSR++ Metal ML package profiler"
echo "Measures each recurrence branch: offset, preparation, DCNv2, backbone, residual."
echo "Generated inputs only; video decode, detector, compositor and encoder are disabled."
echo "Production scheduling and restoration defaults are unchanged."

exec "$ROOT_DIR/script/test_metal_ml_graph_lifecycle.sh" "$1"
