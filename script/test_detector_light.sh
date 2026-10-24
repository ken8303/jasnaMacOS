#!/bin/bash
# Detector-only comparison; does not restore video or change app defaults.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 SOURCE_SBS_VIDEO NEW_RESULT_DIRECTORY [START_SECONDS=0] [SPAN_SECONDS=30]" >&2
  exit 2
fi
PYTHON="$ROOT/.venv-rfdetr/bin/python"
"$PYTHON" "$ROOT/tools/prepare_rfdetr_mlx_light.py" "$1" --output "$2" --start "${3:-0}" --span "${4:-30}"
for variant in baseline compiled fp16 fp16-compiled; do
  "$PYTHON" "$ROOT/tools/benchmark_rfdetr_mlx_light.py" --inputs "$2/inputs.npz" --output "$2" --variant "$variant" --repeats 3
done
"$PYTHON" "$ROOT/tools/compare_rfdetr_mlx_light.py" "$2"
