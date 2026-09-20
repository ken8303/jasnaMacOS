#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 || $# -eq 3 ]] || {
  echo "usage: $0 REFERENCE_WORK_DIR NEW_OUTPUT_PREFIX [START_TIME]" >&2
  echo "Compares disk vs bounded-memory crop handoff using the same cached 30 seconds." >&2
  echo "Both runs: 1024px crop grid, model batch 2, four windows/process." >&2
  echo "Memory variant: 512 MiB per eye/window with existing disk fallback." >&2
  echo "JASNA_AB_ORDER=candidate-first reverses the order; JASNA_RESTORE_ONLY_VALIDATE=1 checks assets only." >&2
  exit 2
}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JASNA_AB_COMPARISON=crop-handoff
export JASNA_MODEL_BATCH="${JASNA_MODEL_BATCH:-2}"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-4}"
exec "$ROOT_DIR/script/test_vr_crop_density_ab_30s.sh" "$@"
