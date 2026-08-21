#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [MOSAIC_RANGES]" >&2
  echo "example ranges: 00:12:00-00:14:00,00:20:30-00:22:00" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Validated macOS 27 / Xcode 27 beta 5 rollout profile. Expert experiments can
# still call restore_vr_sparse_sbs.sh directly with different settings.
export JASNA_METAL_WINDOWS_PER_PROCESS=2
export JASNA_MODEL_BATCH=2
export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_DEVICE=auto
export JASNA_STEREO_MANIFEST_RECONCILE=1
export JASNA_ADAPTIVE_DETECT=0
export JASNA_DIRECT_SBS_OUTPUT=1
export JASNA_SHARED_SBS_SOURCE=1
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_FAST_ENCODE=1
export JASNA_LOG_PEAK_MEMORY=1

echo "Jasna macOS 27 rollout profile"
echo "Detector: RF-DETR on automatic MPS/CPU; model batch: 2; Metal windows/process: 2"
echo "Direct/shared SBS, bounded crop handoff, stereo reconciliation: enabled"
echo "Peak-memory telemetry: enabled"
echo "Adaptive detection: disabled"

if [[ $# -eq 3 ]]; then
  exec "$ROOT_DIR/script/restore_vr_sparse_ranges.sh" "$1" "$2" "$3"
fi

exec "$ROOT_DIR/script/restore_vr_sparse_sbs.sh" "$1" "$2"
