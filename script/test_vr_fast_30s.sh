#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-fast.mov 00:12:00" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fast, quality-preserving 30-second profile. It keeps the validated detector
# and model settings while removing redundant eye encodes and bounded disk I/O.
export JASNA_TEST_SECONDS=30
export JASNA_DIRECT_SBS_OUTPUT=1
export JASNA_SHARED_SBS_SOURCE=1
export JASNA_MODEL_BATCH=2
export JASNA_DETECTOR=rfdetr-vr-v1
export JASNA_DETECT_DEVICE=auto
export JASNA_DETECT_SAMPLE_STRIDE=0.1
export JASNA_ADAPTIVE_DETECT=0
export JASNA_STEREO_MANIFEST_RECONCILE=1
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_METAL_WINDOWS_PER_PROCESS=2
export JASNA_GRAPH_PHASE_TELEMETRY=1
export JASNA_LOG_PEAK_MEMORY=1

echo "Jasna fast 30-second quality profile"
echo "Shared SBS source: enabled; temporary 4K eye encodes: disabled"
echo "RF-DETR 10 Hz; model batch 2; bounded memory handoff: 512 MiB"

exec "$ROOT_DIR/script/test_vr_sparse_30s.sh" "$@"
