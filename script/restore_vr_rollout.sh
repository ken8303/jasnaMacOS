#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [MOSAIC_RANGES]" >&2
  echo "example ranges: 00:12:00-00:14:00,00:20:30-00:22:00" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_MODELS="$ROOT_DIR/Models/MetalML"
BASELINE_BATCH2_MODELS="$ROOT_DIR/Models/MetalMLBatch2"

[[ -d "$BASELINE_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: rollout baseline Metal packages are unavailable: $BASELINE_MODELS" >&2
  exit 1
}
if [[ -n "${JASNA_MODELS_DIR:-}" && "$JASNA_MODELS_DIR" != "$BASELINE_MODELS" ]]; then
  echo "WARNING: ignoring JASNA_MODELS_DIR for baseline rollout: $JASNA_MODELS_DIR" >&2
  echo "Use script/test_vr_finetuned_30s.sh or script/test_vr_restore_ab.sh for candidate weights" >&2
fi

# Validated macOS 27 rollout profile. Expert experiments can
# still call restore_vr_sparse_sbs.sh directly with different settings.
export JASNA_MODELS_DIR="$BASELINE_MODELS"
export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
# Prefer the validated baseline batch-2 set. A checkout without generated
# batch-2 packages remains usable through batch 1.
if [[ "${JASNA_MODEL_BATCH:-auto}" == "auto" ]]; then
  if [[ -d "$BASELINE_BATCH2_MODELS/feature_extract.mtlpackage" ]]; then
    export JASNA_MODEL_BATCH=2
    export JASNA_BATCH2_MODELS_DIR="$BASELINE_BATCH2_MODELS"
  else
    export JASNA_MODEL_BATCH=1
    unset JASNA_BATCH2_MODELS_DIR
  fi
elif [[ "$JASNA_MODEL_BATCH" == "2" ]]; then
  [[ -d "$BASELINE_BATCH2_MODELS/feature_extract.mtlpackage" ]] || {
    echo "error: rollout baseline batch-2 packages are unavailable: $BASELINE_BATCH2_MODELS" >&2
    exit 1
  }
  export JASNA_BATCH2_MODELS_DIR="$BASELINE_BATCH2_MODELS"
elif [[ "$JASNA_MODEL_BATCH" == "1" ]]; then
  unset JASNA_BATCH2_MODELS_DIR
else
  echo "error: JASNA_MODEL_BATCH must be 1, 2, or unset" >&2
  exit 1
fi
export JASNA_DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
export JASNA_DETECT_DEVICE="${JASNA_DETECT_DEVICE:-auto}"
export JASNA_STEREO_MANIFEST_RECONCILE="${JASNA_STEREO_MANIFEST_RECONCILE:-1}"
export JASNA_ADAPTIVE_DETECT="${JASNA_ADAPTIVE_DETECT:-0}"
export JASNA_DIRECT_SBS_OUTPUT="${JASNA_DIRECT_SBS_OUTPUT:-1}"
export JASNA_SHARED_SBS_SOURCE="${JASNA_SHARED_SBS_SOURCE:-1}"
export JASNA_IN_MEMORY_CROP_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-1}"
export JASNA_IN_MEMORY_CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-512}"
export JASNA_FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
export JASNA_LOG_PEAK_MEMORY="${JASNA_LOG_PEAK_MEMORY:-1}"
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS="${JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS:-1}"
export JASNA_TEMPORAL_WARMUP_FRAMES="${JASNA_TEMPORAL_WARMUP_FRAMES:-5}"

echo "Jasna macOS 27 rollout profile"
echo "Model track: baseline (fine-tuned candidates are test-only)"
echo "Models: $JASNA_MODELS_DIR"
echo "Detector: $JASNA_DETECTOR on $JASNA_DETECT_DEVICE; model batch: $JASNA_MODEL_BATCH"
echo "Metal windows/process: $JASNA_METAL_WINDOWS_PER_PROCESS"
echo "Ordinary mask-hole recovery: $JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS"
echo "Temporal crop warm-up: $JASNA_TEMPORAL_WARMUP_FRAMES frame(s)"
echo "Direct/shared SBS, bounded crop handoff, stereo reconciliation: enabled"
echo "Peak-memory telemetry: enabled"
echo "Adaptive detection: disabled"

if [[ $# -eq 3 ]]; then
  exec "$ROOT_DIR/script/restore_vr_sparse_ranges.sh" "$1" "$2" "$3"
fi

exec "$ROOT_DIR/script/restore_vr_sparse_sbs.sh" "$1" "$2"
