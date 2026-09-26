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
RESTORATION_BACKEND="${JASNA_RESTORATION_BACKEND:-metal}"
case "$RESTORATION_BACKEND" in
  metal) ;;
  mlx)
    for required in \
      "$ROOT_DIR/Models/MLX/basicvsrpp-v1.2.safetensors" \
      "$ROOT_DIR/Models/MLXRuntime/mlx" \
      "$ROOT_DIR/tools/restore_mlx_crop.py" \
      "$ROOT_DIR/.venv-rfdetr/bin/python"; do
      [[ -e "$required" ]] || {
        echo "error: MLX restoration component is missing: $required" >&2
        exit 1
      }
    done
    ;;
  *) echo "error: JASNA_RESTORATION_BACKEND must be metal or mlx" >&2; exit 1 ;;
esac
export JASNA_RESTORATION_BACKEND="$RESTORATION_BACKEND"

metal_model_packages_available() {
  local directory="$1"
  local package
  local packages=(
    feature_extract
    spynet_level_0 spynet_level_1 spynet_level_2
    spynet_level_3 spynet_level_4 spynet_level_5
    offset_backward_1 offset_forward_1 offset_backward_2 offset_forward_2
    backbone_backward_1 backbone_forward_1 backbone_backward_2 backbone_forward_2
    upsample
  )
  [[ -d "$directory" ]] || return 1
  for package in "${packages[@]}"; do
    [[ -d "$directory/$package.mtlpackage" ]] || return 1
  done
}

metal_model_packages_available "$BASELINE_MODELS" || {
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
PERFORMANCE_PROFILE="${JASNA_PERFORMANCE_PROFILE:-balanced}"
case "$PERFORMANCE_PROFILE" in
  fast)
    export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-8}"
    export JASNA_REGION_PREPARE_DEPTH="${JASNA_REGION_PREPARE_DEPTH:-2}"
    export JASNA_DETECT_BATCH_SIZE="${JASNA_DETECT_BATCH_SIZE:-2}"
    export JASNA_IN_MEMORY_CROP_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-1}"
    export JASNA_IN_MEMORY_CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-512}"
    ;;
  balanced)
    export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"
    export JASNA_REGION_PREPARE_DEPTH="${JASNA_REGION_PREPARE_DEPTH:-1}"
    export JASNA_DETECT_BATCH_SIZE="${JASNA_DETECT_BATCH_SIZE:-1}"
    export JASNA_IN_MEMORY_CROP_CACHE="${JASNA_IN_MEMORY_CROP_CACHE:-0}"
    export JASNA_IN_MEMORY_CACHE_LIMIT_MB="${JASNA_IN_MEMORY_CACHE_LIMIT_MB:-128}"
    ;;
  *)
    echo "error: JASNA_PERFORMANCE_PROFILE must be fast, balanced, or unset" >&2
    exit 1
    ;;
esac
ROLLOUT_OUTPUT_NAME="$(basename "$2")"
ROLLOUT_OUTPUT_STEM="${ROLLOUT_OUTPUT_NAME%.*}"
ROLLOUT_RESUME_CONFIG="$(dirname "$2")/${ROLLOUT_OUTPUT_STEM}.jasna-vr-full-v22-work/run-config.txt"
RECORDED_MODEL_BATCH=""
if [[ -s "$ROLLOUT_RESUME_CONFIG" ]]; then
  RECORDED_MODEL_BATCH="$(
    /usr/bin/awk -F= '$1 == "model_batch" { print $2; exit }' \
      "$ROLLOUT_RESUME_CONFIG"
  )"
fi
# Batch 2 is the validated M4 default. Auto keeps the launcher usable if its
# matching fixed-batch packages have not been generated.
REQUESTED_MODEL_BATCH="${JASNA_MODEL_BATCH:-auto}"
if [[ "$REQUESTED_MODEL_BATCH" == "auto" \
  && ( "$RECORDED_MODEL_BATCH" == "1" || "$RECORDED_MODEL_BATCH" == "2" ) ]]; then
  REQUESTED_MODEL_BATCH="$RECORDED_MODEL_BATCH"
  echo "Resume profile: retaining recorded model batch $RECORDED_MODEL_BATCH"
fi
if [[ "$REQUESTED_MODEL_BATCH" == "auto" ]]; then
  if metal_model_packages_available "$BASELINE_BATCH2_MODELS"; then
    export JASNA_MODEL_BATCH=2
    export JASNA_BATCH2_MODELS_DIR="$BASELINE_BATCH2_MODELS"
  else
    export JASNA_MODEL_BATCH=1
    unset JASNA_BATCH2_MODELS_DIR
  fi
elif [[ "$REQUESTED_MODEL_BATCH" == "2" ]]; then
  metal_model_packages_available "$BASELINE_BATCH2_MODELS" || {
    echo "error: rollout baseline batch-2 packages are unavailable: $BASELINE_BATCH2_MODELS" >&2
    exit 1
  }
  export JASNA_MODEL_BATCH=2
  export JASNA_BATCH2_MODELS_DIR="$BASELINE_BATCH2_MODELS"
elif [[ "$REQUESTED_MODEL_BATCH" == "1" ]]; then
  export JASNA_MODEL_BATCH=1
  unset JASNA_BATCH2_MODELS_DIR
else
  echo "error: JASNA_MODEL_BATCH must be 1, 2, auto, or unset" >&2
  exit 1
fi
if [[ "$RESTORATION_BACKEND" == "mlx" ]]; then
  export JASNA_MODEL_BATCH=1
  unset JASNA_BATCH2_MODELS_DIR
fi
export JASNA_DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
if [[ "$RESTORATION_BACKEND" == "mlx" ]]; then
  export JASNA_DETECT_DEVICE="${JASNA_DETECT_DEVICE:-mlx}"
else
  export JASNA_DETECT_DEVICE="${JASNA_DETECT_DEVICE:-auto}"
fi
export JASNA_DETECT_DECODE_MODE="${JASNA_DETECT_DECODE_MODE:-sequential}"
if [[ "$JASNA_DETECT_DEVICE" == "mlx" ]]; then
  [[ "$JASNA_DETECTOR" == "rfdetr-vr-v1" ]] || {
    echo "error: MLX detection supports rfdetr-vr-v1 only" >&2
    exit 1
  }
  for required in \
    "$ROOT_DIR/Models/MLXDetector/rfdetr-vr-v1.safetensors" \
    "$ROOT_DIR/Models/MLXRuntime/mlx"; do
    [[ -e "$required" ]] || {
      echo "error: MLX detector component is missing: $required" >&2
      exit 1
    }
  done
fi
if [[ -z "${JASNA_WORK_CONTAINER+x}" ]]; then
  RECORDED_WORK_CONTAINER=""
  if [[ -s "$ROLLOUT_RESUME_CONFIG" ]]; then
    RECORDED_WORK_CONTAINER="$(
      /usr/bin/awk -F= '$1 == "work_container" { print $2; exit }' \
        "$ROLLOUT_RESUME_CONFIG"
    )"
  fi
  case "$RECORDED_WORK_CONTAINER" in
    mov|mp4) export JASNA_WORK_CONTAINER="$RECORDED_WORK_CONTAINER" ;;
    *) export JASNA_WORK_CONTAINER=mp4 ;;
  esac
else
  export JASNA_WORK_CONTAINER
fi
export JASNA_STEREO_MANIFEST_RECONCILE="${JASNA_STEREO_MANIFEST_RECONCILE:-1}"
export JASNA_ADAPTIVE_DETECT="${JASNA_ADAPTIVE_DETECT:-0}"
export JASNA_DIRECT_SBS_OUTPUT="${JASNA_DIRECT_SBS_OUTPUT:-1}"
export JASNA_SHARED_SBS_SOURCE="${JASNA_SHARED_SBS_SOURCE:-1}"
export JASNA_COMPOSITE_CONCURRENCY="${JASNA_COMPOSITE_CONCURRENCY:-1}"
export JASNA_RUNTIME_SCRATCH_ON_OUTPUT="${JASNA_RUNTIME_SCRATCH_ON_OUTPUT:-1}"
# Reusing cached crops across restoration-code changes can mix two algorithms in
# one output. Require an explicit opt-in for known-compatible updates.
export JASNA_ALLOW_IMPLEMENTATION_RESUME="${JASNA_ALLOW_IMPLEMENTATION_RESUME:-0}"
export JASNA_FAST_ENCODE="${JASNA_FAST_ENCODE:-1}"
export JASNA_LOG_PEAK_MEMORY="${JASNA_LOG_PEAK_MEMORY:-1}"
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS="${JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS:-1}"
export JASNA_TEMPORAL_WARMUP_FRAMES="${JASNA_TEMPORAL_WARMUP_FRAMES:-5}"

echo "Jasna macOS 27 rollout profile"
echo "Performance profile: $PERFORMANCE_PROFILE"
echo "Model track: baseline (fine-tuned candidates are test-only)"
echo "Restoration backend: $RESTORATION_BACKEND"
echo "Models: $JASNA_MODELS_DIR"
echo "Detector: $JASNA_DETECTOR on $JASNA_DETECT_DEVICE; model batch: $JASNA_MODEL_BATCH"
echo "Detector memory: batch $JASNA_DETECT_BATCH_SIZE, $JASNA_DETECT_DECODE_MODE decode"
echo "Container profile: $JASNA_WORK_CONTAINER working source; user-selected final extension"
echo "Metal windows/process: $JASNA_METAL_WINDOWS_PER_PROCESS"
echo "Crop preparation pipeline depth: $JASNA_REGION_PREPARE_DEPTH"
echo "Memory profile: bounded (batch $JASNA_MODEL_BATCH, crop handoff $([[ "$JASNA_IN_MEMORY_CROP_CACHE" == "1" ]] && echo memory-up-to-${JASNA_IN_MEMORY_CACHE_LIMIT_MB}MiB || echo output-disk))"
echo "Frame compositing concurrency: $JASNA_COMPOSITE_CONCURRENCY"
echo "Runtime scratch beside output: $JASNA_RUNTIME_SCRATCH_ON_OUTPUT"
echo "Restart data after validated success: preserved"
echo "Accelerators: RF-DETR=$JASNA_DETECT_DEVICE; restoration=$RESTORATION_BACKEND; composite=Metal; encode=VideoToolbox"
echo "CPU boundary: 8192x4096/4096x4096 HEVC decode and FFmpeg spatial/time filters"
echo "Ordinary mask-hole recovery: $JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS"
echo "Temporal crop warm-up: $JASNA_TEMPORAL_WARMUP_FRAMES frame(s)"
echo "Direct/shared SBS, bounded crop handoff, stereo reconciliation: enabled"
echo "Peak-memory telemetry: enabled"
if [[ "$JASNA_ADAPTIVE_DETECT" == "1" ]]; then
  echo "Adaptive detection: enabled (coarse gate plus dense flagged-range refinement)"
else
  echo "Adaptive detection: disabled (validated paired 10 Hz quality path)"
fi

if [[ $# -eq 3 ]]; then
  exec "$ROOT_DIR/script/restore_vr_sparse_ranges.sh" "$1" "$2" "$3"
fi

exec "$ROOT_DIR/script/restore_vr_sparse_sbs.sh" "$1" "$2"
