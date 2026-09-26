#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_PREFIX" >&2
  echo "example: $0 previous.jasna-vr30-v22-work /path/to/metal-windows-ab" >&2
  echo "Runs restoration only with identical prepared source and detector manifests." >&2
  echo "Defaults to four versus eight Metal windows/process." >&2
  echo "optional: JASNA_WINDOWS_AB_CONTROL=4 JASNA_WINDOWS_AB_CANDIDATE=8" >&2
  echo "          JASNA_AB_ORDER=candidate-first|control-first" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_WORK_DIR="$1"
OUTPUT_PREFIX="${2%.mov}"
BATCH2_MODELS="$ROOT_DIR/Models/MetalMLBatch2"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
[[ -d "$BATCH2_MODELS/feature_extract.mtlpackage" ]] || {
  echo "error: batch-2 Metal packages are unavailable: $BATCH2_MODELS" >&2
  exit 1
}
[[ "$OUTPUT_PREFIX" != *[[:space:]] ]] || {
  echo "error: output prefix ends with whitespace: '$OUTPUT_PREFIX'" >&2
  exit 1
}

export JASNA_RESTORE_ONLY_START_SECOND="${JASNA_RESTORE_ONLY_START_SECOND:-0}"
export JASNA_RESTORE_ONLY_SECONDS="${JASNA_RESTORE_ONLY_SECONDS:-30}"
export JASNA_MODEL_BATCH=2
export JASNA_BATCH2_MODELS_DIR="$BATCH2_MODELS"
export JASNA_IN_MEMORY_CROP_CACHE=1
export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512
export JASNA_TEMPORAL_WARMUP_FRAMES=5
export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1
export JASNA_GPU_TIMEOUT_RETRIES=0
export JASNA_GRAPH_PHASE_TELEMETRY=1
export JASNA_LOG_PEAK_MEMORY=1

CONTROL_WINDOWS="${JASNA_WINDOWS_AB_CONTROL:-4}"
CANDIDATE_WINDOWS="${JASNA_WINDOWS_AB_CANDIDATE:-8}"
AB_ORDER="${JASNA_AB_ORDER:-candidate-first}"

[[ "$CONTROL_WINDOWS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_WINDOWS_AB_CONTROL must be a positive integer" >&2
  exit 1
}
[[ "$CANDIDATE_WINDOWS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_WINDOWS_AB_CANDIDATE must be a positive integer" >&2
  exit 1
}
[[ "$AB_ORDER" == "candidate-first" || "$AB_ORDER" == "control-first" ]] || {
  echo "error: JASNA_AB_ORDER must be candidate-first or control-first" >&2
  exit 1
}
[[ "$CONTROL_WINDOWS" != "$CANDIDATE_WINDOWS" ]] || {
  echo "error: control and candidate window counts must differ" >&2
  exit 1
}

CONTROL_OUTPUT="${OUTPUT_PREFIX}-windows${CONTROL_WINDOWS}.mov"
CANDIDATE_OUTPUT="${OUTPUT_PREFIX}-windows${CANDIDATE_WINDOWS}.mov"

if [[ -z "${JASNA_APP_BINARY:-}" ]]; then
  echo "Building one shared optimized Swift executable before timing either candidate"
  mkdir -p "$ROOT_DIR/.build/ModuleCache"
  export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
  export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
  (
    cd "$ROOT_DIR"
    swift build --disable-sandbox -c release
  )
  JASNA_APP_BINARY="$(
    cd "$ROOT_DIR"
    swift build --disable-sandbox -c release --show-bin-path
  )/JasnaMetalPoC"
  export JASNA_APP_BINARY
fi
[[ -x "$JASNA_APP_BINARY" ]] || {
  echo "error: optimized JasnaMetalPoC executable was not produced" >&2
  exit 1
}

echo "Jasna ${JASNA_RESTORE_ONLY_SECONDS}-second Metal process-window A/B"
echo "Prepared source and detector manifests: reused"
echo "Model batch: 2; temporal warm-up: 5; in-memory handoff: 512 MiB"
echo "GPU timeout retry: disabled so a watchdog reset cannot distort timing"
echo "Control/candidate: ${CONTROL_WINDOWS}/${CANDIDATE_WINDOWS} windows/process"
echo "Order: $AB_ORDER"

run_candidate() {
  local label="$1"
  local windows="$2"
  local output="$3"
  echo
  echo "$label: $windows Metal windows/process"
  JASNA_METAL_WINDOWS_PER_PROCESS="$windows" \
    "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
      "$REFERENCE_WORK_DIR" "$output"
}

if [[ "$AB_ORDER" == "candidate-first" ]]; then
  run_candidate "A/B 1/2 candidate" "$CANDIDATE_WINDOWS" "$CANDIDATE_OUTPUT"
  run_candidate "A/B 2/2 control" "$CONTROL_WINDOWS" "$CONTROL_OUTPUT"
else
  run_candidate "A/B 1/2 control" "$CONTROL_WINDOWS" "$CONTROL_OUTPUT"
  run_candidate "A/B 2/2 candidate" "$CANDIDATE_WINDOWS" "$CANDIDATE_OUTPUT"
fi

summarize_log() {
  local label="$1"
  local windows="$2"
  local output="$3"
  local log_path="${output%.mov}.jasna-restore-only.log"
  /usr/bin/python3 - "$label" "$windows" "$log_path" <<'PY'
import re
import sys
from pathlib import Path

label, windows, log_name = sys.argv[1:]
text = Path(log_name).read_text(errors="replace")
sessions = text.split("===== Jasna restoration-only session ")
session = sessions[-1] if len(sessions) > 1 else text

def values(pattern, flags=0):
    return [float(value) for value in re.findall(pattern, session, flags)]

wall = values(r"^Wall time: ([0-9.]+)s$", re.M)
hot_paths = re.findall(
    r"sparse hot-path phases: ([0-9]+) model frames,.*?"
    r"graph wall ([0-9.]+) ms, GPU ([0-9.]+) ms",
    session,
)
memory = values(r"Runtime memory: peak resident ([0-9.]+) GiB")
safety = len(re.findall(
    r"discarded safety execution completed:.*?reason first-family correctness",
    session,
))
failures = len(re.findall(r"GPU Timeout Error|non-finite|restoration failed", session, re.I))

model_frames = sum(float(item[0]) for item in hot_paths)
graph_milliseconds = sum(float(item[1]) for item in hot_paths)
gpu_milliseconds = sum(float(item[2]) for item in hot_paths)

wall_text = f"{wall[-1]:.0f}s" if wall else "unavailable"
memory_text = f"{max(memory):.2f} GiB" if memory else "unavailable"
print(
    f"{label} ({windows}): wall {wall_text}; model frames {model_frames:.0f}; "
    f"graph {graph_milliseconds / 1000:.3f}s; GPU {gpu_milliseconds / 1000:.3f}s; "
    f"first-family executions {safety}; peak {memory_text}; failures {failures}"
)
PY
}

echo
echo "Metal process-window A/B: PASS"
summarize_log "Control" "$CONTROL_WINDOWS" "$CONTROL_OUTPUT"
summarize_log "Candidate" "$CANDIDATE_WINDOWS" "$CANDIDATE_OUTPUT"
echo "Control:   $CONTROL_OUTPUT"
echo "Candidate: $CANDIDATE_OUTPUT"
echo "Promote the candidate only if it is faster, stays below the memory limit,"
echo "and reports no GPU timeout or non-finite fallback."
