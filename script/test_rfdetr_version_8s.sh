#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: ./script/test_rfdetr_version_8s.sh RESTORE_ONLY_WORK_DIR [OUTPUT_DIR]

Runs a detection-only RF-DETR ABBA test on the first 8 seconds of the original
SBS source referenced by a Jasna restore-only cache. Restoration, compositing,
and video encoding are disabled.

Optional environment variables:
  JASNA_RFDETR_TEST_SECONDS=8       detection range in seconds
  JASNA_RFDETR_CANDIDATE=1.10.0     candidate RF-DETR version
  JASNA_RFDETR_AB_ROUNDS=2          alternating rounds; 2 produces ABBA
  JASNA_ALLOW_CPU_DETECTOR_TEST=0   set to 1 only for a CPU comparison
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1%/}"
TEST_SECONDS="${JASNA_RFDETR_TEST_SECONDS:-8}"
CANDIDATE_VERSION="${JASNA_RFDETR_CANDIDATE:-1.10.0}"
AB_ROUNDS="${JASNA_RFDETR_AB_ROUNDS:-2}"
ALLOW_CPU="${JASNA_ALLOW_CPU_DETECTOR_TEST:-0}"
RUN_ID="$(date -u '+%Y%m%d-%H%M%S')"
OUTPUT_DIR="${2:-${WORK_DIR}.rfdetr-${TEST_SECONDS}s-ab-${RUN_ID}}"
PYTHON_PATH="$ROOT_DIR/.venv-rfdetr/bin/python"
MODEL_PATH="$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt"
SCANNER_PATH="$ROOT_DIR/tools/scan_mosaic_regions.py"

[[ -d "$WORK_DIR" ]] || {
  echo "error: restore-only work directory not found: $WORK_DIR" >&2
  exit 1
}
[[ -x "$PYTHON_PATH" && -s "$MODEL_PATH" && -f "$SCANNER_PATH" ]] || {
  echo "error: the RF-DETR detector environment, model, or scanner is missing" >&2
  exit 1
}
/usr/bin/awk -v value="$TEST_SECONDS" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' || {
  echo "error: JASNA_RFDETR_TEST_SECONDS must be greater than zero" >&2
  exit 1
}
[[ "$ALLOW_CPU" == "0" || "$ALLOW_CPU" == "1" ]] || {
  echo "error: JASNA_ALLOW_CPU_DETECTOR_TEST must be 0 or 1" >&2
  exit 1
}
[[ "$AB_ROUNDS" =~ ^[1-5]$ ]] || {
  echo "error: JASNA_RFDETR_AB_ROUNDS must be an integer from 1 through 5" >&2
  exit 1
}

shopt -s nullglob
SEGMENT_LOGS=("$WORK_DIR"/segments/*.jasna.log)
shopt -u nullglob
[[ ${#SEGMENT_LOGS[@]} -gt 0 ]] || {
  echo "error: no restore-only segment logs were found in $WORK_DIR/segments" >&2
  exit 1
}

# Restore-only output MOVs have already passed through the model and are not a
# valid detector fixture. Resolve the original, unrecovered SBS source recorded
# on the Direct SBS batch line instead.
SOURCE_VIDEO="$(/usr/bin/awk '
  /^Direct SBS batch job [0-9]+\/[0-9]+: / {
    line = $0
    sub(/^.*: /, "", line)
    sub(/ \+ .*$/, "", line)
    print line
    exit
  }
' "${SEGMENT_LOGS[@]}")"
[[ -n "$SOURCE_VIDEO" && -f "$SOURCE_VIDEO" ]] || {
  echo "error: could not resolve the original SBS source from the restore logs" >&2
  exit 1
}

FFPROBE_PATH="$(command -v ffprobe || true)"
[[ -n "$FFPROBE_PATH" ]] || {
  echo "error: ffprobe is required to validate the source video" >&2
  exit 1
}
VIDEO_INFO="$($FFPROBE_PATH -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$SOURCE_VIDEO")"
IFS=',' read -r SOURCE_WIDTH SOURCE_HEIGHT SOURCE_RATE <<<"$VIDEO_INFO"
[[ "$SOURCE_WIDTH" =~ ^[0-9]+$ && "$SOURCE_HEIGHT" =~ ^[0-9]+$ \
  && "$SOURCE_RATE" == "30/1" && $((SOURCE_WIDTH % 2)) -eq 0 ]] || {
  echo "error: expected an even-width 30 fps SBS source, got $VIDEO_INFO" >&2
  exit 1
}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "error: test output already exists: $OUTPUT_DIR" >&2
  echo "use a new output directory to keep the measurements independent" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR/runs"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
RUNTIME_DIR="$OUTPUT_DIR/runtime"
CANDIDATE_DIR="$OUTPUT_DIR/rfdetr-$CANDIDATE_VERSION"
mkdir -p "$RUNTIME_DIR/tmp" "$RUNTIME_DIR/cache" "$RUNTIME_DIR/pip-cache"
export TMPDIR="$RUNTIME_DIR/tmp/"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export PIP_CACHE_DIR="$RUNTIME_DIR/pip-cache"

MASTER_LOG="$OUTPUT_DIR/detector-ab.log"
exec > >(/usr/bin/tee "$MASTER_LOG") 2>&1

echo "===== Jasna RF-DETR detection-only A/B $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Restore cache: $WORK_DIR"
echo "Original SBS:  $SOURCE_VIDEO"
echo "Source format: $SOURCE_WIDTH x $SOURCE_HEIGHT at $SOURCE_RATE"
echo "Active range:  0-${TEST_SECONDS}s"
echo "Detector:      rfdetr-vr-v1, paired eyes, 10 Hz, batch 2"
echo "Run order:     alternating across $AB_ROUNDS round(s); default is ABBA"
echo "Output:        $OUTPUT_DIR"
echo "Disabled:      restoration, compositing, and video encoding"

MPS_AVAILABLE="$($PYTHON_PATH -c \
  'import torch; print("1" if torch.backends.mps.is_available() else "0")')"
if [[ "$MPS_AVAILABLE" == "1" ]]; then
  DETECT_DEVICE=mps
  echo "Device:        MPS GPU"
elif [[ "$ALLOW_CPU" == "1" ]]; then
  DETECT_DEVICE=cpu
  echo "WARNING: MPS is unavailable; running the explicitly allowed CPU comparison"
else
  echo "error: PyTorch MPS is unavailable in this terminal" >&2
  echo "rerun from a normal macOS Terminal, or set JASNA_ALLOW_CPU_DETECTOR_TEST=1" >&2
  exit 1
fi

CURRENT_VERSION="$($PYTHON_PATH -c \
  'import importlib.metadata; print(importlib.metadata.version("rfdetr"))')"
echo "Current RF-DETR:   $CURRENT_VERSION"
echo "Candidate RF-DETR: $CANDIDATE_VERSION (isolated; production environment unchanged)"

"$PYTHON_PATH" -m pip install --disable-pip-version-check --no-deps \
  --target "$CANDIDATE_DIR" "rfdetr==$CANDIDATE_VERSION"
PYTHONPATH="$CANDIDATE_DIR" "$PYTHON_PATH" -c \
  'import importlib.metadata, rfdetr; print("Candidate import:", importlib.metadata.version("rfdetr"), rfdetr.__file__)'

run_scan() {
  local run_number="$1"
  local label="$2"
  local package_path="$3"
  local run_name
  run_name="$(printf 'run-%02d-%s' "$run_number" "$label")"
  local result_dir="$OUTPUT_DIR/runs/$run_name"
  local scan_log="$result_dir/scan.log"
  mkdir -p "$result_dir"

  echo
  echo "Starting $run_name detector scan"
  local started=$SECONDS
  (
    if [[ -n "$package_path" ]]; then
      export PYTHONPATH="$package_path"
    else
      unset PYTHONPATH || true
    fi
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    "$PYTHON_PATH" "$SCANNER_PATH" \
      "$SOURCE_VIDEO" "$result_dir/left.json" \
      --stereo-right-manifest "$result_dir/right.json" \
      --model "$MODEL_PATH" \
      --backend rfdetr \
      --rfdetr-variant large \
      --crop-eye both \
      --stereo-sample-mode paired \
      --active-ranges "0/$TEST_SECONDS" \
      --device "$DETECT_DEVICE" \
      --batch-size 2 \
      --max-detections 64 \
      --decode-mode sequential \
      --sample-stride 0.1 \
      --region-duration 1.0 \
      --confidence 0.15 \
      --temporal-padding 1.0 \
      --region-nms-iou 0.45 \
      --mask-expansion 0.10 \
      --mask-size 128
  ) 2>&1 | /usr/bin/tee "$scan_log"
  echo "$run_name wall time: $((SECONDS - started))s"
}

RUN_NUMBER=0
for ((ROUND = 1; ROUND <= AB_ROUNDS; ROUND++)); do
  if (( ROUND % 2 == 1 )); then
    RUN_ORDER=(current candidate)
  else
    RUN_ORDER=(candidate current)
  fi
  for LABEL in "${RUN_ORDER[@]}"; do
    RUN_NUMBER=$((RUN_NUMBER + 1))
    if [[ "$LABEL" == "candidate" ]]; then
      run_scan "$RUN_NUMBER" "$LABEL" "$CANDIDATE_DIR"
    else
      run_scan "$RUN_NUMBER" "$LABEL" ""
    fi
  done
done

summarize_scans() {
  /usr/bin/awk '
    /Detector scan:/ {
      value = $3
      sub(/s,$/, "", value)
      values[++count] = value + 0
      total += value
      if (count == 1 || value < best) best = value
      if (count == 1 || value > worst) worst = value
    }
    END {
      if (count > 0) printf "%.3f %.3f %.3f %d", total / count, best, worst, count
    }
  ' "$@"
}

CURRENT_SUMMARY="$(summarize_scans "$OUTPUT_DIR"/runs/run-*-current/scan.log)"
CANDIDATE_SUMMARY="$(summarize_scans "$OUTPUT_DIR"/runs/run-*-candidate/scan.log)"
read -r CURRENT_SCAN CURRENT_BEST CURRENT_WORST CURRENT_COUNT <<<"$CURRENT_SUMMARY"
read -r CANDIDATE_SCAN CANDIDATE_BEST CANDIDATE_WORST CANDIDATE_COUNT \
  <<<"$CANDIDATE_SUMMARY"

echo
echo "===== Balanced A/B summary ====="
echo "Current $CURRENT_VERSION:   average ${CURRENT_SCAN}s, best/worst ${CURRENT_BEST}/${CURRENT_WORST}s, runs $CURRENT_COUNT"
echo "Candidate $CANDIDATE_VERSION: average ${CANDIDATE_SCAN}s, best/worst ${CANDIDATE_BEST}/${CANDIDATE_WORST}s, runs $CANDIDATE_COUNT"
/usr/bin/awk -v old="$CURRENT_SCAN" -v new="$CANDIDATE_SCAN" '
  BEGIN {
    if (old > 0 && new > 0) {
      printf "Candidate speed change: %.2f%% (positive is faster)\n", (old - new) * 100 / old
    }
  }
'

REFERENCE_DIR="$OUTPUT_DIR/runs/run-01-current"
FIRST_CANDIDATE_DIR="$OUTPUT_DIR/runs/run-02-candidate"
CURRENT_LEFT_REGIONS="$(/usr/bin/jq '.regions | length' "$REFERENCE_DIR/left.json")"
CURRENT_RIGHT_REGIONS="$(/usr/bin/jq '.regions | length' "$REFERENCE_DIR/right.json")"
CANDIDATE_LEFT_REGIONS="$(/usr/bin/jq '.regions | length' "$FIRST_CANDIDATE_DIR/left.json")"
CANDIDATE_RIGHT_REGIONS="$(/usr/bin/jq '.regions | length' "$FIRST_CANDIDATE_DIR/right.json")"
echo "Current regions:   left $CURRENT_LEFT_REGIONS, right $CURRENT_RIGHT_REGIONS"
echo "Candidate regions: left $CANDIDATE_LEFT_REGIONS, right $CANDIDATE_RIGHT_REGIONS"

PARITY=PASS
for RESULT_DIR in "$OUTPUT_DIR"/runs/run-*; do
  if ! cmp -s "$REFERENCE_DIR/left.json" "$RESULT_DIR/left.json" \
    || ! cmp -s "$REFERENCE_DIR/right.json" "$RESULT_DIR/right.json"; then
    PARITY=DIFFERENT
    break
  fi
done
if [[ "$PARITY" == "PASS" ]]; then
  echo "Detection parity: PASS (all manifests are byte-identical)"
else
  echo "Detection parity: DIFFERENT (inspect manifests before upgrading)"
fi
echo "Detection-only A/B: COMPLETE"
echo "Results: $OUTPUT_DIR"
echo "Log:     $MASTER_LOG"
