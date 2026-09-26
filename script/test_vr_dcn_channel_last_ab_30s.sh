#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REFERENCE_WORK_DIR OUTPUT_DIRECTORY" >&2
  echo "example: $0 previous-test.jasna-vr30-v22-work dcn-channel-last-ab" >&2
  echo "" >&2
  echo "Runs restoration only against the same prepared source and manifests." >&2
  echo "Detection and source preparation are not repeated." >&2
  echo "" >&2
  echo "optional: JASNA_DCN_AB_SECONDS=30" >&2
  echo "          JASNA_DCN_AB_START_SECOND=0" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

REFERENCE_WORK_DIR="$1"
OUTPUT_DIRECTORY="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTED_SECONDS="${JASNA_DCN_AB_SECONDS:-30}"
START_SECOND="${JASNA_DCN_AB_START_SECOND:-0}"

[[ -d "$REFERENCE_WORK_DIR" ]] || {
  echo "error: reference work directory not found: $REFERENCE_WORK_DIR" >&2
  exit 1
}
[[ "$SELECTED_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: JASNA_DCN_AB_SECONDS must be a positive integer" >&2
  exit 1
}
[[ "$START_SECOND" =~ ^[0-9]+$ ]] || {
  echo "error: JASNA_DCN_AB_START_SECOND must be a non-negative integer" >&2
  exit 1
}
[[ ! -e "$OUTPUT_DIRECTORY" ]] || {
  echo "error: output directory already exists: $OUTPUT_DIRECTORY" >&2
  echo "use a new directory so previous checkpoints cannot affect the result" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIRECTORY"
OUTPUT_DIRECTORY="$(cd "$OUTPUT_DIRECTORY" && pwd)"
COMPARISON_LOG="$OUTPUT_DIRECTORY/comparison.log"
exec > >(/usr/bin/tee -a "$COMPARISON_LOG") 2>&1

echo "===== Jasna real-video DCNv2 channel-last A/B $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Reference: $REFERENCE_WORK_DIR"
echo "Range:     relative seconds $START_SECOND-$((START_SECOND + SELECTED_SECONDS))"
echo "Control:   original NCHW gather"
echo "Candidate: channel-last staged gather"
echo "Detection/source preparation: reused"

mkdir -p "$ROOT_DIR/.build/ModuleCache"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT_DIR/.build/ModuleCache"
echo "Building one shared optimized executable"
(
  cd "$ROOT_DIR"
  swift build --disable-sandbox -c release
)
JASNA_APP_BINARY="$(
  cd "$ROOT_DIR"
  swift build --disable-sandbox -c release --show-bin-path
)/JasnaMetalPoC"
[[ -x "$JASNA_APP_BINARY" ]] || {
  echo "error: optimized JasnaMetalPoC executable was not produced" >&2
  exit 1
}
export JASNA_APP_BINARY

# Keep failures visible in the measurement rather than silently retrying a
# different amount of work. The same recorded batch/window settings are used
# for both sides unless the caller explicitly overrides them.
export JASNA_GPU_TIMEOUT_RETRIES=0
export JASNA_LOG_PEAK_MEMORY=1
export JASNA_RESTORE_ONLY_START_SECOND="$START_SECOND"
export JASNA_RESTORE_ONLY_SECONDS="$SELECTED_SECONDS"

run_variant() {
  local name="$1"
  local enabled="$2"
  local output="$OUTPUT_DIRECTORY/$name.mov"

  echo
  echo "----- $name -----"
  JASNA_DCN_CHANNEL_LAST_GATHER="$enabled" \
    "$ROOT_DIR/script/test_vr_restore_only_30s.sh" \
      "$REFERENCE_WORK_DIR" "$output"
}

run_variant control-nchw 0
run_variant candidate-channel-last 1

python3 - \
  "$OUTPUT_DIRECTORY/control-nchw.jasna-restore-only.log" \
  "$OUTPUT_DIRECTORY/candidate-channel-last.jasna-restore-only.log" <<'PY'
import re
import sys

def read_metrics(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    walls = re.findall(r"^Wall time:\s*(\d+)s$", text, re.MULTILINE)
    peaks = re.findall(r"Runtime memory: peak resident ([0-9.]+) GiB", text)
    hot_paths = re.findall(
        r"sparse hot-path phases: (\d+) model frames, extraction [0-9.]+ ms, "
        r"graph wall ([0-9.]+) ms, GPU ([0-9.]+) ms, cache writes [0-9.]+ ms",
        text,
    )
    passed = "Restoration-only selected-range test: PASS" in text
    if not walls:
        raise SystemExit(f"error: total wall time missing from {path}")
    return {
        "wall": int(walls[-1]),
        "peak": max(map(float, peaks)) if peaks else None,
        "model_frames": sum(int(row[0]) for row in hot_paths),
        "graph": sum(float(row[1]) for row in hot_paths) / 1000.0,
        "gpu": sum(float(row[2]) for row in hot_paths) / 1000.0,
        "passed": passed,
    }

control = read_metrics(sys.argv[1])
candidate = read_metrics(sys.argv[2])
delta = candidate["wall"] - control["wall"]
percent = 100.0 * delta / control["wall"]

print("\n===== Real-video comparison =====")
print(f"Control wall:   {control['wall']}s")
print(f"Candidate wall: {candidate['wall']}s")
print(f"Difference:     {delta:+d}s ({percent:+.1f}%; negative is faster)")
print(f"Model frames:   {control['model_frames']} / {candidate['model_frames']}")
print(f"Graph wall:     {control['graph']:.3f} / {candidate['graph']:.3f}s")
print(f"GPU time:       {control['gpu']:.3f} / {candidate['gpu']:.3f}s")
if control["peak"] is not None and candidate["peak"] is not None:
    print(f"Peak resident:  {control['peak']:.2f} / {candidate['peak']:.2f} GiB (control/candidate)")
print("Validation:     " + ("PASS" if control["passed"] and candidate["passed"] else "INCOMPLETE"))
if not (control["passed"] and candidate["passed"]):
    raise SystemExit(1)
PY

echo "Comparison log: $COMPARISON_LOG"
