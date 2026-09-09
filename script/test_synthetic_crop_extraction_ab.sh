#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [RESULTS_DIRECTORY]"
  echo "Generated 4K eye frames through the application crop extractor; no video or models."
}
if [[ $# -eq 1 && "$1" == --help ]]; then usage; exit 0; fi
if [[ $# -gt 1 || (${1:-} == -*) ]]; then usage >&2; exit 2; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIRECTORY="${1-$ROOT_DIR/.test-results/synthetic-crop-extraction-ab}"
if [[ -z "$RESULTS_DIRECTORY" || "$RESULTS_DIRECTORY" == *[[:space:]] ]]; then
  echo "error: results directory is empty or ends with whitespace" >&2
  exit 2
fi
command -v swift >/dev/null || { echo "error: Swift toolchain is unavailable" >&2; exit 2; }
mkdir -p "$RESULTS_DIRECTORY"
RESULTS_DIRECTORY="$(cd "$RESULTS_DIRECTORY" && pwd -P)"
PACKAGE_CACHE_ID="$(printf '%s' "$ROOT_DIR" | /usr/bin/shasum -a 256 | /usr/bin/cut -c 1-16)"
BUILD_DIRECTORY="/private/tmp/jasna-synthetic-crop-ab-$(id -u)-$PACKAGE_CACHE_ID"
if [[ ! -e "$BUILD_DIRECTORY" && ! -L "$BUILD_DIRECTORY" ]]; then
  (umask 077; mkdir "$BUILD_DIRECTORY")
fi
if [[ -L "$BUILD_DIRECTORY" || ! -d "$BUILD_DIRECTORY" || ! -O "$BUILD_DIRECTORY" ]]; then
  echo "error: unsafe compiler-cache directory: $BUILD_DIRECTORY" >&2
  exit 2
fi
mkdir -p "$BUILD_DIRECTORY/ModuleCache" "$RESULTS_DIRECTORY/swift-cache" \
  "$RESULTS_DIRECTORY/config" "$RESULTS_DIRECTORY/security" "$RESULTS_DIRECTORY/tmp"
RUN_DIRECTORY="$(mktemp -d "$RESULTS_DIRECTORY/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
REPORT_FILE="$RUN_DIRECTORY/report.json"
LOG_FILE="$RUN_DIRECTORY/comparison.log"
SUMMARY_FILE="$RUN_DIRECTORY/summary.txt"
printf 'status=INCOMPLETE\nreason=run has not completed\nreport=%s\n' "$REPORT_FILE" > "$SUMMARY_FILE"
printf 'Synthetic application crop-extraction A/B\nResults: %s\nCompiler cache: %s\nGenerated frames only; no media, detector, models, restoration graph, compositor, or encoder.\n' \
  "$RUN_DIRECTORY" "$BUILD_DIRECTORY" | tee "$LOG_FILE"

export TMPDIR="$RESULTS_DIRECTORY/tmp/"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIRECTORY/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIRECTORY/ModuleCache"
swift --version 2>&1 | tee -a "$LOG_FILE"

set +e
swift run --package-path "$ROOT_DIR" --scratch-path "$BUILD_DIRECTORY" \
  --cache-path "$RESULTS_DIRECTORY/swift-cache" --config-path "$RESULTS_DIRECTORY/config" \
  --security-path "$RESULTS_DIRECTORY/security" --disable-sandbox \
  --disable-automatic-resolution -c release JasnaMetalPoC \
  --synthetic-crop-extraction-ab "$REPORT_FILE" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUSES=("${PIPESTATUS[@]}")
set -e
APP_STATUS="${RUN_STATUSES[0]}"
LOG_STATUS="${RUN_STATUSES[1]}"
REPORT_STATUS="$(/usr/bin/plutil -extract status raw -o - "$REPORT_FILE" 2>/dev/null || true)"
RESULT=INCOMPLETE
EXIT_STATUS=2
if [[ "$APP_STATUS" -eq 0 && "$LOG_STATUS" -eq 0 && "$REPORT_STATUS" == PASS ]]; then
  RESULT=PASS
  EXIT_STATUS=0
elif [[ "$LOG_STATUS" -eq 0 && -f "$REPORT_FILE" ]]; then
  RESULT=FAIL
  EXIT_STATUS=1
fi
printf 'status=%s\napp_exit=%s\nlogging_exit=%s\nreport=%s\nlog=%s\nbuild_directory=%s\n' \
  "$RESULT" "$APP_STATUS" "$LOG_STATUS" "$REPORT_FILE" "$LOG_FILE" "$BUILD_DIRECTORY" \
  > "$SUMMARY_FILE"
if ! printf '\nSynthetic crop-extraction A/B result: %s\nSummary: %s\n' \
  "$RESULT" "$SUMMARY_FILE" | tee -a "$LOG_FILE"; then
  printf 'status=INCOMPLETE\nreason=final log write failed\n' > "$SUMMARY_FILE" || true
  exit 2
fi
exit "$EXIT_STATUS"
