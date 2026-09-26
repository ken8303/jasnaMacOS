#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && "$1" == --help ]]; then
  echo "usage: $0 [--mixed-only | --held-out-only | --job-profile-only | --selector-candidate-only | --boundary-holdout-only] [RESULTS_DIRECTORY]"
  echo "Standalone generated-pattern CPU/Metal comparison; no videos, models or restoration."
  echo "--mixed-only runs the original mixed workloads and correctness gates only."
  echo "--held-out-only runs new sizes/changing reuse plus correctness gates, skipping earlier timings."
  echo "--job-profile-only splits held-out job costs and pairs instrumented/plain runs; no rule changes."
  echo "--selector-candidate-only compares a 640x512 reuse candidate with the frozen selector and controls."
  echo "--boundary-holdout-only tests a predeclared pixel-area candidate on unseen rectangular sizes."
  exit 0
fi
BENCHMARK_SCOPE=all
if [[ ${1:-} == --mixed-only || ${1:-} == --held-out-only || ${1:-} == --job-profile-only || ${1:-} == --selector-candidate-only || ${1:-} == --boundary-holdout-only ]]; then BENCHMARK_SCOPE="$1"; shift; fi
if [[ $# -gt 1 || (${1:-} == -*) ]]; then echo "usage: $0 [--mixed-only | --held-out-only | --job-profile-only | --selector-candidate-only | --boundary-holdout-only] [RESULTS_DIRECTORY]" >&2; exit 2; fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LAB_DIRECTORY="$ROOT_DIR/Diagnostics/SyntheticResampling"
RESULTS_DIRECTORY="${1-$ROOT_DIR/.test-results/synthetic-metal}"
if [[ -z "$RESULTS_DIRECTORY" || "$RESULTS_DIRECTORY" == *[[:space:]] ]]; then
  echo "error: results directory is empty or ends with whitespace" >&2
  exit 2
fi
command -v swift >/dev/null || { echo "error: Swift toolchain is unavailable" >&2; exit 2; }
mkdir -p "$RESULTS_DIRECTORY" || exit 2
RESULTS_DIRECTORY="$(cd "$RESULTS_DIRECTORY" && pwd -P)"
PACKAGE_CACHE_ID="$(printf '%s' "$LAB_DIRECTORY" | /usr/bin/shasum -a 256 | /usr/bin/cut -c 1-16)"
BUILD_DIRECTORY="/private/tmp/jasna-synthetic-metal-$(id -u)-$PACKAGE_CACHE_ID"
if [[ ! -e "$BUILD_DIRECTORY" && ! -L "$BUILD_DIRECTORY" ]]; then
  (umask 077; mkdir "$BUILD_DIRECTORY") || exit 2
fi
if [[ -L "$BUILD_DIRECTORY" || ! -d "$BUILD_DIRECTORY" || ! -O "$BUILD_DIRECTORY" ]]; then
  echo "error: unsafe compiler-cache directory: $BUILD_DIRECTORY" >&2
  exit 2
fi
mkdir -p "$BUILD_DIRECTORY/ModuleCache" "$RESULTS_DIRECTORY/swift-cache" \
  "$RESULTS_DIRECTORY/config" "$RESULTS_DIRECTORY/security" || exit 2
RUN_DIRECTORY="$(mktemp -d "$RESULTS_DIRECTORY/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
REPORT_FILE="$RUN_DIRECTORY/report.json"
LOG_FILE="$RUN_DIRECTORY/comparison.log"
SUMMARY_FILE="$RUN_DIRECTORY/summary.txt"
printf 'status=INCOMPLETE\nreason=run has not completed\nreport=%s\n' "$REPORT_FILE" > "$SUMMARY_FILE" || exit 2
printf 'Standalone synthetic CPU/Metal comparison\nResults: %s\nCompiler cache: %s\nNo restoration modules or model packages are used.\n' \
  "$RUN_DIRECTORY" "$BUILD_DIRECTORY" | tee "$LOG_FILE" || exit 2
export CLANG_MODULE_CACHE_PATH="$BUILD_DIRECTORY/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIRECTORY/ModuleCache"
swift --version 2>&1 | tee -a "$LOG_FILE" || exit 2
LAB_ARGUMENTS=(--report "$REPORT_FILE")
if [[ "$BENCHMARK_SCOPE" != all ]]; then LAB_ARGUMENTS+=("$BENCHMARK_SCOPE"); fi

set +e
swift run --package-path "$LAB_DIRECTORY" --scratch-path "$BUILD_DIRECTORY" \
  --cache-path "$RESULTS_DIRECTORY/swift-cache" --config-path "$RESULTS_DIRECTORY/config" \
  --security-path "$RESULTS_DIRECTORY/security" --disable-sandbox --disable-automatic-resolution \
  -c release SyntheticResamplingLab "${LAB_ARGUMENTS[@]}" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUSES=("${PIPESTATUS[@]}")
set -e
LAB_STATUS="${RUN_STATUSES[0]}"
LOG_STATUS="${RUN_STATUSES[1]}"
REPORT_STATUS="$(/usr/bin/plutil -extract status raw -o - "$REPORT_FILE" 2>/dev/null || true)"
RESULT=INCOMPLETE
EXIT_STATUS=2
if [[ "$LOG_STATUS" -eq 0 && "$REPORT_STATUS" == PASS && "$LAB_STATUS" -eq 0 ]]; then
  RESULT=PASS
  EXIT_STATUS=0
elif [[ "$LOG_STATUS" -eq 0 && "$REPORT_STATUS" == FAIL ]]; then
  RESULT=FAIL
  EXIT_STATUS=1
fi
printf 'status=%s\nlab_exit=%s\nlogging_exit=%s\nreport=%s\nlog=%s\nbuild_directory=%s\n' \
  "$RESULT" "$LAB_STATUS" "$LOG_STATUS" "$REPORT_FILE" "$LOG_FILE" "$BUILD_DIRECTORY" > "$SUMMARY_FILE" || exit 2
if ! printf '\nSynthetic CPU/Metal result: %s\nSummary: %s\n' "$RESULT" "$SUMMARY_FILE" | tee -a "$LOG_FILE"; then
  printf 'status=INCOMPLETE\nreason=final log write failed\n' > "$SUMMARY_FILE" || true
  exit 2
fi
exit "$EXIT_STATUS"
