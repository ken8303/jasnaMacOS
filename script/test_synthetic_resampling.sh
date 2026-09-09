#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [RESULTS_DIRECTORY]"
  echo "Generated patterns only; no videos, detector, model loading or restoration."
  echo "Each run saves permanent logs/results; only the rebuildable compiler cache uses /private/tmp."
  echo "Exit codes: 0 PASS, 1 test failure, 2 incomplete run/setup/logging error."
}
if [[ $# -eq 1 && "$1" == --help ]]; then usage; exit 0; fi
if [[ $# -gt 1 || (${1:-} == -*) ]]; then usage >&2; exit 2; fi

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIRECTORY="${1-$PACKAGE_ROOT/.test-results/synthetic-resampling}"
if [[ -z "$RESULTS_DIRECTORY" || "$RESULTS_DIRECTORY" == *[[:space:]] ]]; then
  echo "error: results directory is empty or ends with whitespace" >&2
  exit 2
fi
SWIFT_BINARY="$(command -v swift)" || { echo "error: Swift toolchain is unavailable" >&2; exit 2; }
mkdir -p "$RESULTS_DIRECTORY" || exit 2
RESULTS_DIRECTORY="$(cd "$RESULTS_DIRECTORY" && pwd -P)" || exit 2
if [[ -n "${JASNA_SYNTHETIC_BUILD_DIR:-}" ]]; then
  BUILD_DIRECTORY="$JASNA_SYNTHETIC_BUILD_DIR"
  mkdir -p "$BUILD_DIRECTORY" || exit 2
else
  # File Provider can repeatedly attach FinderInfo to bundles under Documents.
  # Keep reports persistent, but build outside that tree with signing enabled.
  # Stable per-user/per-package location avoids accumulating a cache per run.
  PACKAGE_CACHE_ID="$(printf '%s' "$PACKAGE_ROOT" | /usr/bin/shasum -a 256 | /usr/bin/cut -c 1-16)" || exit 2
  BUILD_DIRECTORY="/private/tmp/jasna-synthetic-build-$(id -u)-$PACKAGE_CACHE_ID"
  if [[ ! -e "$BUILD_DIRECTORY" && ! -L "$BUILD_DIRECTORY" ]]; then
    (umask 077; mkdir "$BUILD_DIRECTORY") || exit 2
  fi
  if [[ -L "$BUILD_DIRECTORY" || ! -d "$BUILD_DIRECTORY" || ! -O "$BUILD_DIRECTORY" ]]; then
    echo "error: unsafe default build-cache directory: $BUILD_DIRECTORY" >&2
    exit 2
  fi
fi
BUILD_DIRECTORY="$(cd "$BUILD_DIRECTORY" && pwd -P)" || exit 2
RUN_DIRECTORY="$(mktemp -d "$RESULTS_DIRECTORY/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")" || exit 2
LOG_FILE="$RUN_DIRECTORY/synthetic-tests.log"
SUMMARY_FILE="$RUN_DIRECTORY/summary.txt"
mkdir -p "$RESULTS_DIRECTORY/tmp" "$RESULTS_DIRECTORY/cache" \
  "$RESULTS_DIRECTORY/config" "$RESULTS_DIRECTORY/security" "$BUILD_DIRECTORY/ModuleCache" || exit 2

# An interrupted process leaves RUNNING, never a stale PASS from an earlier run.
printf 'status=RUNNING\nlog=%s\nbuild_directory=%s\n' "$LOG_FILE" "$BUILD_DIRECTORY" > "$SUMMARY_FILE" || exit 2
printf 'Synthetic resampling diagnostic\nResults: %s\nStrict alignment check: enabled\nNo media files or models are used.\nRebuildable compiler cache: %s\nLogs and summaries remain in the results folder.\n' \
  "$RUN_DIRECTORY" "$BUILD_DIRECTORY" | tee "$LOG_FILE" || exit 2

# Keep the override for compatibility with earlier diagnostic builds. The
# corrected production transform now enforces the same gate in every test run.
export JASNA_SYNTHETIC_STRICT_ALIGNMENT=1
export TMPDIR="$RESULTS_DIRECTORY/tmp/"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIRECTORY/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIRECTORY/ModuleCache"
FILTER='^JasnaMetalPoCTests[./](syntheticMovingRawRampKeepsPixelCentres|syntheticMovingFisheyeRampKeepsPixelCentres|syntheticMovingDetailExtractionMatchesBilinearOracle|syntheticReferenceMatchesAnalyticAxes|syntheticReferenceRoundTripsBorderPixelCentres|syntheticReferenceSubpixelMotionSurvivesCropChanges)'

run_synthetic_tests() {
  set +e
  "$SWIFT_BINARY" test --package-path "$PACKAGE_ROOT" --disable-sandbox \
    --scratch-path "$BUILD_DIRECTORY" --cache-path "$RESULTS_DIRECTORY/cache" \
    --config-path "$RESULTS_DIRECTORY/config" --security-path "$RESULTS_DIRECTORY/security" \
    --disable-automatic-resolution --disable-xctest --enable-swift-testing \
    --filter "$FILTER" 2>&1 | tee "$ATTEMPT_LOG" | tee -a "$LOG_FILE"
  RUN_STATUSES=("${PIPESTATUS[@]}")
  set -e
  SWIFT_STATUS="${RUN_STATUSES[0]}"
  TEE_STATUS=0
  if [[ "${RUN_STATUSES[1]}" -ne 0 || "${RUN_STATUSES[2]}" -ne 0 ]]; then TEE_STATUS=1; fi
}

repair_generated_test_metadata() {
  local bundle_name bundle physical_bundle attributes attribute
  local repaired=0
  command -v xattr >/dev/null || return 2
  # Only the two generated bundle roots: no recursive clearing, source edits,
  # quarantine removal, entitlement changes or signing-disable workarounds.
  for bundle_name in JasnaAppSupportTests.xctest JasnaMetalPoCTests.xctest; do
    bundle="$BUILD_DIRECTORY/out/Products/Debug/$bundle_name"
    [[ -d "$bundle" ]] || continue
    physical_bundle="$(cd "$bundle" && pwd -P)" || return 2
    if [[ -L "$bundle" || "$physical_bundle" != "$bundle" ]]; then
      echo "Refusing metadata repair through a symlink: $bundle" >&2
      return 2
    fi
    attributes="$(xattr "$bundle")" || return 2
    for attribute in com.apple.FinderInfo com.apple.ResourceFork; do
      if printf '%s\n' "$attributes" | /usr/bin/grep -Fxq -- "$attribute"; then
        xattr -d "$attribute" "$bundle" || return 2
        repaired=$((repaired + 1))
        echo "Removed $attribute from generated test bundle: $bundle"
      fi
    done
  done
  [[ "$repaired" -gt 0 ]]
}

ATTEMPTS=1
METADATA_REPAIR_STATUS=not-needed
ATTEMPT_LOG="$RUN_DIRECTORY/attempt-1.log"
run_synthetic_tests
if [[ "$SWIFT_STATUS" -ne 0 && "$TEE_STATUS" -eq 0 ]] \
  && /usr/bin/grep -Fq 'resource fork, Finder information, or similar detritus not allowed' "$ATTEMPT_LOG"; then
  set +e
  repair_generated_test_metadata 2>&1 | tee -a "$LOG_FILE"
  REPAIR_STATUSES=("${PIPESTATUS[@]}")
  set -e
  METADATA_REPAIR_STATUS="${REPAIR_STATUSES[0]}"
  if [[ "${REPAIR_STATUSES[1]}" -ne 0 ]]; then
    TEE_STATUS=1
  elif [[ "$METADATA_REPAIR_STATUS" -eq 0 ]]; then
    printf '\nRetrying once after removing generated bundle metadata; signing remains enabled.\n' | tee -a "$LOG_FILE" || exit 2
    ATTEMPTS=2
    ATTEMPT_LOG="$RUN_DIRECTORY/attempt-2.log"
    run_synthetic_tests
  fi
fi

# Swift can succeed when a filter selects no tests. Require every intended case's
# completion marker as well as its exit status; never infer PASS from metrics alone.
EXPECTED_MARKERS=(
  'Synthetic raw moving ramp:'
  'Synthetic fisheye 512px moving ramp:'
  'Synthetic fisheye 4096px moving ramp:'
  'Synthetic reference axes: 10 anchors checked'
  'Synthetic reference borders 512px: 780 points'
  'Synthetic reference borders 4096px: 780 points'
  'Synthetic reference motion: 60 frames, 5 crop changes'
  'Synthetic raw moving detail at 0:'
  'Synthetic raw moving detail at 64:'
  'Synthetic raw moving detail at 128:'
  'Synthetic fisheye moving detail at 0:'
  'Synthetic fisheye moving detail at 64:'
  'Synthetic fisheye moving detail at 128:'
)
MISSING_CASES=0
for marker in "${EXPECTED_MARKERS[@]}"; do
  if ! /usr/bin/grep -Fq -- "$marker" "$ATTEMPT_LOG"; then
    MISSING_CASES=$((MISSING_CASES + 1))
  fi
done

RESULT=PASS
EXIT_STATUS=0
if [[ "$TEE_STATUS" -ne 0 || "$MISSING_CASES" -ne 0 ]]; then
  RESULT=INCOMPLETE
  EXIT_STATUS=2
elif [[ "$SWIFT_STATUS" -ne 0 ]]; then
  RESULT=FAIL
  EXIT_STATUS=1
fi
{
  printf 'status=%s\nswift_exit=%s\nlogging_exit=%s\nmissing_cases=%s\n' \
    "$RESULT" "$SWIFT_STATUS" "$TEE_STATUS" "$MISSING_CASES"
  printf 'strict_alignment=1\nlog=%s\nbuild_directory=%s\n' "$LOG_FILE" "$BUILD_DIRECTORY"
  printf 'attempts=%s\nmetadata_repair_status=%s\nfinal_attempt_log=%s\n' \
    "$ATTEMPTS" "$METADATA_REPAIR_STATUS" "$ATTEMPT_LOG"
  printf 'scope=synthetic CPU geometry and resampling; no restoration or throughput certification\n'
} > "$SUMMARY_FILE" || exit 2
if ! printf '\nSynthetic diagnostic: %s\nSummary: %s\n' "$RESULT" "$SUMMARY_FILE" | tee -a "$LOG_FILE"; then
  printf 'status=INCOMPLETE\nreason=final log write failed\n' > "$SUMMARY_FILE" || true
  exit 2
fi
if [[ "$RESULT" == FAIL ]]; then
  echo "The tests failed. Inspect the log for the actual assertions; this baseline has a known fisheye alignment defect."
elif [[ "$RESULT" == INCOMPLETE ]]; then
  echo "Not all intended cases completed, or logging failed. This is not a validation pass."
fi
exit "$EXIT_STATUS"
