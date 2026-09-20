#!/usr/bin/env bash
set -euo pipefail
SOURCE_NAME=main
if [[ ${1:-} == --gpu-written ]]; then SOURCE_NAME=gpu-written; shift; fi
if [[ "$SOURCE_NAME" == main && ${1:-} == --grouped ]]; then SOURCE_NAME=grouped; shift; fi
if [[ "$SOURCE_NAME" == main && ${1:-} == --pipeline ]]; then SOURCE_NAME=pipeline; shift; fi
[[ $# -le 1 && ${1:-} != --* ]] || { echo "usage: bash $0 [--gpu-written | --grouped | --pipeline] [RESULTS_DIRECTORY]" >&2; exit 2; }
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIR="${1:-$ROOT_DIR/.test-results/synthetic-output-copy}"
mkdir -p "$RESULTS_DIR"
RUN_DIR="$(mktemp -d "$RESULTS_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
BUILD_DIR="$(mktemp -d /private/tmp/synthetic-output-copy.XXXXXX)"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/module-cache"
echo "Results: $RUN_DIR"
xcrun swiftc -O "$ROOT_DIR/Diagnostics/OutputCopy/$SOURCE_NAME.swift" -o "$BUILD_DIR/output-copy" 2>&1 | tee "$RUN_DIR/build.log"
"$BUILD_DIR/output-copy" "$RUN_DIR/report.json" 2>&1 | tee "$RUN_DIR/comparison.log"
echo "Report: $RUN_DIR/report.json"
