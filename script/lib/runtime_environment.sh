#!/usr/bin/env bash

# App resources are read-only. Packaged workflows use the shipped executable;
# only a source checkout may invoke SwiftPM or create a local build directory.
export PYTHONDONTWRITEBYTECODE=1
if [[ "$ROOT_DIR" == *.app/Contents/Resources/Runtime ]]; then
  export JASNA_APP_BINARY="${JASNA_APP_BINARY:-${ROOT_DIR%/Resources/Runtime}/MacOS/JasnaMetalPoC}"
  [[ -x "$JASNA_APP_BINARY" ]] || {
    echo "error: bundled restoration engine is missing or not executable: $JASNA_APP_BINARY" >&2
    exit 1
  }
fi
