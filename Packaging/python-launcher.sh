#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENTS_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONHOME="$CONTENTS_DIR/Resources/Runtime/python"
export PYTHONPATH="$SCRIPT_DIR/../lib/python3.13/site-packages"
exec "$PYTHONHOME/bin/python3.13" "$@"
