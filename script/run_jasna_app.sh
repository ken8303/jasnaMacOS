#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JASNA_PROJECT_ROOT="$ROOT_DIR"
exec /usr/bin/swift run --package-path "$ROOT_DIR" JasnaMacApp
