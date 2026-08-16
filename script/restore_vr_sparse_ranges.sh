#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO MOSAIC_RANGES" >&2
  echo "example ranges: 00:12:00-00:14:00,00:20:30-00:22:00" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JASNA_MOSAIC_RANGES="$3"
exec "$ROOT_DIR/script/restore_vr_sparse_sbs.sh" "$1" "$2"
