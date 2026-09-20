#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO OUTPUT_SBS_VIDEO [START_TIME]" >&2
  echo "example: $0 input.mp4 restored-alternating.mov 00:25:57" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Retain the 10 Hz timeline but alternate real left/right observations. Each
# eye is scanned at 5 Hz and the total RF-DETR image count is halved.
export JASNA_STEREO_SAMPLE_MODE=alternating
export JASNA_ADAPTIVE_DETECT=0

echo "Jasna alternating-eye 30-second experiment"
echo "RF-DETR input: 10 samples/s total, alternating to 5 samples/s per eye"
echo "Restoration and direct SBS output: unchanged"

exec "$ROOT_DIR/script/test_vr_fast_30s.sh" "$@"
