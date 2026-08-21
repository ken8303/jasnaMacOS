#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: $0 INPUT_SBS_VIDEO left|right OUTPUT_EYE_VIDEO" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTOR="${JASNA_DETECTOR:-rfdetr-vr-v1}"
case "$DETECTOR" in
  rfdetr-v6)
    [[ -x "$ROOT_DIR/.venv-rfdetr/bin/python" \
      && -s "$ROOT_DIR/Models/MosaicDetection/rfdetr-v6.pt" ]] || {
      echo "error: first run $ROOT_DIR/script/setup_rfdetr_detector.sh DOWNLOAD_DIRECTORY" >&2
      exit 1
    }
    ;;
  rfdetr-vr-v1)
    [[ -x "$ROOT_DIR/.venv-rfdetr/bin/python" \
      && -s "$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt" ]] || {
      echo "error: first run $ROOT_DIR/script/setup_rfdetr_detector.sh DOWNLOAD_DIRECTORY" >&2
      exit 1
    }
    ;;
  yolo-v2-fast)
    [[ -x "$ROOT_DIR/.venv-mosaic/bin/python" ]] || {
      echo "error: first run $ROOT_DIR/script/setup_mosaic_detector.sh" >&2
      exit 1
    }
    ;;
  *)
    echo "error: JASNA_DETECTOR must be rfdetr-v6, rfdetr-vr-v1, or yolo-v2-fast" >&2
    exit 1
    ;;
esac

if [[ "${JASNA_SHARED_SBS_SOURCE:-0}" == "1" ]]; then
  DETECT_EYE="$2"
else
  DETECT_EYE=none
fi

JASNA_SPARSE_MOSAIC=1 \
JASNA_DETECTOR="$DETECTOR" \
JASNA_DETECT_EYE="$DETECT_EYE" \
JASNA_VR_PROJECTION="${JASNA_VR_PROJECTION:-fisheye}" \
  "$ROOT_DIR/script/restore_vr_eye_segments.sh" "$@"
