#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/.venv-mosaic"
MODEL_DIR="$ROOT_DIR/Models/MosaicDetection"
MODEL_PATH="$MODEL_DIR/lada_vr_mosaic_detection_model_v2_fast.pt"
MODEL_URL="https://huggingface.co/zelefans/vrmr/resolve/main/lada_vr_mosaic_detection_model_v2_fast.pt"

mkdir -p "$MODEL_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install 'ultralytics==8.4.4'

if [[ ! -s "$MODEL_PATH" ]]; then
  curl -L --fail --show-error "$MODEL_URL" -o "$MODEL_PATH"
fi

MODEL_BYTES="$(stat -f '%z' "$MODEL_PATH")"
(( MODEL_BYTES > 1000000 )) || {
  echo "error: downloaded detector model is incomplete" >&2
  exit 1
}

echo "Mosaic detector environment ready"
echo "Python: $ENV_DIR/bin/python"
echo "Model:  $MODEL_PATH"
