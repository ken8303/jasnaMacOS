#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 DIRECTORY_WITH_JASNA_0.10_AMD_PARTS" >&2
  exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$1"
PART_ZERO="$DOWNLOAD_DIR/jasna-linux-amd-0.10.0.tar.zst.part000"
PART_ONE="$DOWNLOAD_DIR/jasna-linux-amd-0.10.0.tar.zst.part001"
MODEL_DIR="$ROOT_DIR/Models/MosaicDetection"
MODEL_PATH="$MODEL_DIR/rfdetr-vr-v1.pt"
PYTHON_BIN="${JASNA_RFDETR_PYTHON:-/opt/homebrew/bin/python3.13}"
VENV_PATH="$ROOT_DIR/.venv-rfdetr"

EXPECTED_PART_ZERO="ea73281ebf73980cc71550a20e888e1c9ab2ba894bdf865b4eef42f671ecfcbd"
EXPECTED_PART_ONE="19c973c24db7caaa48cf68bd1c33c3fe1db675495fbae70db2dc0d78d4a227c2"
EXPECTED_MODEL="55543c83911921ef79cd8cae8540bd25e34c7daf488e77f79d233d6926973a2e"

for path in "$PART_ZERO" "$PART_ONE"; do
  [[ -s "$path" ]] || {
    echo "error: missing Jasna v0.10 AMD archive part: $path" >&2
    exit 1
  }
done
command -v zstd >/dev/null 2>&1 || {
  echo "error: zstd is required (brew install zstd)" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "error: Python 3.10 or newer is required; expected $PYTHON_BIN" >&2
  exit 1
}

verify_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(/usr/bin/shasum -a 256 "$path" | /usr/bin/awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "error: checksum mismatch for $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  }
}

echo "Verifying the official Jasna v0.10 AMD archive"
verify_hash "$PART_ZERO" "$EXPECTED_PART_ZERO"
verify_hash "$PART_ONE" "$EXPECTED_PART_ONE"

if [[ -s "$MODEL_PATH" ]]; then
  verify_hash "$MODEL_PATH" "$EXPECTED_MODEL"
  echo "RF-DETR VR model is already extracted"
else
  mkdir -p "$MODEL_DIR"
  echo "Streaming only rfdetr-vr-v1.pt from the split archive"
  /bin/cat "$PART_ZERO" "$PART_ONE" \
    | zstd -dc \
    | /usr/bin/tar -xvf - -C "$MODEL_DIR" --strip-components=2 \
      jasna-linux-amd-0.10.0/model_weights/rfdetr-vr-v1.pt
  verify_hash "$MODEL_PATH" "$EXPECTED_MODEL"
fi

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi
"$VENV_PATH/bin/python" -m pip install --disable-pip-version-check \
  'rfdetr==1.8.3' 'transformers==5.1.0' 'opencv-python-headless'
"$VENV_PATH/bin/python" -c \
  'import cv2, rfdetr, torch; print(f"RF-DETR ready: torch {torch.__version__}, MPS {torch.backends.mps.is_available()}")'

echo "Model:       $MODEL_PATH"
echo "Environment: $VENV_PATH"
echo "RF-DETR is now the default for sparse VR restoration"
echo "Use JASNA_DETECTOR=yolo-v2-fast when faster scanning is preferred"
