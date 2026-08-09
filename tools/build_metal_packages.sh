#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COREML_DIR="${1:-$ROOT_DIR/Models/CoreML}"
METAL_DIR="${2:-$ROOT_DIR/Models/MetalML}"
SHIM_DIR="$ROOT_DIR/work/metal-toolchain-shim"

BUILDER="$SHIM_DIR/Metal.xctoolchain/usr/bin/metal-package-builder"
if BUILDER_PATH="$(xcrun --find metal-package-builder 2>/dev/null)"; then
  METAL_TOOLCHAIN="$(cd "$(dirname "$BUILDER_PATH")/../.." && pwd)"
  DEFAULT_TOOLCHAIN="$(xcrun --toolchain XcodeDefault --find coremlcompiler)"
  DEFAULT_TOOLCHAIN="$(cd "$(dirname "$DEFAULT_TOOLCHAIN")/../.." && pwd)"

  # Xcode 27 beta 4's package builder looks for XcodeDefault.xctoolchain beside
  # its downloaded Metal.xctoolchain. Recreate that expected layout locally so
  # no installed Xcode files need to be modified.
  mkdir -p "$SHIM_DIR/Metal.xctoolchain/usr/bin"
  cp "$BUILDER_PATH" "$BUILDER"
  ln -sfn "$METAL_TOOLCHAIN/System" "$SHIM_DIR/Metal.xctoolchain/System"
  ln -sfn "$METAL_TOOLCHAIN/usr/lib" "$SHIM_DIR/Metal.xctoolchain/usr/lib"
  ln -sfn "$DEFAULT_TOOLCHAIN" "$SHIM_DIR/XcodeDefault.xctoolchain"
elif [[ ! -x "$BUILDER" ]]; then
  echo "metal-package-builder is unavailable and no workspace shim exists" >&2
  exit 1
fi

mkdir -p "$METAL_DIR"
found=0
for package in "$COREML_DIR"/*.mlpackage; do
  [[ -e "$package" ]] || continue
  found=1
  name="$(basename "$package" .mlpackage)"
  "$BUILDER" -ml "$package" -o "$METAL_DIR/$name.mtlpackage" --mtargetos macos26.0
done

if [[ "$found" -eq 0 ]]; then
  echo "No .mlpackage files found in $COREML_DIR" >&2
  exit 1
fi
