#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COREML_DIR="${1:-$ROOT_DIR/Models/CoreML}"
METAL_DIR="${2:-$ROOT_DIR/Models/MetalML}"
SHIM_DIR="$ROOT_DIR/work/metal-toolchain-shim"

BUILDER="$SHIM_DIR/Metal.xctoolchain/usr/bin/metal-package-builder"
if [[ -n "${JASNA_METAL_TOOLCHAIN_ROOT:-}" ]]; then
  METAL_TOOLCHAIN="${JASNA_METAL_TOOLCHAIN_ROOT%/}"
  BUILDER_PATH="$METAL_TOOLCHAIN/usr/bin/metal-package-builder"
  [[ -x "$BUILDER_PATH" ]] || {
    echo "metal-package-builder is unavailable under JASNA_METAL_TOOLCHAIN_ROOT: $METAL_TOOLCHAIN" >&2
    exit 1
  }
elif BUILDER_PATH="$(xcrun --find metal-package-builder 2>/dev/null)"; then
  METAL_TOOLCHAIN="$(cd "$(dirname "$BUILDER_PATH")/../.." && pwd)"
fi

if [[ -n "${METAL_TOOLCHAIN:-}" ]]; then
  DEFAULT_TOOLCHAIN="$(xcrun --toolchain XcodeDefault --find coremlcompiler)"
  DEFAULT_TOOLCHAIN="$(cd "$(dirname "$DEFAULT_TOOLCHAIN")/../.." && pwd)"

  # Xcode 27's package builder looks for XcodeDefault.xctoolchain beside its
  # downloaded Metal.xctoolchain. Recreate that expected layout locally so no
  # installed Xcode files need to be modified. Recopy the executable and update
  # both runtime links so a retained shim cannot mix different beta toolchains.
  mkdir -p "$SHIM_DIR/Metal.xctoolchain/usr/bin"
  cp "$BUILDER_PATH" "$BUILDER"
  ln -sfn "$METAL_TOOLCHAIN/System" "$SHIM_DIR/Metal.xctoolchain/System"
  ln -sfn "$METAL_TOOLCHAIN/usr/lib" "$SHIM_DIR/Metal.xctoolchain/usr/lib"
  ln -sfn "$DEFAULT_TOOLCHAIN" "$SHIM_DIR/XcodeDefault.xctoolchain"
else
  echo "a matching metal-package-builder is unavailable" >&2
  echo "install Xcode's MetalToolchain component, or set JASNA_METAL_TOOLCHAIN_ROOT" >&2
  echo "refusing to reuse a shim from a different Xcode beta" >&2
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
