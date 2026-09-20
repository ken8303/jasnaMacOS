#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${1:-$ROOT_DIR/dist}"
APP_NAME="Jasna VR Restoration.app"
EXPECTED_RFDETR_VERSION="1.10.0"
BUILD_DIR="$(mktemp -d /private/tmp/jasna-app-build.XXXXXX)"
MODULE_CACHE_DIR="$BUILD_DIR/module-cache"
APP_PATH="$BUILD_DIR/$APP_NAME"
ZIP_PATH="$DIST_DIR/Jasna-VR-Restoration-macOS27-Test.zip"
STAGED_ZIP="$BUILD_DIR/Jasna-VR-Restoration-macOS27-Test.zip"

cleanup() {
  [[ "$BUILD_DIR" == /private/tmp/jasna-app-build.* ]] || return 0
  /bin/rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

mkdir -p "$MODULE_CACHE_DIR"
export CLANG_MODULE_CACHE_PATH="$MODULE_CACHE_DIR"
export SWIFTPM_MODULECACHE_OVERRIDE="$MODULE_CACHE_DIR"

for required in \
  "$ROOT_DIR/Models/MetalML/feature_extract.mtlpackage" \
  "$ROOT_DIR/Models/MetalMLBatch2/feature_extract.mtlpackage" \
  "$ROOT_DIR/Models/DeformConv/manifest.json" \
  "$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt" \
  "$ROOT_DIR/.venv-rfdetr/lib/python3.13/site-packages"; do
  [[ -e "$required" ]] || {
    echo "error: required runtime component is missing: $required" >&2
    exit 1
  }
done
INSTALLED_RFDETR_VERSION="$("$ROOT_DIR/.venv-rfdetr/bin/python" -c \
  'import importlib.metadata; print(importlib.metadata.version("rfdetr"))')"
[[ "$INSTALLED_RFDETR_VERSION" == "$EXPECTED_RFDETR_VERSION" ]] || {
  echo "error: packaging requires RF-DETR $EXPECTED_RFDETR_VERSION; installed $INSTALLED_RFDETR_VERSION" >&2
  echo "run script/setup_rfdetr_detector.sh before packaging" >&2
  exit 1
}
echo "Bundling validated RF-DETR $INSTALLED_RFDETR_VERSION"
BREW_PYTHON="/opt/homebrew/opt/python@3.13/bin/python3.13"
[[ -x "$BREW_PYTHON" ]] || {
  echo "error: packaging requires Homebrew python@3.13: $BREW_PYTHON" >&2
  exit 1
}
PYTHON_FRAMEWORK_VERSION="$(/usr/bin/python3 -c \
  'import os,sys; print(os.path.realpath(sys.argv[1]))' \
  '/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13')"
FFMPEG_BINARY="$(/usr/bin/python3 -c \
  'import os,sys; print(os.path.realpath(sys.argv[1]))' "$(command -v ffmpeg)")"
FFPROBE_BINARY="$(/usr/bin/python3 -c \
  'import os,sys; print(os.path.realpath(sys.argv[1]))' "$(command -v ffprobe)")"
[[ -d "$PYTHON_FRAMEWORK_VERSION" && -x "$FFMPEG_BINARY" && -x "$FFPROBE_BINARY" ]] || {
  echo "error: packaging requires the local Python framework, FFmpeg, and FFprobe" >&2
  exit 1
}

echo "Building optimized macOS 27 executables"
swift build --package-path "$ROOT_DIR" --scratch-path "$BUILD_DIR" \
  --disable-sandbox -c release --product JasnaMacApp
swift build --package-path "$ROOT_DIR" --scratch-path "$BUILD_DIR" \
  --disable-sandbox -c release --product JasnaMetalPoC
BIN_DIR="$(swift build --package-path "$ROOT_DIR" --scratch-path "$BUILD_DIR" \
  --disable-sandbox -c release --show-bin-path)"

mkdir -p "$DIST_DIR"
PREVIOUS_ZIP_PATH="$DIST_DIR/Jasna-VR-Restoration-macOS27-Test.previous.zip"
# A synced distribution directory can attach Finder metadata to a loose app
# after signing and invalidate it. Publish ZIPs only, retaining one previous
# archive, and remove loose apps left by older packagers.
find "$DIST_DIR" -maxdepth 1 -type d \
  -name 'Jasna VR Restoration.previous-*.app' -exec /bin/rm -rf {} +
find "$DIST_DIR" -maxdepth 1 -type d \
  \( -name "$APP_NAME" -o -name 'Jasna VR Restoration.previous.app' \) \
  -exec /bin/rm -rf {} +
find "$DIST_DIR" -maxdepth 1 -type f \
  -name 'Jasna-VR-Restoration-macOS27-Test.zip.previous-*' -delete
if [[ -e "$ZIP_PATH" ]]; then
  /bin/rm -f "$PREVIOUS_ZIP_PATH"
  mv "$ZIP_PATH" "$PREVIOUS_ZIP_PATH"
fi

CONTENTS="$APP_PATH/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
RUNTIME="$RESOURCES/Runtime"
FRAMEWORKS="$CONTENTS/Frameworks"
mkdir -p "$MACOS_DIR" "$FRAMEWORKS" "$RUNTIME/bin" "$RUNTIME/Models/MosaicDetection"

/usr/bin/ditto --noextattr --noqtn "$BIN_DIR/JasnaMacApp" "$MACOS_DIR/JasnaMacApp"
/usr/bin/ditto --noextattr --noqtn "$BIN_DIR/JasnaMetalPoC" "$MACOS_DIR/JasnaMetalPoC"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Packaging/Info.plist" "$CONTENTS/Info.plist"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Packaging/TESTING.md" "$RESOURCES/TESTING.md"
/usr/bin/ditto --noextattr --noqtn \
  "$ROOT_DIR/Packaging/THIRD_PARTY_NOTICES.md" "$RESOURCES/THIRD_PARTY_NOTICES.md"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Package.swift" "$RUNTIME/Package.swift"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Sources/JasnaMetalPoC" "$RUNTIME/Sources/JasnaMetalPoC"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/script" "$RUNTIME/script"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/tools" "$RUNTIME/tools"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Models/MetalML" "$RUNTIME/Models/MetalML"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Models/MetalMLBatch2" "$RUNTIME/Models/MetalMLBatch2"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/Models/DeformConv" "$RUNTIME/Models/DeformConv"
/usr/bin/ditto --noextattr --noqtn \
  "$ROOT_DIR/Models/MosaicDetection/rfdetr-vr-v1.pt" \
  "$RUNTIME/Models/MosaicDetection/rfdetr-vr-v1.pt"
/usr/bin/ditto --noextattr --noqtn "$ROOT_DIR/.venv-rfdetr" "$RUNTIME/.venv-rfdetr"
/usr/bin/ditto --noextattr --noqtn "$PYTHON_FRAMEWORK_VERSION" "$RUNTIME/python"
/bin/rm -f \
  "$RUNTIME/python/lib/python3.13/site-packages"
# Framework Python launches this helper relative to the loaded Python library.
# The relocator places that library in RuntimeLibraries, so preserve the helper
# at the corresponding location as well as in the standalone Python home.
mkdir -p "$FRAMEWORKS/RuntimeLibraries"
/usr/bin/ditto --noextattr --noqtn \
  "$RUNTIME/python/Resources/Python.app" \
  "$FRAMEWORKS/RuntimeLibraries/Resources/Python.app"
/usr/bin/ditto --noextattr --noqtn "$FFMPEG_BINARY" "$RUNTIME/bin/ffmpeg"
/usr/bin/ditto --noextattr --noqtn "$FFPROBE_BINARY" "$RUNTIME/bin/ffprobe"
/bin/rm -f "$RUNTIME/.venv-rfdetr/bin/python" \
  "$RUNTIME/.venv-rfdetr/bin/python3" \
  "$RUNTIME/.venv-rfdetr/bin/python3.13"
/usr/bin/ditto --noextattr --noqtn \
  "$ROOT_DIR/Packaging/python-launcher.sh" "$RUNTIME/.venv-rfdetr/bin/python"
/bin/ln -s python "$RUNTIME/.venv-rfdetr/bin/python3"
/bin/ln -s python "$RUNTIME/.venv-rfdetr/bin/python3.13"

chmod 755 "$MACOS_DIR/JasnaMacApp" "$MACOS_DIR/JasnaMetalPoC" \
  "$RUNTIME/bin/ffmpeg" "$RUNTIME/bin/ffprobe" \
  "$RUNTIME/.venv-rfdetr/bin/python"
find "$RUNTIME/script" -type f -name '*.sh' -exec chmod 755 {} +
plutil -lint "$CONTENTS/Info.plist"

echo "Relocating bundled Python and FFmpeg libraries"
/usr/bin/python3 "$ROOT_DIR/tools/relocate_macos_bundle.py" \
  --app "$APP_PATH" --sign-identity -
find "$APP_PATH" -type f -exec chmod u+w {} +

echo "Applying an ad-hoc test signature"
xattr -cr "$APP_PATH"
/usr/bin/codesign --force --deep --sign - --timestamp=none "$APP_PATH"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_PATH"

echo "Creating shareable ZIP archive"
/usr/bin/ditto -c -k --norsrc --noextattr --noqtn --keepParent \
  "$APP_PATH" "$STAGED_ZIP"
if /usr/bin/unzip -Z1 "$STAGED_ZIP" | /usr/bin/grep -Eq '(^|/)\._'; then
  echo "error: packaged archive contains disallowed AppleDouble metadata" >&2
  exit 1
fi
/usr/bin/ditto --noextattr --noqtn "$STAGED_ZIP" "$ZIP_PATH"
/usr/bin/unzip -tq "$ZIP_PATH"

APP_SIZE="$(du -sh "$APP_PATH" | awk '{print $1}')"
ZIP_SIZE="$(du -sh "$ZIP_PATH" | awk '{print $1}')"
echo "Bundled app size:  $APP_SIZE"
echo "Share:             $ZIP_PATH ($ZIP_SIZE)"
echo "Target prerequisites: Apple silicon and macOS 27 (no Homebrew required)"
