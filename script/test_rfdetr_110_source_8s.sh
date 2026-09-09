#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: ./script/test_rfdetr_110_source_8s.sh INPUT_SBS_VIDEO [OUTPUT_DIRECTORY] [START_TIME]

Creates one short 30 fps Main 8 SBS fixture, then compares the installed
RF-DETR with RF-DETR 1.10.0 in ABBA order. Only detection is measured;
restoration, compositing, audio, and final-video encoding are disabled.

START_TIME accepts FFmpeg time syntax such as 00:19:50, 19:50, or seconds.

Optional environment variables:
  JASNA_RFDETR_TEST_SECONDS=8       fixture/detection duration
  JASNA_RFDETR_CANDIDATE=1.10.0     isolated candidate package
  JASNA_RFDETR_AB_ROUNDS=2          alternating rounds; 2 produces ABBA
  JASNA_RFDETR_FIXTURE_BITRATE=40M  short 8K Main 8 fixture bitrate
  JASNA_ALLOW_CPU_DETECTOR_TEST=0   allow CPU only when MPS is unavailable
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 3 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_VIDEO="$1"
TEST_SECONDS="${JASNA_RFDETR_TEST_SECONDS:-8}"
CANDIDATE_VERSION="${JASNA_RFDETR_CANDIDATE:-1.10.0}"
START_TIME="${3:-0}"
FIXTURE_BITRATE="${JASNA_RFDETR_FIXTURE_BITRATE:-40M}"
RUN_ID="$(date -u '+%Y%m%d-%H%M%S')"

[[ -f "$INPUT_VIDEO" ]] || {
  echo "error: input SBS video not found: $INPUT_VIDEO" >&2
  exit 1
}
INPUT_VIDEO="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)/$(basename "$INPUT_VIDEO")"
OUTPUT_ROOT="${2:-${INPUT_VIDEO%.*}.rfdetr-${CANDIDATE_VERSION}-${TEST_SECONDS}s-${RUN_ID}}"

/usr/bin/awk -v value="$TEST_SECONDS" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' || {
  echo "error: JASNA_RFDETR_TEST_SECONDS must be greater than zero" >&2
  exit 1
}
[[ -n "$START_TIME" ]] || {
  echo "error: START_TIME cannot be empty" >&2
  exit 1
}
[[ "$CANDIDATE_VERSION" =~ ^[0-9]+([.][0-9]+){1,2}([a-zA-Z0-9.-]+)?$ ]] || {
  echo "error: invalid RF-DETR candidate version: $CANDIDATE_VERSION" >&2
  exit 1
}
[[ "$FIXTURE_BITRATE" =~ ^[0-9]+([kKmM])?$ ]] || {
  echo "error: JASNA_RFDETR_FIXTURE_BITRATE must look like 40000000 or 40M" >&2
  exit 1
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "error: test output already exists: $OUTPUT_ROOT" >&2
  echo "use a new output directory to keep the comparison independent" >&2
  exit 1
}

FFMPEG_PATH="$(command -v ffmpeg || true)"
FFPROBE_PATH="$(command -v ffprobe || true)"
[[ -n "$FFMPEG_PATH" && -n "$FFPROBE_PATH" ]] || {
  echo "error: ffmpeg and ffprobe are required" >&2
  exit 1
}

REFERENCE_WORK="$OUTPUT_ROOT/fixture-reference.jasna-restore-only-work"
FIXTURE_VIDEO="$OUTPUT_ROOT/fixture-30fps-main8.mp4"
RESULTS_DIR="$OUTPUT_ROOT/results"
mkdir -p "$REFERENCE_WORK/segments"
MASTER_LOG="$OUTPUT_ROOT/source-detector-ab.log"
exec > >(/usr/bin/tee "$MASTER_LOG") 2>&1

echo "===== Jasna source-video RF-DETR detection A/B $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
echo "Input SBS:     $INPUT_VIDEO"
echo "Start:         $START_TIME"
echo "Duration:      ${TEST_SECONDS}s"
echo "Candidate:     RF-DETR $CANDIDATE_VERSION"
echo "Output:        $OUTPUT_ROOT"
echo "Disabled:      restoration, compositing, audio, and final-video encoding"
echo
echo "Preparing one shared 30 fps HEVC Main 8 detector fixture"
FIXTURE_STARTED=$SECONDS
"$FFMPEG_PATH" -hide_banner -y \
  -ss "$START_TIME" -i "$INPUT_VIDEO" -t "$TEST_SECONDS" \
  -map 0:v:0 -an \
  -vf "fps=30,format=yuv420p" \
  -c:v hevc_videotoolbox -profile:v main -b:v "$FIXTURE_BITRATE" \
  -tag:v hvc1 -movflags +faststart \
  "$FIXTURE_VIDEO"
echo "Fixture preparation wall time: $((SECONDS - FIXTURE_STARTED))s (excluded from detector timings)"

FIXTURE_INFO="$($FFPROBE_PATH -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_frames \
  -of csv=p=0 "$FIXTURE_VIDEO")"
IFS=',' read -r FIXTURE_CODEC FIXTURE_PROFILE FIXTURE_WIDTH FIXTURE_HEIGHT \
  FIXTURE_PIXEL_FORMAT FIXTURE_RATE FIXTURE_FRAMES <<<"$FIXTURE_INFO"
EXPECTED_FRAMES="$(/usr/bin/awk -v seconds="$TEST_SECONDS" \
  'BEGIN { printf "%d", int(seconds * 30 + 0.5) }')"
[[ "$FIXTURE_CODEC" == "hevc" \
  && "$FIXTURE_PROFILE" == "Main" \
  && "$FIXTURE_PIXEL_FORMAT" == "yuv420p" \
  && "$FIXTURE_WIDTH" =~ ^[0-9]+$ \
  && "$FIXTURE_HEIGHT" =~ ^[0-9]+$ \
  && $((FIXTURE_WIDTH % 2)) -eq 0 \
  && "$FIXTURE_RATE" == "30/1" \
  && "$FIXTURE_FRAMES" == "$EXPECTED_FRAMES" ]] || {
  echo "error: detector fixture failed format/frame validation: $FIXTURE_INFO" >&2
  exit 1
}
echo "Fixture validated: $FIXTURE_WIDTH x $FIXTURE_HEIGHT, $FIXTURE_FRAMES frames, $FIXTURE_RATE, HEVC Main 8"

# Reuse the existing balanced A/B runner without requiring old restoration
# output. It resolves only this newly prepared source path from the synthetic
# reference log; no restoration cache payload is created or consumed.
/usr/bin/printf 'Direct SBS batch job 1/1: %s + %s\n' \
  "$FIXTURE_VIDEO" "$FIXTURE_VIDEO" \
  > "$REFERENCE_WORK/segments/source.jasna.log"

export JASNA_RFDETR_TEST_SECONDS="$TEST_SECONDS"
export JASNA_RFDETR_CANDIDATE="$CANDIDATE_VERSION"
"$ROOT_DIR/script/test_rfdetr_version_8s.sh" "$REFERENCE_WORK" "$RESULTS_DIR"

echo
echo "Source-video detection-only A/B: COMPLETE"
echo "Comparison log: $RESULTS_DIR/detector-ab.log"
echo "Master log:     $MASTER_LOG"
echo "Fixture:        $FIXTURE_VIDEO"
