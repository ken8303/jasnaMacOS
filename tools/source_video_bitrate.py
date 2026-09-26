#!/usr/bin/env python3
"""Measure the average video bitrate of a source file for output encoding."""

import argparse
import json
import math
import subprocess
import sys


def positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def video_bitrate(probe):
    streams = probe.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not videos:
        raise ValueError("source has no video stream")
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_rates = [positive_number(stream.get("bit_rate")) for stream in audio]
    metadata_video_rate = positive_number(videos[0].get("bit_rate"))
    container = probe.get("format", {})
    size = positive_number(container.get("size"))
    duration = positive_number(container.get("duration"))
    total_rate = size * 8 / duration if size and duration else positive_number(container.get("bit_rate"))
    if total_rate and all(rate is not None for rate in audio_rates):
        measured_video_rate = total_rate - sum(audio_rates)
        if measured_video_rate > 0:
            return round(measured_video_rate)
    if metadata_video_rate:
        return round(metadata_video_rate)
    if total_rate and not audio:
        return round(total_rate)
    raise ValueError("cannot determine video bitrate separately from audio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffprobe", help="path to ffprobe")
    parser.add_argument("source", help="source video")
    parser.add_argument("--eye", action="store_true", help="half the SBS bitrate for one eye")
    args = parser.parse_args()
    result = subprocess.run(
        [args.ffprobe, "-v", "error", "-show_entries",
         "format=size,duration,bit_rate:stream=codec_type,bit_rate", "-of", "json", args.source],
        check=True, capture_output=True, text=True,
    )
    bitrate = video_bitrate(json.loads(result.stdout))
    print(max(1, bitrate // 2) if args.eye else bitrate)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: unable to measure source video bitrate: {error}", file=sys.stderr)
        raise SystemExit(1)
