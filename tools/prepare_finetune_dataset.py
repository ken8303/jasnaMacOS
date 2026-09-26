#!/usr/bin/env python3
"""Extract deterministic clean 256x256 video clips for Jasna fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract clean frame sequences. Fine-tuning creates synthetic mosaics "
            "from these clips at training time; do not use already-mosaicked video."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips-per-video", type=int, default=20)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--source-crop-size",
        type=int,
        default=0,
        help=(
            "square source area scaled to --size; use about 1024 for 8K VR "
            "to match large restoration regions (default: --size)"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=8303)
    parser.add_argument(
        "--exclude-ranges",
        default="",
        help=(
            "comma-separated source ranges that may contain mosaics, for example "
            "00:25:57-00:26:27"
        ),
    )
    parser.add_argument(
        "--exclude-padding",
        type=float,
        default=2.0,
        help="extra clean guard in seconds on both sides of every excluded range",
    )
    parser.add_argument(
        "--sbs-eyes",
        action="store_true",
        help="sample within one half of exact side-by-side source frames",
    )
    return parser.parse_args()


def discover_videos(inputs: list[Path]) -> list[Path]:
    videos: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
            videos.add(item.resolve())
        elif item.is_dir():
            videos.update(
                path.resolve()
                for path in item.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            )
        else:
            raise ValueError(f"input is not a supported video or directory: {item}")
    if not videos:
        raise ValueError("no supported clean videos found")
    return sorted(videos)


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(payload["format"]["duration"]),
    }


def parse_timestamp(value: str) -> float:
    fields = value.strip().split(":")
    if not fields or len(fields) > 3:
        raise ValueError(f"invalid timestamp: {value}")
    try:
        numbers = [float(field) for field in fields]
    except ValueError as error:
        raise ValueError(f"invalid timestamp: {value}") from error
    if any(number < 0 for number in numbers):
        raise ValueError(f"negative timestamp: {value}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def parse_excluded_ranges(value: str) -> list[tuple[float, float]]:
    if not value.strip():
        return []
    ranges = []
    for item in value.split(","):
        fields = item.strip().split("-")
        if len(fields) != 2:
            raise ValueError(f"invalid excluded range: {item}")
        start, end = (parse_timestamp(field) for field in fields)
        if end <= start:
            raise ValueError(f"excluded range must end after it starts: {item}")
        ranges.append((start, end))
    return sorted(ranges)


def allowed_start_intervals(
    duration: float,
    clip_seconds: float,
    excluded: list[tuple[float, float]],
    padding: float,
) -> list[tuple[float, float]]:
    latest_start = duration - clip_seconds
    if latest_start < 0:
        return []
    blocked = []
    for start, end in excluded:
        start = max(0.0, start - padding)
        end = min(duration, end + padding)
        if end > start:
            blocked.append((start, end))
    merged = []
    for start, end in sorted(blocked):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    allowed = []
    cursor = 0.0
    for start, end in merged:
        last_start_before_block = min(latest_start, start - clip_seconds)
        if last_start_before_block >= cursor:
            allowed.append((cursor, last_start_before_block))
        cursor = max(cursor, end)
    if cursor <= latest_start:
        allowed.append((cursor, latest_start))
    return allowed


def choose_interval_start(
    intervals: list[tuple[float, float]], rng: random.Random
) -> float:
    if not intervals:
        raise ValueError("no clean interval is long enough for one training clip")
    lengths = [max(0.0, end - start) for start, end in intervals]
    total = sum(lengths)
    if total == 0:
        return intervals[rng.randrange(len(intervals))][0]
    position = rng.random() * total
    for (start, end), length in zip(intervals, lengths):
        if position <= length:
            return start + position
        position -= length
    return intervals[-1][1]


def choose_sample(
    metadata: dict,
    frames: int,
    fps: float,
    size: int,
    sbs_eyes: bool,
    rng: random.Random,
    start_intervals: list[tuple[float, float]] | None = None,
    source_crop_size: int | None = None,
) -> dict:
    width = metadata["width"]
    height = metadata["height"]
    eye = None
    crop_width = width
    crop_offset = 0
    if sbs_eyes:
        if width % 2:
            raise ValueError("--sbs-eyes requires an even source width")
        crop_width = width // 2
        eye = rng.choice(("left", "right"))
        crop_offset = 0 if eye == "left" else crop_width
    source_size = source_crop_size or size
    if crop_width < source_size or height < source_size:
        raise ValueError(
            f"source area {crop_width}x{height} is smaller than "
            f"{source_size}x{source_size}"
        )
    clip_seconds = frames / fps
    if start_intervals is None:
        start_intervals = allowed_start_intervals(
            metadata["duration"], clip_seconds, [], 0.0
        )
    return {
        "start": choose_interval_start(start_intervals, rng),
        "x": crop_offset + rng.randrange(0, crop_width - source_size + 1),
        "y": rng.randrange(0, height - source_size + 1),
        "eye": eye,
        "sourceSize": source_size,
    }


def extraction_command(
    source: Path,
    destination: Path,
    sample: dict,
    frames: int,
    fps: float,
    size: int,
) -> list[str]:
    source_size = sample.get("sourceSize", size)
    video_filter = (
        f"fps={fps:.8g},crop={source_size}:{source_size}:"
        f"{sample['x']}:{sample['y']}"
    )
    if source_size != size:
        video_filter += f",scale={size}:{size}:flags=lanczos"
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{sample['start']:.6f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-frames:v",
        str(frames),
        "-fps_mode",
        "passthrough",
        str(destination / "frame-%04d.png"),
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.clips_per_video <= 0:
        raise ValueError("--clips-per-video must be positive")
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.size < 256 or args.size % 4:
        raise ValueError("--size must be at least 256 and divisible by four")
    if args.source_crop_size and args.source_crop_size < args.size:
        raise ValueError("--source-crop-size cannot be smaller than --size")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.exclude_padding < 0:
        raise ValueError("--exclude-padding cannot be negative")
    parse_excluded_ranges(args.exclude_ranges)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("ffmpeg and ffprobe must be available on PATH")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_split_order(
    clips_per_video: int,
    validation_fraction: float,
    rng: random.Random,
) -> list[str]:
    """Keep every source represented in validation when isolation is possible."""
    if clips_per_video < 2:
        raise ValueError(
            "--clips-per-video must be at least two so every source can have "
            "separate train and validation clips"
        )
    validation_count = max(1, round(clips_per_video * validation_fraction))
    validation_count = min(validation_count, clips_per_video - 1)
    splits = ["validation"] * validation_count + ["train"] * (
        clips_per_video - validation_count
    )
    rng.shuffle(splits)
    return splits


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        videos = discover_videos(args.inputs)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f"error: output path is not a directory: {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"error: output dataset is not empty: {args.output}\n"
            "choose a new directory so an existing dataset is never overwritten"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    excluded_ranges = parse_excluded_ranges(args.exclude_ranges)
    source_crop_size = args.source_crop_size or args.size
    requested = len(videos) * args.clips_per_video

    clips = []
    clip_index = 0
    for source in videos:
        try:
            split_order = source_split_order(
                args.clips_per_video,
                args.validation_fraction,
                rng,
            )
        except ValueError as error:
            raise SystemExit(f"error: {error}") from error
        metadata = probe_video(source)
        start_intervals = allowed_start_intervals(
            metadata["duration"],
            args.frames / args.fps,
            excluded_ranges,
            args.exclude_padding,
        )
        if not start_intervals:
            raise SystemExit(
                f"error: no clean interval in {source} is long enough after exclusions"
            )
        for source_clip_index in range(args.clips_per_video):
            sample = choose_sample(
                metadata,
                args.frames,
                args.fps,
                args.size,
                args.sbs_eyes,
                rng,
                start_intervals,
                source_crop_size,
            )
            split = split_order[source_clip_index]
            name = f"clip-{clip_index:06d}"
            destination = args.output / split / name
            destination.mkdir(parents=True)
            command = extraction_command(
                source, destination, sample, args.frames, args.fps, args.size
            )
            subprocess.run(command, check=True)
            frame_paths = sorted(destination.glob("frame-*.png"))
            if len(frame_paths) != args.frames:
                raise SystemExit(
                    f"error: {name} produced {len(frame_paths)} of {args.frames} frames"
                )
            clips.append(
                {
                    "name": name,
                    "split": split,
                    "source": str(source),
                    "startSeconds": round(sample["start"], 6),
                    "sourceCrop": [
                        sample["x"],
                        sample["y"],
                        source_crop_size,
                        source_crop_size,
                    ],
                    "outputSize": args.size,
                    "eye": sample["eye"],
                }
            )
            clip_index += 1
            print(f"Prepared {clip_index}/{requested}: {split}/{name}", flush=True)

    write_json_atomic(
        args.output / "dataset.json",
        {
            "version": 1,
            "seed": args.seed,
            "frames": args.frames,
            "fps": args.fps,
            "size": args.size,
            "sourceCropSize": source_crop_size,
            "syntheticMosaicAtTrainingTime": True,
            "excludedRanges": args.exclude_ranges,
            "excludePaddingSeconds": args.exclude_padding,
            "clips": clips,
        },
    )
    validation_count = sum(clip["split"] == "validation" for clip in clips)
    print(
        f"Dataset ready: {requested - validation_count} train and "
        f"{validation_count} validation clips in {args.output}"
    )


if __name__ == "__main__":
    main()
