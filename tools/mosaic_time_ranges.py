#!/usr/bin/env python3
"""Normalize user mosaic time ranges onto a selected clip and its segments."""

from __future__ import annotations

import argparse
import math


def parse_time(value):
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"invalid time: {value}")
    numbers = [float(part) for part in parts]
    if any(number < 0 for number in numbers):
        raise ValueError(f"negative time: {value}")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def parse_user_ranges(spec):
    ranges = []
    for item in spec.split(","):
        parts = item.strip().split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid range: {item}")
        start, end = map(parse_time, parts)
        if end <= start:
            raise ValueError(f"range end must follow start: {item}")
        ranges.append((start, end))
    return merge_ranges(ranges)


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def selected_ranges(spec, selection_start, selection_duration=None):
    clip_end = math.inf if selection_duration is None else selection_start + selection_duration
    result = []
    for start, end in parse_user_ranges(spec):
        clipped_start = max(start, selection_start)
        clipped_end = min(end, clip_end)
        if clipped_end > clipped_start:
            result.append((clipped_start - selection_start, clipped_end - selection_start))
    return merge_ranges(result)


def segment_ranges(ranges, offset, duration):
    result = []
    for start, end in ranges:
        clipped_start = max(start, offset)
        clipped_end = min(end, offset + duration)
        if clipped_end > clipped_start:
            result.append((clipped_start - offset, clipped_end - offset))
    return result


def active_segment_indices(ranges, duration, segment_duration):
    """Return timeline segment indices that intersect at least one active range."""
    if duration <= 0 or segment_duration <= 0:
        return []
    segment_count = int(math.ceil(duration / segment_duration))
    return [
        index
        for index in range(segment_count)
        if segment_ranges(
            ranges,
            index * segment_duration,
            min(segment_duration, duration - index * segment_duration),
        )
    ]


def encode_internal(ranges):
    return ",".join(f"{start:.6f}/{end:.6f}" for start, end in ranges)


def decode_internal(spec):
    if not spec:
        return []
    return [(float(a), float(b)) for a, b in (item.split("/") for item in spec.split(","))]


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("ranges")
    normalize.add_argument("selection_start")
    normalize.add_argument("selection_duration")
    segment = subparsers.add_parser("segment")
    segment.add_argument("ranges")
    segment.add_argument("offset", type=float)
    segment.add_argument("duration", type=float)
    indices = subparsers.add_parser("indices")
    indices.add_argument("ranges")
    indices.add_argument("duration", type=float)
    indices.add_argument("segment_duration", type=float)
    args = parser.parse_args()
    if args.command == "normalize":
        duration = None if args.selection_duration == "full" else float(args.selection_duration)
        print(encode_internal(selected_ranges(
            args.ranges, parse_time(args.selection_start), duration
        )))
    elif args.command == "segment":
        print(encode_internal(segment_ranges(
            decode_internal(args.ranges), args.offset, args.duration
        )))
    else:
        print(" ".join(str(index) for index in active_segment_indices(
            decode_internal(args.ranges), args.duration, args.segment_duration
        )))


if __name__ == "__main__":
    main()
