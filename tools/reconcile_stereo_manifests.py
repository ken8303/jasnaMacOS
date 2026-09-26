#!/usr/bin/env python3
"""Fill completely missed stereo-eye windows from the counterpart manifest."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics


WINDOW_FRAMES = 30


def center(region):
    blend_x = region.get("blendX")
    blend_y = region.get("blendY")
    blend_width = region.get("blendWidth")
    blend_height = region.get("blendHeight")
    return (
        float(
            (region["x"] if blend_x is None else blend_x)
            + (region["width"] if blend_width is None else blend_width) / 2
        ),
        float(
            (region["y"] if blend_y is None else blend_y)
            + (region["height"] if blend_height is None else blend_height) / 2
        ),
    )


def overlaps(left, right):
    return int(left["startFrame"]) < int(right["endFrame"]) \
        and int(right["startFrame"]) < int(left["endFrame"])


def estimate_disparity(left_regions, right_regions, width, height):
    """Estimate right-minus-left pixel displacement from nearby temporal pairs."""
    candidates = []
    diagonal = math.hypot(width, height)
    for left in left_regions:
        left_center = center(left)
        nearby = []
        for right in right_regions:
            if not overlaps(left, right):
                continue
            right_center = center(right)
            distance = math.hypot(
                right_center[0] - left_center[0],
                right_center[1] - left_center[1],
            )
            if distance <= diagonal * 0.12:
                nearby.append((distance, right_center))
        if nearby:
            _, right_center = min(nearby)
            candidates.append(
                (right_center[0] - left_center[0], right_center[1] - left_center[1])
            )
    if not candidates:
        return 0, 0
    return (
        int(round(statistics.median(item[0] for item in candidates))),
        int(round(statistics.median(item[1] for item in candidates))),
    )


def regions_by_window(regions, frame_count):
    window_count = math.ceil(frame_count / WINDOW_FRAMES)
    result = [[] for _ in range(window_count)]
    for region in regions:
        start = max(0, min(frame_count, int(region["startFrame"])))
        end = max(start, min(frame_count, int(region["endFrame"])))
        if end <= start:
            continue
        for window in range(start // WINDOW_FRAMES, (end - 1) // WINDOW_FRAMES + 1):
            result[window].append(region)
    return result


def shifted_region(region, dx, dy, width, height, start_frame, end_frame):
    inferred = copy.deepcopy(region)
    inferred["startFrame"] = max(start_frame, int(region["startFrame"]))
    inferred["endFrame"] = min(end_frame, int(region["endFrame"]))
    crop_x = max(0, min(width - int(region["width"]), int(region["x"]) + dx))
    crop_y = max(0, min(height - int(region["height"]), int(region["y"]) + dy))
    applied_dx = crop_x - int(region["x"])
    applied_dy = crop_y - int(region["y"])
    inferred["x"] = crop_x
    inferred["y"] = crop_y
    blend_x = region.get("blendX")
    blend_y = region.get("blendY")
    inferred["blendX"] = int(region["x"] if blend_x is None else blend_x) + applied_dx
    inferred["blendY"] = int(region["y"] if blend_y is None else blend_y) + applied_dy
    if "maskKeyframes" in inferred:
        keyframes = [
            keyframe
            for keyframe in inferred["maskKeyframes"]
            if inferred["startFrame"] <= int(keyframe["frame"]) < inferred["endFrame"]
        ]
        if keyframes:
            inferred["maskKeyframes"] = keyframes
        else:
            inferred.pop("maskKeyframes")
    inferred["stereoInferred"] = True
    return inferred


def reconcile_manifests(left, right):
    header = ("width", "height", "frameCount", "framesPerSecond")
    mismatches = [
        f"{key} {left.get(key)!r} != {right.get(key)!r}"
        for key in header
        if left.get(key) != right.get(key)
    ]
    if mismatches:
        raise ValueError(
            "left/right manifest headers do not match: " + ", ".join(mismatches)
        )
    width = int(left["width"])
    height = int(left["height"])
    frame_count = int(left["frameCount"])
    left_regions = list(left.get("regions", []))
    right_regions = list(right.get("regions", []))
    dx, dy = estimate_disparity(left_regions, right_regions, width, height)
    left_windows = regions_by_window(left_regions, frame_count)
    right_windows = regions_by_window(right_regions, frame_count)
    inferred_left = []
    inferred_right = []
    for window, (left_active, right_active) in enumerate(
        zip(left_windows, right_windows)
    ):
        start = window * WINDOW_FRAMES
        end = min(frame_count, start + WINDOW_FRAMES)
        if left_active and not right_active:
            inferred_right.extend(
                shifted_region(region, dx, dy, width, height, start, end)
                for region in left_active
            )
        elif right_active and not left_active:
            inferred_left.extend(
                shifted_region(region, -dx, -dy, width, height, start, end)
                for region in right_active
            )
    output_left = copy.deepcopy(left)
    output_right = copy.deepcopy(right)
    output_left["regions"] = left_regions + inferred_left
    output_right["regions"] = right_regions + inferred_right
    return output_left, output_right, dx, dy, len(inferred_left), len(inferred_right)


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_manifest", type=Path)
    parser.add_argument("right_manifest", type=Path)
    parser.add_argument("output_left", type=Path)
    parser.add_argument("output_right", type=Path)
    args = parser.parse_args()
    left = json.loads(args.left_manifest.read_text(encoding="utf-8"))
    right = json.loads(args.right_manifest.read_text(encoding="utf-8"))
    reconciled = reconcile_manifests(left, right)
    output_left, output_right, dx, dy, inferred_left, inferred_right = reconciled
    write_manifest(args.output_left, output_left)
    write_manifest(args.output_right, output_right)
    print(
        f"Stereo manifest reconciliation: disparity {dx:+d}/{dy:+d}px, "
        f"inferred left/right regions {inferred_left}/{inferred_right}"
    )


if __name__ == "__main__":
    main()
