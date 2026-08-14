#!/usr/bin/env python3
"""Report the active/empty run beginning at one temporal manifest window."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


WINDOW_FRAMES = 30


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def combined_window_activity(manifests):
    frame_counts = {int(manifest["frameCount"]) for manifest in manifests}
    if len(frame_counts) != 1:
        raise ValueError("left/right manifests have different frame counts")
    frame_count = frame_counts.pop()
    window_count = math.ceil(frame_count / WINDOW_FRAMES)
    active = [False] * window_count
    for manifest in manifests:
        for region in manifest.get("regions", []):
            start = max(0, min(frame_count, int(region.get("startFrame", 0))))
            end = max(start, min(frame_count, int(region.get("endFrame", start))))
            if end <= start:
                continue
            first_window = start // WINDOW_FRAMES
            last_window = (end - 1) // WINDOW_FRAMES
            for window in range(first_window, last_window + 1):
                active[window] = True
    return active


def activity_run(active, start_window):
    if start_window < 0 or start_window >= len(active):
        raise ValueError("start window is outside the manifest")
    state = active[start_window]
    end_window = start_window + 1
    while end_window < len(active) and active[end_window] == state:
        end_window += 1
    return ("active" if state else "empty", end_window - start_window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_manifest", type=Path)
    parser.add_argument("right_manifest", type=Path)
    parser.add_argument("start_window", type=int)
    args = parser.parse_args()
    active = combined_window_activity(
        [load_manifest(args.left_manifest), load_manifest(args.right_manifest)]
    )
    state, count = activity_run(active, args.start_window)
    print(f"{state}\t{count}")


if __name__ == "__main__":
    main()
