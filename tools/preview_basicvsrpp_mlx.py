#!/usr/bin/env python3
"""Preview an experimental MLX BasicVSR++ restoration on a short video crop."""

import argparse
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from basicvsrpp_mlx_segments import BasicVSRSegments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--start", required=True, type=float, help="seconds into the source video")
    parser.add_argument("--x", required=True, type=int)
    parser.add_argument("--y", required=True, type=int)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 2 or args.frames > 8 or args.x < 0 or args.y < 0 or args.start < 0:
        parser.error("use 2-8 frames and nonnegative start/crop coordinates")
    size = 256
    command = [args.ffmpeg, "-v", "error", "-ss", str(args.start), "-i", str(args.input),
               "-vf", f"crop={size}:{size}:{args.x}:{args.y}", "-frames:v", str(args.frames),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr.decode("utf-8", "replace"))
    frame_bytes = size * size * 3
    if len(result.stdout) != args.frames * frame_bytes:
        raise SystemExit(f"expected {args.frames} frames; got {len(result.stdout) // frame_bytes}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(args.frames, size, size, 3)
    restored = BasicVSRSegments(args.archive).restore_frames(mx.array(frames[None].astype(np.float32) / 255))
    mx.eval(restored)
    output = np.clip(np.array(restored)[0] * 255, 0, 255).round().astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.frames):
        Image.fromarray(frames[index]).save(args.output / f"source-{index:02d}.png")
        Image.fromarray(output[index]).save(args.output / f"mlx-restored-{index:02d}.png")
    print(f"saved {args.frames} source/restored frame pairs to {args.output}")


if __name__ == "__main__":
    main()
