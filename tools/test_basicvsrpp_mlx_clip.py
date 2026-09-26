#!/usr/bin/env python3
"""Restore a short 256-pixel crop and encode source|MLX comparison video."""

import argparse
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from basicvsrpp_mlx_segments import BasicVSRSegments


def run_command(command, input_bytes=None):
    result = subprocess.run(command, input=input_bytes, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--start-frame", type=int, default=198)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--x", type=int, default=608)
    parser.add_argument("--y", type=int, default=989)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.start_frame < 0 or not 2 <= args.frames <= 30 or args.x < 0 or args.y < 0:
        parser.error("use 2-30 frames and nonnegative frame/crop coordinates")
    end = args.start_frame + args.frames - 1
    crop = f"select=between(n\\,{args.start_frame}\\,{end}),crop=256:256:{args.x}:{args.y}"
    raw = run_command([args.ffmpeg, "-v", "error", "-i", str(args.input),
                       "-vf", crop, "-vsync", "0", "-frames:v", str(args.frames),
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frame_bytes = 256 * 256 * 3
    if len(raw) != frame_bytes * args.frames:
        raise RuntimeError(f"expected {args.frames} frames; decoded {len(raw) // frame_bytes}")
    source = np.frombuffer(raw, dtype=np.uint8).reshape(args.frames, 256, 256, 3)
    model = BasicVSRSegments(args.archive)
    start = time.monotonic()
    restored = model.restore_frames(mx.array(source[None].astype(np.float32) / 255))
    mx.eval(restored)
    seconds = time.monotonic() - start
    output = np.clip(np.array(restored)[0] * 255, 0, 255).round().astype(np.uint8)
    comparison = np.concatenate((source, output), axis=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_command([args.ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "512x256", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
                 "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags",
                 "+faststart", str(args.output)], comparison.tobytes())
    print(f"source left | MLX right: {args.output}")
    print(f"{args.frames} frames restored in {seconds:.2f}s; output {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
