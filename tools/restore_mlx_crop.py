#!/usr/bin/env python3
"""Restore one planar-FP16 256×256 crop sequence for the Swift video pipeline."""

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from basicvsrpp_mlx_segments import BasicVSRSegments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        model = BasicVSRSegments(args.archive)
        print("READY", flush=True)
        for line in sys.stdin:
            try:
                input_path, output_path, count = line.rstrip("\n").split("\t")
                started = time.monotonic()
                restore(Path(input_path), Path(output_path), int(count), model)
                print(f"OK\t{(time.monotonic() - started) * 1000:.3f}", flush=True)
            except Exception as error:
                print(f"ERROR\t{type(error).__name__}: {error}", flush=True)
        return
    if args.input is None or args.output is None or args.frames is None:
        parser.error("--input, --output, and --frames are required outside --serve")
    restore(args.input, args.output, args.frames, BasicVSRSegments(args.archive))
    print(f"MLX restored {args.frames} crop frames", flush=True)


def restore(input_path, output_path, frame_count, model):
    if not 3 <= frame_count <= 40:
        raise ValueError("crop frame count must be 3-40")
    values = np.fromfile(input_path, dtype="<f2")
    expected = frame_count * 3 * 256 * 256
    if values.size != expected:
        raise ValueError(f"expected {expected} FP16 values; got {values.size}")
    frames = values.reshape(frame_count, 3, 256, 256).transpose(0, 2, 3, 1)
    result = model.restore_frames(mx.array(frames[None].astype(np.float32)))
    mx.eval(result)
    output = np.array(result)[0]
    if output.shape != (frame_count, 256, 256, 3) or not np.isfinite(output).all():
        raise ValueError("MLX returned invalid crop frames")
    planar = output.transpose(0, 3, 1, 2).astype("<f2")
    pending = output_path.with_name(output_path.name + f".writing-{os.getpid()}")
    try:
        planar.tofile(pending)
        os.replace(pending, output_path)
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
