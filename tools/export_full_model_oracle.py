#!/usr/bin/env python3
"""Export an independent three-frame BasicVSR++ oracle for the Metal graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def synthetic_frame(frame_index: int) -> np.ndarray:
    count = 3 * 256 * 256
    indices = np.arange(count, dtype=np.int64)
    values = (
        (indices * (29 + frame_index * 6) + 17 + frame_index * 23) % 1021
    ).astype(np.float32) / np.float32(1020.0)
    # The Metal probe stores its input as FP16 before either feature extraction
    # or bicubic flow preprocessing. Match that conversion before PyTorch runs.
    return values.astype(np.float16).astype(np.float32).reshape(3, 256, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    generator = generator.cpu().eval()
    if args.frames < 3:
        raise SystemExit("--frames must be at least 3")
    frames = torch.from_numpy(
        np.stack([synthetic_frame(index) for index in range(args.frames)])
    )[None]
    with torch.inference_mode():
        restored = generator(frames).detach().cpu().numpy()[0]

    if restored.shape != (args.frames, 3, 256, 256):
        raise SystemExit(f"unexpected restored shape: {restored.shape}")
    args.output.mkdir(parents=True, exist_ok=True)
    fp32 = restored.astype("<f4", copy=False)
    fp16 = restored.astype("<f2", copy=False)
    (args.output / "restored.f32").write_bytes(fp32.tobytes(order="C"))
    (args.output / "restored.f16").write_bytes(fp16.tobytes(order="C"))
    print(
        f"restored: shape={restored.shape}, min={restored.min():.7g}, "
        f"max={restored.max():.7g}, fp32_bytes={fp32.nbytes}, fp16_bytes={fp16.nbytes}"
    )
    for frame_index, frame in enumerate(restored):
        checksum = frame.reshape(-1)[::257].astype(np.float64).sum()
        print(f"frame {frame_index}: checksum={checksum:.6f}")


if __name__ == "__main__":
    main()
