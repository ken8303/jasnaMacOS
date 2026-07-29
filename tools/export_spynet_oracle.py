#!/usr/bin/env python3
"""Export deterministic bidirectional SPyNet outputs from the Jasna checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def deterministic_frame(multiplier: int, addend: int, modulus: int, divisor: int) -> torch.Tensor:
    indices = np.arange(3 * 64 * 64, dtype=np.int64)
    # Match Swift's Float -> Float16 input conversion before returning to FP32
    # for portable CPU convolution.
    values = (((indices * multiplier + addend) % modulus).astype(np.float32) / divisor).astype(np.float16)
    return torch.from_numpy(values.astype(np.float32).reshape(1, 3, 64, 64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    generator.eval()
    reference = deterministic_frame(29, 17, 1021, 1020)
    support = deterministic_frame(43, 31, 1019, 1018)
    with torch.inference_mode():
        backward = generator.spynet(reference, support)
        forward = generator.spynet(support, reference)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("backward", backward), ("forward", forward)):
        values = tensor.detach().cpu().numpy().astype("<f2", copy=False)
        (args.output / f"{name}.f16").write_bytes(values.tobytes(order="C"))
        print(f"{name}: max={np.abs(values.astype(np.float32)).max():.7g}, bytes={values.nbytes}")


if __name__ == "__main__":
    main()
