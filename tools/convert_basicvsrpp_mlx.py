#!/usr/bin/env python3
"""Export the v1.2 BasicVSR++ checkpoint for an experimental MLX backend.

This is a weight archive, not a replacement for the Metal runtime: flow warp,
deformable alignment, and temporal scheduling still need an MLX implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional


def convert(weights: Path, output: Path, validate: bool) -> None:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    prefix = "generator_ema." if any(k.startswith("generator_ema.") for k in checkpoint) else "generator."
    tensors = {}
    layouts = {}
    for key, value in checkpoint.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        array = value.detach().contiguous().numpy()
        # MLX convolutions use NHWC inputs and O,H,W,I filters. DCNv2 weights
        # remain O,I,H,W because the existing custom Metal kernel owns them.
        is_dcn_weight = name.startswith("deform_align.") and ".conv_offset." not in name
        if array.ndim == 4 and name.endswith(".weight") and not is_dcn_weight:
            array = np.transpose(array, (0, 2, 3, 1)).copy()
            layouts[name] = "OHWI"
        elif array.ndim == 4:
            layouts[name] = "OIHW"
        tensors[name] = mx.array(array)
    if not tensors:
        raise ValueError(f"no {prefix} tensors in {weights}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "basicvsrpp-v1.2.safetensors"
    mx.save_safetensors(str(archive), tensors)
    manifest = {
        "format": "jasna-basicvsrpp-mlx-weights-v1",
        "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "source_branch": prefix[:-1],
        "tensor_count": len(tensors),
        "layout_exceptions": layouts,
        "runtime_status": "experimental MLX inference in basicvsrpp_mlx_segments.py; app not integrated",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    loaded = mx.load(str(archive))
    if set(loaded) != set(tensors):
        raise AssertionError("saved MLX archive is incomplete")
    if validate:
        # Check a real v1.2 SPyNet convolution chain, including layout and
        # activation, against PyTorch on the same deterministic input.
        rng = np.random.default_rng(7)
        source = rng.normal(size=(1, 8, 16, 16)).astype(np.float32)
        torch_result = torch.from_numpy(source)
        mlx_result = mx.array(np.transpose(source, (0, 2, 3, 1)).copy())
        for layer in range(5):
            base = f"spynet.basic_module.0.basic_module.{layer}.conv"
            w = checkpoint[prefix + base + ".weight"]
            b = checkpoint[prefix + base + ".bias"]
            torch_result = functional.conv2d(torch_result, w, b, padding=3)
            mlx_result = mx.conv2d(mlx_result, loaded[base + ".weight"], padding=3)
            mlx_result = mlx_result + loaded[base + ".bias"]
            if layer < 4:
                torch_result = functional.relu(torch_result)
                mlx_result = mx.maximum(mlx_result, 0)
        actual = np.transpose(np.array(mlx_result), (0, 3, 1, 2))
        expected = torch_result.detach().numpy()
        error = np.abs(actual - expected)
        print(f"SPyNet level 0: max error {error.max():.6g}, mean error {error.mean():.6g}")
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError("MLX SPyNet output differs from PyTorch")
    print(f"exported {len(tensors)} tensors to {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    convert(args.weights, args.output, args.validate)
