#!/usr/bin/env python3
"""Export Jasna's RF-DETR weights for an experimental MLX detector port.

This exports parameters only. RF-DETR's backbone, deformable attention,
transformer decoder, and segmentation head must be implemented and checked
against PyTorch before this archive can be used for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(checkpoint_path: Path, output_dir: Path, validate_patch: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    tensors = {}
    skipped = []
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"unexpected checkpoint entry: {name}")
        if value.numel() == 0:
            skipped.append(name)
            continue
        array = value.detach().contiguous().cpu().numpy()
        tensors[name] = mx.array(array)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "rfdetr-vr-v1.safetensors"
    mx.save_safetensors(str(archive), tensors)
    loaded = mx.load(str(archive))
    if set(loaded) != set(tensors):
        raise AssertionError("saved archive has missing or unexpected tensors")
    for name, source in tensors.items():
        actual = np.asarray(loaded[name])
        expected = np.asarray(source)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise AssertionError(f"weight mismatch after export: {name}")

    if validate_patch:
        prefix = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection"
        weight = np.asarray(loaded[prefix + ".weight"])
        bias = np.asarray(loaded[prefix + ".bias"])
        rng = np.random.default_rng(17)
        source = rng.normal(size=(1, 3, 24, 24)).astype(np.float32)
        expected = functional.conv2d(
            torch.from_numpy(source), state[prefix + ".weight"],
            state[prefix + ".bias"], stride=12,
        ).detach().numpy()
        actual = mx.conv2d(
            mx.array(np.transpose(source, (0, 2, 3, 1)).copy()),
            mx.array(np.transpose(weight, (0, 2, 3, 1)).copy()), stride=12,
        ) + mx.array(bias)
        actual = np.transpose(np.asarray(actual), (0, 3, 1, 2))
        error = np.max(np.abs(actual - expected))
        print(f"RF-DETR patch projection PyTorch/MLX max error: {error:.6g}")
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError("MLX patch projection differs from PyTorch")

    manifest = {
        "format": "jasna-rfdetr-mlx-weights-v1",
        "source_sha256": sha256(checkpoint_path),
        "archive_sha256": sha256(archive),
        "tensor_count": len(tensors),
        "skipped_empty_tensors": skipped,
        "weight_layout": "PyTorch native; transpose convolution weights when implementing MLX layers",
        "runtime_status": "weights only; detector inference and parity validation pending",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"RF-DETR MLX weight export: PASS ({len(tensors)} tensors, {archive})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--validate-patch", action="store_true")
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir, args.validate_patch)
