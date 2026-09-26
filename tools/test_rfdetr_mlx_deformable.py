#!/usr/bin/env python3
"""Compare MLX deformable bilinear sampling with PyTorch grid_sample."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional

from rfdetr_mlx_deformable import deformable_attention, sample_attention
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    rng = np.random.default_rng(47)
    batch, height, width, heads, channels, queries, points = 2, 16, 12, 8, 32, 23, 4
    value = rng.normal(size=(batch, height * width, heads, channels)).astype(np.float32)
    locations = rng.uniform(-0.2, 1.2, size=(batch, queries, heads, points, 2)).astype(np.float32)
    attention = rng.uniform(size=(batch, queries, heads, points)).astype(np.float32)
    attention /= attention.sum(axis=-1, keepdims=True)
    actual = np.asarray(sample_attention(mx.array(value), mx.array(locations),
                                         mx.array(attention), height=height, width=width))
    image = torch.from_numpy(value).permute(0, 2, 3, 1).reshape(batch * heads, channels, height, width)
    grid = torch.from_numpy(locations).permute(0, 2, 1, 3, 4).reshape(batch * heads, queries, points, 2)
    sampled = functional.grid_sample(image, 2 * grid - 1, padding_mode="zeros", align_corners=False)
    sampled = sampled.reshape(batch, heads, channels, queries, points).permute(0, 3, 1, 4, 2)
    expected = (sampled * torch.from_numpy(attention)[..., None]).sum(dim=3)
    expected = expected.reshape(batch, queries, heads * channels).numpy()
    error = np.abs(actual - expected)
    print(f"Deformable sampler: max error {error.max():.7g}, mean error {error.mean():.7g}")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    root = Path(__file__).resolve().parents[1]
    detector = RFDetrMPSDetector(
        root / "Models/MosaicDetection/rfdetr-vr-v1.pt", device="cpu",
    )
    module = detector._core.transformer.decoder.layers[0].cross_attn
    weights = mx.load(str(root / "Models/MLXDetector/rfdetr-vr-v1.safetensors"))
    batch, height, width, queries, channels = 1, 16, 16, 20, 256
    query = rng.normal(size=(batch, queries, channels)).astype(np.float32)
    memory = rng.normal(size=(batch, height * width, channels)).astype(np.float32)
    reference = rng.uniform(.1, .9, size=(batch, queries, 1, 4)).astype(np.float32)
    with torch.inference_mode():
        expected = module(torch.from_numpy(query), torch.from_numpy(reference),
                          torch.from_numpy(memory), torch.tensor([[height, width]]),
                          torch.tensor([0]), input_spatial_shapes_hw=[(height, width)]).numpy()
    actual = np.asarray(deformable_attention(mx.array(query), mx.array(reference),
                                             mx.array(memory), weights, layer=0,
                                             height=height, width=width))
    module_error = np.abs(actual - expected)
    print(f"Cross-attention module: max error {module_error.max():.7g}, "
          f"mean error {module_error.mean():.7g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    print("RF-DETR MLX deformable sampling parity: PASS")


if __name__ == "__main__":
    main()
