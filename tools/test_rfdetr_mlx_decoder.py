#!/usr/bin/env python3
"""Compare a full MLX RF-DETR decoder layer with its PyTorch counterpart."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch

from rfdetr_mlx_decoder import decoder_layer, transformer
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    root = Path(__file__).resolve().parents[1]
    detector = RFDetrMPSDetector(root / "Models/MosaicDetection/rfdetr-vr-v1.pt",
                                device="cpu")
    weights = mx.load(str(root / "Models/MLXDetector/rfdetr-vr-v1.safetensors"))
    module = detector._core.transformer.decoder.layers[0]
    rng = np.random.default_rng(53)
    batch, queries, channels, height, width = 1, 20, 256, 16, 16
    target = rng.normal(size=(batch, queries, channels)).astype(np.float32)
    memory = rng.normal(size=(batch, height * width, channels)).astype(np.float32)
    query_pos = rng.normal(size=(batch, queries, channels)).astype(np.float32)
    refs = rng.uniform(.1, .9, size=(batch, queries, 1, 4)).astype(np.float32)
    with torch.inference_mode():
        expected = module(torch.from_numpy(target), torch.from_numpy(memory),
                          query_pos=torch.from_numpy(query_pos),
                          reference_points=torch.from_numpy(refs),
                          spatial_shapes=torch.tensor([[height, width]]),
                          spatial_shapes_hw=[(height, width)],
                          level_start_index=torch.tensor([0])).numpy()
    actual = np.asarray(decoder_layer(mx.array(target), mx.array(memory),
                                      mx.array(query_pos), mx.array(refs), weights,
                                      layer=0, height=height, width=width))
    error = np.abs(actual - expected)
    print(f"Decoder layer 0: max error {error.max():.7g}, mean error {error.mean():.7g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    feature = rng.normal(size=(batch, channels, height, width)).astype(np.float32)
    with torch.inference_mode():
        reference_transformer = detector._core.transformer(
            [torch.from_numpy(feature)],
            [torch.zeros((batch, height, width), dtype=torch.bool)],
            [torch.zeros_like(torch.from_numpy(feature))],
            detector._core.refpoint_embed.weight[:detector._core.num_queries],
            detector._core.query_feat.weight[:detector._core.num_queries],
        )
    flattened = np.transpose(feature, (0, 2, 3, 1)).reshape(batch, height * width, channels)
    actual_transformer = transformer(mx.array(flattened), weights,
                                     height=height, width=width)
    for name, actual_value, expected_value in zip(
        ("decoder features", "initial references", "encoder features", "encoder boxes"),
        actual_transformer, reference_transformer[:4],
    ):
        actual_value = np.asarray(actual_value)
        expected_value = expected_value.detach().numpy()
        difference = np.abs(actual_value - expected_value)
        print(f"{name}: shape={actual_value.shape}, max error {difference.max():.7g}, "
              f"mean error {difference.mean():.7g}")
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-3, atol=3e-3)
    print("RF-DETR MLX decoder layer parity: PASS")


if __name__ == "__main__":
    main()
