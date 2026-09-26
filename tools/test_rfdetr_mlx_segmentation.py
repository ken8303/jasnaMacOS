#!/usr/bin/env python3
"""Compare all five MLX RF-DETR segmentation stages against PyTorch."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch

from rfdetr_mlx_segmentation import segmentation_masks
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    root = Path(__file__).resolve().parents[1]
    detector = RFDetrMPSDetector(root / "Models/MosaicDetection/rfdetr-vr-v1.pt",
                                device="cpu")
    weights = mx.load(str(root / "Models/MLXDetector/rfdetr-vr-v1.safetensors"))
    rng = np.random.default_rng(59)
    spatial = rng.normal(size=(1, 256, 64, 64)).astype(np.float32)
    queries = rng.normal(size=(5, 1, 20, 256)).astype(np.float32)
    with torch.inference_mode():
        expected = detector._core.segmentation_head(
            torch.from_numpy(spatial),
            [torch.from_numpy(layer) for layer in queries],
            (768, 768),
        )
    actual = segmentation_masks(
        mx.array(np.transpose(spatial, (0, 2, 3, 1)).copy()),
        mx.array(queries), weights,
    )
    for index, (result, reference) in enumerate(zip(actual, expected)):
        result = np.asarray(result)
        reference = reference.numpy()
        error = np.abs(result - reference)
        print(f"Mask head {index}: max error {error.max():.7g}, mean error {error.mean():.7g}")
        np.testing.assert_allclose(result, reference, rtol=2e-3, atol=2e-3)
    print("RF-DETR MLX segmentation head parity: PASS")


if __name__ == "__main__":
    main()
