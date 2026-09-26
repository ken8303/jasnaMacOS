#!/usr/bin/env python3
"""Check one MLX DINOv2 encoder block against Jasna's loaded RF-DETR model."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch

from rfdetr_mlx_backbone import backbone_features, encoder_block, patch_embeddings
from rfdetr_mlx_projector import project_features
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also compare all four backbone feature maps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    detector = RFDetrMPSDetector(
        root / "Models/MosaicDetection/rfdetr-vr-v1.pt", device="cpu",
    )
    reference = detector._core.backbone[0].encoder.encoder.encoder.layer[0]
    weights = mx.load(str(root / "Models/MLXDetector/rfdetr-vr-v1.safetensors"))
    source = np.random.default_rng(31).normal(size=(4, 32, 384)).astype(np.float32)
    with torch.inference_mode():
        expected = reference(torch.from_numpy(source))[0].numpy()
    actual = np.asarray(encoder_block(mx.array(source), weights, 0))
    error = np.abs(actual - expected)
    print(f"DINOv2 block 0: max error {error.max():.7g}, mean error {error.mean():.7g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    image = np.random.default_rng(32).normal(size=(1, 3, 768, 768)).astype(np.float32)
    with torch.inference_mode():
        expected_patch = detector._core.backbone[0].encoder.encoder.embeddings(
            torch.from_numpy(image)
        ).numpy()
    actual_patch = np.asarray(patch_embeddings(
        mx.array(np.transpose(image, (0, 2, 3, 1)).copy()), weights,
    ))
    patch_error = np.abs(actual_patch - expected_patch)
    print(f"Windowed patch embeddings: max error {patch_error.max():.7g}, "
          f"mean error {patch_error.mean():.7g}")
    np.testing.assert_allclose(actual_patch, expected_patch, rtol=2e-4, atol=2e-4)
    if args.full:
        with torch.inference_mode():
            expected_features = detector._core.backbone[0].encoder.encoder(
                torch.from_numpy(image)
            )[0]
        actual_features = backbone_features(
            mx.array(np.transpose(image, (0, 2, 3, 1)).copy()), weights,
        )
        for stage, actual_stage, expected_stage in zip((3, 6, 9, 12), actual_features, expected_features):
            actual_stage = np.asarray(actual_stage)
            expected_stage = np.transpose(expected_stage.numpy(), (0, 2, 3, 1))
            error = np.abs(actual_stage - expected_stage)
            print(f"Stage {stage}: max error {error.max():.7g}, mean error {error.mean():.7g}")
            np.testing.assert_allclose(actual_stage, expected_stage, rtol=5e-3, atol=5e-3)
        with torch.inference_mode():
            expected_projected = detector._core.backbone[0].projector(list(expected_features))[0]
        input_features = [mx.array(np.transpose(f.numpy(), (0, 2, 3, 1)).copy())
                          for f in expected_features]
        actual_projected = np.asarray(project_features(input_features, weights))
        expected_projected = np.transpose(expected_projected.numpy(), (0, 2, 3, 1))
        projector_error = np.abs(actual_projected - expected_projected)
        print(f"Feature projector: max error {projector_error.max():.7g}, "
              f"mean error {projector_error.mean():.7g}")
        np.testing.assert_allclose(actual_projected, expected_projected,
                                   rtol=5e-3, atol=5e-3)
    print("RF-DETR MLX backbone block parity: PASS")


if __name__ == "__main__":
    main()
