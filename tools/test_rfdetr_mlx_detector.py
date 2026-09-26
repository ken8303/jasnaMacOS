#!/usr/bin/env python3
"""Compare complete MLX RF-DETR inference against the current MPS path."""

from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch

from rfdetr_mlx_detector import RFDetrMLXDetector
from rfdetr_mlx_backbone import backbone_features
from rfdetr_mlx_projector import project_features
from rfdetr_mlx_decoder import transformer
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    root = Path(__file__).resolve().parents[1]
    reference = RFDetrMPSDetector(
        root / "Models/MosaicDetection/rfdetr-vr-v1.pt", device="mps",
    )
    candidate = RFDetrMLXDetector(
        root / "Models/MLXDetector/rfdetr-vr-v1.safetensors",
    )
    source = np.random.default_rng(61).normal(size=(1, 3, 768, 768)).astype(np.float32)
    with torch.inference_mode():
        torch_features = reference._core.backbone[0].encoder.encoder(
            torch.from_numpy(source).to("mps"))[0]
        torch_projected = reference._core.backbone[0].projector(list(torch_features))[0]
    mlx_features = backbone_features(
        mx.array(np.transpose(source, (0, 2, 3, 1)).copy()), candidate.weights)
    for stage, actual_feature, expected_feature in zip((3, 6, 9, 12), mlx_features, torch_features):
        difference = np.abs(np.asarray(actual_feature)
                            - np.transpose(expected_feature.cpu().numpy(), (0, 2, 3, 1)))
        print(f"Stage {stage} on MPS: max {difference.max():.7g}, mean {difference.mean():.7g}", flush=True)
    mlx_projected = project_features(mlx_features, candidate.weights)
    projection_difference = np.abs(np.asarray(mlx_projected)
                                   - np.transpose(torch_projected.cpu().numpy(), (0, 2, 3, 1)))
    print(f"Full projected feature: max {projection_difference.max():.7g}, "
          f"mean {projection_difference.mean():.7g}", flush=True)
    with torch.inference_mode():
        torch_transformer = reference._core.transformer(
            [torch_projected],
            [torch.zeros((1, 64, 64), dtype=torch.bool, device="mps")],
            [torch.zeros_like(torch_projected)],
            reference._core.refpoint_embed.weight[:200],
            reference._core.query_feat.weight[:200],
        )
    mlx_transformer = transformer(mx.reshape(mlx_projected, (1, 64 * 64, 256)),
                                  candidate.weights, height=64, width=64)
    for name, left, right in zip(("transformer features", "references", "encoder features", "encoder boxes"),
                                 mlx_transformer, torch_transformer[:4]):
        difference = np.abs(np.asarray(left) - right.float().cpu().numpy())
        print(f"Full {name}: max {difference.max():.7g}, mean {difference.mean():.7g}", flush=True)
    with torch.inference_mode():
        torch_delta = reference._core.transformer.enc_out_bbox_embed[0](torch_transformer[2])
    mx_delta = mlx_transformer[2]
    for index in (0, 1):
        base = f"transformer.enc_out_bbox_embed.0.layers.{index}"
        mx_delta = mx.maximum(mx_delta @ mx.transpose(candidate.weights[base + ".weight"])
                              + candidate.weights[base + ".bias"], 0)
    base = "transformer.enc_out_bbox_embed.0.layers.2"
    mx_delta = mx_delta @ mx.transpose(candidate.weights[base + ".weight"])
    mx_delta = mx_delta + candidate.weights[base + ".bias"]
    delta_error = np.abs(np.asarray(mx_delta) - torch_delta.float().cpu().numpy())
    print(f"Encoder box delta: max {delta_error.max():.7g}, mean {delta_error.mean():.7g}", flush=True)
    expected_boxes = torch_transformer[3].float().cpu().numpy()
    expected_delta = torch_delta.float().cpu().numpy()
    expected_wh = expected_boxes[..., 2:] / np.exp(expected_delta[..., 2:])
    expected_xy = expected_boxes[..., :2] - expected_delta[..., :2] * expected_wh
    print("Torch selected proposal first:", np.concatenate((expected_xy, expected_wh), axis=-1)[0, :3], flush=True)
    actual_boxes = np.asarray(mlx_transformer[3])
    actual_delta = np.asarray(mx_delta)
    actual_wh = actual_boxes[..., 2:] / np.exp(actual_delta[..., 2:])
    actual_xy = actual_boxes[..., :2] - actual_delta[..., :2] * actual_wh
    print("MLX selected proposal first:", np.concatenate((actual_xy, actual_wh), axis=-1)[0, :3], flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        expected = reference._core(torch.from_numpy(source).to("mps"))
    print(f"PyTorch MPS forward: {time.perf_counter() - started:.2f}s", flush=True)
    started = time.perf_counter()
    actual = candidate.predict_raw(
        mx.array(np.transpose(source, (0, 2, 3, 1)).copy()),
    )
    print(f"MLX forward: {time.perf_counter() - started:.2f}s", flush=True)
    failed = []
    for name in ("pred_logits", "pred_boxes", "pred_masks"):
        candidate_array = np.asarray(actual[name])
        reference_array = expected[name].float().cpu().numpy()
        difference = np.abs(candidate_array - reference_array)
        print(f"{name}: shape={candidate_array.shape}, "
              f"max error={difference.max():.7g}, mean error={difference.mean():.7g}",
              flush=True)
        if name == "pred_logits":
            np.save("/private/tmp/jasna-rfdetr-mlx-logits.npy", candidate_array)
            np.save("/private/tmp/jasna-rfdetr-torch-logits.npy", reference_array)
        try:
            np.testing.assert_allclose(candidate_array, reference_array,
                                       rtol=5e-3, atol=5e-3)
        except AssertionError:
            failed.append(name)
            print(f"{name} parity: FAIL", flush=True)
    if failed:
        raise AssertionError(f"RF-DETR end-to-end MLX parity failed: {', '.join(failed)}")
    print("RF-DETR end-to-end MLX parity: PASS")


if __name__ == "__main__":
    main()
