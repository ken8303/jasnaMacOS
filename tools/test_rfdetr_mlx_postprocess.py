#!/usr/bin/env python3
"""Compare MLX RF-DETR output selection against the current PyTorch path."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import numpy as np
import torch

from rfdetr_mlx_postprocess import select_predictions
from rfdetr_mps_detector import RFDetrMPSDetector


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    detector = RFDetrMPSDetector(
        root / "Models/MosaicDetection/rfdetr-vr-v1.pt",
        device="mps", max_select=64,
    )
    rng = np.random.default_rng(21)
    image = rng.random((1, 3, 768, 768), dtype=np.float32)
    source = torch.from_numpy(image).to("mps")
    with torch.inference_mode():
        output = detector._core(source)
    boxes = output["pred_boxes"].float().cpu().numpy()
    logits = output["pred_logits"].float().cpu().numpy()
    masks = output["pred_masks"].float().cpu().numpy()
    values, selected_boxes, selected_masks, valid = select_predictions(
        mx.array(boxes), mx.array(logits), mx.array(masks),
        threshold=0.5, max_select=64,
    )
    probability = torch.from_numpy(logits).sigmoid().amax(dim=2)
    expected_values, indices = torch.topk(probability, 64, dim=1, sorted=True)
    cx, cy, width, height = torch.from_numpy(boxes).unbind(-1)
    xyxy = torch.stack((cx-width/2, cy-height/2, cx+width/2, cy+height/2), dim=-1)
    expected_boxes = xyxy.gather(1, indices[..., None].expand(-1, -1, 4))
    expected_masks = torch.from_numpy(masks).gather(
        1, indices[..., None, None].expand(-1, -1, masks.shape[2], masks.shape[3])
    )
    checks = (
        ("score", np.asarray(values), expected_values.numpy()),
        ("box", np.asarray(selected_boxes), expected_boxes.numpy()),
        ("mask", np.asarray(selected_masks), expected_masks.numpy()),
        ("threshold", np.asarray(valid), (expected_values > .5).numpy()),
    )
    for name, actual, expected in checks:
        max_error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        print(f"{name}: shape={actual.shape}, max error={max_error:.7g}")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("RF-DETR MLX postprocessing parity: PASS")


if __name__ == "__main__":
    main()
