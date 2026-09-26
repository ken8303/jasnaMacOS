#!/usr/bin/env python3
"""Compare MLX and MPS RF-DETR polygons on a real saved VR eye frame."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from rfdetr_mlx_detector import RFDetrMLXDetector
from rfdetr_mps_detector import RFDetrMPSDetector


def main():
    root = Path(__file__).resolve().parents[1]
    clip = Path("/Users/kenlo/Documents/Codex/2026-09-12/pl/work/mlx-switch-eye-12.mov")
    capture = cv2.VideoCapture(str(clip))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode {clip}")
    print(f"Real frame: {frame.shape[1]}x{frame.shape[0]}", flush=True)
    reference = RFDetrMPSDetector(root / "Models/MosaicDetection/rfdetr-vr-v1.pt",
                                  device="mps", max_select=64)
    candidate = RFDetrMLXDetector(
        root / "Models/MLXDetector/rfdetr-vr-v1.safetensors")
    results = []
    for name, detector in (("MPS", reference), ("MLX", candidate)):
        started = time.perf_counter()
        batch_results = detector.predict([frame, frame], score_threshold=.05)
        result = batch_results[0]
        assert batch_results[0] == batch_results[1]
        print(f"{name}: {len(result.boxes_xyxy)} boxes per image in "
              f"{time.perf_counter() - started:.2f}s; "
              f"scores {result.confidences[:6]}", flush=True)
        results.append(result)
    mps, mlx = results
    assert len(mps.boxes_xyxy) == len(mlx.boxes_xyxy)
    np.testing.assert_allclose(mlx.boxes_xyxy, mps.boxes_xyxy, rtol=.01, atol=1)
    np.testing.assert_allclose(mlx.confidences, mps.confidences, rtol=.01, atol=.01)
    assert len(mps.polygons) == len(mlx.polygons)
    for left, right in zip(mps.polygons, mlx.polygons):
        assert len(left) == len(right)
    print("RF-DETR MLX real-frame prediction parity: PASS")


if __name__ == "__main__":
    main()
