"""MLX score, box, and native-resolution mask selection for RF-DETR.

Consumes raw model outputs. The RF-DETR network itself is not yet ported.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx


def select_predictions(pred_boxes, pred_logits, pred_masks, *, threshold, max_select):
    """Return sorted unique-query scores, xyxy boxes, and native masks."""
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes must have shape [batch, queries, 4]")
    batch, queries, _ = pred_boxes.shape
    if pred_logits.shape[:2] != (batch, queries):
        raise ValueError("pred_logits must match batch and query counts")
    if pred_masks.shape[:2] != (batch, queries):
        raise ValueError("pred_masks must match batch and query counts")
    if max_select < 1:
        raise ValueError("max_select must be positive")
    count = min(max_select, queries)
    scores = mx.max(mx.sigmoid(pred_logits.astype(mx.float32)), axis=2)
    indices = mx.argsort(-scores, axis=1)[:, :count]
    values = mx.take_along_axis(scores, indices, axis=1)
    cx, cy, width, height = mx.split(pred_boxes.astype(mx.float32), 4, axis=-1)
    boxes = mx.concatenate(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        axis=-1,
    )
    boxes = mx.take_along_axis(boxes, indices[:, :, None], axis=1)
    masks = mx.take_along_axis(pred_masks.astype(mx.float32), indices[:, :, None, None], axis=1)
    return values, boxes, masks, values > threshold
