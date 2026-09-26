"""Experimental end-to-end MLX inference for Jasna's RF-DETR detector."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx

from rfdetr_mlx_backbone import backbone_features
from rfdetr_mlx_decoder import transformer
from rfdetr_mlx_projector import project_features
from rfdetr_mlx_segmentation import segmentation_masks
from rfdetr_mlx_postprocess import select_predictions
from rfdetr_mps_detector import RFDetrPrediction, valid_prefix_counts


class RFDetrMLXDetector:
    """MLX RF-DETR inference for the 768-pixel Jasna VR checkpoint."""

    def __init__(self, archive: Path, *, max_select: int = 64) -> None:
        self.weights = mx.load(str(archive))
        self.max_select = max_select

    def predict_raw(self, image):
        """Accept normalized RGB NHWC [B,768,768,3] and return model outputs."""
        if image.ndim != 4 or image.shape[1:] != (768, 768, 3):
            raise ValueError("expected normalized RGB 768x768 image batch")
        weights = self.weights
        features = backbone_features(image, weights)
        projected = project_features(features, weights)
        batch, height, width, channels = projected.shape
        memory = mx.reshape(projected, (batch, height * width, channels))
        decoder_features, references, _, _ = transformer(
            memory, weights, height=height, width=width,
        )

        def linear(x, name):
            return x @ mx.transpose(weights[name + ".weight"]) + weights[name + ".bias"]

        boxes = decoder_features
        for index in (0, 1):
            boxes = mx.maximum(linear(boxes, f"bbox_embed.layers.{index}"), 0)
        delta = linear(boxes, "bbox_embed.layers.2")[-1]
        reference = references[-1]
        predicted_boxes = mx.concatenate((
            delta[..., :2] * reference[..., 2:] + reference[..., :2],
            mx.exp(delta[..., 2:]) * reference[..., 2:]), axis=-1)
        logits = linear(decoder_features[-1], "class_embed")
        masks = segmentation_masks(projected, decoder_features, weights)[-1]
        mx.eval(predicted_boxes, logits, masks)
        return {"pred_boxes": predicted_boxes, "pred_logits": logits,
                "pred_masks": masks}

    def predict(self, frames, *, score_threshold: float):
        """Return the same native-mask polygon format as the MPS detector."""
        import cv2
        import numpy as np

        if not frames:
            return []
        sizes = [(int(frame.shape[0]), int(frame.shape[1])) for frame in frames]
        resized = [cv2.cvtColor(cv2.resize(frame, (768, 768),
                                          interpolation=cv2.INTER_LINEAR),
                                cv2.COLOR_BGR2RGB) for frame in frames]
        image = np.stack(resized).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        image = image / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        raw = self.predict_raw(mx.array(image))
        values, boxes, masks, valid = select_predictions(
            raw["pred_boxes"], raw["pred_logits"], raw["pred_masks"],
            threshold=score_threshold, max_select=self.max_select,
        )
        values_cpu, boxes_cpu, valid_cpu = (np.asarray(item) for item in
                                             (values, boxes, valid))
        counts = valid_prefix_counts(valid_cpu)
        transfer_count = max(counts, default=0)
        masks_cpu = np.asarray(masks[:, :transfer_count] > 0) if transfer_count else None
        results = []
        for batch_index, (height, width) in enumerate(sizes):
            output_boxes, confidences, polygons = [], [], []
            for selected in range(counts[batch_index]):
                box = boxes_cpu[batch_index, selected]
                output_boxes.append([
                    float(np.clip(box[0] * width, 0, width)),
                    float(np.clip(box[1] * height, 0, height)),
                    float(np.clip(box[2] * width, 0, width)),
                    float(np.clip(box[3] * height, 0, height)),
                ])
                confidences.append(float(values_cpu[batch_index, selected]))
                mask = masks_cpu[batch_index, selected].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                meaningful = [c for c in contours if cv2.contourArea(c) >= 4.0]
                if not meaningful and contours:
                    meaningful = [max(contours, key=cv2.contourArea)]
                meaningful.sort(key=cv2.contourArea, reverse=True)
                scale_x = (width - 1) / max(mask.shape[1] - 1, 1)
                scale_y = (height - 1) / max(mask.shape[0] - 1, 1)
                polygons.append([[
                    [float(point[0]) * scale_x, float(point[1]) * scale_y]
                    for point in contour.reshape(-1, 2)
                ] for contour in meaningful])
            results.append(RFDetrPrediction(output_boxes, confidences, polygons))
        return results
