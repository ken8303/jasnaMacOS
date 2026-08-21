#!/usr/bin/env python3
"""Memory-bounded RF-DETR segmentation inference for Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RFDetrPrediction:
    boxes_xyxy: list[list[float]]
    confidences: list[float]
    polygons: list[list[list[list[float]]]]


class RFDetrMPSDetector:
    """Run Jasna's RF-DETR checkpoint without full-resolution mask tensors.

    The upstream convenience ``predict`` method resizes every selected mask to
    the source image before returning. A 4096x4096 eye can therefore request
    more than 12 GiB for one frame. This adapter traces contours at the native
    192x192 mask resolution and scales only the polygon coordinates.
    """

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str,
        variant: str = "large",
        resolution: int | None = None,
        max_select: int = 64,
    ) -> None:
        import rfdetr
        import torch

        self._torch = torch
        self.device = torch.device(device)
        variants = {
            "medium": (rfdetr.RFDETRSegMedium, 432),
            "large": (rfdetr.RFDETRSegLarge, 768),
        }
        if variant not in variants:
            raise ValueError(f"unsupported RF-DETR variant: {variant}")
        wrapper_type, default_resolution = variants[variant]
        self.resolution = int(resolution or default_resolution)
        self.max_select = int(max_select)
        if self.resolution <= 0 or self.max_select <= 0:
            raise ValueError("resolution and max_select must be positive")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        num_classes = int(state["class_embed.weight"].shape[0]) - 1
        wrapper = wrapper_type(
            num_classes=num_classes,
            resolution=self.resolution,
            pretrain_weights=str(weights_path),
            device=str(self.device),
        )
        core = wrapper.model.model
        if core is None:
            raise RuntimeError("RF-DETR model is unavailable after checkpoint load")
        self._wrapper = wrapper
        self._core = core.to(self.device).eval()
        self._mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self._std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def predict(self, frames, *, score_threshold: float) -> list[RFDetrPrediction]:
        import cv2
        import numpy as np

        torch = self._torch
        if not frames:
            return []

        sizes = [(int(frame.shape[0]), int(frame.shape[1])) for frame in frames]
        resized = [
            cv2.cvtColor(
                cv2.resize(
                    frame,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_LINEAR,
                ),
                cv2.COLOR_BGR2RGB,
            )
            for frame in frames
        ]
        array = np.stack(resized).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        x = torch.from_numpy(array)
        x = (x - self._mean) / self._std
        x = x.to(self.device)

        with torch.inference_mode():
            output = self._core(x)

        pred_boxes = output["pred_boxes"].float()
        pred_logits = output["pred_logits"].float()
        pred_masks = output["pred_masks"].float()
        probability = pred_logits.sigmoid()
        batch_size, query_count, _ = probability.shape
        select_count = min(self.max_select, query_count)
        # Select unique object queries. Flattening query/class scores can pick
        # the same query more than once when a checkpoint gains extra classes.
        query_probability = probability.amax(dim=2)
        values, selected_queries = torch.topk(
            query_probability, select_count, dim=1
        )

        center_x, center_y, width, height = pred_boxes.unbind(-1)
        boxes = torch.stack(
            (
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                center_x + 0.5 * width,
                center_y + 0.5 * height,
            ),
            dim=-1,
        )
        boxes = boxes.gather(
            1, selected_queries.unsqueeze(-1).expand(batch_size, select_count, 4)
        )
        mask_height, mask_width = pred_masks.shape[-2:]
        masks = pred_masks.gather(
            1,
            selected_queries[:, :, None, None].expand(
                batch_size, select_count, mask_height, mask_width
            ),
        )
        valid = values > float(score_threshold)

        boxes_cpu = boxes.cpu().numpy()
        values_cpu = values.cpu().numpy()
        masks_cpu = (masks > 0).cpu().numpy()
        valid_cpu = valid.cpu().numpy()
        predictions = []
        for batch_index, (target_height, target_width) in enumerate(sizes):
            output_boxes = []
            confidences = []
            polygons = []
            for select_index in range(select_count):
                if not valid_cpu[batch_index, select_index]:
                    continue
                normalized_box = boxes_cpu[batch_index, select_index]
                output_boxes.append(
                    [
                        float(np.clip(normalized_box[0] * target_width, 0, target_width)),
                        float(np.clip(normalized_box[1] * target_height, 0, target_height)),
                        float(np.clip(normalized_box[2] * target_width, 0, target_width)),
                        float(np.clip(normalized_box[3] * target_height, 0, target_height)),
                    ]
                )
                confidences.append(float(values_cpu[batch_index, select_index]))
                mask = masks_cpu[batch_index, select_index].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    polygons.append([])
                    continue
                meaningful = [
                    contour for contour in contours if cv2.contourArea(contour) >= 4.0
                ]
                if not meaningful:
                    meaningful = [max(contours, key=cv2.contourArea)]
                meaningful.sort(key=cv2.contourArea, reverse=True)
                scale_x = (target_width - 1) / max(mask_width - 1, 1)
                scale_y = (target_height - 1) / max(mask_height - 1, 1)
                polygons.append(
                    [
                        [
                            [float(point[0]) * scale_x, float(point[1]) * scale_y]
                            for point in contour.reshape(-1, 2)
                        ]
                        for contour in meaningful
                    ]
                )
            predictions.append(
                RFDetrPrediction(output_boxes, confidences, polygons)
            )
        return predictions
