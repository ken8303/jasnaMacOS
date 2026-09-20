#!/usr/bin/env python3
"""Compare an experimental RF-DETR Core ML export with eager PyTorch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("coreml_package", type=Path)
    parser.add_argument("sbs_video", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--compute-units",
        choices=("cpu-ne", "all", "cpu-gpu", "cpu"),
        default="cpu-ne",
    )
    return parser.parse_args()


def prepare_eye_batch(video_path: Path, frame_index: int, resolution: int):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index} from {video_path}")
    height, width = frame.shape[:2]
    if width % 2 != 0:
        raise RuntimeError(f"SBS frame width must be even, got {width}")
    eye_width = width // 2
    eyes = (frame[:, :eye_width], frame[:, eye_width:])
    resized = [
        cv2.cvtColor(
            cv2.resize(eye, (resolution, resolution), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        for eye in eyes
    ]
    array = np.stack(resized).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[None, :, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[None, :, None, None]
    return (array - mean) / std, (height, eye_width)


def iou(box_a, box_b) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def xyxy(boxes):
    import numpy as np

    center_x, center_y, width, height = np.moveaxis(boxes, -1, 0)
    return np.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        axis=-1,
    )


def main() -> None:
    arguments = parse_arguments()
    for path in (arguments.checkpoint, arguments.coreml_package, arguments.sbs_video):
        if not path.exists():
            raise SystemExit(f"error: input not found: {path}")
    if arguments.frame < 0 or arguments.repeats <= 0:
        raise SystemExit("error: --frame must be non-negative and --repeats positive")

    import coremltools as ct
    import numpy as np
    import rfdetr
    import torch

    inputs, eye_size = prepare_eye_batch(arguments.sbs_video, arguments.frame, 768)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    num_classes = int(state["class_embed.weight"].shape[0]) - 1
    wrapper = rfdetr.RFDETRSegLarge(
        num_classes=num_classes,
        resolution=768,
        pretrain_weights=str(arguments.checkpoint),
        device="cpu",
    )
    eager = wrapper.model.model
    if eager is None:
        raise RuntimeError("eager RF-DETR model failed to load")
    eager.eval()
    with torch.inference_mode():
        started = time.perf_counter()
        eager_output = eager(torch.from_numpy(inputs))
        eager_ms = (time.perf_counter() - started) * 1000
    eager_boxes = eager_output["pred_boxes"].float().numpy()
    eager_logits = eager_output["pred_logits"].float().numpy()
    eager_masks = eager_output["pred_masks"].float().numpy()
    del eager_output, eager, wrapper

    compute_units = {
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[arguments.compute_units]
    model = ct.models.MLModel(str(arguments.coreml_package), compute_units=compute_units)
    output_shapes = {
        (2, 200, 4): "boxes",
        (2, 200, 2): "logits",
        (2, 200, 192, 192): "masks",
    }
    core_outputs = None
    timings = []
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = model.predict({"tensors": inputs})
        timings.append((time.perf_counter() - started) * 1000)
        core_outputs = {output_shapes[value.shape]: value for value in result.values()}
    assert core_outputs is not None
    core_boxes = core_outputs["boxes"].astype(np.float32)
    core_logits = core_outputs["logits"].astype(np.float32)
    core_masks = core_outputs["masks"].astype(np.float32)

    print(f"RF-DETR Core ML parity ({arguments.compute_units})")
    print(f"Fixture: {arguments.sbs_video}, frame {arguments.frame}, eyes {eye_size[1]}x{eye_size[0]}")
    print(f"PyTorch CPU inference: {eager_ms:.3f} ms")
    print(
        f"Core ML {arguments.compute_units} inference: "
        f"cold {timings[0]:.3f} ms; warm median {np.median(timings[1:] or timings):.3f} ms; "
        f"samples {len(timings)}"
    )
    print(
        "Raw mean/max error: "
        f"boxes {np.mean(np.abs(core_boxes - eager_boxes)):.6f}/"
        f"{np.max(np.abs(core_boxes - eager_boxes)):.6f}; "
        f"logits {np.mean(np.abs(core_logits - eager_logits)):.6f}/"
        f"{np.max(np.abs(core_logits - eager_logits)):.6f}; "
        f"masks {np.mean(np.abs(core_masks - eager_masks)):.6f}/"
        f"{np.max(np.abs(core_masks - eager_masks)):.6f}"
    )

    eager_scores = 1.0 / (1.0 + np.exp(-eager_logits))
    core_scores = 1.0 / (1.0 + np.exp(-core_logits))
    eager_scores = eager_scores.max(axis=2)
    core_scores = core_scores.max(axis=2)
    eager_xyxy = xyxy(eager_boxes)
    core_xyxy = xyxy(core_boxes)
    parity_ok = True
    for batch_index, eye in enumerate(("left", "right")):
        eager_selected = set(np.flatnonzero(eager_scores[batch_index] > arguments.threshold).tolist())
        core_selected = set(np.flatnonzero(core_scores[batch_index] > arguments.threshold).tolist())
        # DETR query slots are not object identities.  Core ML may choose the
        # same detections in different query slots, so compare detections using
        # greedy box matching instead of requiring identical query indices.
        unmatched_core = set(core_selected)
        matches = []
        for eager_query in sorted(
            eager_selected,
            key=lambda query: float(eager_scores[batch_index, query]),
            reverse=True,
        ):
            candidates = [
                (
                    iou(
                        eager_xyxy[batch_index, eager_query],
                        core_xyxy[batch_index, core_query],
                    ),
                    core_query,
                )
                for core_query in unmatched_core
            ]
            if not candidates:
                break
            best_iou, core_query = max(candidates)
            unmatched_core.remove(core_query)
            matches.append((eager_query, core_query, best_iou))
        box_ious = [match[2] for match in matches]
        mask_ious = []
        for eager_query, core_query, _ in matches:
            eager_mask = eager_masks[batch_index, eager_query] > 0
            core_mask = core_masks[batch_index, core_query] > 0
            union = np.logical_or(eager_mask, core_mask).sum()
            intersection = np.logical_and(eager_mask, core_mask).sum()
            mask_ious.append(float(intersection / union) if union else 1.0)
        matched_eager = {match[0] for match in matches if match[2] >= 0.50}
        missing = eager_selected - matched_eager
        extra = len(core_selected) - len(matched_eager)
        minimum_box_iou = min(box_ious, default=1.0)
        minimum_mask_iou = min(mask_ious, default=1.0)
        eye_ok = not missing and extra == 0 and minimum_box_iou >= 0.90 and minimum_mask_iou >= 0.90
        parity_ok = parity_ok and eye_ok
        print(
            f"{eye}: eager/core/matched {len(eager_selected)}/{len(core_selected)}/{len(matches)}; "
            f"missing {len(missing)}; extra {extra}; "
            f"minimum box/mask IoU {minimum_box_iou:.4f}/{minimum_mask_iou:.4f}; "
            f"{'PASS' if eye_ok else 'FAIL'}"
        )
    print(f"Detection parity: {'PASS' if parity_ok else 'FAIL'}")
    if not parity_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
