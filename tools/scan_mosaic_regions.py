#!/usr/bin/env python3
"""Create a sparse restoration manifest using VR Video Toolbox's YOLO workflow."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("yolo", "rfdetr"), default="yolo"
    )
    parser.add_argument("--sample-stride", type=float, default=0.1)
    parser.add_argument(
        "--coarse-stride",
        type=float,
        default=1.0,
        help="seconds between coarse gate samples before dense refinement",
    )
    parser.add_argument(
        "--coarse-confidence",
        type=float,
        default=0.05,
        help="sensitive gate threshold; final regions still use --confidence",
    )
    parser.add_argument(
        "--refine-padding",
        type=float,
        default=1.0,
        help="seconds added around coarse detections for dense temporal tracking",
    )
    parser.add_argument(
        "--adaptive-scan",
        action="store_true",
        help="use a once-per-second gate before dense temporal refinement",
    )
    parser.add_argument(
        "--region-duration",
        type=float,
        default=1.0,
        help=(
            "temporal restoration clip length in seconds; the default keeps a "
            "tracked mosaic active for the complete 30-frame processing window"
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=2048)
    parser.add_argument("--rect-expand", type=float, default=1.5)
    parser.add_argument("--minimum-rect", type=int, default=512)
    parser.add_argument("--temporal-padding", type=float, default=0.75)
    parser.add_argument("--region-nms-iou", type=float, default=0.45)
    parser.add_argument(
        "--mask-expansion",
        type=float,
        default=0.10,
        help="soft segmentation-mask expansion as a fraction of mask resolution",
    )
    parser.add_argument(
        "--mask-size",
        type=int,
        default=128,
        help="square segmentation-mask resolution; power of two from 32 through 256",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-detections",
        type=int,
        default=64,
        help="maximum RF-DETR object queries retained per sampled frame",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("sequential", "seek"),
        default="sequential",
        help="sequential avoids expensive random seeks in HEVC video",
    )
    return parser.parse_args()


def aligned_rect(boxes, frame_width, frame_height, expand, minimum, alignment=16):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    target_width = min(frame_width, max(minimum, (x2 - x1) * expand))
    target_height = min(frame_height, max(minimum, (y2 - y1) * expand))
    left = max(0, min(frame_width - target_width, center_x - target_width / 2))
    top = max(0, min(frame_height - target_height, center_y - target_height / 2))
    left = int(math.floor(left / alignment) * alignment)
    top = int(math.floor(top / alignment) * alignment)
    right = int(math.ceil((left + target_width) / alignment) * alignment)
    bottom = int(math.ceil((top + target_height) / alignment) * alignment)
    right = min(frame_width, right)
    bottom = min(frame_height, bottom)
    if right - left < minimum:
        left = max(0, min(left, frame_width - minimum))
        right = min(frame_width, left + minimum)
    if bottom - top < minimum:
        top = max(0, min(top, frame_height - minimum))
        bottom = min(frame_height, top + minimum)
    return left, top, right - left, bottom - top


def box_iou(left, right):
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def tracking_distance(left, right):
    left_width, left_height = left[2] - left[0], left[3] - left[1]
    right_width, right_height = right[2] - right[0], right[3] - right[1]
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    distance = math.hypot(
        left_center[0] - right_center[0], left_center[1] - right_center[1]
    )
    scale = max(left_width, left_height, right_width, right_height, 1.0)
    return distance / scale


def detector_coverage_metrics(regions, frame_width, frame_height, frame_count):
    """Return normalized scheduling load without rasterizing full-resolution masks."""
    if frame_width <= 0 or frame_height <= 0 or frame_count <= 0:
        return 0.0, 0.0
    active_region_frames = 0
    scheduled_blend_pixel_frames = 0
    for region in regions:
        start = max(0, min(frame_count, int(region.get("startFrame", 0))))
        end = max(start, min(frame_count, int(region.get("endFrame", start))))
        duration = end - start
        blend_width = max(0, int(region.get("blendWidth", region.get("width", 0))))
        blend_height = max(0, int(region.get("blendHeight", region.get("height", 0))))
        active_region_frames += duration
        scheduled_blend_pixel_frames += blend_width * blend_height * duration
    average_active_regions = active_region_frames / frame_count
    scheduled_blend_percent = (
        100.0
        * scheduled_blend_pixel_frames
        / (frame_width * frame_height * frame_count)
    )
    return average_active_regions, scheduled_blend_percent


def coarse_sample_indices(frame_count, source_fps, coarse_stride):
    """Choose one representative center frame from each coarse time interval."""
    if frame_count <= 0 or source_fps <= 0 or coarse_stride <= 0:
        return []
    interval_frames = max(1, int(round(coarse_stride * source_fps)))
    first_frame = min(frame_count - 1, interval_frames // 2)
    return list(range(first_frame, frame_count, interval_frames))


def refinement_sample_indices(
    frame_count, source_fps, dense_stride, detected_frame_indices, padding_seconds
):
    """Return dense samples only in whole seconds surrounding coarse detections."""
    if frame_count <= 0 or source_fps <= 0 or dense_stride <= 0:
        return []
    dense_frames = max(1, int(round(dense_stride * source_fps)))
    second_frames = max(1, int(round(source_fps)))
    padding_frames = max(0, int(round(padding_seconds * source_fps)))
    intervals = []
    for frame_index in sorted(set(int(frame) for frame in detected_frame_indices)):
        second_start = max(0, (frame_index // second_frames) * second_frames)
        intervals.append(
            (
                max(0, second_start - padding_frames),
                min(frame_count, second_start + second_frames + padding_frames),
            )
        )
    if not intervals:
        return []
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    samples = []
    for start, end in merged:
        first = ((start + dense_frames - 1) // dense_frames) * dense_frames
        samples.extend(range(first, end, dense_frames))
    return samples


def track_boxes(boxes):
    """Associate detections over time without merging two subjects in one frame."""
    by_frame = {}
    for box in boxes:
        by_frame.setdefault(int(box[5]), []).append(box)
    tracks = []
    for frame_index, detections in sorted(by_frame.items()):
        available = {
            index for index, track in enumerate(tracks)
            if frame_index - track["last_frame"] <= 30
        }
        for detection in sorted(detections, key=lambda item: item[4], reverse=True):
            candidates = []
            for track_index in available:
                previous = tracks[track_index]["last"]
                iou = box_iou(previous, detection)
                distance = tracking_distance(previous, detection)
                # VR mosaics can travel several of their own widths between
                # 0.1-second samples during quick camera or body motion.
                if iou >= 0.05 or distance <= 3.0:
                    candidates.append((-(iou * 4.0) + distance, track_index))
            if candidates:
                _, track_index = min(candidates)
                track = tracks[track_index]
                track["boxes"].append(detection)
                track["last"] = detection
                track["last_frame"] = frame_index
                available.remove(track_index)
            else:
                tracks.append({
                    "boxes": [detection], "last": detection, "last_frame": frame_index
                })
    return [track["boxes"] for track in tracks]


def interpolated_box(boxes, frame_index):
    ordered = sorted(boxes, key=lambda box: box[5])
    before = [box for box in ordered if box[5] <= frame_index]
    after = [box for box in ordered if box[5] >= frame_index]
    left = before[-1] if before else ordered[0]
    right = after[0] if after else ordered[-1]
    if right[5] == left[5]:
        return tuple(left[:5]) + (frame_index,)
    amount = (frame_index - left[5]) / (right[5] - left[5])
    values = tuple(left[index] + (right[index] - left[index]) * amount for index in range(5))
    return values + (frame_index,)


def rectangles_overlap(left, right):
    return (
        left[0] < right[0] + right[2]
        and left[0] + left[2] > right[0]
        and left[1] < right[1] + right[3]
        and left[1] + left[3] > right[1]
    )


def intersection_fraction_of_smaller(small, large):
    """Return how much of small's blend rectangle is covered by large's."""
    sx, sy = small.get("blendX", small["x"]), small.get("blendY", small["y"])
    sw = small.get("blendWidth", small["width"])
    sh = small.get("blendHeight", small["height"])
    lx, ly = large.get("blendX", large["x"]), large.get("blendY", large["y"])
    lw = large.get("blendWidth", large["width"])
    lh = large.get("blendHeight", large["height"])
    intersection_width = max(0, min(sx + sw, lx + lw) - max(sx, lx))
    intersection_height = max(0, min(sy + sh, ly + lh) - max(sy, ly))
    small_area = sw * sh
    return intersection_width * intersection_height / small_area if small_area > 0 else 0.0


def blend_iou(left, right):
    """Return intersection-over-union for two visible blend rectangles."""
    lx, ly = left.get("blendX", left["x"]), left.get("blendY", left["y"])
    lw = left.get("blendWidth", left["width"])
    lh = left.get("blendHeight", left["height"])
    rx, ry = right.get("blendX", right["x"]), right.get("blendY", right["y"])
    rw = right.get("blendWidth", right["width"])
    rh = right.get("blendHeight", right["height"])
    intersection = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def suppress_nested_regions(regions, coverage=0.8, area_ratio=1.5):
    """Drop duplicate inner crops that would overwrite a larger restoration.

    Distinct nearby subjects are preserved. A crop is removed only when one
    larger region covers its complete active time and nearly all of its blend
    rectangle.
    """
    kept = []
    for index, small in enumerate(regions):
        small_area = small.get("blendWidth", small["width"]) * small.get(
            "blendHeight", small["height"]
        )
        nested = False
        for other_index, large in enumerate(regions):
            if index == other_index:
                continue
            large_area = large.get("blendWidth", large["width"]) * large.get(
                "blendHeight", large["height"]
            )
            if (
                large["startFrame"] <= small["startFrame"]
                and large["endFrame"] >= small["endFrame"]
                and large_area >= area_ratio * small_area
                and intersection_fraction_of_smaller(small, large) >= coverage
            ):
                nested = True
                break
        if not nested:
            kept.append(small)
    return kept


def suppress_duplicate_regions(regions, overlap=0.45):
    """Apply contained-crop removal, then temporal region NMS.

    The detector can create several tracks for one large moving mosaic. When
    those tracks are restored independently, their projected deltas form a
    stack of translucent rectangles. Prefer the most confident track only when
    it covers the candidate's complete active interval and their visible areas
    have strong IoU. Distinct subjects and temporal continuations remain.
    """
    unnested = suppress_nested_regions(regions)
    ranked = sorted(
        enumerate(unnested),
        key=lambda item: (
            -(item[1]["endFrame"] - item[1]["startFrame"]),
            -item[1].get("confidence", 0.0),
            -item[1].get("blendWidth", item[1]["width"])
            * item[1].get("blendHeight", item[1]["height"]),
            item[0],
        ),
    )
    kept = []
    for original_index, candidate in ranked:
        duplicate = any(
            winner["startFrame"] <= candidate["startFrame"]
            and winner["endFrame"] >= candidate["endFrame"]
            and blend_iou(candidate, winner) >= overlap
            for _, winner in kept
        )
        if not duplicate:
            kept.append((original_index, candidate))
    return [region for _, region in sorted(kept)]


def tight_rect(boxes, frame_width, frame_height):
    left = max(0, int(math.floor(min(box[0] for box in boxes))))
    top = max(0, int(math.floor(min(box[1] for box in boxes))))
    right = min(frame_width, int(math.ceil(max(box[2] for box in boxes))))
    bottom = min(frame_height, int(math.ceil(max(box[3] for box in boxes))))
    return left, top, max(1, right - left), max(1, bottom - top)


def vr_model_crop(boxes, frame_width, frame_height, target_size=256):
    """Approximate Lada crop_to_box_v3: context crop, aspect fit, reflect padding."""
    left, top, width, height = tight_rect(boxes, frame_width, frame_height)
    original_right = left + width
    original_bottom = top + height
    border = max(20, int(max(width, height) * 0.06))
    left = max(0, left - border)
    top = max(0, top - border)
    right = min(frame_width, original_right + border)
    bottom = min(frame_height, original_bottom + border)
    width = right - left
    height = bottom - top

    scale = min(target_size / width, target_size / height, 1.0)
    missing_width = max(0, int((target_size - width * scale) / scale))
    missing_height = max(0, int((target_size - height * scale) / scale))
    grow_left = min(left, missing_width // 2, width)
    grow_right = min(frame_width - right, missing_width - grow_left, width - grow_left)
    grow_top = min(top, missing_height // 2, height)
    grow_bottom = min(frame_height - bottom, missing_height - grow_top, height - grow_top)
    left -= grow_left
    right += grow_right
    top -= grow_top
    bottom += grow_bottom
    return left, top, right - left, bottom - top


def mask_expansion_radius(size, expansion_fraction):
    return max(1, int(math.ceil(size * expansion_fraction)))


def box_polygon_groups(box):
    """Normalize one legacy polygon or several RF-DETR mask islands."""
    if len(box) <= 6 or not box[6]:
        return []
    payload = box[6]
    first = payload[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and all(isinstance(value, (int, float)) for value in first)
    ):
        return [payload]
    return [
        polygon
        for polygon in payload
        if isinstance(polygon, (list, tuple)) and len(polygon) >= 3
    ]


def has_box_polygons(box):
    return any(len(polygon) >= 3 for polygon in box_polygon_groups(box))


def segmentation_alpha_mask(
    boxes, rectangle, cv2, np, size=64, expansion_fraction=0.10
):
    """Rasterize tracked YOLO polygons into a compact soft region mask."""
    polygons = [
        polygon for box in boxes for polygon in box_polygon_groups(box)
        if len(polygon) >= 3
    ]
    if not polygons:
        return None
    x, y, width, height = rectangle
    mask = np.zeros((size, size), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).copy()
        points[:, 0] = (points[:, 0] - x) * (size - 1) / max(width - 1, 1)
        points[:, 1] = (points[:, 1] - y) * (size - 1) / max(height - 1, 1)
        points = np.rint(np.clip(points, 0, size - 1)).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    if not np.any(mask):
        return None
    # Mosaic encoders operate in large blocks that extend beyond the detector's
    # semantic contour. Expand proportionally so large VR crops do not retain a
    # staircase of original blocks, while keeping a rounded boundary.
    radius = mask_expansion_radius(size, expansion_fraction)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)
    blur_radius = max(2, int(math.ceil(radius / 2)))
    blur_size = blur_radius * 2 + 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), radius / 3)
    return {
        "maskWidth": size,
        "maskHeight": size,
        "maskData": base64.b64encode(mask.tobytes()).decode("ascii"),
    }


def mask_source_boxes(cluster, nearby, start_frame, end_frame):
    """Ensure padded/interpolated segments inherit the nearest real polygon."""
    if any(has_box_polygons(box) for box in nearby):
        return nearby
    polygon_boxes = [box for box in cluster if has_box_polygons(box)]
    if not polygon_boxes:
        return nearby
    midpoint = (start_frame + end_frame - 1) / 2
    return [min(polygon_boxes, key=lambda box: abs(float(box[5]) - midpoint))]


def mask_keyframe_box_groups(cluster, start_frame, end_frame, stride_frames):
    """Group real detector polygons into ordered, clamped mask keyframes."""
    polygon_boxes = [box for box in cluster if has_box_polygons(box)]
    selected = [
        box for box in polygon_boxes
        if start_frame - stride_frames <= int(box[5]) < end_frame + stride_frames
    ]
    if not selected and polygon_boxes:
        midpoint = (start_frame + end_frame - 1) / 2
        selected = [min(polygon_boxes, key=lambda box: abs(float(box[5]) - midpoint))]
    groups = {}
    for box in selected:
        frame = min(max(int(box[5]), start_frame), end_frame - 1)
        groups.setdefault(frame, []).append(box)
    return sorted(groups.items())


def segmentation_mask_keyframes(
    cluster,
    rectangle,
    start_frame,
    end_frame,
    stride_frames,
    cv2,
    np,
    size,
    expansion_fraction,
):
    keyframes = []
    for frame, boxes in mask_keyframe_box_groups(
        cluster, start_frame, end_frame, stride_frames
    ):
        mask = segmentation_alpha_mask(
            boxes,
            rectangle,
            cv2,
            np,
            size=size,
            expansion_fraction=expansion_fraction,
        )
        if mask is not None:
            keyframes.append({"frame": frame, "maskData": mask["maskData"]})
    return keyframes


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    if not args.input_video.is_file():
        raise SystemExit(f"input video not found: {args.input_video}")
    # Core ML exports are directory-backed .mlpackage bundles, whereas the
    # PyTorch detector is a regular .pt file.
    if not args.model.exists():
        raise SystemExit(f"mosaic detector not found: {args.model}")
    if (
        args.sample_stride <= 0
        or args.coarse_stride <= 0
        or args.coarse_confidence <= 0
        or args.coarse_confidence > 1
        or args.refine_padding < 0
        or args.region_duration <= 0
        or args.region_duration > 1.0
        or args.temporal_padding < 0
        or args.region_nms_iou <= 0
        or args.region_nms_iou > 1
        or args.mask_expansion <= 0
        or args.mask_expansion > 0.25
        or args.mask_size < 32
        or args.mask_size > 256
        or args.mask_size & (args.mask_size - 1) != 0
    ):
        raise SystemExit(
            "sample and coarse strides must be positive, coarse confidence must be "
            "in (0, 1], refinement padding cannot be negative, region duration must "
            "be in (0, 1], "
            "temporal padding cannot be negative, region NMS IoU must be in (0, 1], "
            "mask expansion must be in (0, 0.25], and mask size must be a power "
            "of two from 32 through 256"
        )
    if args.batch_size <= 0 or args.max_detections <= 0 or args.max_detections > 200:
        raise SystemExit("batch size must be positive and max detections must be from 1 to 200")

    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as error:
        raise SystemExit(
            "mosaic detector dependencies are missing; run script/setup_mosaic_detector.sh"
        ) from error

    if args.backend == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise SystemExit(
                "YOLO detector dependencies are missing; run "
                "script/setup_mosaic_detector.sh"
            ) from error
    else:
        try:
            from rfdetr_mps_detector import RFDetrMPSDetector
        except ImportError as error:
            raise SystemExit(
                "RF-DETR dependencies are missing from .venv-rfdetr"
            ) from error

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise SystemExit(f"unable to decode video: {args.input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or source_fps <= 0 or frame_count <= 0:
        raise SystemExit("video metadata is incomplete")
    if abs(source_fps - 30.0) > 0.05:
        raise SystemExit(f"sparse restoration requires a 30 fps eye video, got {source_fps:.3f}")
    capture.release()

    device = choose_device(torch, args.device)
    print(
        f"Scanning {width}x{height}, {frame_count} frames at {source_fps:.3f} fps "
        f"on {device}; {args.backend} detector, {args.decode_mode} decode, "
        f"batch {args.batch_size}",
        flush=True,
    )
    if args.backend == "yolo":
        # Exported Core ML packages do not reliably retain enough metadata for
        # Ultralytics to infer that this checkpoint is a segmentation model.
        model = YOLO(str(args.model), task="segment")
    else:
        model = RFDetrMPSDetector(
            args.model, device=device, max_select=args.max_detections
        )
    stride_frames = max(1, int(round(args.sample_stride * source_fps)))
    region_frames = max(stride_frames, int(round(args.region_duration * source_fps)))
    padding_frames = int(round(args.temporal_padding * source_fps))
    window_frames = 30
    window_boxes: dict[int, list[tuple[float, float, float, float, float]]] = {}
    scan_started = time.perf_counter()

    def predict(frames, confidence):
        nonlocal device, model
        try:
            if args.backend == "rfdetr":
                return model.predict(frames, score_threshold=confidence)
            source = frames if len(frames) > 1 else frames[0]
            return model.predict(
                source, imgsz=args.image_size, conf=confidence,
                device=device, verbose=False,
            )
        except Exception:
            if device != "mps":
                raise
            print("MPS detector failed; retrying the scan on CPU", file=sys.stderr, flush=True)
            device = "cpu"
            if args.backend == "rfdetr":
                model = RFDetrMPSDetector(
                    args.model, device=device, max_select=args.max_detections
                )
                return model.predict(frames, score_threshold=confidence)
            source = frames if len(frames) > 1 else frames[0]
            return model.predict(
                source,
                imgsz=args.image_size,
                conf=confidence,
                device=device,
                verbose=False,
            )

    def scan_samples(sample_indices, phase_name, confidence):
        phase_started = time.perf_counter()
        phase_inference_seconds = 0.0
        phase_boxes = []
        phase_scanned_samples = 0
        phase_capture = cv2.VideoCapture(str(args.input_video))
        if not phase_capture.isOpened():
            raise RuntimeError(f"unable to reopen video for {phase_name} scan")

        def process_batch(frames, frame_indices):
            nonlocal phase_inference_seconds, phase_scanned_samples
            inference_started = time.perf_counter()
            predictions = predict(frames, confidence)
            phase_inference_seconds += time.perf_counter() - inference_started
            if len(predictions) != len(frame_indices):
                raise RuntimeError(
                    f"detector returned {len(predictions)} results for "
                    f"{len(frame_indices)} frames"
                )
            for result, frame_index in zip(predictions, frame_indices):
                boxes = []
                if args.backend == "rfdetr":
                    boxes = [
                        tuple(coords) + (float(conf), frame_index, polygon)
                        for coords, conf, polygon in zip(
                            result.boxes_xyxy, result.confidences, result.polygons
                        )
                    ]
                elif result.boxes is not None:
                    coordinates = result.boxes.xyxy.detach().cpu().tolist()
                    confidences = result.boxes.conf.detach().cpu().tolist()
                    polygons = result.masks.xy if result.masks is not None else []
                    boxes = [
                        tuple(coords)
                        + (
                            float(conf),
                            frame_index,
                            polygons[index].tolist() if index < len(polygons) else [],
                        )
                        for index, (coords, conf) in enumerate(
                            zip(coordinates, confidences)
                        )
                    ]
                phase_boxes.extend(boxes)
                phase_scanned_samples += 1
                if (
                    phase_scanned_samples == 1
                    or phase_scanned_samples == len(sample_indices)
                    or phase_scanned_samples % 10 == 0
                ):
                    print(
                        f"{phase_name} sample {phase_scanned_samples}/"
                        f"{len(sample_indices)} at {frame_index / source_fps:.2f}s; "
                        f"detections {len(boxes)}",
                        flush=True,
                    )

        batch_frames = []
        batch_indices = []

        def append_sample(frame, frame_index):
            batch_frames.append(frame)
            batch_indices.append(frame_index)
            if len(batch_frames) >= args.batch_size:
                process_batch(batch_frames, batch_indices)
                batch_frames.clear()
                batch_indices.clear()

        if args.decode_mode == "seek":
            for frame_index in sample_indices:
                phase_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = phase_capture.read()
                if not ok:
                    print(
                        f"warning: unable to decode sampled frame {frame_index}",
                        file=sys.stderr,
                    )
                    continue
                append_sample(frame, frame_index)
        else:
            next_sample = iter(sample_indices)
            target_frame = next(next_sample, None)
            for frame_index in range(frame_count):
                ok = phase_capture.grab()
                if not ok:
                    print(
                        f"warning: decoding stopped at frame {frame_index}",
                        file=sys.stderr,
                    )
                    break
                if frame_index != target_frame:
                    continue
                ok, frame = phase_capture.retrieve()
                if not ok:
                    print(
                        f"warning: unable to retrieve sampled frame {frame_index}",
                        file=sys.stderr,
                    )
                else:
                    append_sample(frame, frame_index)
                target_frame = next(next_sample, None)
                if target_frame is None:
                    break
        if batch_frames:
            process_batch(batch_frames, batch_indices)
        phase_capture.release()
        return (
            phase_boxes,
            phase_scanned_samples,
            phase_inference_seconds,
            time.perf_counter() - phase_started,
        )

    if not args.adaptive_scan:
        sample_indices = list(range(0, frame_count, stride_frames))
        boxes, scanned_samples, inference_seconds, dense_seconds = scan_samples(
            sample_indices, "Dense", args.confidence
        )
        coarse_seconds = 0.0
        coarse_sample_count = 0
        active_coarse_seconds = 0
        refinement_sample_count = scanned_samples
    else:
        coarse_indices = coarse_sample_indices(
            frame_count, source_fps, args.coarse_stride
        )
        coarse_boxes, coarse_sample_count, coarse_inference, coarse_seconds = (
            scan_samples(coarse_indices, "Coarse", args.coarse_confidence)
        )
        detected_frames = [int(box[5]) for box in coarse_boxes]
        active_coarse_seconds = len(
            {frame_index // window_frames for frame_index in detected_frames}
        )
        refinement_indices = refinement_sample_indices(
            frame_count,
            source_fps,
            args.sample_stride,
            detected_frames,
            args.refine_padding,
        )
        if refinement_indices:
            (
                refined_boxes,
                refinement_sample_count,
                refine_inference,
                dense_seconds,
            ) = scan_samples(refinement_indices, "Refine", args.confidence)
        else:
            refined_boxes = []
            refinement_sample_count = 0
            refine_inference = 0.0
            dense_seconds = 0.0
        boxes = refined_boxes
        scanned_samples = coarse_sample_count + refinement_sample_count
        inference_seconds = coarse_inference + refine_inference

    for box in boxes:
        frame_index = int(box[5])
        first_window = max(0, frame_index - padding_frames) // window_frames
        last_window = min(frame_count - 1, frame_index + padding_frames) // window_frames
        for window_index in range(first_window, last_window + 1):
            window_boxes.setdefault(window_index, []).append(box)
    scan_seconds = time.perf_counter() - scan_started

    regions = []
    for window_index, boxes in sorted(window_boxes.items()):
        start_frame = window_index * window_frames
        end_frame = min(frame_count, start_frame + window_frames)
        for cluster in track_boxes(boxes):
            active_start = max(start_frame, int(min(box[5] for box in cluster)) - padding_frames)
            active_end = min(
                end_frame, int(max(box[5] for box in cluster)) + padding_frames + 1
            )
            first_segment = (active_start // region_frames) * region_frames
            for segment_start in range(first_segment, active_end, region_frames):
                segment_end = min(end_frame, active_end, segment_start + region_frames)
                clipped_start = max(start_frame, active_start, segment_start)
                if segment_end <= clipped_start:
                    continue
                nearby = [
                    box for box in cluster
                    if clipped_start - stride_frames <= int(box[5]) < segment_end + stride_frames
                ]
                nearby.extend(
                    [
                        interpolated_box(cluster, clipped_start),
                        interpolated_box(cluster, segment_end - 1),
                    ]
                )
                rectangle = vr_model_crop(nearby, width, height)
                blend_rectangle = tight_rect(nearby, width, height)
                confidence = max(box[4] for box in nearby)
                x, y, region_width, region_height = rectangle
                blend_x, blend_y, blend_width, blend_height = blend_rectangle
                region = {
                        "startFrame": clipped_start,
                        "endFrame": segment_end,
                        "x": x,
                        "y": y,
                        "width": region_width,
                        "height": region_height,
                        "confidence": confidence,
                        "blendX": blend_x,
                        "blendY": blend_y,
                        "blendWidth": blend_width,
                        "blendHeight": blend_height,
                    }
                mask_boxes = mask_source_boxes(cluster, nearby, clipped_start, segment_end)
                mask = segmentation_alpha_mask(
                    mask_boxes,
                    rectangle,
                    cv2,
                    np,
                    size=args.mask_size,
                    expansion_fraction=args.mask_expansion,
                )
                if mask is not None:
                    region.update(mask)
                    keyframes = segmentation_mask_keyframes(
                        cluster,
                        rectangle,
                        clipped_start,
                        segment_end,
                        stride_frames,
                        cv2,
                        np,
                        size=args.mask_size,
                        expansion_fraction=args.mask_expansion,
                    )
                    if keyframes:
                        region["maskKeyframes"] = keyframes
                regions.append(region)

    unsuppressed_region_count = len(regions)
    regions = suppress_duplicate_regions(regions, overlap=args.region_nms_iou)
    suppressed_region_count = unsuppressed_region_count - len(regions)
    masked_region_count = sum("maskData" in region for region in regions)
    temporal_mask_region_count = sum("maskKeyframes" in region for region in regions)
    temporal_mask_count = sum(
        len(region.get("maskKeyframes", [])) for region in regions
    )
    average_active_regions, scheduled_blend_percent = detector_coverage_metrics(
        regions, width, height, frame_count
    )

    manifest = {
        "version": 1,
        "width": width,
        "height": height,
        "framesPerSecond": 30.0,
        "frameCount": frame_count,
        "regions": regions,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_manifest)
    total_windows = math.ceil(frame_count / window_frames)
    affected_windows = len({region["startFrame"] // window_frames for region in regions})
    print(
        f"Saved {len(regions)} regions across {affected_windows} affected windows "
        f"out of {total_windows} to "
        f"{args.output_manifest}",
        flush=True,
    )
    print(
        f"Suppressed {suppressed_region_count} nested/overlapping duplicate regions",
        flush=True,
    )
    print(
        f"Segmentation masks: {masked_region_count}/{len(regions)} regions",
        flush=True,
    )
    print(
        f"Temporal masks: {temporal_mask_count} keyframes across "
        f"{temporal_mask_region_count}/{len(regions)} regions",
        flush=True,
    )
    print(
        f"Detector coverage: {average_active_regions:.2f} active regions/frame, "
        f"{scheduled_blend_percent:.3f}% scheduled blend area/eye-frame",
        flush=True,
    )
    if args.adaptive_scan:
        print(
            f"Detector coarse gate: {coarse_seconds:.3f}s, "
            f"{coarse_sample_count} gate samples, "
            f"{active_coarse_seconds}/{total_windows} seconds flagged",
            flush=True,
        )
        print(
            f"Detector refinement: {dense_seconds:.3f}s, "
            f"{refinement_sample_count} dense samples around flagged seconds",
            flush=True,
        )
    print(
        f"Detector scan: {scan_seconds:.3f}s, "
        f"{scanned_samples / scan_seconds:.2f} sampled frames/s",
        flush=True,
    )
    print(
        f"Detector phases: inference {inference_seconds:.3f}s, "
        f"decode/batching {max(0.0, scan_seconds - inference_seconds):.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
