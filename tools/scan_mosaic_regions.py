#!/usr/bin/env python3
"""Create a sparse restoration manifest using VR Video Toolbox's YOLO workflow."""

from __future__ import annotations

import argparse
import base64
import copy
import gc
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
    parser.add_argument(
        "--stereo-right-manifest",
        type=Path,
        help="with --crop-eye both, write the right-eye manifest here",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("yolo", "rfdetr"), default="yolo"
    )
    parser.add_argument(
        "--rfdetr-variant",
        choices=("medium", "large"),
        default="large",
        help="RF-DETR architecture matching the selected checkpoint",
    )
    parser.add_argument("--sample-stride", type=float, default=0.1)
    parser.add_argument(
        "--crop-eye",
        choices=("none", "left", "right", "both"),
        default="none",
        help="scan one or both halves of an SBS source without intermediate eye videos",
    )
    parser.add_argument(
        "--stereo-sample-mode",
        choices=("paired", "alternating"),
        default="paired",
        help=(
            "with --crop-eye both, scan both eyes at every sample or alternate "
            "eyes between consecutive samples"
        ),
    )
    parser.add_argument(
        "--active-ranges",
        help="optional comma-separated segment-relative start/end seconds",
    )
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
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
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


def write_manifest(path, width, height, frame_count, regions):
    manifest = {
        "version": 1,
        "width": width,
        "height": height,
        "framesPerSecond": 30.0,
        "frameCount": frame_count,
        "regions": regions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def active_frame_intervals(spec, source_fps, frame_count):
    if spec is None:
        return [(0, frame_count)]
    intervals = []
    for item in filter(None, spec.split(",")):
        try:
            start_text, end_text = item.split("/")
            start = max(0, int(math.floor(float(start_text) * source_fps)))
            end = min(frame_count, int(math.ceil(float(end_text) * source_fps)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid active range: {item}") from error
        if end > start:
            intervals.append((start, end))
    return intervals


def samples_in_intervals(indices, intervals):
    return [index for index in indices if any(start <= index < end for start, end in intervals)]


def stereo_sample_eyes(sample_ordinal, mode):
    """Return the SBS eye crops scheduled for one temporal detector sample."""
    if mode == "paired":
        return ("left", "right")
    if mode == "alternating":
        return ("left",) if sample_ordinal % 2 == 0 else ("right",)
    raise ValueError(f"unsupported stereo sample mode: {mode}")


def clip_regions_to_intervals(regions, intervals):
    clipped = []
    for region in regions:
        for start, end in intervals:
            clipped_start = max(int(region["startFrame"]), start)
            clipped_end = min(int(region["endFrame"]), end)
            if clipped_end <= clipped_start:
                continue
            item = copy.deepcopy(region)
            item["startFrame"] = clipped_start
            item["endFrame"] = clipped_end
            if "maskKeyframes" in item:
                keyframes = [
                    keyframe for keyframe in item["maskKeyframes"]
                    if clipped_start <= int(keyframe["frame"]) < clipped_end
                ]
                if keyframes:
                    item["maskKeyframes"] = keyframes
                else:
                    item.pop("maskKeyframes")
            clipped.append(item)
    return clipped


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


def decoded_mask_payload(region, payload):
    width = region.get("maskWidth")
    height = region.get("maskHeight")
    if not width or not height or not payload:
        return None
    try:
        data = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        return None
    if len(data) != width * height:
        return None
    return width, height, data


def decoded_region_mask(region):
    return decoded_mask_payload(region, region.get("maskData"))


def mask_support_coverage(small, small_mask, large, large_mask, threshold=128):
    small_width, small_height, small_data = small_mask
    large_width, large_height, large_data = large_mask
    active = 0
    covered = 0
    sample_step = max(1, min(small_width, small_height) // 32)
    sample_offset = sample_step // 2
    for mask_y in range(sample_offset, small_height, sample_step):
        for mask_x in range(sample_offset, small_width, sample_step):
            if small_data[mask_y * small_width + mask_x] < threshold:
                continue
            active += 1
            pixel_x = small["x"] + mask_x * max(small["width"] - 1, 0) / max(
                small_width - 1, 1
            )
            pixel_y = small["y"] + mask_y * max(small["height"] - 1, 0) / max(
                small_height - 1, 1
            )
            large_x = round(
                (pixel_x - large["x"])
                * max(large_width - 1, 0)
                / max(large["width"] - 1, 1)
            )
            large_y = round(
                (pixel_y - large["y"])
                * max(large_height - 1, 0)
                / max(large["height"] - 1, 1)
            )
            if (
                0 <= large_x < large_width
                and 0 <= large_y < large_height
                and large_data[large_y * large_width + large_x] >= threshold
            ):
                covered += 1
    return covered / active if active else None


def nested_mask_coverage(small, large, threshold=128):
    """Measure how much of a small mask is covered by a containing mask.

    ``None`` means rectangle-only legacy suppression remains appropriate. An
    unmasked outer region blends its complete rectangle, while an unmasked
    inner region cannot safely be discarded in favour of a masked outer one.
    """
    small_mask = decoded_region_mask(small)
    large_mask = decoded_region_mask(large)
    if large_mask is None:
        return None
    if small_mask is None:
        return 0.0
    aggregate_coverage = mask_support_coverage(
        small, small_mask, large, large_mask, threshold
    )
    return aggregate_coverage or 0.0


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
                mask_coverage = nested_mask_coverage(small, large)
                if mask_coverage is not None and mask_coverage < coverage:
                    continue
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
        duplicate = False
        for _, winner in kept:
            if not (
                winner["startFrame"] <= candidate["startFrame"]
                and winner["endFrame"] >= candidate["endFrame"]
                and blend_iou(candidate, winner) >= overlap
            ):
                continue
            mask_coverage = nested_mask_coverage(candidate, winner)
            if mask_coverage is None or mask_coverage >= 0.8:
                duplicate = True
                break
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


def interpolated_polygon_box(boxes, frame_index):
    """Move the nearest real polygon onto an interpolated tracked box."""
    polygon_boxes = [box for box in boxes if has_box_polygons(box)]
    if not polygon_boxes:
        return None
    target = interpolated_box(polygon_boxes, frame_index)
    source = min(
        polygon_boxes,
        key=lambda box: abs(float(box[5]) - frame_index),
    )
    source_width = max(float(source[2]) - float(source[0]), 1.0)
    source_height = max(float(source[3]) - float(source[1]), 1.0)
    target_width = max(float(target[2]) - float(target[0]), 1.0)
    target_height = max(float(target[3]) - float(target[1]), 1.0)
    transformed = []
    for polygon in box_polygon_groups(source):
        transformed.append([
            [
                float(target[0])
                + (float(point[0]) - float(source[0])) * target_width / source_width,
                float(target[1])
                + (float(point[1]) - float(source[1])) * target_height / source_height,
            ]
            for point in polygon
        ])
    payload = transformed[0] if len(transformed) == 1 else transformed
    return tuple(target) + (payload,)


def dense_mask_keyframe_box_groups(cluster, start_frame, end_frame, stride_frames):
    """Create a polygon mask at every output frame along the tracked motion."""
    sparse_groups = mask_keyframe_box_groups(
        cluster, start_frame, end_frame, stride_frames
    )
    if not sparse_groups:
        return []
    exact_by_frame = dict(sparse_groups)
    polygon_boxes = [box for _, boxes in sparse_groups for box in boxes]
    groups = []
    for frame in range(start_frame, end_frame):
        exact = exact_by_frame.get(frame)
        if exact:
            groups.append((frame, exact))
            continue
        interpolated = interpolated_polygon_box(polygon_boxes, frame)
        if interpolated is not None:
            groups.append((frame, [interpolated]))
    return groups


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
    for frame, boxes in dense_mask_keyframe_box_groups(
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
    if args.crop_eye == "both" and args.stereo_right_manifest is None:
        raise SystemExit("--crop-eye both requires --stereo-right-manifest")
    if args.crop_eye != "both" and args.stereo_right_manifest is not None:
        raise SystemExit("--stereo-right-manifest requires --crop-eye both")
    if args.stereo_sample_mode == "alternating" and args.crop_eye != "both":
        raise SystemExit("--stereo-sample-mode alternating requires --crop-eye both")
    if args.stereo_sample_mode == "alternating" and args.adaptive_scan:
        raise SystemExit(
            "--stereo-sample-mode alternating cannot be combined with --adaptive-scan"
        )

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
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or height <= 0 or source_fps <= 0 or frame_count <= 0:
        raise SystemExit("video metadata is incomplete")
    if args.crop_eye != "none" and source_width % 2 != 0:
        raise SystemExit("side-by-side eye cropping requires an even source width")
    width = source_width // 2 if args.crop_eye != "none" else source_width
    if abs(source_fps - 30.0) > 0.05:
        raise SystemExit(f"sparse restoration requires a 30 fps eye video, got {source_fps:.3f}")
    capture.release()
    try:
        allowed_intervals = active_frame_intervals(
            args.active_ranges, source_fps, frame_count
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.active_ranges is not None:
        print(
            f"Manual range gate: {len(allowed_intervals)} active interval(s)",
            flush=True,
        )
    if not allowed_intervals:
        write_manifest(args.output_manifest, width, height, frame_count, [])
        if args.stereo_right_manifest is not None:
            write_manifest(args.stereo_right_manifest, width, height, frame_count, [])
        print(
            f"Manual range gate: clean segment, detector bypassed; saved "
            f"{args.output_manifest}"
            + (
                f" and {args.stereo_right_manifest}"
                if args.stereo_right_manifest is not None
                else ""
            ),
            flush=True,
        )
        return 0

    device = choose_device(torch, args.device)
    detector_description = args.backend
    if args.backend == "rfdetr":
        detector_description += f"/{args.rfdetr_variant}"
    print(
        f"Scanning {width}x{height}, {frame_count} frames at {source_fps:.3f} fps "
        f"on {device}; {detector_description} detector, {args.decode_mode} decode, "
        f"batch {args.batch_size}"
        + (
            f"; shared stereo model/decode, {args.stereo_sample_mode} eye sampling"
            if args.crop_eye == "both"
            else ""
        ),
        flush=True,
    )
    if args.backend == "yolo":
        # Exported Core ML packages do not reliably retain enough metadata for
        # Ultralytics to infer that this checkpoint is a segmentation model.
        model = YOLO(str(args.model), task="segment")
    else:
        model = RFDetrMPSDetector(
            args.model,
            device=device,
            variant=args.rfdetr_variant,
            max_select=args.max_detections,
        )
    stride_frames = max(1, int(round(args.sample_stride * source_fps)))
    region_frames = max(stride_frames, int(round(args.region_duration * source_fps)))
    padding_frames = int(round(args.temporal_padding * source_fps))
    window_frames = 30
    scan_eyes = ("left", "right") if args.crop_eye == "both" else ("single",)
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
            model = None
            gc.collect()
            torch.mps.empty_cache()
            device = "cpu"
            if args.backend == "rfdetr":
                model = RFDetrMPSDetector(
                    args.model,
                    device=device,
                    variant=args.rfdetr_variant,
                    max_select=args.max_detections,
                )
                return model.predict(frames, score_threshold=confidence)
            model = YOLO(str(args.model), task="segment")
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
        phase_boxes = {eye: [] for eye in scan_eyes}
        phase_scanned_samples = 0
        total_sample_images = len(sample_indices) * len(scan_eyes)
        if args.crop_eye == "both" and args.stereo_sample_mode == "alternating":
            total_sample_images = len(sample_indices)
        phase_capture = cv2.VideoCapture(str(args.input_video))
        if not phase_capture.isOpened():
            raise RuntimeError(f"unable to reopen video for {phase_name} scan")

        def process_batch(frames, frame_samples):
            nonlocal phase_inference_seconds, phase_scanned_samples
            inference_started = time.perf_counter()
            predictions = predict(frames, confidence)
            phase_inference_seconds += time.perf_counter() - inference_started
            if len(predictions) != len(frame_samples):
                raise RuntimeError(
                    f"detector returned {len(predictions)} results for "
                    f"{len(frame_samples)} frames"
                )
            for result, (frame_index, eye) in zip(predictions, frame_samples):
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
                phase_boxes[eye].extend(boxes)
                phase_scanned_samples += 1
                if (
                    phase_scanned_samples == 1
                    or phase_scanned_samples == total_sample_images
                    or phase_scanned_samples % 10 == 0
                ):
                    print(
                        f"{phase_name} sample {phase_scanned_samples}/"
                        f"{total_sample_images} at {frame_index / source_fps:.2f}s"
                        + (f" ({eye})" if args.crop_eye == "both" else "")
                        + f"; detections {len(boxes)}",
                        flush=True,
                    )

        batch_frames = []
        batch_samples = []

        def append_sample(frame, frame_index, eye="single"):
            if args.crop_eye == "left":
                frame = frame[:, :width]
            elif args.crop_eye == "right":
                frame = frame[:, source_width - width:]
            batch_frames.append(frame)
            batch_samples.append((frame_index, eye))
            if len(batch_frames) >= args.batch_size:
                process_batch(batch_frames, batch_samples)
                batch_frames.clear()
                batch_samples.clear()

        def append_source_sample(frame, frame_index, sample_ordinal):
            if args.crop_eye == "both":
                for eye in stereo_sample_eyes(sample_ordinal, args.stereo_sample_mode):
                    if eye == "left":
                        append_sample(frame[:, :width], frame_index, eye)
                    else:
                        append_sample(frame[:, source_width - width:], frame_index, eye)
            else:
                append_sample(frame, frame_index)

        if args.decode_mode == "seek":
            for sample_ordinal, frame_index in enumerate(sample_indices):
                phase_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = phase_capture.read()
                if not ok:
                    print(
                        f"warning: unable to decode sampled frame {frame_index}",
                        file=sys.stderr,
                    )
                    continue
                append_source_sample(frame, frame_index, sample_ordinal)
        else:
            next_sample = iter(enumerate(sample_indices))
            target_sample = next(next_sample, None)
            for frame_index in range(frame_count):
                ok = phase_capture.grab()
                if not ok:
                    print(
                        f"warning: decoding stopped at frame {frame_index}",
                        file=sys.stderr,
                    )
                    break
                if target_sample is None or frame_index != target_sample[1]:
                    continue
                ok, frame = phase_capture.retrieve()
                if not ok:
                    print(
                        f"warning: unable to retrieve sampled frame {frame_index}",
                        file=sys.stderr,
                    )
                else:
                    append_source_sample(frame, frame_index, target_sample[0])
                target_sample = next(next_sample, None)
                if target_sample is None:
                    break
        if batch_frames:
            process_batch(batch_frames, batch_samples)
        phase_capture.release()
        return (
            phase_boxes,
            phase_scanned_samples,
            phase_inference_seconds,
            time.perf_counter() - phase_started,
        )

    if not args.adaptive_scan:
        sample_indices = samples_in_intervals(
            range(0, frame_count, stride_frames), allowed_intervals
        )
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
        coarse_indices = samples_in_intervals(coarse_indices, allowed_intervals)
        coarse_boxes, coarse_sample_count, coarse_inference, coarse_seconds = (
            scan_samples(coarse_indices, "Coarse", args.coarse_confidence)
        )
        detected_frames = [
            int(box[5])
            for eye_boxes in coarse_boxes.values()
            for box in eye_boxes
        ]
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
        refinement_indices = samples_in_intervals(
            refinement_indices, allowed_intervals
        )
        if refinement_indices:
            (
                refined_boxes,
                refinement_sample_count,
                refine_inference,
                dense_seconds,
            ) = scan_samples(refinement_indices, "Refine", args.confidence)
        else:
            refined_boxes = {eye: [] for eye in scan_eyes}
            refinement_sample_count = 0
            refine_inference = 0.0
            dense_seconds = 0.0
        boxes = refined_boxes
        scanned_samples = coarse_sample_count + refinement_sample_count
        inference_seconds = coarse_inference + refine_inference

    scan_seconds = time.perf_counter() - scan_started

    def build_regions(detected_boxes):
        window_boxes = {}
        for box in detected_boxes:
            frame_index = int(box[5])
            first_window = max(0, frame_index - padding_frames) // window_frames
            last_window = min(frame_count - 1, frame_index + padding_frames) // window_frames
            for window_index in range(first_window, last_window + 1):
                window_boxes.setdefault(window_index, []).append(box)
        regions = []
        for window_index, window_detections in sorted(window_boxes.items()):
            start_frame = window_index * window_frames
            end_frame = min(frame_count, start_frame + window_frames)
            for cluster in track_boxes(window_detections):
                regions.extend(
                    regions_for_cluster(cluster, start_frame, end_frame)
                )
        return regions

    def regions_for_cluster(cluster, start_frame, end_frame):
        cluster_regions = []
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
            cluster_regions.append(region)
        return cluster_regions

    regions_by_eye = {eye: build_regions(boxes[eye]) for eye in scan_eyes}

    for eye, regions in regions_by_eye.items():
        regions = clip_regions_to_intervals(regions, allowed_intervals)
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
        output_manifest = (
            args.stereo_right_manifest
            if eye == "right"
            else args.output_manifest
        )
        write_manifest(output_manifest, width, height, frame_count, regions)
        total_windows = math.ceil(frame_count / window_frames)
        affected_windows = len({region["startFrame"] // window_frames for region in regions})
        eye_suffix = f" ({eye} eye)" if args.crop_eye == "both" else ""
        print(
            f"Saved {len(regions)} regions across {affected_windows} affected windows "
            f"out of {total_windows} to {output_manifest}",
            flush=True,
        )
        print(
            f"Suppressed {suppressed_region_count} nested/overlapping duplicate "
            f"regions{eye_suffix}",
            flush=True,
        )
        print(
            f"Segmentation masks: {masked_region_count}/{len(regions)} "
            f"regions{eye_suffix}",
            flush=True,
        )
        print(
            f"Temporal masks: {temporal_mask_count} keyframes across "
            f"{temporal_mask_region_count}/{len(regions)} regions{eye_suffix}",
            flush=True,
        )
        print(
            f"Detector coverage: {average_active_regions:.2f} active regions/frame, "
            f"{scheduled_blend_percent:.3f}% scheduled blend area/eye-frame"
            f"{eye_suffix}",
            flush=True,
        )
    total_windows = math.ceil(frame_count / window_frames)
    sample_label = "eye-images" if args.crop_eye == "both" else "frames"
    if args.adaptive_scan:
        print(
            f"Detector coarse gate: {coarse_seconds:.3f}s, "
            f"{coarse_sample_count} gate {sample_label}, "
            f"{active_coarse_seconds}/{total_windows} seconds flagged",
            flush=True,
        )
        print(
            f"Detector refinement: {dense_seconds:.3f}s, "
            f"{refinement_sample_count} dense {sample_label} around flagged seconds",
            flush=True,
        )
    print(
        f"Detector scan: {scan_seconds:.3f}s, "
        f"{scanned_samples / scan_seconds:.2f} sampled {sample_label}/s",
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
