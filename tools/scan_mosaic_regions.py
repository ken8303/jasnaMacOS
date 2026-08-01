#!/usr/bin/env python3
"""Create a sparse restoration manifest using VR Video Toolbox's YOLO workflow."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--sample-stride", type=float, default=0.1)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
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
    if not args.model.is_file():
        raise SystemExit(f"mosaic detector not found: {args.model}")
    if (
        args.sample_stride <= 0
        or args.region_duration <= 0
        or args.region_duration > 1.0
        or args.temporal_padding < 0
    ):
        raise SystemExit(
            "sample stride must be positive, region duration must be in (0, 1], "
            "and temporal padding cannot be negative"
        )
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")

    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "mosaic detector dependencies are missing; run script/setup_mosaic_detector.sh"
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

    device = choose_device(torch, args.device)
    print(
        f"Scanning {width}x{height}, {frame_count} frames at {source_fps:.3f} fps "
        f"on {device}; {args.decode_mode} decode, batch {args.batch_size}",
        flush=True,
    )
    model = YOLO(str(args.model))
    stride_frames = max(1, int(round(args.sample_stride * source_fps)))
    region_frames = max(stride_frames, int(round(args.region_duration * source_fps)))
    padding_frames = int(round(args.temporal_padding * source_fps))
    window_frames = 30
    window_boxes: dict[int, list[tuple[float, float, float, float, float]]] = {}
    sample_indices = list(range(0, frame_count, stride_frames))
    scan_started = time.perf_counter()

    def predict(frames):
        nonlocal device
        source = frames if len(frames) > 1 else frames[0]
        try:
            return model.predict(
                source,
                imgsz=args.image_size,
                conf=args.confidence,
                device=device,
                verbose=False,
            )
        except Exception:
            if device != "mps":
                raise
            print("MPS detector failed; retrying the scan on CPU", file=sys.stderr, flush=True)
            device = "cpu"
            return model.predict(
                source,
                imgsz=args.image_size,
                conf=args.confidence,
                device=device,
                verbose=False,
            )

    scanned_samples = 0

    def process_batch(frames, frame_indices):
        nonlocal scanned_samples
        predictions = predict(frames)
        if len(predictions) != len(frame_indices):
            raise RuntimeError(
                f"detector returned {len(predictions)} results for "
                f"{len(frame_indices)} frames"
            )
        for result, frame_index in zip(predictions, frame_indices):
            boxes = []
            if result.boxes is not None:
                coordinates = result.boxes.xyxy.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                boxes = [
                    tuple(coords) + (float(conf), frame_index)
                    for coords, conf in zip(coordinates, confidences)
                ]
            if boxes:
                first_window = max(0, frame_index - padding_frames) // window_frames
                last_window = min(frame_count - 1, frame_index + padding_frames) // window_frames
                for window_index in range(first_window, last_window + 1):
                    window_boxes.setdefault(window_index, []).extend(boxes)
            scanned_samples += 1
            if (
                scanned_samples == 1
                or scanned_samples == len(sample_indices)
                or scanned_samples % 10 == 0
            ):
                print(
                    f"Scanned sample {scanned_samples}/{len(sample_indices)} at "
                    f"{frame_index / source_fps:.2f}s; detections {len(boxes)}",
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
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                print(f"warning: unable to decode sampled frame {frame_index}", file=sys.stderr)
                continue
            append_sample(frame, frame_index)
    else:
        next_sample = iter(sample_indices)
        target_frame = next(next_sample, None)
        for frame_index in range(frame_count):
            ok = capture.grab()
            if not ok:
                print(f"warning: decoding stopped at frame {frame_index}", file=sys.stderr)
                break
            if frame_index != target_frame:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                print(f"warning: unable to retrieve sampled frame {frame_index}", file=sys.stderr)
            else:
                append_sample(frame, frame_index)
            target_frame = next(next_sample, None)
            if target_frame is None:
                break

    if batch_frames:
        process_batch(batch_frames, batch_indices)
    capture.release()
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
                regions.append(
                    {
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
        f"Detector scan: {scan_seconds:.3f}s, "
        f"{scanned_samples / scan_seconds:.2f} sampled frames/s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
