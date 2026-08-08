#!/usr/bin/env python3
"""Composite saved DeepMosaics region results into a full-resolution preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_region_result(value: str) -> tuple[int, Path, float, float]:
    pieces = value.split(",")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError(
            "region-result must be REGION,NPY_PATH,EXPANSION,POST_FILTER_SIGMA"
        )
    try:
        return int(pieces[0]), Path(pieces[1]), float(pieces[2]), float(pieces[3])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument(
        "--region-result", type=parse_region_result, action="append", required=True
    )
    parser.add_argument("--feather-pixels", type=int, default=16)
    return parser.parse_args()


def expanded_square(
    region: dict, frame_width: int, frame_height: int, expansion: float
) -> tuple[int, int, int]:
    x = int(region.get("blendX", region["x"]))
    y = int(region.get("blendY", region["y"]))
    width = int(region.get("blendWidth", region["width"]))
    height = int(region.get("blendHeight", region["height"]))
    side = min(max(2, int(max(width, height) * expansion)), frame_width, frame_height)
    left = max(0, min(frame_width - side, int(round(x + width / 2 - side / 2))))
    top = max(0, min(frame_height - side, int(round(y + height / 2 - side / 2))))
    return left, top, side


def composite_region(
    frame_bgr: np.ndarray,
    restored_rgb: np.ndarray,
    region: dict,
    expansion: float,
    feather_pixels: int,
) -> None:
    frame_height, frame_width = frame_bgr.shape[:2]
    left, top, side = expanded_square(region, frame_width, frame_height, expansion)
    restored_bgr = cv2.resize(
        restored_rgb[:, :, ::-1], (side, side), interpolation=cv2.INTER_CUBIC
    )
    blend_x = int(region.get("blendX", region["x"]))
    blend_y = int(region.get("blendY", region["y"]))
    blend_width = int(region.get("blendWidth", region["width"]))
    blend_height = int(region.get("blendHeight", region["height"]))
    x0 = max(0, blend_x)
    y0 = max(0, blend_y)
    x1 = min(frame_width, blend_x + blend_width)
    y1 = min(frame_height, blend_y + blend_height)
    if x1 <= x0 or y1 <= y0:
        return

    restored_x0 = x0 - left
    restored_y0 = y0 - top
    restored_roi = restored_bgr[
        restored_y0:restored_y0 + (y1 - y0),
        restored_x0:restored_x0 + (x1 - x0),
    ].astype(np.float32)
    source_roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    roi_height, roi_width = source_roi.shape[:2]
    yy, xx = np.mgrid[0:roi_height, 0:roi_width]
    edge_distance = np.minimum.reduce(
        (xx + 1, roi_width - xx, yy + 1, roi_height - yy)
    ).astype(np.float32)
    alpha = np.clip(
        edge_distance / max(1, feather_pixels), 0.0, 1.0
    )[..., None]
    frame_bgr[y0:y1, x0:x1] = np.clip(
        source_roi * (1.0 - alpha) + restored_roi * alpha, 0, 255
    ).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.frame_count <= 0 or args.feather_pixels < 0:
        raise ValueError("invalid frame range or feather size")
    manifest = json.loads(args.manifest.read_text())
    regions = manifest["regions"]
    prepared = []
    for region_number, restored_path, expansion, sigma in args.region_result:
        if region_number <= 0 or region_number > len(regions):
            raise ValueError(f"region {region_number} is not in the manifest")
        restored = np.load(restored_path)
        if restored.shape != (args.frame_count, 256, 256, 3):
            raise ValueError(f"unexpected array shape {restored.shape} at {restored_path}")
        if sigma > 0.0:
            restored = np.stack(
                [cv2.GaussianBlur(frame, (0, 0), sigma) for frame in restored]
            )
        prepared.append((regions[region_number - 1], restored, expansion))

    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(manifest.get("framesPerSecond", capture.get(cv2.CAP_PROP_FPS) or 30.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to create {args.output}")
    written = 0
    try:
        while written < args.frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            for region, restored, expansion in prepared:
                composite_region(
                    frame, restored[written], region, expansion, args.feather_pixels
                )
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != args.frame_count:
        raise RuntimeError(f"wrote {written} of {args.frame_count} requested frames")
    print(f"Wrote {written} full-resolution frames to {args.output}")


if __name__ == "__main__":
    main()
