#!/usr/bin/env python3
"""Test DeepMosaics' video checkpoint on Jasna's detected mosaic regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmosaics-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--expansion", type=float, default=1.5)
    parser.add_argument("--consistent-rgb-previous", action="store_true")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--region", type=int)
    parser.add_argument("--reuse-restored-from", type=Path)
    parser.add_argument("--post-filter-sigma", type=float, default=0.0)
    return parser.parse_args()


def expanded_square(region: dict, frame_width: int, frame_height: int,
                    expansion: float = 1.5) -> tuple[int, int, int]:
    x = int(region.get("blendX", region["x"]))
    y = int(region.get("blendY", region["y"]))
    width = int(region.get("blendWidth", region["width"]))
    height = int(region.get("blendHeight", region["height"]))
    side = max(2, int(max(width, height) * expansion))
    side = min(side, frame_width, frame_height)
    left = int(round(x + width / 2 - side / 2))
    top = int(round(y + height / 2 - side / 2))
    left = max(0, min(frame_width - side, left))
    top = max(0, min(frame_height - side, top))
    return left, top, side


def to_model_crop(frame_bgr: np.ndarray, square: tuple[int, int, int]) -> np.ndarray:
    left, top, side = square
    crop = frame_bgr[top:top + side, left:left + side]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)


def normalized_rgb_stream(crops_bgr: np.ndarray, indices: list[int]) -> torch.Tensor:
    rgb = crops_bgr[indices, :, :, ::-1].copy()
    values = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(values).permute(3, 0, 1, 2).unsqueeze(0)


def initial_previous(crop_bgr: np.ndarray) -> torch.Tensor:
    # DeepMosaics converts the crop to RGB before calling im2tensor(), whose
    # default bgr2rgb conversion reverses it once more. Preserve that behavior.
    values = (crop_bgr.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)


def tensor_rgb_u8(value: torch.Tensor) -> np.ndarray:
    array = value.squeeze(0).detach().cpu().float().numpy().transpose(1, 2, 0)
    return np.clip((array * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def write_comparison(path: Path, before_bgr: np.ndarray, after_rgb: np.ndarray) -> None:
    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
    divider = np.full((256, 4, 3), 255, dtype=np.uint8)
    comparison = np.concatenate((before_rgb, divider, after_rgb), axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def write_progression(path: Path, panels_rgb: list[np.ndarray]) -> None:
    divider = np.full((256, 4, 3), 255, dtype=np.uint8)
    pieces = []
    for index, panel in enumerate(panels_rgb):
        labeled = panel.copy()
        label = "source" if index == 0 else f"pass {index}"
        cv2.putText(
            labeled, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        if pieces:
            pieces.append(divider)
        pieces.append(labeled)
    comparison = np.concatenate(pieces, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def write_preview_video(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to create preview video at {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def feathered_composite(
    source_bgr: np.ndarray,
    restored_rgb: np.ndarray,
    square: tuple[int, int, int],
    region: dict,
    feather_pixels: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source/restored RGB square panels with only the blend box replaced."""
    left, top, side = square
    source_rgb = cv2.cvtColor(
        source_bgr[top:top + side, left:left + side], cv2.COLOR_BGR2RGB
    )
    restored_square = cv2.resize(
        restored_rgb, (side, side), interpolation=cv2.INTER_CUBIC
    )
    result = source_rgb.astype(np.float32)

    blend_x = int(region.get("blendX", region["x"])) - left
    blend_y = int(region.get("blendY", region["y"])) - top
    blend_width = int(region.get("blendWidth", region["width"]))
    blend_height = int(region.get("blendHeight", region["height"]))
    x0 = max(0, blend_x)
    y0 = max(0, blend_y)
    x1 = min(side, blend_x + blend_width)
    y1 = min(side, blend_y + blend_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("blend rectangle does not intersect the expanded crop")

    roi_height = y1 - y0
    roi_width = x1 - x0
    yy, xx = np.mgrid[0:roi_height, 0:roi_width]
    edge_distance = np.minimum.reduce(
        (xx + 1, roi_width - xx, yy + 1, roi_height - yy)
    ).astype(np.float32)
    alpha = np.clip(edge_distance / max(1, feather_pixels), 0.0, 1.0)[..., None]
    source_roi = result[y0:y1, x0:x1]
    restored_roi = restored_square[y0:y1, x0:x1].astype(np.float32)
    result[y0:y1, x0:x1] = source_roi * (1.0 - alpha) + restored_roi * alpha
    return source_rgb, np.clip(result, 0, 255).astype(np.uint8)


def labeled_panel(image_rgb: np.ndarray, label: str, size: int = 512) -> np.ndarray:
    panel = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size, 36), (0, 0, 0), -1)
    cv2.putText(
        panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return panel


def write_composited_review(
    output: Path,
    frames_bgr: list[np.ndarray],
    region_results: list[tuple[int, dict, tuple[int, int, int], np.ndarray]],
    fps: float,
) -> None:
    review_frames = []
    middle_rows = []
    for frame_index, source_frame in enumerate(frames_bgr):
        rows = []
        for region_number, region, square, restored in region_results:
            before, after = feathered_composite(
                source_frame, restored[frame_index], square, region
            )
            row = np.concatenate(
                (
                    labeled_panel(before, f"Region {region_number} - source"),
                    labeled_panel(after, f"Region {region_number} - restored"),
                ),
                axis=1,
            )
            rows.append(row)
        review = np.concatenate(rows, axis=0)
        review_frames.append(review)
        if frame_index == len(frames_bgr) // 2:
            middle_rows = rows
    review_array = np.stack(review_frames)
    write_preview_video(output / "composited-before-after.mp4", review_array, fps)
    middle = np.concatenate(middle_rows, axis=0)
    cv2.imwrite(
        str(output / "composited-before-after-middle.png"),
        cv2.cvtColor(middle, cv2.COLOR_RGB2BGR),
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if (args.start_frame < 0 or args.frame_count <= 0 or args.expansion < 1.0
            or args.passes <= 0 or args.post_filter_sigma < 0.0):
        raise ValueError("start-frame must be nonnegative and frame-count must be positive")
    end_frame = args.start_frame + args.frame_count
    indexed_regions = [
        (index, region) for index, region in enumerate(manifest["regions"], start=1)
        if int(region["startFrame"]) < end_frame and int(region["endFrame"]) > args.start_frame
    ]
    if args.region is not None:
        matching = [item for item in indexed_regions if item[0] == args.region]
        if not matching:
            available = ", ".join(str(index) for index, _ in indexed_regions)
            raise ValueError(f"region {args.region} is unavailable; choose from {available}")
        indexed_regions = matching

    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frames_bgr = []
    while len(frames_bgr) < args.frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        frames_bgr.append(frame)
    capture.release()
    if not frames_bgr:
        raise RuntimeError(f"no frames decoded from {args.video}")

    sys.path.insert(0, str(args.deepmosaics_source.resolve()))
    from models.BVDNet import BVDNet

    model = BVDNet(N=2, n_blocks=4)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    frame_count = len(frames_bgr)
    frame_height, frame_width = frames_bgr[0].shape[:2]
    offsets = (-6, -3, 0, 3, 6)
    print(f"Decoded {frame_count} frames; testing {len(indexed_regions)} DeepMosaics crops")
    region_results = []

    for region_index, region in indexed_regions:
        square = expanded_square(
            region, frame_width, frame_height, expansion=args.expansion
        )
        source_crops = np.stack([to_model_crop(frame, square) for frame in frames_bgr])
        started = time.monotonic()
        if args.reuse_restored_from:
            restored_path = (
                args.reuse_restored_from / f"crop-{region_index:02d}-restored.npy"
            )
            restored = np.load(restored_path)
            if restored.shape != (frame_count, 256, 256, 3):
                raise ValueError(
                    f"unexpected restored array shape {restored.shape} at {restored_path}"
                )
            pass_outputs = [restored]
        else:
            current_crops = source_crops
            pass_outputs = []
            with torch.inference_mode():
                for _ in range(args.passes):
                    outputs = []
                    previous_crop = (
                        current_crops[0, :, :, ::-1].copy()
                        if args.consistent_rgb_previous else current_crops[0]
                    )
                    previous = initial_previous(previous_crop)
                    for frame_index in range(frame_count):
                        indices = [
                            min(frame_count - 1, max(0, frame_index + offset))
                            for offset in offsets
                        ]
                        stream = normalized_rgb_stream(current_crops, indices)
                        prediction = model(stream, previous)
                        previous = prediction
                        outputs.append(tensor_rgb_u8(prediction))
                    pass_output = np.stack(outputs)
                    pass_outputs.append(pass_output)
                    current_crops = pass_output[:, :, :, ::-1].copy()
            restored = pass_outputs[-1]
        elapsed = time.monotonic() - started
        if args.post_filter_sigma > 0.0:
            restored = np.stack([
                cv2.GaussianBlur(frame, (0, 0), args.post_filter_sigma)
                for frame in restored
            ])
        source_rgb = source_crops[:, :, :, ::-1]
        absolute = np.abs(restored.astype(np.int16) - source_rgb.astype(np.int16))
        middle = frame_count // 2
        write_comparison(
            args.output / f"crop-{region_index:02d}-middle-before-after.png",
            source_crops[middle],
            restored[middle],
        )
        write_progression(
            args.output / f"crop-{region_index:02d}-middle-progression.png",
            [source_rgb[middle]] + [output[middle] for output in pass_outputs],
        )
        np.save(args.output / f"crop-{region_index:02d}-restored.npy", restored)
        write_preview_video(
            args.output / f"crop-{region_index:02d}-restored.mp4",
            restored,
            float(manifest.get("framesPerSecond", 30.0)),
        )
        region_results.append((region_index, region, square, restored))
        correction = restored.astype(np.float32) - source_rgb.astype(np.float32)
        correction_flicker = (
            np.abs(np.diff(correction, axis=0)).mean() if frame_count > 1 else 0.0
        )
        source_motion = (
            np.abs(np.diff(source_rgb.astype(np.float32), axis=0)).mean()
            if frame_count > 1 else 0.0
        )
        restored_motion = (
            np.abs(np.diff(restored.astype(np.float32), axis=0)).mean()
            if frame_count > 1 else 0.0
        )
        left, top, side = square
        print(
            f"Crop {region_index}: source square {left},{top} {side}x{side}; "
            f"{args.passes} pass(es), {elapsed:.2f}s; "
            f"post-filter sigma {args.post_filter_sigma:.2f}; "
            f"mean change {absolute.mean():.3f}/255, "
            f"p99 {np.quantile(absolute, 0.99):.1f}, max {absolute.max()}; "
            f"correction flicker {correction_flicker:.3f}, "
            f"motion source/restored {source_motion:.3f}/{restored_motion:.3f}"
        )

    write_composited_review(
        args.output,
        frames_bgr,
        region_results,
        float(manifest.get("framesPerSecond", 30.0)),
    )
    print(f"Composited review: {args.output / 'composited-before-after.mp4'}")


if __name__ == "__main__":
    main()
