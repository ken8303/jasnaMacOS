#!/usr/bin/env python3
"""Run an official Jasna checkpoint on the exact crops used by the Metal app.

This is a small correctness oracle, not a benchmark.  It emits before/after
crop images and reports how much the trained model changed the input.
"""

from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def enable_postponed_annotations_for_legacy_python() -> None:
    """Allow the Python 3.12 Jasna tree to load in our isolated Python 3.9 venv."""
    if sys.version_info >= (3, 10):
        return
    import __future__

    original_compile = builtins.compile

    def compile_with_annotations(source, filename, mode, flags=0,
                                 dont_inherit=False, optimize=-1, **kwargs):
        future_flags = (
            __future__.annotations.compiler_flag
            if "/work/jasna/" in str(filename)
            else 0
        )
        return original_compile(
            source,
            filename,
            mode,
            flags | future_flags,
            dont_inherit,
            optimize,
            **kwargs,
        )

    builtins.compile = compile_with_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tight-blend", action="store_true")
    parser.add_argument(
        "--projection",
        choices=("raw", "fisheye", "gnomonic"),
        default="raw",
        help="VR180 region projection used before restoration",
    )
    parser.add_argument(
        "--source-space-preview",
        action="store_true",
        help="also write middle-frame comparisons after inverse VR projection",
    )
    return parser.parse_args()


def model_crop(frame_rgb: np.ndarray, region: dict, size: int = 256,
               tight_blend: bool = False) -> np.ndarray:
    prefix = "blend" if tight_blend else ""
    x_key = f"{prefix}X" if prefix else "x"
    y_key = f"{prefix}Y" if prefix else "y"
    width_key = f"{prefix}Width" if prefix else "width"
    height_key = f"{prefix}Height" if prefix else "height"
    x, y = int(region[x_key]), int(region[y_key])
    width, height = int(region[width_key]), int(region[height_key])
    crop = frame_rgb[y:y + height, x:x + width]
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, int(width * scale)))
    resized_height = max(1, min(size, int(height * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    right = size - resized_width - left
    top = (size - resized_height) // 2
    bottom = size - resized_height - top
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT)


def write_comparison(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    divider = np.full((before.shape[0], 4, 3), 255, dtype=np.uint8)
    comparison = np.concatenate((before, divider, after), axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def region_bbox(region: dict, tight_blend: bool) -> np.ndarray:
    prefix = "blend" if tight_blend else ""
    x_key = f"{prefix}X" if prefix else "x"
    y_key = f"{prefix}Y" if prefix else "y"
    width_key = f"{prefix}Width" if prefix else "width"
    height_key = f"{prefix}Height" if prefix else "height"
    x = int(region[x_key])
    y = int(region[y_key])
    return np.array(
        [x, y, x + int(region[width_key]), y + int(region[height_key])],
        dtype=np.float32,
    )


def feathered_box_mask(
    bbox: np.ndarray,
    enlarged_bbox: tuple[int, int, int, int],
    feather: int = 12,
) -> torch.Tensor:
    x1, y1, x2, y2 = enlarged_bbox
    height, width = y2 - y1, x2 - x1
    mask = np.zeros((height, width), dtype=np.float32)
    bx1 = max(0, int(np.floor(bbox[0])) - x1)
    by1 = max(0, int(np.floor(bbox[1])) - y1)
    bx2 = min(width, int(np.ceil(bbox[2])) - x1)
    by2 = min(height, int(np.ceil(bbox[3])) - y1)
    if bx2 > bx1 and by2 > by1:
        mask[by1:by2, bx1:bx2] = 1.0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather / 3.0)
        peak = float(mask.max())
        if peak > 0:
            mask /= peak
    return torch.from_numpy(mask)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    regions = manifest["regions"]

    enable_postponed_annotations_for_legacy_python()
    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.crop_buffer import extract_crop, prepare_crops_for_restoration
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.vr_projection import build_vr_projector

    frame_width = int(manifest["width"])
    frame_height = int(manifest["height"])
    projector = build_vr_projector(
        args.projection,
        eye_width=frame_width,
        height=frame_height,
        device=torch.device("cpu"),
    )
    raw_crops: list[list[object]] = [[] for _ in regions]
    decoded_frames: list[torch.Tensor] = []

    capture = cv2.VideoCapture(str(args.video))
    frame_count = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb.copy()).permute(2, 0, 1)
        decoded_frames.append(frame_tensor)
        for index, region in enumerate(regions):
            bbox = region_bbox(region, args.tight_blend)
            if projector is None:
                raw_crop = extract_crop(
                    frame_tensor, bbox, frame_height, frame_width
                )
            else:
                raw_crop = projector.extract_region_crop(
                    frame_tensor, bbox, frame_height, frame_width
                )
            raw_crops[index].append(raw_crop)
        frame_count += 1
    capture.release()
    if frame_count == 0:
        raise RuntimeError(f"no frames decoded from {args.video}")

    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    generator.eval()

    print(
        f"Decoded {frame_count} frames; testing {len(regions)} mosaic crops; "
        f"projection {args.projection}"
    )
    for index, region_crops in enumerate(raw_crops, start=1):
        prepared, pad_offsets, resize_shapes = prepare_crops_for_restoration(
            region_crops, torch.device("cpu"), torch.float32
        )
        source = (
            torch.stack(prepared)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        inputs = torch.from_numpy(source).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0)
        started = time.monotonic()
        with torch.inference_mode():
            output = generator(inputs).squeeze(0).clamp(0, 1)
        elapsed = time.monotonic() - started
        restored = output.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        absolute = np.abs(restored.astype(np.int16) - source.astype(np.int16))
        middle = frame_count // 2
        write_comparison(args.output / f"crop-{index:02d}-middle-before-after.png",
                         source[middle], restored[middle])
        if args.source_space_preview:
            raw_crop = region_crops[middle]
            pad_left, pad_top = pad_offsets[middle]
            resize_height, resize_width = resize_shapes[middle]
            restored_patch = output[middle].mul(255.0)
            unpadded = restored_patch[
                :,
                pad_top:pad_top + resize_height,
                pad_left:pad_left + resize_width,
            ]
            resized_back = F.interpolate(
                unpadded.unsqueeze(0),
                size=raw_crop.crop_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            original_frame = decoded_frames[middle]
            x1, y1, x2, y2 = raw_crop.enlarged_bbox
            original_patch = original_frame[:, y1:y2, x1:x2].float()
            if projector is not None:
                original_projected = projector.project_region(
                    original_frame, raw_crop.enlarged_bbox
                )
                delta = projector.source_region_from_patch(
                    resized_back - original_projected,
                    raw_crop.enlarged_bbox,
                )
                candidate = original_patch + delta
            else:
                candidate = resized_back
            mask = feathered_box_mask(
                region_bbox(regions[index - 1], args.tight_blend),
                raw_crop.enlarged_bbox,
            ).unsqueeze(0)
            composited = original_patch.lerp(candidate, mask).round().clamp(0, 255)
            before_source = original_patch.byte().permute(1, 2, 0).numpy()
            after_source = composited.byte().permute(1, 2, 0).numpy()
            write_comparison(
                args.output / f"crop-{index:02d}-source-space-before-after.png",
                before_source,
                after_source,
            )
        np.save(args.output / f"crop-{index:02d}-restored.npy", restored)
        print(
            f"Crop {index}: {elapsed:.2f}s, mean change {absolute.mean():.3f}/255, "
            f"p99 {np.quantile(absolute, 0.99):.1f}, max {absolute.max()}"
        )


if __name__ == "__main__":
    main()
