#!/usr/bin/env python3
"""Create a short VR-eye proof using Jasna's projection-aware Lada path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from validate_restoration_checkpoint import (
    enable_postponed_annotations_for_legacy_python,
    feathered_box_mask,
    region_bbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projection", choices=("raw", "fisheye", "gnomonic"), default="fisheye"
    )
    parser.add_argument("--clip-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=20_000_000)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def active_regions(manifest: dict, start: int, end: int) -> list[dict]:
    return [
        region
        for region in manifest["regions"]
        if int(region["startFrame"]) < end and int(region["endFrame"]) > start
    ]


def extract_raw_crops(frames: list[torch.Tensor], region: dict, projector, extract_crop):
    height, width = frames[0].shape[1:]
    bbox = region_bbox(region, True)
    result = []
    for frame in frames:
        if projector is None:
            result.append(extract_crop(frame, bbox, height, width))
        else:
            result.append(projector.extract_region_crop(frame, bbox, height, width))
    return result


def composite_result(
    frames: list[torch.Tensor],
    restored: torch.Tensor,
    raw_crops,
    pad_offsets,
    resize_shapes,
    region: dict,
    projector,
) -> None:
    bbox = region_bbox(region, True)
    for index, frame in enumerate(frames):
        raw_crop = raw_crops[index]
        pad_left, pad_top = pad_offsets[index]
        resize_height, resize_width = resize_shapes[index]
        restored_patch = restored[index].mul(255.0)
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
        x1, y1, x2, y2 = raw_crop.enlarged_bbox
        original_patch = frame[:, y1:y2, x1:x2].float()
        if projector is not None:
            original_projected = projector.project_region(frame, raw_crop.enlarged_bbox)
            delta = projector.source_region_from_patch(
                resized_back - original_projected, raw_crop.enlarged_bbox
            )
            candidate = original_patch + delta
        else:
            candidate = resized_back
        mask = feathered_box_mask(bbox, raw_crop.enlarged_bbox).unsqueeze(0)
        frame[:, y1:y2, x1:x2] = (
            original_patch.lerp(candidate, mask).round().clamp(0, 255).byte()
        )


def start_encoder(args: argparse.Namespace, width: int, height: int, fps: float):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.stem}.encoding{args.output.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.6f}", "-i", "pipe:0", "-an",
        "-c:v", "hevc_videotoolbox", "-allow_sw", "1", "-pix_fmt", "yuv420p",
        "-b:v", str(args.bitrate), "-maxrate", str(args.bitrate * 3 // 2),
        "-bufsize", str(args.bitrate * 3), "-g", str(args.max_frames), "-bf", "0",
        "-tag:v", "hvc1", "-movflags", "+faststart", str(temporary),
    ]
    return temporary, subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    args = parse_args()
    if args.clip_frames <= 0 or args.max_frames <= 0 or args.bitrate <= 0:
        raise ValueError("clip frames, max frames, and bitrate must be positive")
    manifest = json.loads(args.manifest.read_text())
    width, height = int(manifest["width"]), int(manifest["height"])
    fps = float(manifest.get("framesPerSecond", 30.0))

    enable_postponed_annotations_for_legacy_python()
    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.crop_buffer import extract_crop, prepare_crops_for_restoration
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.vr_projection import build_vr_projector

    device = torch.device("cpu")
    model = load_model(None, str(args.weights), device, False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    generator.eval()
    projector = build_vr_projector(
        args.projection, eye_width=width, height=height, device=device
    )

    temporary, encoder = start_encoder(args, width, height, fps)
    capture = cv2.VideoCapture(str(args.video))
    written = 0
    started = time.monotonic()
    try:
        assert encoder.stdin is not None
        while written < args.max_frames:
            frames = []
            while len(frames) < min(args.clip_frames, args.max_frames - written):
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
            if not frames:
                break
            for region in active_regions(manifest, written, written + len(frames)):
                raw_crops = extract_raw_crops(
                    frames, region, projector, extract_crop
                )
                prepared, pad_offsets, resize_shapes = prepare_crops_for_restoration(
                    raw_crops, device, torch.float32
                )
                inputs = torch.stack(prepared).div_(255.0).unsqueeze(0)
                with torch.inference_mode():
                    restored = generator(inputs).squeeze(0).clamp(0, 1)
                composite_result(
                    frames, restored, raw_crops, pad_offsets, resize_shapes,
                    region, projector,
                )
            for frame in frames:
                encoder.stdin.write(
                    np.ascontiguousarray(frame.permute(1, 2, 0).numpy()).tobytes()
                )
            written += len(frames)
            print(f"Restored {written}/{args.max_frames} frames", flush=True)
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace")
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {stderr.strip()}")
    os.replace(temporary, args.output)
    elapsed = time.monotonic() - started
    print(f"Saved {written} frames to {args.output} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
