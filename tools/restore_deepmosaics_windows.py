#!/usr/bin/env python3
"""Restore detected mosaic regions as resumable one-second eye-video windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from composite_deepmosaics_preview import composite_region, expanded_square
from validate_deepmosaics_checkpoint import (
    initial_previous,
    normalized_rgb_stream,
    tensor_rgb_u8,
    to_model_crop,
)


WINDOW_FRAMES = 30
TEMPORAL_OFFSETS = (-6, -3, 0, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deepmosaics-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--bitrate", type=int, default=20_000_000)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_settings(args: argparse.Namespace, manifest: dict) -> None:
    settings = {
        "version": 1,
        "input": str(args.video.resolve()),
        "manifest": str(args.manifest.resolve()),
        "weights": str(args.weights.resolve()),
        "weightsSHA256": file_sha256(args.weights),
        "passes": args.passes,
        "windowFrames": WINDOW_FRAMES,
        "frameCount": int(manifest["frameCount"]),
        "adaptiveRule": "wide=expansion2+sigma2.5; square=expansion3",
    }
    path = args.work_dir / "settings.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != settings:
            raise RuntimeError(
                f"run settings changed; use a new work directory instead of {args.work_dir}"
            )
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(settings, indent=2) + "\n")
        os.replace(temporary, path)


def adaptive_parameters(region: dict) -> tuple[float, float]:
    width = float(region.get("blendWidth", region["width"]))
    height = float(region.get("blendHeight", region["height"]))
    aspect = max(width, height) / max(1.0, min(width, height))
    return (2.0, 2.5) if aspect >= 1.2 else (3.0, 0.0)


def decode_window(video: Path, start_frame: int, frame_count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(
            f"decoded {len(frames)} of {frame_count} frames at frame {start_frame}"
        )
    return frames


def restore_region(
    model: torch.nn.Module,
    frames_bgr: list[np.ndarray],
    region: dict,
    expansion: float,
    passes: int,
) -> np.ndarray:
    frame_height, frame_width = frames_bgr[0].shape[:2]
    square = expanded_square(region, frame_width, frame_height, expansion)
    current_crops = np.stack([to_model_crop(frame, square) for frame in frames_bgr])
    with torch.inference_mode():
        for _ in range(passes):
            outputs = []
            previous = initial_previous(current_crops[0, :, :, ::-1].copy())
            for frame_index in range(len(frames_bgr)):
                indices = [
                    min(len(frames_bgr) - 1, max(0, frame_index + offset))
                    for offset in TEMPORAL_OFFSETS
                ]
                stream = normalized_rgb_stream(current_crops, indices)
                prediction = model(stream, previous)
                previous = prediction
                outputs.append(tensor_rgb_u8(prediction))
            restored = np.stack(outputs)
            current_crops = restored[:, :, :, ::-1].copy()
    return restored


def tensor_batch_rgb_u8(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    return np.clip((array * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def restore_regions_batched(
    model: torch.nn.Module,
    frames_bgr: list[np.ndarray],
    regions: list[tuple[int, dict, float]],
    passes: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Restore every uncached region in a window in one model batch."""
    frame_height, frame_width = frames_bgr[0].shape[:2]
    crops = []
    for _, region, expansion in regions:
        square = expanded_square(region, frame_width, frame_height, expansion)
        crops.append(np.stack([to_model_crop(frame, square) for frame in frames_bgr]))
    current_crops = np.stack(crops)
    with torch.inference_mode():
        for _ in range(passes):
            previous = torch.cat(
                [
                    initial_previous(crop[0, :, :, ::-1].copy())
                    for crop in current_crops
                ],
                dim=0,
            ).to(device)
            output_frames = []
            for frame_index in range(len(frames_bgr)):
                indices = [
                    min(len(frames_bgr) - 1, max(0, frame_index + offset))
                    for offset in TEMPORAL_OFFSETS
                ]
                stream = torch.cat(
                    [normalized_rgb_stream(crop, indices) for crop in current_crops],
                    dim=0,
                ).to(device)
                prediction = model(stream, previous)
                previous = prediction
                output_frames.append(tensor_batch_rgb_u8(prediction))
            restored = np.stack(output_frames, axis=1)
            current_crops = restored[..., ::-1].copy()
    return {
        region_number: restored[index]
        for index, (region_number, _, _) in enumerate(regions)
    }


def active_regions(manifest: dict, start_frame: int, end_frame: int):
    return [
        (index, region)
        for index, region in enumerate(manifest["regions"], start=1)
        if int(region["startFrame"]) < end_frame
        and int(region["endFrame"]) > start_frame
    ]


def encode_window(
    ffmpeg: str,
    frames: list[np.ndarray],
    output: Path,
    fps: float,
    bitrate: int,
) -> None:
    height, width = frames[0].shape[:2]
    temporary = output.with_name(f".{output.stem}.encoding{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:.6f}",
        "-i", "pipe:0", "-an", "-c:v", "hevc_videotoolbox",
        "-allow_sw", "1", "-pix_fmt", "yuv420p", "-b:v", str(bitrate),
        "-maxrate", str(bitrate * 3 // 2), "-bufsize", str(bitrate * 3),
        "-g", str(len(frames)), "-bf", "0", "-tag:v", "hvc1",
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"FFmpeg window encoding failed: {stderr.strip()}")
    os.replace(temporary, output)


def validate_window(ffprobe: str, path: Path, expected_frames: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == str(expected_frames)


def main() -> None:
    args = parse_args()
    if args.passes <= 0 or args.bitrate <= 0 or args.torch_threads <= 0:
        raise ValueError("passes, bitrate, and torch threads must be positive")
    for path in (args.video, args.manifest, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = args.work_dir / "windows"
    results_dir = args.work_dir / "region-results"
    windows_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    write_settings(args, manifest)
    torch.set_num_threads(args.torch_threads)
    if args.device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(device_name)
    fps = float(manifest.get("framesPerSecond", 30.0))
    frame_count = int(manifest["frameCount"])
    if abs(fps - 30.0) > 0.05:
        raise ValueError(f"expected 30 fps input, got {fps}")

    sys.path.insert(0, str(args.deepmosaics_source.resolve()))
    from models.BVDNet import BVDNet

    model = BVDNet(N=2, n_blocks=4)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    total_windows = (frame_count + WINDOW_FRAMES - 1) // WINDOW_FRAMES
    if args.max_windows is not None:
        total_windows = min(total_windows, args.max_windows)
    print(
        f"DeepMosaics plan: {frame_count} frames, {total_windows} one-second windows; "
        f"device {device_name}",
        flush=True,
    )

    for window_index in range(total_windows):
        start_frame = window_index * WINDOW_FRAMES
        count = min(WINDOW_FRAMES, frame_count - start_frame)
        output = windows_dir / f"window-{window_index:05d}.mov"
        done = windows_dir / f"window-{window_index:05d}.done"
        if done.exists() and validate_window(args.ffprobe, output, count):
            print(f"Window {window_index + 1}/{total_windows}: already complete", flush=True)
            continue
        done.unlink(missing_ok=True)
        frames = decode_window(args.video, start_frame, count)
        regions = active_regions(manifest, start_frame, start_frame + count)
        print(
            f"Window {window_index + 1}/{total_windows}: {len(regions)} detected region(s)",
            flush=True,
        )
        prepared = []
        missing = []
        for region_number, region in regions:
            expansion, sigma = adaptive_parameters(region)
            region_dir = results_dir / f"window-{window_index:05d}"
            region_dir.mkdir(exist_ok=True)
            result_path = region_dir / f"region-{region_number:04d}.npy"
            if result_path.exists():
                restored = np.load(result_path)
                if restored.shape != (count, 256, 256, 3):
                    raise ValueError(f"invalid saved region result: {result_path}")
                print(f"  Region {region_number}: reused saved result", flush=True)
                prepared.append((region_number, region, expansion, sigma, restored))
            else:
                missing.append((region_number, region, expansion))

        if missing:
            if device_name == "mps":
                torch.mps.synchronize()
            started = time.monotonic()
            batch_results = restore_regions_batched(
                model, frames, missing, args.passes, device
            )
            if device_name == "mps":
                torch.mps.synchronize()
            elapsed = time.monotonic() - started
            print(
                f"  Restored batch of {len(missing)} region(s) in {elapsed:.1f}s",
                flush=True,
            )
            for region_number, region, expansion in missing:
                restored = batch_results[region_number]
                _, sigma = adaptive_parameters(region)
                result_path = (
                    results_dir / f"window-{window_index:05d}"
                    / f"region-{region_number:04d}.npy"
                )
                temporary_result = result_path.with_suffix(".npy.tmp")
                with temporary_result.open("wb") as handle:
                    np.save(handle, restored)
                os.replace(temporary_result, result_path)
                prepared.append((region_number, region, expansion, sigma, restored))

        for region_number, region, expansion, sigma, restored in sorted(prepared):
            if sigma > 0.0:
                restored = np.stack(
                    [cv2.GaussianBlur(frame, (0, 0), sigma) for frame in restored]
                )
            for local_frame, frame in enumerate(frames):
                composite_region(frame, restored[local_frame], region, expansion, 16)
        encode_window(args.ffmpeg, frames, output, fps, args.bitrate)
        if not validate_window(args.ffprobe, output, count):
            raise RuntimeError(f"encoded window failed validation: {output}")
        done.touch()
        print(f"Window {window_index + 1}/{total_windows}: saved and validated", flush=True)

    print(f"Completed {total_windows} window(s) in {windows_dir}", flush=True)


if __name__ == "__main__":
    main()
