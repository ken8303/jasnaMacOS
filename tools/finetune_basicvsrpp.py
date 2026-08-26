#!/usr/bin/env python3
"""Conservatively fine-tune Jasna's BasicVSR++ generator on clean clips."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.0,
        help="masked spatial-gradient reconstruction weight",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-clips", type=int, default=8)
    parser.add_argument("--minimum-psnr-gain", type=float, default=0.10)
    parser.add_argument("--maximum-temporal-regression", type=float, default=1.02)
    parser.add_argument(
        "--corruption-profile",
        choices=("legacy", "moving-vr", "lada-vr", "lada-vr-v4"),
        default="legacy",
        help="synthetic mosaic recipe used for both training and validation",
    )
    parser.add_argument(
        "--reset-quality-baseline",
        action="store_true",
        help="measure a new gate baseline when resuming into a different recipe",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8303)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--allow-experimental-non-cuda", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--skip-dataset-audit",
        action="store_true",
        help="diagnostic only: allow training without a passing mosaic-audit.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "steps",
        "frames",
        "batch_size",
        "save_every",
        "validate_every",
        "validation_clips",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.gradient_weight < 0:
        raise ValueError("--gradient-weight cannot be negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be between zero and one")
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation must be positive")
    if args.steps % args.gradient_accumulation:
        raise ValueError("--steps must be divisible by --gradient-accumulation")
    if args.minimum_psnr_gain < 0:
        raise ValueError("--minimum-psnr-gain cannot be negative")
    if args.maximum_temporal_regression < 1.0:
        raise ValueError("--maximum-temporal-regression must be at least 1.0")
    for path in (args.jasna_source, args.weights, args.dataset):
        if not path.exists():
            raise ValueError(f"required path does not exist: {path}")
    if args.resume is not None and not args.resume.is_file():
        raise ValueError(f"resume checkpoint does not exist: {args.resume}")
    if args.reset_quality_baseline and args.resume is None:
        raise ValueError("--reset-quality-baseline requires --resume")
    if args.output.exists() and not args.output.is_dir():
        raise ValueError(f"output path is not a directory: {args.output}")
    if (
        args.output.is_dir()
        and any(path.name != "training.log" for path in args.output.iterdir())
        and args.resume is None
    ):
        raise ValueError(
            f"output directory is not empty: {args.output}; "
            "choose a new directory or pass --resume explicitly"
        )
    if args.device != "cuda" and not args.allow_experimental_non_cuda:
        raise ValueError(
            "BasicVSR++ training is validated for CUDA only; pass "
            "--allow-experimental-non-cuda for an explicit diagnostic attempt"
        )


def list_clips(dataset: Path, split: str) -> list[Path]:
    root = dataset / split
    clips = sorted(path for path in root.glob("clip-*") if path.is_dir())
    if not clips:
        raise ValueError(f"dataset contains no {split} clips: {root}")
    return clips


def validate_dataset_audit(
    dataset: Path, train_clips: list[Path], validation_clips: list[Path]
) -> dict:
    path = dataset / "mosaic-audit.json"
    if not path.is_file():
        raise ValueError(
            f"dataset audit is missing: {path}; run script/audit_finetune_dataset.sh"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("accepted", False):
        raise ValueError(f"dataset audit did not pass: {path}")
    expected = {
        f"train/{clip.name}" for clip in train_clips
    } | {f"validation/{clip.name}" for clip in validation_clips}
    audited = set(report.get("clipQuality", {}))
    if audited != expected:
        missing = sorted(expected - audited)
        stale = sorted(audited - expected)
        raise ValueError(
            "dataset changed after its audit; "
            f"missing={missing or 'none'}, stale={stale or 'none'}"
        )
    return report


def select_frame_window(paths: list[Path], frames: int, rng: random.Random) -> list[Path]:
    if len(paths) < frames:
        raise ValueError(f"clip has {len(paths)} frames, fewer than requested {frames}")
    start = rng.randrange(0, len(paths) - frames + 1)
    return paths[start : start + frames]


def model_state_dict(generator, ema_generator) -> OrderedDict:
    state = OrderedDict()
    for key, value in generator.state_dict().items():
        state[f"generator.{key}"] = value.detach().cpu()
    for key, value in ema_generator.state_dict().items():
        state[f"generator_ema.{key}"] = value.detach().cpu()
    return state


def atomic_torch_save(torch, payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_metric_or_none(value: float) -> float | None:
    """Keep telemetry strict-JSON compatible before a best metric exists."""
    return value if math.isfinite(value) else None


def ema_stage_fraction(
    decay: float, start_step: int, target_step: int, gradient_accumulation: int
) -> float:
    """Approximate how much of a resumed EMA comes from the new stage."""
    updates = target_step // gradient_accumulation - start_step // gradient_accumulation
    return 1.0 - decay ** max(0, updates)


def passes_quality_gate(
    baseline: dict,
    candidate: dict,
    minimum_psnr_gain: float,
    maximum_temporal_regression: float,
) -> bool:
    return (
        candidate["maskedPSNR"] >= baseline["maskedPSNR"] + minimum_psnr_gain
        and candidate["temporalMAE"]
        <= baseline["temporalMAE"] * maximum_temporal_regression
    )


def quality_baseline(
    measured: dict, resume_checkpoint: dict | None, reset: bool = False
) -> dict:
    if resume_checkpoint is None or reset:
        return measured
    baseline = resume_checkpoint.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("resume checkpoint does not contain the original baseline")
    for key in ("maskedPSNR", "temporalMAE"):
        if key not in baseline:
            raise ValueError(f"resume checkpoint baseline is missing {key}")
    return baseline


def set_trainable_scope(generator) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for name, parameter in generator.named_parameters():
        parameter.requires_grad_(not name.startswith("spynet."))
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return trainable, frozen


def update_ema(torch, ema_generator, generator, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_generator.parameters(), generator.parameters()
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(
            ema_generator.buffers(), generator.buffers()
        ):
            ema_buffer.copy_(buffer)


def synthetic_mosaic(
    torch, functional, clean, rng: random.Random, profile: str = "legacy"
):
    """Return pixelated input and the exact affected-pixel mask."""
    if profile == "moving-vr":
        return synthetic_moving_vr_mosaic(torch, functional, clean, rng)
    if profile in ("lada-vr", "lada-vr-v4"):
        return synthetic_lada_vr_mosaic(
            torch, functional, clean, rng, aggressive=profile == "lada-vr-v4"
        )
    if profile != "legacy":
        raise ValueError(f"unknown corruption profile: {profile}")
    low_quality = clean.clone()
    batch, frame_count, _, height, width = clean.shape
    mask = clean.new_zeros(batch, frame_count, 1, height, width)
    for batch_index in range(batch):
        region_width = rng.randrange(width // 3, (width * 4) // 5 + 1)
        region_height = rng.randrange(height // 3, (height * 4) // 5 + 1)
        base_x = rng.randrange(0, width - region_width + 1)
        base_y = rng.randrange(0, height - region_height + 1)
        velocity_x = rng.randint(-3, 3)
        velocity_y = rng.randint(-3, 3)
        block = rng.randint(8, 24)
        for frame_index in range(frame_count):
            centered_time = frame_index - (frame_count - 1) / 2.0
            x = round(base_x + velocity_x * centered_time)
            y = round(base_y + velocity_y * centered_time)
            x = max(0, min(width - region_width, x))
            y = max(0, min(height - region_height, y))
            patch = clean[
                batch_index : batch_index + 1,
                frame_index,
                :,
                y : y + region_height,
                x : x + region_width,
            ]
            tiny_height = max(1, math.ceil(region_height / block))
            tiny_width = max(1, math.ceil(region_width / block))
            tiny = functional.interpolate(
                patch,
                size=(tiny_height, tiny_width),
                mode="area",
            )
            pixelated = functional.interpolate(
                tiny,
                size=(region_height, region_width),
                mode="nearest",
            )
            low_quality[
                batch_index,
                frame_index,
                :,
                y : y + region_height,
                x : x + region_width,
            ] = pixelated[0]
            mask[
                batch_index,
                frame_index,
                :,
                y : y + region_height,
                x : x + region_width,
            ] = 1
    return low_quality, mask


def synthetic_moving_vr_mosaic(torch, functional, clean, rng: random.Random):
    """Create faster, shape-changing mosaics resembling tracked VR subjects."""
    low_quality = clean.clone()
    batch, frame_count, _, height, width = clean.shape
    mask = clean.new_zeros(batch, frame_count, 1, height, width)
    for batch_index in range(batch):
        region_count = 2 if rng.random() < 0.35 else 1
        for _ in range(region_count):
            region_width = rng.randrange(width // 4, (width * 3) // 4 + 1)
            region_height = rng.randrange(height // 4, (height * 3) // 4 + 1)
            base_x = rng.randrange(0, width - region_width + 1)
            base_y = rng.randrange(0, height - region_height + 1)
            velocity_x = rng.uniform(-10.0, 10.0)
            velocity_y = rng.uniform(-10.0, 10.0)
            if abs(velocity_x) + abs(velocity_y) < 5.0:
                velocity_x += 5.0 if velocity_x >= 0 else -5.0
            acceleration_x = rng.uniform(-0.35, 0.35)
            acceleration_y = rng.uniform(-0.35, 0.35)
            curve_x = rng.uniform(-4.0, 4.0)
            curve_y = rng.uniform(-4.0, 4.0)
            phase = rng.uniform(0.0, math.tau)
            base_block = rng.randint(6, 28)
            ellipse = rng.random() < 0.75
            for frame_index in range(frame_count):
                centered_time = frame_index - (frame_count - 1) / 2.0
                curve = math.sin(phase + frame_index * 0.7)
                x = round(
                    base_x
                    + velocity_x * centered_time
                    + 0.5 * acceleration_x * centered_time * centered_time
                    + curve_x * curve
                )
                y = round(
                    base_y
                    + velocity_y * centered_time
                    + 0.5 * acceleration_y * centered_time * centered_time
                    + curve_y * curve
                )
                x = max(0, min(width - region_width, x))
                y = max(0, min(height - region_height, y))
                patch = clean[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ]
                block = max(4, base_block + rng.randint(-3, 3))
                tiny = functional.interpolate(
                    patch,
                    size=(
                        max(1, math.ceil(region_height / block)),
                        max(1, math.ceil(region_width / block)),
                    ),
                    mode="area",
                )
                pixelated = functional.interpolate(
                    tiny,
                    size=(region_height, region_width),
                    mode="nearest",
                )
                if ellipse:
                    vertical = torch.linspace(
                        -1.0, 1.0, region_height, device=clean.device
                    ).view(1, 1, region_height, 1)
                    horizontal = torch.linspace(
                        -1.0, 1.0, region_width, device=clean.device
                    ).view(1, 1, 1, region_width)
                    shape = (
                        horizontal.square()
                        + (vertical + 0.08 * curve).square()
                        <= 1.0
                    ).to(clean.dtype)
                else:
                    shape = clean.new_ones(1, 1, region_height, region_width)
                destination = low_quality[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ]
                low_quality[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ] = pixelated * shape + destination * (1.0 - shape)
                mask[
                    batch_index,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ] = torch.maximum(
                    mask[
                        batch_index,
                        frame_index,
                        :,
                        y : y + region_height,
                        x : x + region_width,
                    ],
                    shape[0],
                )
    return low_quality, mask


def synthetic_lada_vr_mosaic(
    torch, functional, clean, rng: random.Random, aggressive: bool = False
):
    """Approximate Lada's block-mask and post-mosaic video degradations."""
    low_quality = clean.clone()
    batch, frame_count, _, height, width = clean.shape
    mask = clean.new_zeros(batch, frame_count, 1, height, width)
    for batch_index in range(batch):
        region_count = 2 if rng.random() < (0.45 if aggressive else 0.25) else 1
        for _ in range(region_count):
            minimum_fraction = 3 if aggressive else 4
            maximum_numerator = 7 if aggressive else 3
            maximum_denominator = 8 if aggressive else 4
            region_width = rng.randrange(
                width // minimum_fraction,
                (width * maximum_numerator) // maximum_denominator + 1,
            )
            region_height = rng.randrange(
                height // minimum_fraction,
                (height * maximum_numerator) // maximum_denominator + 1,
            )
            base_x = rng.randrange(0, width - region_width + 1)
            base_y = rng.randrange(0, height - region_height + 1)
            velocity_limit = 13.0 if aggressive else 9.0
            velocity_x = rng.uniform(-velocity_limit, velocity_limit)
            velocity_y = rng.uniform(-velocity_limit, velocity_limit)
            if abs(velocity_x) + abs(velocity_y) < 4.0:
                velocity_y += 4.0 if velocity_y >= 0 else -4.0
            curve_x = rng.uniform(-3.0, 3.0)
            curve_y = rng.uniform(-3.0, 3.0)
            phase = rng.uniform(0.0, math.tau)
            block_height = rng.randint(12, 44) if aggressive else rng.randint(7, 28)
            rectangle_ratio = rng.uniform(0.65, 1.85) if aggressive else rng.uniform(0.8, 1.65)
            block_width = max(4, round(block_height * rectangle_ratio))
            sample_mode = "area" if rng.random() < 0.75 else "nearest"
            ellipse = rng.random() < 0.55
            feather = rng.random() < (0.20 if aggressive else 0.45)
            for frame_index in range(frame_count):
                centered_time = frame_index - (frame_count - 1) / 2.0
                curve = math.sin(phase + frame_index * 0.55)
                x = round(base_x + velocity_x * centered_time + curve_x * curve)
                y = round(base_y + velocity_y * centered_time + curve_y * curve)
                x = max(0, min(width - region_width, x))
                y = max(0, min(height - region_height, y))
                patch = clean[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ]
                tiny_size = (
                    max(1, math.ceil(region_height / block_height)),
                    max(1, math.ceil(region_width / block_width)),
                )
                tiny = functional.interpolate(patch, size=tiny_size, mode=sample_mode)
                pixelated = functional.interpolate(
                    tiny, size=(region_height, region_width), mode="nearest"
                )
                if ellipse:
                    vertical = torch.linspace(
                        -1.0, 1.0, region_height, device=clean.device
                    ).view(1, 1, region_height, 1)
                    horizontal = torch.linspace(
                        -1.0, 1.0, region_width, device=clean.device
                    ).view(1, 1, 1, region_width)
                    shape = (
                        horizontal.square()
                        + (vertical + 0.06 * curve).square()
                        <= 1.0
                    ).to(clean.dtype)
                else:
                    shape = clean.new_ones(1, 1, region_height, region_width)

                # Lada mosaics whole grid cells selected by a mask instead of
                # cutting smooth partial blocks at the subject boundary.
                block_shape = functional.interpolate(
                    shape, size=tiny_size, mode="area"
                )
                block_shape = (block_shape >= 0.5).to(clean.dtype)
                block_shape = functional.interpolate(
                    block_shape,
                    size=(region_height, region_width),
                    mode="nearest",
                )
                blend_shape = block_shape
                if feather:
                    kernel = max(3, min(block_height, block_width))
                    if kernel % 2 == 0:
                        kernel += 1
                    blend_shape = functional.avg_pool2d(
                        block_shape, kernel, stride=1, padding=kernel // 2
                    )
                destination = low_quality[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ]
                low_quality[
                    batch_index : batch_index + 1,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ] = pixelated * blend_shape + destination * (1.0 - blend_shape)
                mask_slice = mask[
                    batch_index,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ]
                mask[
                    batch_index,
                    frame_index,
                    :,
                    y : y + region_height,
                    x : x + region_width,
                ] = torch.maximum(mask_slice, blend_shape[0])

        # Approximate Lada's post-mosaic video compression, occasional blur,
        # and a second compression pass without invoking a codec per step.
        clip = low_quality[batch_index]
        if rng.random() < (0.45 if aggressive else 0.30):
            kernel = 3 if rng.random() < 0.75 else 5
            clip = functional.avg_pool2d(
                clip, kernel, stride=1, padding=kernel // 2
            )
        quantization_levels = rng.randint(72, 192) if aggressive else rng.randint(128, 240)
        clip = torch.round(clip * quantization_levels) / quantization_levels
        if rng.random() < (0.30 if aggressive else 0.15):
            second_levels = rng.randint(56, 128) if aggressive else rng.randint(80, 160)
            clip = torch.round(clip * second_levels) / second_levels
        low_quality[batch_index] = clip.clamp(0, 1)
    return low_quality, mask


def masked_spatial_gradient_loss(torch, prediction, target, mask):
    epsilon = 1e-6
    channels = prediction.shape[2]
    prediction_x = prediction[..., 1:] - prediction[..., :-1]
    target_x = target[..., 1:] - target[..., :-1]
    mask_x = torch.maximum(mask[..., 1:], mask[..., :-1])
    difference_x = torch.sqrt((prediction_x - target_x).square() + epsilon)
    loss_x = (difference_x * mask_x).sum() / (mask_x.sum() * channels).clamp_min(1.0)

    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    mask_y = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    difference_y = torch.sqrt((prediction_y - target_y).square() + epsilon)
    loss_y = (difference_y * mask_y).sum() / (mask_y.sum() * channels).clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def restoration_loss(
    torch,
    prediction,
    target,
    mask,
    background_weight,
    temporal_weight,
    gradient_weight=0.0,
):
    epsilon = 1e-6
    difference = torch.sqrt((prediction - target).square() + epsilon)
    channels = prediction.shape[2]
    masked_denominator = mask.sum() * channels
    background = 1.0 - mask
    background_denominator = background.sum() * channels
    masked = (difference * mask).sum() / masked_denominator.clamp_min(1.0)
    clean_area = (difference * background).sum() / background_denominator.clamp_min(1.0)
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    temporal_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
    temporal_difference = torch.sqrt(
        (prediction_delta - target_delta).square() + epsilon
    )
    temporal_denominator = temporal_mask.sum() * channels
    temporal = (
        (temporal_difference * temporal_mask).sum()
        / temporal_denominator.clamp_min(1.0)
    )
    gradient = masked_spatial_gradient_loss(torch, prediction, target, mask)
    total = (
        masked
        + background_weight * clean_area
        + temporal_weight * temporal
        + gradient_weight * gradient
    )
    return total, masked, clean_area, temporal, gradient


def load_clip(torch, np, image_module, clip: Path, frames: int, rng: random.Random):
    paths = select_frame_window(sorted(clip.glob("frame-*.png")), frames, rng)
    arrays = []
    for path in paths:
        with image_module.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        arrays.append(torch.from_numpy(rgb).permute(2, 0, 1).div_(255.0))
    return torch.stack(arrays)


def evaluate(
    torch,
    functional,
    np,
    image_module,
    generator,
    clips,
    frames,
    device,
    seed,
    corruption_profile="legacy",
):
    generator.eval()
    rng = random.Random(seed)
    squared_error = 0.0
    pixel_count = 0
    temporal_error = 0.0
    temporal_count = 0
    with torch.inference_mode():
        for clip in clips:
            clean = load_clip(torch, np, image_module, clip, frames, rng).unsqueeze(0)
            clean = clean.to(device)
            low_quality, mask = synthetic_mosaic(
                torch, functional, clean, rng, corruption_profile
            )
            prediction = generator(low_quality).clamp(0, 1)
            error = (prediction - clean).square() * mask
            squared_error += error.sum().item()
            pixel_count += int(mask.sum().item()) * clean.shape[2]
            delta_error = (
                (prediction[:, 1:] - prediction[:, :-1])
                - (clean[:, 1:] - clean[:, :-1])
            ).abs()
            temporal_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
            temporal_error += (delta_error * temporal_mask).sum().item()
            temporal_count += int(temporal_mask.sum().item()) * clean.shape[2]
    mse = squared_error / max(1, pixel_count)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    temporal_mae = temporal_error / max(1, temporal_count)
    generator.train()
    return {"maskedPSNR": psnr, "temporalMAE": temporal_mae}


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        train_clips = list_clips(args.dataset, "train")
        validation_clips = list_clips(args.dataset, "validation")
        if not args.skip_dataset_audit:
            validate_dataset_audit(args.dataset, train_clips, validation_clips)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            "error: training requires torch, numpy and Pillow in the Jasna environment"
        ) from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("error: MPS is not available")

    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(None, str(args.weights), device, False)
    generator = model.generator
    if model.generator_ema is None:
        raise SystemExit("error: source checkpoint does not contain an EMA generator")
    ema_generator = model.generator_ema
    generator.load_state_dict(ema_generator.state_dict())
    ema_generator.requires_grad_(False).eval()
    trainable, frozen = set_trainable_scope(generator)
    optimizer = torch.optim.Adam(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
    )
    start_step = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        generator.load_state_dict(resume_checkpoint["generator"])
        ema_generator.load_state_dict(resume_checkpoint["generatorEMA"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        # The requested learning rate is part of the new run configuration.
        # torch's optimizer checkpoint otherwise silently restores the old one.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        start_step = int(resume_checkpoint["step"])
        if start_step >= args.steps:
            raise SystemExit(
                f"error: resume step {start_step} is not below target step {args.steps}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in configuration.items()
    }
    write_json_atomic(args.output / "training-config.json", configuration)
    selected_validation = validation_clips[: args.validation_clips]
    measured_start = evaluate(
        torch,
        functional,
        np,
        Image,
        ema_generator,
        selected_validation,
        args.frames,
        device,
        args.seed + 1,
        args.corruption_profile,
    )
    try:
        baseline = quality_baseline(
            measured_start, resume_checkpoint, args.reset_quality_baseline
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"Trainable parameters: {trainable:,}; frozen SPyNet: {frozen:,}\n"
        f"Original baseline: PSNR {baseline['maskedPSNR']:.3f} dB, "
        f"temporal MAE {baseline['temporalMAE']:.6f}",
        flush=True,
    )
    if resume_checkpoint is not None:
        print(
            f"Resume checkpoint validation: PSNR "
            f"{measured_start['maskedPSNR']:.3f} dB, temporal MAE "
            f"{measured_start['temporalMAE']:.6f}",
            flush=True,
        )
        stage_fraction = ema_stage_fraction(
            args.ema_decay,
            start_step,
            args.steps,
            args.gradient_accumulation,
        )
        print(
            f"EMA new-stage contribution at target: {stage_fraction * 100:.1f}% "
            f"(decay {args.ema_decay})",
            flush=True,
        )

    rng = random.Random(args.seed + start_step)
    generator.train()
    optimizer.zero_grad(set_to_none=True)
    best_psnr = (
        float(resume_checkpoint.get("bestPSNR", float("-inf")))
        if resume_checkpoint is not None and not args.reset_quality_baseline
        else float("-inf")
    )
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        batch = []
        for _ in range(args.batch_size):
            clip = rng.choice(train_clips)
            batch.append(load_clip(torch, np, Image, clip, args.frames, rng))
        clean = torch.stack(batch).to(device)
        low_quality, mask = synthetic_mosaic(
            torch, functional, clean, rng, args.corruption_profile
        )
        prediction = generator(low_quality)
        loss, masked, background, temporal, gradient = restoration_loss(
            torch,
            prediction,
            clean,
            mask,
            args.background_weight,
            args.temporal_weight,
            args.gradient_weight,
        )
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(torch, ema_generator, generator, args.ema_decay)

        if step == 1 or step % 25 == 0:
            elapsed = time.monotonic() - started
            print(
                f"step {step}/{args.steps}: loss {loss.item():.6f}, "
                f"masked {masked.item():.6f}, background {background.item():.6f}, "
                f"temporal {temporal.item():.6f}, gradient {gradient.item():.6f}, "
                f"{elapsed / (step - start_step):.2f}s/step",
                flush=True,
            )

        validation = None
        if step % args.validate_every == 0 or step == args.steps:
            validation = evaluate(
                torch,
                functional,
                np,
                Image,
                ema_generator,
                selected_validation,
                args.frames,
                device,
                args.seed + 1,
                args.corruption_profile,
            )
            print(
                f"validation {step}: PSNR {validation['maskedPSNR']:.3f} dB, "
                f"temporal MAE {validation['temporalMAE']:.6f}",
                flush=True,
            )
            accepted = passes_quality_gate(
                baseline,
                validation,
                args.minimum_psnr_gain,
                args.maximum_temporal_regression,
            )
            print(
                "quality gate: " + ("PASS" if accepted else "REJECT"),
                flush=True,
            )
            if accepted and validation["maskedPSNR"] > best_psnr:
                best_psnr = validation["maskedPSNR"]
                atomic_torch_save(
                    torch,
                    model_state_dict(generator, ema_generator),
                    args.output / "best-inference.pth",
                )
            write_json_atomic(
                args.output / "latest-validation.json",
                {
                    "step": step,
                    "baseline": baseline,
                    "bestPSNR": finite_metric_or_none(best_psnr),
                    "candidate": validation,
                    "minimumPSNRGain": args.minimum_psnr_gain,
                    "maximumTemporalRegression": args.maximum_temporal_regression,
                    "accepted": accepted,
                },
            )

        if step % args.save_every == 0 or step == args.steps:
            atomic_torch_save(
                torch,
                {
                    "version": 2,
                    "step": step,
                    "generator": generator.state_dict(),
                    "generatorEMA": ema_generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "baseline": baseline,
                    "bestPSNR": best_psnr,
                    "validation": validation,
                    "configuration": configuration,
                },
                args.output / f"checkpoint-{step:06d}.pth",
            )
            atomic_torch_save(
                torch,
                model_state_dict(generator, ema_generator),
                args.output / f"inference-{step:06d}.pth",
            )


if __name__ == "__main__":
    main()
