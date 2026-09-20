#!/usr/bin/env python3
"""Reject fine-tuning clips that still contain detectable mosaic regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--minimum-skin-fraction", type=float, default=0.08)
    parser.add_argument("--minimum-luma", type=float, default=0.03)
    parser.add_argument("--minimum-peak-detail", type=float, default=80.0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help=(
            "move flagged clips into rejected/<split> after a complete audit; "
            "nothing is deleted and a second audit is still required"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sample_frame_paths(clip: Path, stride: int) -> list[Path]:
    frames = sorted(clip.glob("frame-*.png"))
    if not frames:
        raise ValueError(f"clip contains no PNG frames: {clip}")
    selected = frames[::stride]
    if frames[-1] not in selected:
        selected.append(frames[-1])
    return selected


def discover_samples(dataset: Path, stride: int) -> list[tuple[str, Path]]:
    samples = []
    for split in ("train", "validation"):
        for clip in sorted((dataset / split).glob("clip-*")):
            if clip.is_dir():
                samples.extend((f"{split}/{clip.name}", path) for path in sample_frame_paths(clip, stride))
    if not samples:
        raise ValueError(f"no training clips found in {dataset}")
    return samples


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def quarantine_clips(dataset: Path, clip_names: list[str]) -> list[dict[str, str]]:
    """Move flagged clips out of the active train/validation trees safely."""
    moves: list[tuple[Path, Path, str]] = []
    for clip_name in clip_names:
        parts = Path(clip_name).parts
        if len(parts) != 2 or parts[0] not in ("train", "validation"):
            raise ValueError(f"invalid active clip name: {clip_name}")
        source = dataset / parts[0] / parts[1]
        destination = dataset / "rejected" / parts[0] / parts[1]
        if not source.is_dir():
            raise ValueError(f"flagged clip no longer exists: {source}")
        if destination.exists():
            raise ValueError(f"quarantine destination already exists: {destination}")
        moves.append((source, destination, clip_name))

    quarantined = []
    for source, destination, clip_name in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        quarantined.append(
            {
                "clip": clip_name,
                "destination": str(destination.relative_to(dataset)),
            }
        )
    return quarantined


def rebalance_dataset_splits(dataset: Path) -> list[dict[str, str]]:
    """Keep every surviving source in both train and validation."""
    manifest_path = dataset / "dataset.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"dataset manifest has no clip list: {manifest_path}")
    by_name = {item.get("name"): item for item in clips}
    if len(by_name) != len(clips) or None in by_name:
        raise ValueError(f"dataset manifest has invalid clip names: {manifest_path}")

    by_source: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "validation"):
        for clip_path in sorted((dataset / split).glob("clip-*")):
            if not clip_path.is_dir():
                continue
            item = by_name.get(clip_path.name)
            if item is None or not item.get("source"):
                raise ValueError(f"active clip is absent from manifest: {clip_path}")
            groups = by_source.setdefault(
                item["source"],
                {"train": [], "validation": []},
            )
            groups[split].append(clip_path.name)

    planned: list[tuple[str, str, str]] = []
    for source, groups in sorted(by_source.items()):
        if not groups["validation"]:
            if len(groups["train"]) < 2:
                raise ValueError(
                    f"source has too few surviving clips for split isolation: {source}"
                )
            planned.append((groups["train"][0], "train", "validation"))
        elif not groups["train"]:
            if len(groups["validation"]) < 2:
                raise ValueError(
                    f"source has too few surviving clips for split isolation: {source}"
                )
            planned.append((groups["validation"][0], "validation", "train"))

    for name, old_split, new_split in planned:
        source_path = dataset / old_split / name
        destination = dataset / new_split / name
        if destination.exists():
            raise ValueError(f"split destination already exists: {destination}")
        if not source_path.is_dir():
            raise ValueError(f"split source no longer exists: {source_path}")

    changes = []
    for name, old_split, new_split in planned:
        source_path = dataset / old_split / name
        destination = dataset / new_split / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(destination)
        item = by_name[name]
        item.setdefault("originalSplit", item.get("split", old_split))
        item["split"] = new_split
        changes.append(
            {
                "clip": name,
                "from": old_split,
                "to": new_split,
            }
        )
    if changes:
        write_json_atomic(manifest_path, manifest)
    return changes


def choose_device(torch, requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return requested


def frame_quality(cv2, frame) -> tuple[float, float, float]:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 135) & (y >= 35)
    detail = cv2.Laplacian(y, cv2.CV_64F).var()
    return float(skin.mean()), float(y.mean() / 255.0), float(detail)


def main() -> None:
    args = parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"error: dataset does not exist: {args.dataset}")
    if not args.model.is_file():
        raise SystemExit(f"error: RF-DETR checkpoint does not exist: {args.model}")
    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("error: --threshold must be between zero and one")
    if args.frame_stride <= 0 or args.batch_size <= 0:
        raise SystemExit("error: --frame-stride and --batch-size must be positive")
    if not 0.0 <= args.minimum_skin_fraction <= 1.0:
        raise SystemExit("error: --minimum-skin-fraction must be between zero and one")
    if not 0.0 <= args.minimum_luma <= 1.0:
        raise SystemExit("error: --minimum-luma must be between zero and one")
    if args.minimum_peak_detail < 0:
        raise SystemExit("error: --minimum-peak-detail cannot be negative")

    try:
        import cv2
        import torch
        from rfdetr_mps_detector import RFDetrMPSDetector
    except ImportError as error:
        raise SystemExit(
            "error: run this tool with the project's .venv-rfdetr Python"
        ) from error
    try:
        device = choose_device(torch, args.device)
        samples = discover_samples(args.dataset, args.frame_stride)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    detector = RFDetrMPSDetector(
        args.model,
        device=device,
        variant="large",
        max_select=16,
    )
    detections = []
    clip_quality = {}
    audited = 0
    for batch_start in range(0, len(samples), args.batch_size):
        batch = samples[batch_start : batch_start + args.batch_size]
        frames = []
        for _, path in batch:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"error: could not decode training frame: {path}")
            frames.append(frame)
        for (clip_name, _), frame in zip(batch, frames):
            skin_fraction, luma, detail = frame_quality(cv2, frame)
            quality = clip_quality.setdefault(
                clip_name,
                {
                    "sampledFrames": 0,
                    "skinFractionTotal": 0.0,
                    "minimumLuma": 1.0,
                    "maximumDetail": 0.0,
                },
            )
            quality["sampledFrames"] += 1
            quality["skinFractionTotal"] += skin_fraction
            quality["minimumLuma"] = min(quality["minimumLuma"], luma)
            quality["maximumDetail"] = max(quality["maximumDetail"], detail)
        predictions = detector.predict(frames, score_threshold=args.threshold)
        for (clip_name, path), prediction in zip(batch, predictions):
            audited += 1
            if not prediction.confidences:
                continue
            detections.append(
                {
                    "clip": clip_name,
                    "frame": path.name,
                    "maximumConfidence": max(prediction.confidences),
                    "boxes": prediction.boxes_xyxy,
                    "confidences": prediction.confidences,
                }
            )
        print(f"Audited {audited}/{len(samples)} frames", flush=True)

    low_content_clips = []
    quality_summary = {}
    for clip_name, values in clip_quality.items():
        average_skin = values["skinFractionTotal"] / values["sampledFrames"]
        quality_summary[clip_name] = {
            "sampledFrames": values["sampledFrames"],
            "averageSkinFraction": average_skin,
            "minimumLuma": values["minimumLuma"],
            "maximumDetail": values["maximumDetail"],
        }
        if (
            average_skin < args.minimum_skin_fraction
            or values["minimumLuma"] < args.minimum_luma
            or values["maximumDetail"] < args.minimum_peak_detail
        ):
            low_content_clips.append(clip_name)
    mosaic_clips = sorted({item["clip"] for item in detections})
    flagged_clips = sorted(set(mosaic_clips) | set(low_content_clips))
    quarantined = []
    rebalanced = []
    if args.quarantine and flagged_clips:
        try:
            quarantined = quarantine_clips(args.dataset, flagged_clips)
            rebalanced = rebalance_dataset_splits(args.dataset)
        except ValueError as error:
            raise SystemExit(f"error: {error}") from error
    report = {
        "version": 1,
        "model": str(args.model.resolve()),
        "device": device,
        "threshold": args.threshold,
        "frameStride": args.frame_stride,
        "minimumSkinFraction": args.minimum_skin_fraction,
        "minimumLuma": args.minimum_luma,
        "minimumPeakDetail": args.minimum_peak_detail,
        "auditedFrames": audited,
        "mosaicClips": mosaic_clips,
        "lowContentClips": sorted(low_content_clips),
        "flaggedClips": flagged_clips,
        "quarantineRequested": args.quarantine,
        "quarantinedClips": quarantined,
        "rebalancedSplits": rebalanced,
        "clipQuality": quality_summary,
        "detections": detections,
        "accepted": not flagged_clips,
    }
    output = args.output or args.dataset / "mosaic-audit.json"
    write_json_atomic(output, report)
    if flagged_clips:
        quarantine_note = ""
        if quarantined:
            quarantine_note = (
                f"; moved {len(quarantined)} clip(s) into "
                f"{args.dataset / 'rejected'}"
            )
            if rebalanced:
                quarantine_note += f"; rebalanced {len(rebalanced)} source split(s)"
            quarantine_note += "; run the audit again"
        print(
            f"REJECTED: {len(flagged_clips)} clip(s) contain possible mosaics "
            "or insufficient target detail; "
            f"see {output}{quarantine_note}",
            flush=True,
        )
        raise SystemExit(2)
    print(f"PASS: no mosaic detected in {audited} sampled frames; report {output}")


if __name__ == "__main__":
    main()
