#!/usr/bin/env python3
"""Export Jasna's RF-DETR segmentation checkpoint for an ANE experiment.

This is deliberately separate from the production MPS detector.  A fixed
batch is required by RF-DETR's Core ML exporter and lets the detector test use
CPU + Neural Engine while Metal restoration remains on the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--variant", choices=("medium", "large"), default="large")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.checkpoint.is_file():
        raise SystemExit(f"error: checkpoint not found: {arguments.checkpoint}")
    if arguments.batch_size <= 0:
        raise SystemExit("error: --batch-size must be positive")
    if arguments.resolution <= 0:
        raise SystemExit("error: --resolution must be positive")

    import coremltools as ct
    import rfdetr
    import torch

    wrappers = {
        "medium": rfdetr.RFDETRSegMedium,
        "large": rfdetr.RFDETRSegLarge,
    }
    name = (
        f"rfdetr-vr-{arguments.variant}-{arguments.resolution}-"
        f"batch{arguments.batch_size}-{arguments.precision}"
    )
    package = arguments.output_directory / f"{name}.mlpackage"
    metadata_path = arguments.output_directory / f"{name}.json"
    if package.exists():
        if not arguments.force:
            raise SystemExit(
                f"error: output already exists: {package}\n"
                "use --force to replace this experimental export"
            )
        shutil.rmtree(package)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    num_classes = int(state["class_embed.weight"].shape[0]) - 1
    print(
        f"Loading {arguments.variant} RF-DETR segmentation checkpoint: "
        f"{num_classes} class(es), {arguments.resolution}px"
    )
    model = wrappers[arguments.variant](
        num_classes=num_classes,
        resolution=arguments.resolution,
        pretrain_weights=str(arguments.checkpoint),
        device="cpu",
    )
    print(
        f"Exporting fixed batch {arguments.batch_size}, {arguments.precision}: "
        f"{package}"
    )
    exported_path = Path(
        model.export(
            output_dir=str(arguments.output_directory),
            format="coreml",
            batch_size=arguments.batch_size,
            dynamic_batch=False,
            coreml_precision=arguments.precision,
            output_name=name,
        )
    )
    if exported_path.resolve() != package.resolve():
        if package.exists():
            shutil.rmtree(package)
        shutil.move(str(exported_path), str(package))

    coreml_model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    specification = coreml_model.get_spec()
    inputs = [
        {
            "name": value.name,
            "description": value.shortDescription,
            "type": str(value.type),
        }
        for value in specification.description.input
    ]
    outputs = [
        {
            "name": value.name,
            "description": value.shortDescription,
            "type": str(value.type),
        }
        for value in specification.description.output
    ]
    metadata = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "package": str(package.resolve()),
        "variant": arguments.variant,
        "resolution": arguments.resolution,
        "batch_size": arguments.batch_size,
        "precision": arguments.precision,
        "num_classes": num_classes,
        "torch_version": torch.__version__,
        "rfdetr_version": __import__("importlib.metadata").metadata.version("rfdetr"),
        "coremltools_version": ct.__version__,
        "compute_units_for_test": "CPU_AND_NE",
        "inputs": inputs,
        "outputs": outputs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Core ML export: PASS")


if __name__ == "__main__":
    main()
