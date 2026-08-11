#!/usr/bin/env python3
"""Export Jasna's feature extractor as a macOS 27 Core AI experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coreai_torch
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing output: {args.output}")

    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    feature_extract = generator.feat_extract.cpu().eval()

    element_count = 3 * 256 * 256
    input_tensor = (
        torch.arange(element_count, dtype=torch.float32).remainder(251).div(250).sub(0.5)
        .reshape(1, 3, 256, 256)
    )
    with torch.inference_mode():
        reference = feature_extract(input_tensor).contiguous()

    exported = torch.export.export(feature_extract, args=(input_tensor,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(
            exported,
            input_names=["frames"],
            output_names=["features"],
        )
        .to_coreai()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(args.output)
    input_path = args.output.with_suffix(".input.f32")
    reference_path = args.output.with_suffix(".reference.f32")
    metadata_path = args.output.with_suffix(".json")
    input_tensor.numpy().astype("<f4", copy=False).tofile(input_path)
    reference.numpy().astype("<f4", copy=False).tofile(reference_path)
    metadata_path.write_text(
        json.dumps(
            {
                "input_name": "frames",
                "input_shape": list(input_tensor.shape),
                "output_name": "features",
                "output_shape": list(reference.shape),
                "input_file": input_path.name,
                "reference_file": reference_path.name,
                "reference_checksum": float(reference.double().sum()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Core AI feature extractor: {args.output}")
    print(f"Input shape:  {tuple(input_tensor.shape)}")
    print(f"Output shape: {tuple(reference.shape)}")
    print(f"Reference checksum: {reference.double().sum().item():.9f}")


if __name__ == "__main__":
    main()
