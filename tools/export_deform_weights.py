#!/usr/bin/env python3
"""Export Jasna's four DCNv2 parameter sets in the Metal kernel's packed layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "jasna-metal-dcnv2-fp16-v1",
        "layout": "weight[input_channel,kernel_element,output_channel], then bias[output_channel]",
        "directions": {},
    }

    for direction in DIRECTIONS:
        module = generator.deform_align[direction]
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy()
        if weight.shape != (64, 128, 3, 3) or bias.shape != (64,):
            raise RuntimeError(f"unexpected {direction} shapes: {weight.shape}, {bias.shape}")
        packed = np.transpose(weight.reshape(64, 128, 9), (1, 2, 0)).astype("<f2", copy=False)
        payload = packed.tobytes(order="C") + bias.astype("<f2", copy=False).tobytes(order="C")
        output = args.output / f"{direction}.dcnfp16"
        output.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest["directions"][direction] = {
            "file": output.name,
            "bytes": len(payload),
            "weight_elements": int(packed.size),
            "bias_elements": int(bias.size),
            "sha256": digest,
        }
        print(f"exported {direction}: {output} ({len(payload)} bytes, sha256 {digest})")

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
