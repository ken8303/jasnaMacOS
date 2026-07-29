#!/usr/bin/env python3
"""Convert Jasna's BasicVSR++ into Metal ML-compatible graph segments.

The deformable convolution remains in the native Metal kernel. Everything
around it is emitted as independent Core ML packages that can share MTLTensor
storage on the Metal command timeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn


DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")


class Upsample(nn.Module):
    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.reconstruction = generator.reconstruction
        self.upsample1 = generator.upsample1
        self.upsample2 = generator.upsample2
        self.conv_hr = generator.conv_hr
        self.conv_last = generator.conv_last
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reconstruction(x)
        x = self.activation(self.upsample1(x))
        x = self.activation(self.upsample2(x))
        x = self.activation(self.conv_hr(x))
        return self.conv_last(x)


class SPyNet(nn.Module):
    """Trace-friendly six-level SPyNet used by the Jasna TensorRT split."""

    def __init__(self, spynet: nn.Module) -> None:
        super().__init__()
        self.register_buffer("mean", spynet.mean.clone())
        self.register_buffer("std", spynet.std.clone())
        self.blocks = spynet.basic_module
        for size in (2, 4, 8, 16, 32, 64):
            theta = torch.eye(2, 3).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, (1, 3, size, size), align_corners=True)
            self.register_buffer(f"grid_{size}", grid)

    @staticmethod
    def warp(
        x: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor, coordinate_scale: float
    ) -> torch.Tensor:
        flow = flow.permute(0, 2, 3, 1)
        normalized = torch.stack(
            (flow[..., 0] * coordinate_scale, flow[..., 1] * coordinate_scale),
            dim=-1,
        )
        return torch.nn.functional.grid_sample(
            x, grid + normalized, mode="bilinear", padding_mode="border", align_corners=True
        )

    def level(
        self,
        block: nn.Module,
        reference: torch.Tensor,
        support: torch.Tensor,
        flow: torch.Tensor,
        grid: torch.Tensor,
        coordinate_scale: float,
    ) -> torch.Tensor:
        flow = torch.nn.functional.interpolate(
            flow, scale_factor=2, mode="bilinear", align_corners=True
        ) * 2.0
        return flow + block(
            torch.cat((reference, self.warp(support, flow, grid, coordinate_scale), flow), dim=1)
        )

    def forward(self, reference: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        r0 = (reference - self.mean) / self.std
        s0 = (support - self.mean) / self.std
        r1, s1 = torch.nn.functional.avg_pool2d(r0, 2, 2), torch.nn.functional.avg_pool2d(s0, 2, 2)
        r2, s2 = torch.nn.functional.avg_pool2d(r1, 2, 2), torch.nn.functional.avg_pool2d(s1, 2, 2)
        r3, s3 = torch.nn.functional.avg_pool2d(r2, 2, 2), torch.nn.functional.avg_pool2d(s2, 2, 2)
        r4, s4 = torch.nn.functional.avg_pool2d(r3, 2, 2), torch.nn.functional.avg_pool2d(s3, 2, 2)
        r5, s5 = torch.nn.functional.avg_pool2d(r4, 2, 2), torch.nn.functional.avg_pool2d(s4, 2, 2)
        flow = torch.zeros_like(reference[:, :2, :2, :2])
        flow = flow + self.blocks[0](
            torch.cat((r5, self.warp(s5, flow, self.grid_2, 2.0), flow), dim=1)
        )
        flow = self.level(self.blocks[1], r4, s4, flow, self.grid_4, 2.0 / 3.0)
        flow = self.level(self.blocks[2], r3, s3, flow, self.grid_8, 2.0 / 7.0)
        flow = self.level(self.blocks[3], r2, s2, flow, self.grid_16, 2.0 / 15.0)
        flow = self.level(self.blocks[4], r1, s1, flow, self.grid_32, 2.0 / 31.0)
        flow = self.level(self.blocks[5], r0, s0, flow, self.grid_64, 2.0 / 63.0)
        return flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jasna-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def convert(
    name: str,
    module: nn.Module,
    examples: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_dir: Path,
    validate: bool,
) -> None:
    output = output_dir / f"{name}.mlpackage"
    module = module.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, examples, strict=False)
        reference = module(*examples).detach().numpy() if validate else None
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=input_name, shape=tuple(example.shape))
            for input_name, example in zip(input_names, examples)
        ],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    model.author = "Jasna Metal PoC"
    model.short_description = f"BasicVSR++ segment: {name}"
    model.save(output)
    print(f"converted {name}: {output}")

    if validate:
        prediction = model.predict(
            {input_name: example.numpy() for input_name, example in zip(input_names, examples)}
        )["output"]
        difference = np.abs(reference - prediction)
        print(
            f"validated {name}: max={difference.max():.6g} "
            f"mean={difference.mean():.6g} p99={np.quantile(difference, 0.99):.6g}"
        )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.jasna_source.resolve()))
    from jasna.models.basicvsrpp.inference import load_model

    torch.manual_seed(7)
    model = load_model(None, str(args.weights), torch.device("cpu"), False)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)

    jobs: list[tuple[str, nn.Module, tuple[torch.Tensor, ...], tuple[str, ...]]] = [
        ("feature_extract", generator.feat_extract, (torch.randn(1, 3, 256, 256),), ("frames",)),
        ("upsample", Upsample(generator), (torch.randn(1, 320, 64, 64),), ("features",)),
    ]
    # Metal ML in Xcode 27 beta 4 rejects SPyNet's dynamic sample_grid op at
    # runtime. Emit its six convolutional residual blocks independently; the
    # pyramid, flow upsample, border warp, concatenation, and residual add are
    # handled on the same GPU timeline by custom Metal kernels.
    for level, size in enumerate((2, 4, 8, 16, 32, 64)):
        jobs.append(
            (
                f"spynet_level_{level}",
                generator.spynet.basic_module[level],
                (torch.randn(1, 8, size, size),),
                ("features",),
            )
        )
    for index, direction in enumerate(DIRECTIONS):
        jobs.append(
            (
                f"offset_{direction}",
                generator.deform_align[direction].conv_offset,
                (torch.randn(1, 196, 64, 64),),
                ("conditions",),
            )
        )
        jobs.append(
            (
                f"backbone_{direction}",
                generator.backbone[direction],
                (torch.randn(1, (2 + index) * 64, 64, 64),),
                ("features",),
            )
        )

    failures = []
    for job in jobs:
        if selected and job[0] not in selected:
            continue
        try:
            convert(*job, output_dir=args.output, validate=args.validate)
        except Exception as error:
            failures.append((job[0], error))
            print(f"FAILED {job[0]}: {error}", file=sys.stderr)

    if failures:
        raise SystemExit("conversion failures: " + ", ".join(name for name, _ in failures))


if __name__ == "__main__":
    main()
