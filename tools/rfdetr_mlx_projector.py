"""MLX single-scale RF-DETR feature projector for the Jasna checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx


PREFIX = "backbone.0.projector.stages.0"


def project_features(features, weights):
    """Fuse four NHWC DINOv2 feature maps into the decoder's NHWC map."""
    if len(features) != 4 or any(x.shape[-1] != 384 for x in features):
        raise ValueError("expected four 384-channel DINOv2 feature maps")

    def conv_norm_silu(x, name):
        base = f"{PREFIX}.0.{name}"
        kernel = weights[base + ".conv.weight"]
        kernel = mx.transpose(kernel, (0, 2, 3, 1))
        padding = kernel.shape[1] // 2
        x = mx.conv2d(x, kernel, padding=padding)
        x = mx.fast.layer_norm(x, weights[base + ".bn.weight"],
                               weights[base + ".bn.bias"], 1e-6)
        return x * mx.sigmoid(x)

    fused = mx.concatenate(features, axis=-1)
    first = conv_norm_silu(fused, "cv1")
    channels = first.shape[-1] // 2
    chunks = [first[..., :channels], first[..., channels:]]
    for index in range(3):
        hidden = conv_norm_silu(chunks[-1], f"m.{index}.cv1")
        chunks.append(conv_norm_silu(hidden, f"m.{index}.cv2"))
    fused = conv_norm_silu(mx.concatenate(chunks, axis=-1), "cv2")
    return mx.fast.layer_norm(fused, weights[PREFIX + ".1.weight"],
                              weights[PREFIX + ".1.bias"], 1e-6)
