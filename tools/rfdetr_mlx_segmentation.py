"""Experimental MLX RF-DETR segmentation head for 768-pixel input."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import mlx.nn as nn


PREFIX = "segmentation_head"


def resize_bilinear(image, height: int, width: int):
    """NHWC resize with PyTorch's align_corners=False and border padding."""
    _, old_height, old_width, _ = image.shape
    ys = (mx.arange(height, dtype=mx.float32) + 0.5) * old_height / height - 0.5
    xs = (mx.arange(width, dtype=mx.float32) + 0.5) * old_width / width - 0.5
    ys = mx.clip(ys, 0, old_height - 1)
    xs = mx.clip(xs, 0, old_width - 1)
    y0, x0 = mx.floor(ys).astype(mx.int32), mx.floor(xs).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, old_height - 1)
    x1 = mx.minimum(x0 + 1, old_width - 1)
    dy = (ys - y0)[None, :, None, None]
    dx = (xs - x0)[None, None, :, None]
    def corner(y, x):
        return mx.take(mx.take(image, y, axis=1), x, axis=2)
    top = corner(y0, x0) * (1 - dx) + corner(y0, x1) * dx
    bottom = corner(y1, x0) * (1 - dx) + corner(y1, x1) * dx
    return top * (1 - dy) + bottom * dy


def segmentation_masks(spatial, query_layers, weights):
    """Return five native 192x192 mask-logit tensors from NHWC features."""
    if spatial.shape[1:3] != (64, 64) or query_layers.shape[0] != 5:
        raise ValueError("expected 64x64 features and five decoder layers")
    x = resize_bilinear(spatial, 192, 192)
    masks = []
    for index, queries in enumerate(query_layers):
        base = f"{PREFIX}.blocks.{index}"
        kernel = mx.transpose(weights[base + ".dwconv.weight"], (0, 2, 3, 1))
        conv = mx.conv2d(x, kernel, padding=1, groups=x.shape[-1])
        conv = conv + weights[base + ".dwconv.bias"]
        conv = mx.fast.layer_norm(conv, weights[base + ".norm.weight"],
                                  weights[base + ".norm.bias"], 1e-6)
        conv = conv @ mx.transpose(weights[base + ".pwconv1.weight"])
        conv = nn.gelu(conv + weights[base + ".pwconv1.bias"])
        x = x + conv
        spatial_kernel = mx.transpose(weights[PREFIX + ".spatial_features_proj.weight"],
                                      (0, 2, 3, 1))
        projected = mx.conv2d(x, spatial_kernel)
        projected = projected + weights[PREFIX + ".spatial_features_proj.bias"]
        query_base = PREFIX + ".query_features_block"
        queries = mx.fast.layer_norm(queries,
                                     weights[query_base + ".norm_in.weight"],
                                     weights[query_base + ".norm_in.bias"], 1e-5)
        queries = queries @ mx.transpose(weights[query_base + ".layers.0.weight"])
        queries = nn.gelu(queries + weights[query_base + ".layers.0.bias"])
        queries = queries @ mx.transpose(weights[query_base + ".layers.2.weight"])
        queries = queries + weights[query_base + ".layers.2.bias"]
        queries = queries + query_layers[index]
        queries = queries @ mx.transpose(weights[PREFIX + ".query_features_proj.weight"])
        queries = queries + weights[PREFIX + ".query_features_proj.bias"]
        batch, height, width, channels = projected.shape
        pixels = mx.reshape(projected, (batch, height * width, channels))
        logits = queries @ mx.transpose(pixels, (0, 2, 1))
        masks.append(mx.reshape(logits, (batch, queries.shape[1], height, width))
                     + weights[PREFIX + ".bias"])
    return masks
