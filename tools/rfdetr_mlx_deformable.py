"""Experimental MLX bilinear sampling for RF-DETR deformable attention."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx


def sample_attention(value, locations, attention, *, height: int, width: int):
    """Sample one feature level with zero padding and align_corners=False.

    value: [B, H*W, heads, channels]
    locations: [B, queries, heads, points, 2], normalized x/y
    attention: [B, queries, heads, points]
    returns [B, queries, heads*channels]
    """
    batch, spatial, heads, channels = value.shape
    if spatial != height * width:
        raise ValueError("value length does not match spatial dimensions")
    queries, points = locations.shape[1], locations.shape[3]
    if locations.shape != (batch, queries, heads, points, 2):
        raise ValueError("invalid sampling location shape")
    if attention.shape != (batch, queries, heads, points):
        raise ValueError("invalid attention weight shape")
    values = mx.reshape(mx.transpose(value, (0, 2, 1, 3)),
                        (batch * heads * spatial, channels))
    x = locations[..., 0] * width - 0.5
    y = locations[..., 1] * height - 0.5
    x0, y0 = mx.floor(x), mx.floor(y)
    dx, dy = x - x0, y - y0
    offset = ((mx.arange(batch, dtype=mx.int32)[:, None, None, None] * heads)
              + mx.arange(heads, dtype=mx.int32)[None, None, :, None]) * spatial

    def corner(ix, iy, weight):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        safe_x = mx.clip(ix.astype(mx.int32), 0, width - 1)
        safe_y = mx.clip(iy.astype(mx.int32), 0, height - 1)
        indices = offset + safe_y * width + safe_x
        samples = mx.take(values, indices, axis=0)
        return samples * (weight * valid)[..., None]

    sampled = (corner(x0, y0, (1 - dx) * (1 - dy))
               + corner(x0 + 1, y0, dx * (1 - dy))
               + corner(x0, y0 + 1, (1 - dx) * dy)
               + corner(x0 + 1, y0 + 1, dx * dy))
    combined = mx.sum(sampled * attention[..., None], axis=3)
    return mx.reshape(combined, (batch, queries, heads * channels))


def deformable_attention(query, reference_points, memory, weights, *,
                         layer: int, height: int, width: int,
                         heads: int = 16, points: int = 2):
    """Evaluate one checkpoint cross-attention module for its single feature level."""
    batch, queries, channels = query.shape
    if memory.shape != (batch, height * width, channels):
        raise ValueError("memory shape does not match feature grid")
    if reference_points.shape not in ((batch, queries, 1, 2), (batch, queries, 1, 4)):
        raise ValueError("invalid reference point shape")
    base = f"transformer.decoder.layers.{layer}.cross_attn"

    def linear(x, name):
        return x @ mx.transpose(weights[f"{base}.{name}.weight"]) + weights[f"{base}.{name}.bias"]

    projected = linear(memory, "value_proj")
    projected = mx.reshape(projected, (batch, height * width, heads, channels // heads))
    offsets = mx.reshape(linear(query, "sampling_offsets"),
                         (batch, queries, heads, points, 2))
    attention = mx.reshape(linear(query, "attention_weights"),
                           (batch, queries, heads, points))
    attention = mx.softmax(attention, axis=-1)
    reference = reference_points[:, :, 0]
    if reference.shape[-1] == 2:
        normalizer = mx.array([width, height], dtype=query.dtype)
        locations = reference[:, :, None, None, :] + offsets / normalizer
    else:
        locations = (reference[:, :, None, None, :2]
                     + offsets / points * reference[:, :, None, None, 2:] * 0.5)
    selected = sample_attention(projected, locations, attention,
                                height=height, width=width)
    return linear(selected, "output_proj")
