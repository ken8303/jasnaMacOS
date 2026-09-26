"""Experimental MLX DINOv2 backbone blocks for Jasna's RF-DETR checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx
import mlx.nn as nn


PREFIX = "backbone.0.encoder.encoder.encoder.layer"
EMBEDDING_PREFIX = "backbone.0.encoder.encoder.embeddings"


def patch_embeddings(image, weights, *, patch_size: int = 12, windows: int = 2):
    """Create class and patch tokens in RF-DETR's window order.

    Input is NHWC. The current checkpoint has 4096 position tokens, which
    exactly matches its 768-pixel input at patch size 12. Other input sizes
    require the checkpoint's bicubic position interpolation and are rejected.
    """
    batch, height, width, channels = image.shape
    if channels != 3 or height != 768 or width != 768:
        raise ValueError("this checkpoint currently supports 768x768 RGB input")
    if height % (patch_size * windows) or width % (patch_size * windows):
        raise ValueError("input dimensions must divide into attention windows")
    key = EMBEDDING_PREFIX + ".patch_embeddings.projection"
    kernel = mx.transpose(weights[key + ".weight"], (0, 2, 3, 1))
    patches = mx.conv2d(image, kernel, stride=patch_size) + weights[key + ".bias"]
    rows, cols = height // patch_size, width // patch_size
    tokens = mx.reshape(patches, (batch, rows * cols, -1))
    cls = mx.broadcast_to(weights[EMBEDDING_PREFIX + ".cls_token"],
                          (batch, 1, tokens.shape[-1]))
    tokens = mx.concatenate((cls, tokens), axis=1)
    tokens = tokens + weights[EMBEDDING_PREFIX + ".position_embeddings"]
    cls, pixels = tokens[:, :1], tokens[:, 1:]
    pixels = mx.reshape(pixels, (batch * windows, rows // windows,
                                 windows, cols // windows, -1))
    pixels = mx.transpose(pixels, (0, 2, 1, 3, 4))
    pixels = mx.reshape(pixels, (batch * windows * windows,
                                 rows * cols // (windows * windows), -1))
    cls = mx.tile(cls, (windows * windows, 1, 1))
    return mx.concatenate((cls, pixels), axis=1)


def encoder_block(hidden, weights, index: int, *, heads: int = 6, eps: float = 1e-6):
    """Evaluate one RF-DETR DINOv2 block; window packing is supplied by caller."""
    base = f"{PREFIX}.{index}"

    def parameter(name):
        return weights[f"{base}.{name}"]

    def linear(x, name):
        return x @ mx.transpose(parameter(name + ".weight")) + parameter(name + ".bias")

    norm = mx.fast.layer_norm(
        hidden, parameter("norm1.weight"), parameter("norm1.bias"), eps,
    )
    batch, tokens, channels = norm.shape
    if channels % heads:
        raise ValueError("hidden size must be divisible by attention heads")
    head_width = channels // heads

    def project(name):
        return mx.transpose(
            mx.reshape(linear(norm, f"attention.attention.{name}"),
                       (batch, tokens, heads, head_width)),
            (0, 2, 1, 3),
        )

    query, key, value = (project(name) for name in ("query", "key", "value"))
    attention = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=head_width ** -0.5,
    )
    attention = mx.reshape(mx.transpose(attention, (0, 2, 1, 3)), (batch, tokens, channels))
    attention = linear(attention, "attention.output.dense")
    hidden = hidden + attention * parameter("layer_scale1.lambda1")
    norm = mx.fast.layer_norm(
        hidden, parameter("norm2.weight"), parameter("norm2.bias"), eps,
    )
    mlp = linear(nn.gelu(linear(norm, "mlp.fc1")), "mlp.fc2")
    return hidden + mlp * parameter("layer_scale2.lambda1")


def backbone_features(image, weights):
    """Run the 12 DINOv2 blocks and return stage 3/6/9/12 feature maps.

    Feature map layout is NHWC for subsequent MLX layers.
    """
    hidden = patch_embeddings(image, weights)
    batch = image.shape[0]
    features = []
    window_blocks = {0, 1, 2, 4, 5, 7, 8, 10, 11}
    for index in range(12):
        if index in window_blocks:
            hidden = encoder_block(hidden, weights, index)
        else:
            window_batch, tokens, channels = hidden.shape
            joined = mx.reshape(hidden, (window_batch // 4, tokens * 4, channels))
            hidden = mx.reshape(encoder_block(joined, weights, index),
                                (window_batch, tokens, channels))
        if index + 1 in (3, 6, 9, 12):
            channels = hidden.shape[-1]
            base = "backbone.0.encoder.encoder.layernorm"
            normalized = mx.fast.layer_norm(
                hidden, weights[base + ".weight"], weights[base + ".bias"], 1e-6,
            )[:, 1:]
            # Reverse the checkpoint's window packing, preserving row order.
            normalized = mx.reshape(normalized, (batch * 2, 2, 32, 32, channels))
            normalized = mx.transpose(normalized, (0, 2, 1, 3, 4))
            features.append(mx.reshape(normalized, (batch, 64, 64, channels)))
    return features
