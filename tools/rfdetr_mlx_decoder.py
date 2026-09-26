"""Experimental MLX RF-DETR decoder layers for the Jasna checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Models/MLXRuntime"))
import mlx.core as mx

from rfdetr_mlx_deformable import deformable_attention


def decoder_layer(target, memory, query_pos, reference_points, weights, *,
                  layer: int, height: int, width: int):
    """Run one detection-only decoder layer with the checkpoint's zero dropout."""
    batch, queries, channels = target.shape
    base = f"transformer.decoder.layers.{layer}"

    def parameter(name):
        return weights[f"{base}.{name}"]

    def linear(x, name):
        return x @ mx.transpose(parameter(name + ".weight")) + parameter(name + ".bias")

    def norm(x, name):
        return mx.fast.layer_norm(x, parameter(name + ".weight"),
                                  parameter(name + ".bias"), 1e-5)

    combined = target + query_pos
    packed_weight = parameter("self_attn.in_proj_weight")
    packed_bias = parameter("self_attn.in_proj_bias")
    head_count, head_width = 8, channels // 8

    def project(x, index):
        w = packed_weight[index * channels:(index + 1) * channels]
        b = packed_bias[index * channels:(index + 1) * channels]
        return mx.transpose(mx.reshape(x @ mx.transpose(w) + b,
                                       (batch, queries, head_count, head_width)),
                            (0, 2, 1, 3))

    q, k, v = project(combined, 0), project(combined, 1), project(target, 2)
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_width ** -0.5)
    attended = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)),
                          (batch, queries, channels))
    target = norm(target + linear(attended, "self_attn.out_proj"), "norm1")
    cross = deformable_attention(target + query_pos, reference_points, memory,
                                 weights, layer=layer, height=height, width=width)
    target = norm(target + cross, "norm2")
    feedforward = linear(mx.maximum(linear(target, "linear1"), 0), "linear2")
    return norm(target + feedforward, "norm3")


def transformer(memory, weights, *, height: int, width: int, queries: int = 200):
    """Run the checkpoint's five-layer, two-stage detection transformer.

    The current production path uses one unpadded feature level and no
    keypoint tokens; those are the only configurations supported here.
    """
    batch, spatial, channels = memory.shape
    if spatial != height * width or channels != 256:
        raise ValueError("expected one 256-channel feature level")

    def linear(x, name):
        return x @ mx.transpose(weights[name + ".weight"]) + weights[name + ".bias"]

    def mlp(x, base):
        x = mx.maximum(linear(x, base + ".layers.0"), 0)
        x = mx.maximum(linear(x, base + ".layers.1"), 0)
        return linear(x, base + ".layers.2")

    grid_y, grid_x = mx.meshgrid(mx.arange(height, dtype=mx.float32),
                                 mx.arange(width, dtype=mx.float32), indexing="ij")
    centers = mx.stack(((grid_x + 0.5) / width, (grid_y + 0.5) / height), axis=-1)
    centers = mx.reshape(centers, (1, spatial, 2))
    proposals = mx.concatenate((mx.broadcast_to(centers, (batch, spatial, 2)),
                                mx.full((batch, spatial, 2), 0.05)), axis=-1)
    valid = mx.all((proposals > 0.01) & (proposals < 0.99), axis=-1, keepdims=True)
    output_memory = mx.where(valid, memory, mx.zeros_like(memory))
    proposals = mx.where(valid, proposals, mx.zeros_like(proposals))
    selected_memory = mx.fast.layer_norm(
        linear(output_memory, "transformer.enc_output.0"),
        weights["transformer.enc_output_norm.0.weight"],
        weights["transformer.enc_output_norm.0.bias"], 1e-5,
    )
    class_scores = linear(selected_memory, "transformer.enc_out_class_embed.0")
    indices = mx.argsort(-mx.max(class_scores, axis=-1), axis=1)[:, :queries]
    initial_memory = mx.take_along_axis(selected_memory, indices[:, :, None], axis=1)
    initial_proposals = mx.take_along_axis(proposals, indices[:, :, None], axis=1)
    delta = mlp(initial_memory, "transformer.enc_out_bbox_embed.0")
    first_boxes = mx.concatenate((delta[..., :2] * initial_proposals[..., 2:]
                                  + initial_proposals[..., :2],
                                  mx.exp(delta[..., 2:]) * initial_proposals[..., 2:]), axis=-1)
    target = mx.broadcast_to(weights["query_feat.weight"][:queries],
                             (batch, queries, channels))
    initial_reference = mx.broadcast_to(weights["refpoint_embed.weight"][:queries],
                                        (batch, queries, 4))
    ref_xy = initial_reference[..., :2] * first_boxes[..., 2:] + first_boxes[..., :2]
    ref_wh = mx.exp(initial_reference[..., 2:]) * first_boxes[..., 2:]
    reference = mx.concatenate((ref_xy, ref_wh), axis=-1)

    dimension = mx.arange(128, dtype=mx.float32)
    divisor = mx.power(mx.array(10000.0), 2 * mx.floor(dimension / 2) / 128)
    def position_component(index):
        angle = reference[..., index, None] * (2 * 3.141592653589793) / divisor
        return mx.reshape(mx.stack((mx.sin(angle[..., 0::2]),
                                    mx.cos(angle[..., 1::2])), axis=-1),
                          (batch, queries, 128))
    sine = mx.concatenate(tuple(position_component(i) for i in (1, 0, 2, 3)), axis=-1)
    query_pos = mx.maximum(linear(sine, "transformer.decoder.ref_point_head.layers.0"), 0)
    query_pos = linear(query_pos, "transformer.decoder.ref_point_head.layers.1")
    outputs = []
    for index in range(5):
        target = decoder_layer(target, memory, query_pos, reference[:, :, None],
                               weights, layer=index, height=height, width=width)
        outputs.append(mx.fast.layer_norm(
            target, weights["transformer.decoder.norm.weight"],
            weights["transformer.decoder.norm.bias"], 1e-5,
        ))
    return mx.stack(outputs), reference[None], initial_memory, first_boxes
