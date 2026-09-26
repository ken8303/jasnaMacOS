#!/usr/bin/env python3
"""Compare every implemented MLX restoration segment with v1.2 PyTorch math."""

import argparse

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision

from basicvsrpp_mlx_segments import BasicVSRSegments, flow_warp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    prefix = "generator_ema."
    model = BasicVSRSegments(args.archive)
    rng = np.random.default_rng(7)

    image = rng.normal(size=(2, 7, 9, 3)).astype(np.float32)
    displacement = rng.normal(size=(2, 7, 9, 2)).astype(np.float32) * 2
    displacement[0, 0, 0] = (50, -50)
    for padding in ("zeros", "border"):
        image_t = torch.from_numpy(np.transpose(image, (0, 3, 1, 2)).copy())
        flow_t = torch.from_numpy(displacement)
        grid = functional.affine_grid(torch.eye(2, 3)[None].expand(2, -1, -1),
                                      image_t.shape, align_corners=True)
        normalized = torch.stack((flow_t[..., 0] * 2 / 8, flow_t[..., 1] * 2 / 6), -1)
        expected_warp = functional.grid_sample(image_t, grid + normalized,
                                                padding_mode=padding, align_corners=True)
        actual_warp = np.transpose(np.array(flow_warp(mx.array(image), mx.array(displacement), padding)),
                                   (0, 3, 1, 2))
        error = np.abs(expected_warp.numpy() - actual_warp)
        print(f"flow_warp_{padding}: max={error.max():.6g} mean={error.mean():.6g}")
        if not np.allclose(expected_warp.numpy(), actual_warp, atol=1e-5, rtol=1e-5):
            raise AssertionError(f"flow warp {padding} differs from PyTorch")

    def conv(x, name, stride=1, padding=1):
        return functional.conv2d(x, state[prefix + name + ".weight"],
                                 state[prefix + name + ".bias"], stride=stride, padding=padding)

    def lrelu(x):
        return functional.leaky_relu(x, negative_slope=0.1)

    def stack(x, name, count):
        x = lrelu(conv(x, name + ".main.0"))
        for index in range(count):
            base = f"{name}.main.2.{index}"
            x = x + conv(functional.relu(conv(x, base + ".conv1")), base + ".conv2")
        return x

    cases = [
        ("feature_extract", (1, 3, 64, 64),
         lambda x: stack(lrelu(conv(lrelu(conv(x, "feat_extract.0", 2)), "feat_extract.2", 2)),
                         "feat_extract.4", 5), model.feature_extract),
        ("spynet_level_0", (1, 8, 16, 16),
         lambda x: spynet_reference(x, conv), lambda x: model.spynet_level(x, 0)),
        ("offset_backward_1", (1, 196, 16, 16),
         lambda x: offset_reference(x, conv, lrelu), lambda x: model.offset(x, "backward_1")),
        ("backbone_backward_1", (1, 128, 16, 16),
         lambda x: stack(x, "backbone.backward_1", 15),
         lambda x: model.backbone(x, "backward_1")),
        ("upsample", (1, 320, 8, 8),
         lambda x: upsample_reference(stack(x, "reconstruction", 5), conv, lrelu),
         model.upsample),
    ]
    for name, shape, reference, mlx_run in cases:
        source = rng.normal(size=shape).astype(np.float32) * 0.1
        with torch.inference_mode():
            expected = reference(torch.from_numpy(source)).numpy()
        actual_nhwc = np.array(mlx_run(mx.array(np.transpose(source, (0, 2, 3, 1)).copy())))
        actual = np.transpose(actual_nhwc, (0, 3, 1, 2))
        error = np.abs(expected - actual)
        print(f"{name}: max={error.max():.6g} mean={error.mean():.6g}")
        if not np.allclose(expected, actual, atol=1e-4, rtol=1e-4):
            raise AssertionError(f"{name} differs from PyTorch")

    ref = rng.uniform(size=(1, 64, 64, 3)).astype(np.float32)
    supp = rng.uniform(size=(1, 64, 64, 3)).astype(np.float32)
    with torch.inference_mode():
        expected = spynet_flow_reference(torch.from_numpy(ref.transpose(0, 3, 1, 2).copy()),
                                         torch.from_numpy(supp.transpose(0, 3, 1, 2).copy()),
                                         state, prefix)
    actual = np.array(model.spynet_flow(mx.array(ref), mx.array(supp)))
    error = np.abs(expected.numpy().transpose(0, 2, 3, 1) - actual)
    print(f"spynet_flow: max={error.max():.6g} mean={error.mean():.6g}")
    if not np.allclose(expected.numpy().transpose(0, 2, 3, 1), actual, atol=1e-4, rtol=1e-4):
        raise AssertionError("full SPyNet flow differs from PyTorch")

    shape = (1, 128, 5, 6)
    source = rng.normal(size=shape).astype(np.float32) * 0.1
    offsets = rng.normal(size=(1, 16 * 18, 5, 6)).astype(np.float32) * 0.3
    mask = rng.uniform(size=(1, 16 * 9, 5, 6)).astype(np.float32)
    with torch.inference_mode():
        expected = torchvision.ops.deform_conv2d(
            torch.from_numpy(source), torch.from_numpy(offsets),
            state[prefix + "deform_align.backward_1.weight"],
            state[prefix + "deform_align.backward_1.bias"],
            padding=1, mask=torch.from_numpy(mask)).numpy()
    actual = np.array(model.deform_conv(
        mx.array(source.transpose(0, 2, 3, 1).copy()),
        mx.array(offsets.transpose(0, 2, 3, 1).copy()),
        mx.array(mask.transpose(0, 2, 3, 1).copy()), "backward_1"))
    actual = actual.transpose(0, 3, 1, 2)
    error = np.abs(expected - actual)
    print(f"deform_conv: max={error.max():.6g} mean={error.mean():.6g}")
    if not np.allclose(expected, actual, atol=1e-4, rtol=1e-4):
        raise AssertionError("deformable convolution differs from PyTorch")

    condition = rng.normal(size=(1, 192, 5, 6)).astype(np.float32) * 0.1
    flow1 = rng.normal(size=(1, 2, 5, 6)).astype(np.float32) * 0.2
    flow2 = rng.normal(size=(1, 2, 5, 6)).astype(np.float32) * 0.2
    with torch.inference_mode():
        aligned_condition = torch.cat((torch.from_numpy(condition), torch.from_numpy(flow1),
                                       torch.from_numpy(flow2)), 1)
        raw = offset_reference(aligned_condition, conv, lrelu)
        one, two, predicted_mask = torch.chunk(raw, 3, dim=1)
        predicted_offsets = 10 * torch.tanh(torch.cat((one, two), 1))
        offset1, offset2 = torch.chunk(predicted_offsets, 2, dim=1)
        offset1 = offset1 + torch.from_numpy(flow1).flip(1).repeat(1, offset1.shape[1] // 2, 1, 1)
        offset2 = offset2 + torch.from_numpy(flow2).flip(1).repeat(1, offset2.shape[1] // 2, 1, 1)
        expected_align = torchvision.ops.deform_conv2d(
            torch.from_numpy(source), torch.cat((offset1, offset2), 1),
            state[prefix + "deform_align.backward_1.weight"],
            state[prefix + "deform_align.backward_1.bias"],
            padding=1, mask=torch.sigmoid(predicted_mask)).numpy()
    actual_align = np.array(model.align(
        mx.array(source.transpose(0, 2, 3, 1).copy()),
        mx.array(condition.transpose(0, 2, 3, 1).copy()),
        mx.array(flow1.transpose(0, 2, 3, 1).copy()),
        mx.array(flow2.transpose(0, 2, 3, 1).copy()), "backward_1")).transpose(0, 3, 1, 2)
    align_error = np.abs(expected_align - actual_align)
    print(f"second_order_align: max={align_error.max():.6g} mean={align_error.mean():.6g}")
    if not np.allclose(expected_align, actual_align, atol=1e-4, rtol=1e-4):
        raise AssertionError("second-order alignment differs from PyTorch")

    frame = rng.uniform(size=(1, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        frame_t = torch.from_numpy(frame)
        spatial = stack(lrelu(conv(lrelu(conv(frame_t, "feat_extract.0", 2)),
                                    "feat_extract.2", 2)), "feat_extract.4", 5)
        features = [spatial]
        for direction in ("backward_1", "forward_1", "backward_2", "forward_2"):
            features.append(stack(torch.cat(features + [torch.zeros_like(spatial)], 1),
                                  "backbone." + direction, 15))
        expected_frame = upsample_reference(stack(torch.cat(features, 1),
                                                  "reconstruction", 5), conv, lrelu) + frame_t
    actual_frame = np.array(model.restore_single_frame(
        mx.array(frame.transpose(0, 2, 3, 1).copy()))).transpose(0, 3, 1, 2)
    frame_error = np.abs(expected_frame.numpy() - actual_frame)
    print(f"single_frame_restoration: max={frame_error.max():.6g} mean={frame_error.mean():.6g}")
    if not np.allclose(expected_frame.numpy(), actual_frame, atol=1e-4, rtol=1e-4):
        raise AssertionError("one-frame restoration differs from PyTorch")

    video = rng.uniform(size=(1, 3, 32, 32, 3)).astype(np.float32)
    forward = [rng.normal(size=(1, 8, 8, 2)).astype(np.float32) * 0.1 for _ in range(2)]
    backward = [rng.normal(size=(1, 8, 8, 2)).astype(np.float32) * 0.1 for _ in range(2)]
    with torch.inference_mode():
        expected_video = restore_video_reference(video, forward, backward, state, prefix,
                                                 conv, lrelu, stack)
    actual_video = np.array(model.restore_frames(
        mx.array(video), [mx.array(flow) for flow in forward],
        [mx.array(flow) for flow in backward]))
    video_error = np.abs(expected_video - actual_video)
    print(f"three_frame_restoration: max={video_error.max():.6g} mean={video_error.mean():.6g}")
    if not np.allclose(expected_video, actual_video, atol=1e-4, rtol=1e-4):
        raise AssertionError("three-frame restoration differs from PyTorch")

    large = rng.uniform(size=(1, 256, 256, 3)).astype(np.float32)
    torch_down = functional.interpolate(torch.from_numpy(large.transpose(0, 3, 1, 2).copy()),
                                        scale_factor=0.25, mode="bicubic")
    mlx_down = nn.Upsample(scale_factor=0.25, mode="cubic", align_corners=False)(mx.array(large))
    down_error = np.abs(torch_down.numpy().transpose(0, 2, 3, 1) - np.array(mlx_down))
    print(f"bicubic_downsample: max={down_error.max():.6g} mean={down_error.mean():.6g}")
    if not np.allclose(torch_down.numpy().transpose(0, 2, 3, 1), np.array(mlx_down),
                       atol=1e-4, rtol=1e-4):
        raise AssertionError("bicubic downsample differs from PyTorch")


def spynet_reference(x, conv):
    for index in range(5):
        x = conv(x, f"spynet.basic_module.0.basic_module.{index}.conv", padding=3)
        if index != 4:
            x = functional.relu(x)
    return x


def offset_reference(x, conv, lrelu):
    for index in (0, 2, 4, 6):
        x = conv(x, f"deform_align.backward_1.conv_offset.{index}")
        if index != 6:
            x = lrelu(x)
    return x


def upsample_reference(x, conv, lrelu):
    for stage in (1, 2):
        x = lrelu(functional.pixel_shuffle(conv(x, f"upsample{stage}.upsample_conv"), 2))
    x = lrelu(conv(x, "conv_hr"))
    return conv(x, "conv_last")


def spynet_flow_reference(ref, supp, state, prefix):
    mean = state[prefix + "spynet.mean"]
    std = state[prefix + "spynet.std"]
    refs = [(ref - mean) / std]
    supps = [(supp - mean) / std]
    for _ in range(5):
        refs.append(functional.avg_pool2d(refs[-1], 2))
        supps.append(functional.avg_pool2d(supps[-1], 2))
    refs.reverse()
    supps.reverse()
    flow = torch.zeros((1, 2, 2, 2))
    for level in range(6):
        if level:
            flow = functional.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2
        grid = functional.affine_grid(torch.eye(2, 3)[None], supps[level].shape, align_corners=True)
        height, width = flow.shape[-2:]
        normalized = torch.stack((flow[:, 0] * 2 / (width - 1),
                                  flow[:, 1] * 2 / (height - 1)), -1)
        warped = functional.grid_sample(supps[level], grid + normalized,
                                        padding_mode="border", align_corners=True)
        x = torch.cat((refs[level], warped, flow), 1)
        for layer in range(5):
            base = f"spynet.basic_module.{level}.basic_module.{layer}.conv"
            x = functional.conv2d(x, state[prefix + base + ".weight"],
                                  state[prefix + base + ".bias"], padding=3)
            if layer < 4:
                x = functional.relu(x)
        flow = flow + x
    return flow


def restore_video_reference(video, forward, backward, state, prefix, conv, lrelu, stack):
    frames = [torch.from_numpy(video[:, i].transpose(0, 3, 1, 2).copy()) for i in range(3)]
    spatial = [stack(lrelu(conv(lrelu(conv(frame, "feat_extract.0", 2)),
                                  "feat_extract.2", 2)), "feat_extract.4", 5) for frame in frames]
    flows_forward = [torch.from_numpy(item.transpose(0, 3, 1, 2).copy()) for item in forward]
    flows_backward = [torch.from_numpy(item.transpose(0, 3, 1, 2).copy()) for item in backward]
    features = {"spatial": spatial}

    def warp(x, flow):
        height, width = x.shape[-2:]
        grid = functional.affine_grid(torch.eye(2, 3)[None], x.shape, align_corners=True)
        norm = torch.stack((flow[:, 0] * 2 / (width - 1),
                            flow[:, 1] * 2 / (height - 1)), -1)
        return functional.grid_sample(x, grid + norm, align_corners=True)

    for direction in ("backward_1", "forward_1", "backward_2", "forward_2"):
        is_backward = direction.startswith("backward")
        order = list(range(2, -1, -1)) if is_backward else list(range(3))
        flows = flows_backward if is_backward else flows_forward
        outputs = [None] * 3
        propagated = torch.zeros_like(spatial[0])
        for position, index in enumerate(order):
            if position:
                flow_index = index if is_backward else index - 1
                flow1 = flows[flow_index]
                cond1 = warp(propagated, flow1)
                prev2 = torch.zeros_like(propagated)
                flow2 = torch.zeros_like(flow1)
                cond2 = torch.zeros_like(cond1)
                if position > 1:
                    prev2 = outputs[order[position - 2]]
                    prior_index = order[position - 1] if is_backward else order[position - 2]
                    flow2 = flow1 + warp(flows[prior_index], flow1)
                    cond2 = warp(prev2, flow2)
                condition = torch.cat((cond1, spatial[index], cond2, flow1, flow2), 1)
                raw = condition
                for layer in (0, 2, 4, 6):
                    raw = conv(raw, f"deform_align.{direction}.conv_offset.{layer}")
                    if layer != 6:
                        raw = lrelu(raw)
                o1, o2, mask = torch.chunk(raw, 3, 1)
                offsets = 10 * torch.tanh(torch.cat((o1, o2), 1))
                off1, off2 = torch.chunk(offsets, 2, 1)
                off1 = off1 + flow1.flip(1).repeat(1, off1.shape[1] // 2, 1, 1)
                off2 = off2 + flow2.flip(1).repeat(1, off2.shape[1] // 2, 1, 1)
                propagated = torchvision.ops.deform_conv2d(
                    torch.cat((propagated, prev2), 1), torch.cat((off1, off2), 1),
                    state[prefix + f"deform_align.{direction}.weight"],
                    state[prefix + f"deform_align.{direction}.bias"],
                    padding=1, mask=torch.sigmoid(mask))
            earlier = [features[name][index] for name in features if name != "spatial"]
            propagated = propagated + stack(torch.cat([spatial[index]] + earlier + [propagated], 1),
                                            "backbone." + direction, 15)
            outputs[index] = propagated
        features[direction] = outputs
    restored = []
    for index in range(3):
        combined = torch.cat([features[name][index] for name in features], 1)
        result = upsample_reference(stack(combined, "reconstruction", 5), conv, lrelu)
        restored.append((result + frames[index]).numpy().transpose(0, 2, 3, 1))
    return np.stack(restored, 1)


if __name__ == "__main__":
    main()
