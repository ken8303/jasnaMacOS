"""Experimental MLX BasicVSR++ inference with optical flow and temporal alignment.

Inputs and outputs are NHWC. The saved weights are exported by
convert_basicvsrpp_mlx.py. Intermediate results are evaluated as the video
advances so full 30-frame windows fit Metal's graph resource limits.
"""

import mlx.core as mx
import mlx.nn as nn


def flow_warp(x, flow, padding="zeros"):
    """Bilinear pixel-space warp, matching PyTorch grid_sample align_corners=True."""
    batch, height, width, channels = x.shape
    if flow.shape != (batch, height, width, 2):
        raise ValueError("flow must be NHW2 and match the feature map")
    if padding not in ("zeros", "border"):
        raise ValueError("padding must be zeros or border")
    xx = mx.arange(width, dtype=flow.dtype)[None, None, :]
    yy = mx.arange(height, dtype=flow.dtype)[None, :, None]
    sample_x = xx + flow[..., 0]
    sample_y = yy + flow[..., 1]
    if padding == "border":
        sample_x = mx.clip(sample_x, 0, width - 1)
        sample_y = mx.clip(sample_y, 0, height - 1)
    x0 = mx.floor(sample_x)
    y0 = mx.floor(sample_y)
    dx = sample_x - x0
    dy = sample_y - y0
    flat = mx.reshape(x, (batch * height * width, channels))
    batch_offset = mx.arange(batch, dtype=mx.int32)[:, None, None] * height * width

    def pixel(xi, yi):
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        clipped_x = mx.clip(xi, 0, width - 1).astype(mx.int32)
        clipped_y = mx.clip(yi, 0, height - 1).astype(mx.int32)
        index = batch_offset + clipped_y * width + clipped_x
        value = mx.take(flat, index, axis=0)
        return value if padding == "border" else value * inside[..., None]

    return ((1 - dx) * (1 - dy))[..., None] * pixel(x0, y0) + \
           (dx * (1 - dy))[..., None] * pixel(x0 + 1, y0) + \
           ((1 - dx) * dy)[..., None] * pixel(x0, y0 + 1) + \
           (dx * dy)[..., None] * pixel(x0 + 1, y0 + 1)


class BasicVSRSegments:
    def __init__(self, archive):
        self.weights = mx.load(str(archive))

    def conv(self, x, name, stride=1, padding=1):
        w = self.weights[name + ".weight"]
        b = self.weights[name + ".bias"]
        return mx.conv2d(x, w, stride=stride, padding=padding) + b

    @staticmethod
    def lrelu(x):
        return mx.maximum(x, 0) + 0.1 * mx.minimum(x, 0)

    def residual_stack(self, x, prefix):
        x = self.lrelu(self.conv(x, prefix + ".main.0"))
        block_prefix = prefix + ".main.2."
        blocks = sorted({int(key[len(block_prefix):].split(".")[0])
                         for key in self.weights if key.startswith(block_prefix)})
        for index in blocks:
            base = block_prefix + str(index)
            delta = mx.maximum(self.conv(x, base + ".conv1"), 0)
            x = x + self.conv(delta, base + ".conv2")
        return x

    def feature_extract(self, x):
        x = self.lrelu(self.conv(x, "feat_extract.0", stride=2))
        x = self.lrelu(self.conv(x, "feat_extract.2", stride=2))
        return self.residual_stack(x, "feat_extract.4")

    def spynet_level(self, x, level):
        for index in range(5):
            x = self.conv(x, f"spynet.basic_module.{level}.basic_module.{index}.conv", padding=3)
            if index != 4:
                x = mx.maximum(x, 0)
        return x

    def spynet_flow(self, reference, support):
        """Six-level SPyNet flow for equal-sized frames divisible by 32."""
        if reference.shape != support.shape or reference.shape[-1] != 3:
            raise ValueError("SPyNet expects matching NHWC RGB frames")
        batch, height, width, _ = reference.shape
        if height % 32 or width % 32 or min(height, width) < 64:
            raise ValueError("SPyNet inputs must be at least 64 and divisible by 32")
        mean = mx.reshape(self.weights["spynet.mean"], (1, 1, 1, 3))
        std = mx.reshape(self.weights["spynet.std"], (1, 1, 1, 3))
        refs = [(reference - mean) / std]
        supps = [(support - mean) / std]
        pool = nn.AvgPool2d(2, stride=2)
        for _ in range(5):
            refs.append(pool(refs[-1]))
            supps.append(pool(supps[-1]))
        refs.reverse()
        supps.reverse()
        flow = mx.zeros((batch, height // 32, width // 32, 2), dtype=reference.dtype)
        resize = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        for level in range(6):
            if level:
                flow = resize(flow) * 2
            condition = mx.concatenate((refs[level], flow_warp(supps[level], flow, "border"), flow), -1)
            flow = flow + self.spynet_level(condition, level)
        return flow

    def offset(self, x, direction):
        for index in (0, 2, 4, 6):
            x = self.conv(x, f"deform_align.{direction}.conv_offset.{index}")
            if index != 6:
                x = self.lrelu(x)
        return x

    def deform_conv(self, x, offsets, mask, direction):
        """Reference MLX DCNv2, vectorized across pixels but not kernel taps."""
        batch, height, width, channels = x.shape
        weight = self.weights[f"deform_align.{direction}.weight"]  # OIHW
        bias = self.weights[f"deform_align.{direction}.bias"]
        out_channels, in_channels, kh, kw = weight.shape
        groups = 16
        taps = kh * kw
        if channels != in_channels or channels % groups:
            raise ValueError("unexpected deformable-convolution channels")
        if offsets.shape != (batch, height, width, 2 * groups * taps):
            raise ValueError("unexpected offset shape")
        if mask.shape != (batch, height, width, groups * taps):
            raise ValueError("unexpected mask shape")
        # A full video window contains many second-order warps. Materialize
        # their inputs before expanding 144 bilinear tap samples; otherwise
        # MLX retains the preceding frames' graphs and exhausts Metal's
        # resource-count limit on 30-frame windows.
        mx.eval(x, offsets, mask)
        patches = []
        for group in range(groups):
            group_x = x[..., group * (channels // groups):(group + 1) * (channels // groups)]
            samples = []
            for tap in range(taps):
                offset_index = (group * taps + tap) * 2
                oy = offsets[..., offset_index]
                ox = offsets[..., offset_index + 1]
                flow = mx.stack((ox + tap % kw - kw // 2,
                                 oy + tap // kw - kh // 2), axis=-1)
                samples.append(flow_warp(group_x, flow) * mask[..., group * taps + tap, None])
            patches.append(mx.reshape(mx.stack(samples, axis=-1),
                                      (batch, height, width, (channels // groups) * taps)))
            mx.eval(patches[-1])
        unfolded = mx.concatenate(patches, axis=-1)
        return mx.matmul(unfolded, mx.transpose(mx.reshape(weight, (out_channels, -1)))) + bias

    def align(self, x, condition, flow_1, flow_2, direction):
        """Second-order offset prediction and modulated deformable alignment."""
        raw = self.offset(mx.concatenate((condition, flow_1, flow_2), -1), direction)
        one, two, mask = mx.split(raw, 3, axis=-1)
        offsets = 10 * mx.tanh(mx.concatenate((one, two), -1))
        offset_1, offset_2 = mx.split(offsets, 2, axis=-1)
        repeats = offset_1.shape[-1] // 2
        offset_1 = offset_1 + mx.tile(flow_1[..., ::-1], (1, 1, 1, repeats))
        offset_2 = offset_2 + mx.tile(flow_2[..., ::-1], (1, 1, 1, repeats))
        return self.deform_conv(x, mx.concatenate((offset_1, offset_2), -1), mx.sigmoid(mask), direction)

    def backbone(self, x, direction):
        return self.residual_stack(x, "backbone." + direction)

    @staticmethod
    def pixel_shuffle_2(x):
        batch, height, width, channels = x.shape
        if channels % 4:
            raise ValueError("pixel shuffle requires channels divisible by four")
        x = mx.reshape(x, (batch, height, width, channels // 4, 2, 2))
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        return mx.reshape(x, (batch, height * 2, width * 2, channels // 4))

    def upsample(self, x):
        x = self.residual_stack(x, "reconstruction")
        for stage in (1, 2):
            x = self.pixel_shuffle_2(self.conv(x, f"upsample{stage}.upsample_conv"))
            x = self.lrelu(x)
        x = self.lrelu(self.conv(x, "conv_hr"))
        return self.conv(x, "conv_last")

    def restore_single_frame(self, frame):
        """Run the complete model's one-frame path (no temporal alignment)."""
        spatial = self.feature_extract(frame)
        features = [spatial]
        for direction in ("backward_1", "forward_1", "backward_2", "forward_2"):
            previous = features + [mx.zeros_like(spatial)]
            propagated = self.backbone(mx.concatenate(previous, -1), direction)
            features.append(propagated)
        return self.upsample(mx.concatenate(features, -1)) + frame

    def restore_frames(self, frames, flows_forward=None, flows_backward=None):
        """Experimental bidirectional BasicVSR++ inference for NHWC video frames."""
        if frames.ndim != 5 or frames.shape[-1] != 3 or frames.shape[1] < 1:
            raise ValueError("frames must be NTHWC RGB")
        batch, count, height, width, _ = frames.shape
        spatial = [self.feature_extract(frames[:, index]) for index in range(count)]
        mx.eval(*spatial)
        _, low_h, low_w, channels = spatial[0].shape
        if flows_forward is None or flows_backward is None:
            if min(height, width) < 256 or height % 128 or width % 128:
                raise ValueError("automatic flow requires frames >=256 and divisible by 128")
            downsample = nn.Upsample(scale_factor=0.25, mode="cubic", align_corners=False)
            low_frames = [downsample(frames[:, index]) for index in range(count)]
            flows_backward, flows_forward = [], []
            for i in range(count - 1):
                backward = self.spynet_flow(low_frames[i], low_frames[i + 1])
                forward = self.spynet_flow(low_frames[i + 1], low_frames[i])
                mx.eval(backward, forward)
                flows_backward.append(backward)
                flows_forward.append(forward)
        if len(flows_forward) != count - 1 or len(flows_backward) != count - 1:
            raise ValueError("one flow per neighboring frame pair is required")
        for flow in list(flows_forward) + list(flows_backward):
            if flow.shape != (batch, low_h, low_w, 2):
                raise ValueError("flow resolution must match extracted features")
        features = {"spatial": spatial}
        for direction in ("backward_1", "forward_1", "backward_2", "forward_2"):
            backward = direction.startswith("backward")
            traversal = list(range(count - 1, -1, -1)) if backward else list(range(count))
            flows = flows_backward if backward else flows_forward
            outputs = [None] * count
            propagated = mx.zeros_like(spatial[0])
            for position, index in enumerate(traversal):
                if position:
                    flow_index = index if backward else index - 1
                    flow_1 = flows[flow_index]
                    condition_1 = flow_warp(propagated, flow_1)
                    previous_2 = mx.zeros_like(propagated)
                    flow_2 = mx.zeros_like(flow_1)
                    condition_2 = mx.zeros_like(condition_1)
                    if position > 1:
                        previous_2 = outputs[traversal[position - 2]]
                        prior_index = traversal[position - 1] if backward else traversal[position - 2]
                        flow_2 = flow_1 + flow_warp(flows[prior_index], flow_1)
                        condition_2 = flow_warp(previous_2, flow_2)
                    condition = mx.concatenate((condition_1, spatial[index], condition_2), -1)
                    propagated = self.align(mx.concatenate((propagated, previous_2), -1),
                                            condition, flow_1, flow_2, direction)
                earlier = [features[name][index] for name in features if name != "spatial"]
                backbone_input = mx.concatenate([spatial[index]] + earlier + [propagated], -1)
                propagated = propagated + self.backbone(backbone_input, direction)
                mx.eval(propagated)
                outputs[index] = propagated
            features[direction] = outputs
        result = []
        for index in range(count):
            concat = mx.concatenate([features[name][index] for name in features], -1)
            result.append(self.upsample(concat) + frames[:, index])
            mx.eval(result[-1])
        return mx.stack(result, axis=1)
