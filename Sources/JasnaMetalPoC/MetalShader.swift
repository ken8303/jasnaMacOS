enum MetalShader {
static let fusedGEMMRowsPerTile = 32
static let source = #"""
#include <metal_stdlib>
using namespace metal;

struct DeformConvShape {
    uint batch;
    uint inputChannels;
    uint inputHeight;
    uint inputWidth;
    uint outputChannels;
    uint outputHeight;
    uint outputWidth;
    uint kernelHeight;
    uint kernelWidth;
    uint padHeight;
    uint padWidth;
    uint strideHeight;
    uint strideWidth;
    uint dilationHeight;
    uint dilationWidth;
    uint groups;
    uint offsetGroups;
    uint hasMask;
};

inline float sample_fp32(
    device const float *input,
    uint base,
    uint height,
    uint width,
    float y,
    float x
) {
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    float ly = y - float(y0);
    float lx = x - float(x0);
    float v00 = (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) ? input[base + uint(y0) * width + uint(x0)] : 0.0f;
    float v01 = (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) ? input[base + uint(y0) * width + uint(x1)] : 0.0f;
    float v10 = (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) ? input[base + uint(y1) * width + uint(x0)] : 0.0f;
    float v11 = (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) ? input[base + uint(y1) * width + uint(x1)] : 0.0f;
    return v00 * (1.0f - ly) * (1.0f - lx)
         + v01 * (1.0f - ly) * lx
         + v10 * ly * (1.0f - lx)
         + v11 * ly * lx;
}

inline float sample_fp16(
    device const half *input,
    uint base,
    uint height,
    uint width,
    float y,
    float x
) {
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    float ly = y - float(y0);
    float lx = x - float(x0);
    float v00 = (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) ? float(input[base + uint(y0) * width + uint(x0)]) : 0.0f;
    float v01 = (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) ? float(input[base + uint(y0) * width + uint(x1)]) : 0.0f;
    float v10 = (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) ? float(input[base + uint(y1) * width + uint(x0)]) : 0.0f;
    float v11 = (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) ? float(input[base + uint(y1) * width + uint(x1)]) : 0.0f;
    return v00 * (1.0f - ly) * (1.0f - lx)
         + v01 * (1.0f - ly) * lx
         + v10 * ly * (1.0f - lx)
         + v11 * ly * lx;
}

kernel void deform_conv2d_fp32(
    device const float *input [[buffer(0)]],
    device const float *offset [[buffer(1)]],
    device const float *mask [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device const float *bias [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant DeformConvShape &s [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outputCount = s.batch * s.outputChannels * s.outputHeight * s.outputWidth;
    if (gid >= outputCount) return;
    uint ox = gid % s.outputWidth;
    uint q = gid / s.outputWidth;
    uint oy = q % s.outputHeight;
    q /= s.outputHeight;
    uint oc = q % s.outputChannels;
    uint n = q / s.outputChannels;
    uint channelsPerGroup = s.inputChannels / s.groups;
    uint outputsPerGroup = s.outputChannels / s.groups;
    uint channelsPerOffsetGroup = s.inputChannels / s.offsetGroups;
    uint group = oc / outputsPerGroup;
    uint inputPlane = s.inputHeight * s.inputWidth;
    uint outputPlane = s.outputHeight * s.outputWidth;
    uint kernelArea = s.kernelHeight * s.kernelWidth;
    uint spatial = oy * s.outputWidth + ox;
    float sum = bias[oc];
    for (uint localIC = 0; localIC < channelsPerGroup; ++localIC) {
        uint ic = group * channelsPerGroup + localIC;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        for (uint ky = 0; ky < s.kernelHeight; ++ky) {
            for (uint kx = 0; kx < s.kernelWidth; ++kx) {
                uint k = ky * s.kernelWidth + kx;
                uint offsetChannel = 2 * (offsetGroup * kernelArea + k);
                uint offsetBase = n * 2 * s.offsetGroups * kernelArea * outputPlane;
                float offY = offset[offsetBase + offsetChannel * outputPlane + spatial];
                float offX = offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial];
                float y = float(int(oy * s.strideHeight + ky * s.dilationHeight) - int(s.padHeight)) + offY;
                float x = float(int(ox * s.strideWidth + kx * s.dilationWidth) - int(s.padWidth)) + offX;
                float sampled = sample_fp32(input, (n * s.inputChannels + ic) * inputPlane, s.inputHeight, s.inputWidth, y, x);
                uint maskIndex = n * s.offsetGroups * kernelArea * outputPlane
                    + (offsetGroup * kernelArea + k) * outputPlane + spatial;
                uint weightIndex = ((oc * channelsPerGroup + localIC) * s.kernelHeight + ky) * s.kernelWidth + kx;
                sum += sampled * mask[maskIndex] * weight[weightIndex];
            }
        }
    }
    output[gid] = sum;
}

kernel void deform_conv2d_fp16(
    device const half *input [[buffer(0)]],
    device const half *offset [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device const half *weight [[buffer(3)]],
    device const half *bias [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant DeformConvShape &s [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outputCount = s.batch * s.outputChannels * s.outputHeight * s.outputWidth;
    if (gid >= outputCount) return;
    uint ox = gid % s.outputWidth;
    uint q = gid / s.outputWidth;
    uint oy = q % s.outputHeight;
    q /= s.outputHeight;
    uint oc = q % s.outputChannels;
    uint n = q / s.outputChannels;
    uint channelsPerGroup = s.inputChannels / s.groups;
    uint outputsPerGroup = s.outputChannels / s.groups;
    uint channelsPerOffsetGroup = s.inputChannels / s.offsetGroups;
    uint group = oc / outputsPerGroup;
    uint inputPlane = s.inputHeight * s.inputWidth;
    uint outputPlane = s.outputHeight * s.outputWidth;
    uint kernelArea = s.kernelHeight * s.kernelWidth;
    uint spatial = oy * s.outputWidth + ox;
    float sum = float(bias[oc]);
    for (uint localIC = 0; localIC < channelsPerGroup; ++localIC) {
        uint ic = group * channelsPerGroup + localIC;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        for (uint ky = 0; ky < s.kernelHeight; ++ky) {
            for (uint kx = 0; kx < s.kernelWidth; ++kx) {
                uint k = ky * s.kernelWidth + kx;
                uint offsetChannel = 2 * (offsetGroup * kernelArea + k);
                uint offsetBase = n * 2 * s.offsetGroups * kernelArea * outputPlane;
                float offY = float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
                float offX = float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
                float y = float(int(oy * s.strideHeight + ky * s.dilationHeight) - int(s.padHeight)) + offY;
                float x = float(int(ox * s.strideWidth + kx * s.dilationWidth) - int(s.padWidth)) + offX;
                float sampled = sample_fp16(input, (n * s.inputChannels + ic) * inputPlane, s.inputHeight, s.inputWidth, y, x);
                uint maskIndex = n * s.offsetGroups * kernelArea * outputPlane
                    + (offsetGroup * kernelArea + k) * outputPlane + spatial;
                uint weightIndex = ((oc * channelsPerGroup + localIC) * s.kernelHeight + ky) * s.kernelWidth + kx;
                sum += sampled * float(mask[maskIndex]) * float(weight[weightIndex]);
            }
        }
    }
    output[gid] = half(sum);
}

// Jasna's propagation body uses 64 output channels and one convolution group.
// One threadgroup owns one output pixel in one batch plane. Its two SIMD
// groups calculate 32 output channels each while broadcasting the common
// bilinear sample across the SIMD lanes. This removes the largest source of
// redundant work in the general kernel.
kernel void deform_conv2d_fp16_jasna_simd(
    device const half *input [[buffer(0)]],
    device const half *offset [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device const half *weight [[buffer(3)]],
    device const half *bias [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant DeformConvShape &s [[buffer(6)]],
    uint spatialGroup [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (s.outputChannels != 64 || s.groups != 1) return;
    uint oc0 = lane;
    uint oc1 = lane + 32;
    uint outputPlane = s.outputHeight * s.outputWidth;
    uint n = spatialGroup / outputPlane;
    uint spatial = spatialGroup % outputPlane;
    if (n >= s.batch) return;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    uint channelsPerOffsetGroup = s.inputChannels / s.offsetGroups;
    uint inputPlane = s.inputHeight * s.inputWidth;
    uint kernelArea = s.kernelHeight * s.kernelWidth;
    float sum0 = float(bias[oc0]);
    float sum1 = float(bias[oc1]);

    for (uint ic = 0; ic < s.inputChannels; ++ic) {
        uint offsetGroup = ic / channelsPerOffsetGroup;
        for (uint ky = 0; ky < s.kernelHeight; ++ky) {
            for (uint kx = 0; kx < s.kernelWidth; ++kx) {
                uint k = ky * s.kernelWidth + kx;
                uint offsetChannel = 2 * (offsetGroup * kernelArea + k);
                uint offsetBase = n * 2 * s.offsetGroups * kernelArea * outputPlane;
                float common = 0.0f;
                if (simd_is_first()) {
                    float offY = float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
                    float offX = float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
                    float y = float(int(oy * s.strideHeight + ky * s.dilationHeight) - int(s.padHeight)) + offY;
                    float x = float(int(ox * s.strideWidth + kx * s.dilationWidth) - int(s.padWidth)) + offX;
                    float sampled = sample_fp16(input, (n * s.inputChannels + ic) * inputPlane, s.inputHeight, s.inputWidth, y, x);
                    uint maskIndex = n * s.offsetGroups * kernelArea * outputPlane
                        + (offsetGroup * kernelArea + k) * outputPlane + spatial;
                    common = sampled * float(mask[maskIndex]);
                }
                common = simd_broadcast_first(common);
                // Prepacked as [input_channel, kernel_element, output_channel]
                // so the 32 SIMD lanes read adjacent weights.
                uint weightIndex0 = (ic * kernelArea + k) * s.outputChannels + oc0;
                uint weightIndex1 = (ic * kernelArea + k) * s.outputChannels + oc1;
                sum0 += common * float(weight[weightIndex0]);
                sum1 += common * float(weight[weightIndex1]);
            }
        }
    }
    output[(n * s.outputChannels + oc0) * outputPlane + spatial] = half(sum0);
    output[(n * s.outputChannels + oc1) * outputPlane + spatial] = half(sum1);
}

// Cooperative version for Jasna's fixed 128×3×3 input tile. Threads first
// calculate the 1,152 sampled-and-masked values in parallel, then 64 threads
// accumulate one output channel each from fast threadgroup memory.
kernel void deform_conv2d_fp16_jasna_tiled(
    device const half *input [[buffer(0)]],
    device const half *offset [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device const half *weight [[buffer(3)]],
    device const half *bias [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant DeformConvShape &s [[buffer(6)]],
    uint spatialGroup [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    if (s.inputChannels != 128 || s.outputChannels != 64 ||
        s.kernelHeight != 3 || s.kernelWidth != 3 || s.groups != 1 ||
        s.offsetGroups == 0) return;
    constexpr uint sampleCount = 128 * 9;
    threadgroup half samples[sampleCount];
    uint outputPlane = s.outputHeight * s.outputWidth;
    uint n = spatialGroup / outputPlane;
    uint spatial = spatialGroup % outputPlane;
    if (n >= s.batch) return;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    uint inputPlane = s.inputHeight * s.inputWidth;
    uint channelsPerOffsetGroup = s.inputChannels / s.offsetGroups;

    for (uint sampleIndex = tid; sampleIndex < sampleCount; sampleIndex += threadsPerGroup) {
        uint ic = sampleIndex / 9;
        uint k = sampleIndex % 9;
        uint ky = k / 3;
        uint kx = k % 3;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        uint offsetChannel = 2 * (offsetGroup * 9 + k);
        uint offsetBase = n * 2 * s.offsetGroups * 9 * outputPlane;
        float offY = float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
        float offX = float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
        float y = float(int(oy * s.strideHeight + ky * s.dilationHeight) - int(s.padHeight)) + offY;
        float x = float(int(ox * s.strideWidth + kx * s.dilationWidth) - int(s.padWidth)) + offX;
        float sampled = sample_fp16(input, (n * 128 + ic) * inputPlane, s.inputHeight, s.inputWidth, y, x);
        uint maskIndex = n * s.offsetGroups * 9 * outputPlane
            + (offsetGroup * 9 + k) * outputPlane + spatial;
        samples[sampleIndex] = half(sampled * float(mask[maskIndex]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        float sum = float(bias[tid]);
        for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
            sum += float(samples[sampleIndex]) * float(weight[sampleIndex * 64 + tid]);
        }
        output[(n * 64 + tid) * outputPlane + spatial] = half(sum);
    }
}

// Materializes deformable im2col as a row-major [output pixel, 1,152]
// matrix. The following dense multiply uses the GPU's SIMD-group matrix
// instructions instead of performing 64 scalar reductions here.
kernel void deform_conv2d_fp16_jasna_gather(
    device const half *input [[buffer(0)]],
    device const half *offset [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device half *gathered [[buffer(3)]],
    constant DeformConvShape &s [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    constexpr uint sampleCount = 128 * 9;
    constexpr uint offsetSampleCount = 16 * 9;
    uint outputPlane = s.outputHeight * s.outputWidth;
    if (row >= s.batch * outputPlane) return;
    uint n = row / outputPlane;
    uint spatial = row % outputPlane;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    uint channelsPerOffsetGroup = 128 / s.offsetGroups;
    uint offsetBase = n * 2 * s.offsetGroups * 9 * outputPlane;
    uint inputPlane = s.inputHeight * s.inputWidth;
    threadgroup int4 sampleNeighbor[offsetSampleCount];
    threadgroup float2 sampleFraction[offsetSampleCount];
    threadgroup float sampleMask[offsetSampleCount];
    for (
        uint offsetSample = tid;
        offsetSample < offsetSampleCount;
        offsetSample += threadsPerGroup
    ) {
        uint k = offsetSample % 9;
        uint ky = k / 3;
        uint kx = k % 3;
        uint offsetChannel = 2 * offsetSample;
        float y = float(int(oy * s.strideHeight + ky * s.dilationHeight)
            - int(s.padHeight))
            + float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
        float x = float(int(ox * s.strideWidth + kx * s.dilationWidth)
            - int(s.padWidth))
            + float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
        int y0 = int(floor(y));
        int x0 = int(floor(x));
        int y1 = y0 + 1;
        int x1 = x0 + 1;
        sampleNeighbor[offsetSample] = int4(
            y0 >= 0 && y0 < int(s.inputHeight) && x0 >= 0 && x0 < int(s.inputWidth)
                ? y0 * int(s.inputWidth) + x0 : -1,
            y0 >= 0 && y0 < int(s.inputHeight) && x1 >= 0 && x1 < int(s.inputWidth)
                ? y0 * int(s.inputWidth) + x1 : -1,
            y1 >= 0 && y1 < int(s.inputHeight) && x0 >= 0 && x0 < int(s.inputWidth)
                ? y1 * int(s.inputWidth) + x0 : -1,
            y1 >= 0 && y1 < int(s.inputHeight) && x1 >= 0 && x1 < int(s.inputWidth)
                ? y1 * int(s.inputWidth) + x1 : -1
        );
        sampleFraction[offsetSample] = float2(y - float(y0), x - float(x0));
        sampleMask[offsetSample] = float(
            mask[n * s.offsetGroups * 9 * outputPlane
                + offsetSample * outputPlane + spatial]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sampleIndex = tid; sampleIndex < sampleCount; sampleIndex += threadsPerGroup) {
        uint ic = sampleIndex / 9;
        uint k = sampleIndex % 9;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        uint offsetSample = offsetGroup * 9 + k;
        uint inputBase = (n * 128 + ic) * inputPlane;
        int4 neighbor = sampleNeighbor[offsetSample];
        float2 fraction = sampleFraction[offsetSample];
        float v00 = neighbor.x >= 0 ? float(input[inputBase + uint(neighbor.x)]) : 0.0f;
        float v01 = neighbor.y >= 0 ? float(input[inputBase + uint(neighbor.y)]) : 0.0f;
        float v10 = neighbor.z >= 0 ? float(input[inputBase + uint(neighbor.z)]) : 0.0f;
        float v11 = neighbor.w >= 0 ? float(input[inputBase + uint(neighbor.w)]) : 0.0f;
        float ly = fraction.x;
        float lx = fraction.y;
        float sampled = v00 * (1.0f - ly) * (1.0f - lx)
            + v01 * (1.0f - ly) * lx
            + v10 * ly * (1.0f - lx)
            + v11 * ly * lx;
        gathered[row * sampleCount + sampleIndex] = half(
            sampled * sampleMask[offsetSample]
        );
    }
}

// NCHW-to-NHWC staging for the deformable-convolution input.
// The padded tile keeps both sides of the 16x16 transpose coalesced.
kernel void transpose_dcn_input_channel_last_fp16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant DeformConvShape &s [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    threadgroup half tile[16][17];
    constexpr uint channels = 128;
    uint plane = s.inputHeight * s.inputWidth;
    uint sourceSpatial = gid.x;
    uint sourceBatchChannel = gid.y;
    uint n = sourceBatchChannel / channels;
    uint channel = sourceBatchChannel % channels;
    if (sourceSpatial < plane && n < s.batch) {
        tile[lid.y][lid.x] = input[(n * channels + channel) * plane + sourceSpatial];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outputSpatial = group.x * 16 + lid.y;
    uint outputChannel = group.y * 16 + lid.x;
    uint outputBatch = group.y / (channels / 16);
    outputChannel %= channels;
    if (outputSpatial < plane && outputChannel < channels && outputBatch < s.batch) {
        output[(outputBatch * plane + outputSpatial) * channels + outputChannel]
            = tile[lid.x][lid.y];
    }
}

// Channel-last gather. Lanes traverse channels for a fixed kernel
// position, turning each offset group's eight channel reads into contiguous
// memory accesses. The gathered matrix retains the production column order.
kernel void deform_conv2d_fp16_jasna_gather_channel_last(
    device const half *input [[buffer(0)]],
    device const half *offset [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device half *gathered [[buffer(3)]],
    constant DeformConvShape &s [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    constexpr uint channels = 128;
    constexpr uint kernelArea = 9;
    constexpr uint sampleCount = channels * kernelArea;
    constexpr uint offsetSampleCount = 16 * kernelArea;
    uint outputPlane = s.outputHeight * s.outputWidth;
    if (row >= s.batch * outputPlane) return;
    uint n = row / outputPlane;
    uint spatial = row % outputPlane;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    uint channelsPerOffsetGroup = channels / s.offsetGroups;
    uint offsetBase = n * 2 * s.offsetGroups * kernelArea * outputPlane;
    uint inputPlane = s.inputHeight * s.inputWidth;
    threadgroup int4 sampleNeighbor[offsetSampleCount];
    threadgroup float2 sampleFraction[offsetSampleCount];
    threadgroup float sampleMask[offsetSampleCount];
    for (
        uint offsetSample = tid;
        offsetSample < offsetSampleCount;
        offsetSample += threadsPerGroup
    ) {
        uint k = offsetSample % kernelArea;
        uint ky = k / 3;
        uint kx = k % 3;
        uint offsetChannel = 2 * offsetSample;
        float y = float(int(oy * s.strideHeight + ky * s.dilationHeight)
            - int(s.padHeight))
            + float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
        float x = float(int(ox * s.strideWidth + kx * s.dilationWidth)
            - int(s.padWidth))
            + float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
        int y0 = int(floor(y));
        int x0 = int(floor(x));
        int y1 = y0 + 1;
        int x1 = x0 + 1;
        sampleNeighbor[offsetSample] = int4(
            y0 >= 0 && y0 < int(s.inputHeight) && x0 >= 0 && x0 < int(s.inputWidth)
                ? y0 * int(s.inputWidth) + x0 : -1,
            y0 >= 0 && y0 < int(s.inputHeight) && x1 >= 0 && x1 < int(s.inputWidth)
                ? y0 * int(s.inputWidth) + x1 : -1,
            y1 >= 0 && y1 < int(s.inputHeight) && x0 >= 0 && x0 < int(s.inputWidth)
                ? y1 * int(s.inputWidth) + x0 : -1,
            y1 >= 0 && y1 < int(s.inputHeight) && x1 >= 0 && x1 < int(s.inputWidth)
                ? y1 * int(s.inputWidth) + x1 : -1
        );
        sampleFraction[offsetSample] = float2(y - float(y0), x - float(x0));
        sampleMask[offsetSample] = float(
            mask[n * s.offsetGroups * kernelArea * outputPlane
                + offsetSample * outputPlane + spatial]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = tid; linear < sampleCount; linear += threadsPerGroup) {
        uint k = linear / channels;
        uint ic = linear % channels;
        uint sampleIndex = ic * kernelArea + k;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        uint offsetSample = offsetGroup * kernelArea + k;
        int4 neighbor = sampleNeighbor[offsetSample];
        float2 fraction = sampleFraction[offsetSample];
        float v00 = neighbor.x >= 0
            ? float(input[(n * inputPlane + uint(neighbor.x)) * channels + ic]) : 0.0f;
        float v01 = neighbor.y >= 0
            ? float(input[(n * inputPlane + uint(neighbor.y)) * channels + ic]) : 0.0f;
        float v10 = neighbor.z >= 0
            ? float(input[(n * inputPlane + uint(neighbor.z)) * channels + ic]) : 0.0f;
        float v11 = neighbor.w >= 0
            ? float(input[(n * inputPlane + uint(neighbor.w)) * channels + ic]) : 0.0f;
        float ly = fraction.x;
        float lx = fraction.y;
        float sampled = v00 * (1.0f - ly) * (1.0f - lx)
            + v01 * (1.0f - ly) * lx
            + v10 * ly * (1.0f - lx)
            + v11 * ly * lx;
        gathered[row * sampleCount + sampleIndex] = half(
            sampled * sampleMask[offsetSample]
        );
    }
}

// Diagnostic-only summary of the learned sampling field. Each invocation
// writes one temporal step's eight counters, avoiding retention/readback of
// the full offset tensor. Positive float bit patterns preserve ordering for
// the atomic maximum.
kernel void summarize_dcn_offset_locality_fp16(
    device const half *offset [[buffer(0)]],
    device atomic_uint *statistics [[buffer(1)]],
    constant DeformConvShape &s [[buffer(2)]],
    constant uint &step [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    constexpr uint kernelArea = 9u;
    constexpr uint offsetSamples = 16u * kernelArea;
    constexpr float fixedScale = 64.0f;
    uint plane = s.outputHeight * s.outputWidth;
    uint count = s.batch * offsetSamples * plane;
    if (gid >= count) return;
    uint spatial = gid % plane;
    uint q = gid / plane;
    uint offsetSample = q % offsetSamples;
    uint n = q / offsetSamples;
    uint offsetChannel = 2u * offsetSample;
    uint offsetBase = n * 2u * offsetSamples * plane;
    float offY = float(offset[offsetBase + offsetChannel * plane + spatial]);
    float offX = float(offset[offsetBase + (offsetChannel + 1u) * plane + spatial]);
    float magnitude = max(abs(offY), abs(offX));
    uint k = offsetSample % kernelArea;
    uint ky = k / 3u;
    uint kx = k % 3u;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    float y = float(int(oy * s.strideHeight + ky * s.dilationHeight)
        - int(s.padHeight)) + offY;
    float x = float(int(ox * s.strideWidth + kx * s.dilationWidth)
        - int(s.padWidth)) + offX;
    float neighborDelta = 0.0f;
    uint neighborCount = 0u;
    if (ox > 0u) {
        uint prior = spatial - 1u;
        neighborDelta += abs(offY - float(offset[offsetBase + offsetChannel * plane + prior]));
        neighborDelta += abs(offX - float(offset[offsetBase + (offsetChannel + 1u) * plane + prior]));
        neighborCount += 2u;
    }
    if (oy > 0u) {
        uint prior = spatial - s.outputWidth;
        neighborDelta += abs(offY - float(offset[offsetBase + offsetChannel * plane + prior]));
        neighborDelta += abs(offX - float(offset[offsetBase + (offsetChannel + 1u) * plane + prior]));
        neighborCount += 2u;
    }
    if (neighborCount > 0u) neighborDelta /= float(neighborCount);

    device atomic_uint *result = statistics + step * 8u;
    atomic_fetch_add_explicit(&result[0], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &result[1], uint(min(magnitude, 32.0f) * fixedScale), memory_order_relaxed
    );
    atomic_fetch_max_explicit(
        &result[2], as_type<uint>(magnitude), memory_order_relaxed
    );
    if (magnitude > 2.0f) atomic_fetch_add_explicit(&result[3], 1u, memory_order_relaxed);
    if (magnitude > 4.0f) atomic_fetch_add_explicit(&result[4], 1u, memory_order_relaxed);
    if (magnitude > 8.0f) atomic_fetch_add_explicit(&result[5], 1u, memory_order_relaxed);
    if (y < 0.0f || x < 0.0f || y > float(s.inputHeight - 1u)
        || x > float(s.inputWidth - 1u)) {
        atomic_fetch_add_explicit(&result[6], 1u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(
        &result[7], uint(min(neighborDelta, 32.0f) * fixedScale), memory_order_relaxed
    );
}

// Eight SIMD groups cooperatively produce a 32x64 output tile using the GPU's
// 8x8 matrix instructions. Each weight tile feeds four row blocks before being
// discarded. The accumulators are staged in threadgroup memory so
// this dispatch can add bias, convert to FP16 and scatter directly to NCHW.
kernel void deform_conv2d_fp16_jasna_simdgroup_gemm_fused(
    device const half *gathered [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant DeformConvShape &s [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    constexpr uint innerColumns = 128 * 9;
    constexpr uint outputColumns = 64;
    constexpr uint rowsPerTile = \#(fusedGEMMRowsPerTile);
    uint row = tile * rowsPerTile;
    uint column = simdgroupIndex * 8;
    simdgroup_half8x8 matrixA0;
    simdgroup_half8x8 matrixA1;
    simdgroup_half8x8 matrixA2;
    simdgroup_half8x8 matrixA3;
    simdgroup_half8x8 matrixB;
    simdgroup_float8x8 matrixC0(0.0f);
    simdgroup_float8x8 matrixC1(0.0f);
    simdgroup_float8x8 matrixC2(0.0f);
    simdgroup_float8x8 matrixC3(0.0f);
    for (uint k = 0; k < innerColumns; k += 8) {
        simdgroup_load(matrixA0, gathered + row * innerColumns + k, innerColumns);
        simdgroup_load(matrixA1, gathered + (row + 8) * innerColumns + k, innerColumns);
        simdgroup_load(matrixA2, gathered + (row + 16) * innerColumns + k, innerColumns);
        simdgroup_load(matrixA3, gathered + (row + 24) * innerColumns + k, innerColumns);
        simdgroup_load(matrixB, weight + k * outputColumns + column, outputColumns);
        simdgroup_multiply_accumulate(matrixC0, matrixA0, matrixB, matrixC0);
        simdgroup_multiply_accumulate(matrixC1, matrixA1, matrixB, matrixC1);
        simdgroup_multiply_accumulate(matrixC2, matrixA2, matrixB, matrixC2);
        simdgroup_multiply_accumulate(matrixC3, matrixA3, matrixB, matrixC3);
    }
    threadgroup float tileOutput[rowsPerTile * outputColumns];
    simdgroup_store(
        matrixC0, tileOutput + column, outputColumns
    );
    simdgroup_store(
        matrixC1, tileOutput + 8 * outputColumns + column, outputColumns
    );
    simdgroup_store(
        matrixC2, tileOutput + 16 * outputColumns + column, outputColumns
    );
    simdgroup_store(
        matrixC3, tileOutput + 24 * outputColumns + column, outputColumns
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outputPlane = s.outputHeight * s.outputWidth;
    uint totalRows = s.batch * outputPlane;
    for (uint index = tid; index < rowsPerTile * outputColumns; index += threadsPerGroup) {
        uint localRow = index / outputColumns;
        uint outputChannel = index % outputColumns;
        uint outputRow = row + localRow;
        if (outputRow < totalRows) {
            uint n = outputRow / outputPlane;
            uint spatial = outputRow % outputPlane;
            output[(n * outputColumns + outputChannel) * outputPlane + spatial] = half(
                tileOutput[index] + float(bias[outputChannel])
            );
        }
    }
}

kernel void flag_non_finite_fp16(
    device const half *values [[buffer(0)]],
    device atomic_uint *flag [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count && !isfinite(float(values[gid]))) {
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
    }
}

struct SPyNetPrepareShape {
    uint width;
    uint height;
    uint sourceFlowWidth;
    uint sourceFlowHeight;
    uint firstLevel;
};

inline float cubic_weight(float distance) {
    constexpr float a = -0.75f;
    float x = abs(distance);
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// Matches torch.nn.functional.interpolate(scale_factor=0.25, mode="bicubic")
// for Jasna's fixed 256×256 NCHW input. The half-pixel source positions are
// 1.5, 5.5, ... 253.5, so every 4×4 footprint is inside the source image.
kernel void jasna_bicubic_downsample_quarter_fp16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    constexpr uint inputSize = 256u;
    constexpr uint outputSize = 64u;
    constexpr uint inputPlane = inputSize * inputSize;
    constexpr uint outputPlane = outputSize * outputSize;
    constexpr uint outputBatchElements = 3u * outputPlane;
    uint n = gid / outputBatchElements;
    uint local = gid % outputBatchElements;
    uint channel = local / outputPlane;
    uint spatial = local % outputPlane;
    uint outputY = spatial / outputSize;
    uint outputX = spatial % outputSize;
    float sourceY = (float(outputY) + 0.5f) * 4.0f - 0.5f;
    float sourceX = (float(outputX) + 0.5f) * 4.0f - 0.5f;
    int baseY = int(floor(sourceY));
    int baseX = int(floor(sourceX));
    float value = 0.0f;
    for (int yy = -1; yy <= 2; ++yy) {
        float wy = cubic_weight(sourceY - float(baseY + yy));
        for (int xx = -1; xx <= 2; ++xx) {
            float wx = cubic_weight(sourceX - float(baseX + xx));
            uint source = n * 3u * inputPlane + channel * inputPlane
                + uint(baseY + yy) * inputSize + uint(baseX + xx);
            value += float(input[source]) * wy * wx;
        }
    }
    output[gid] = half(value);
}

// Builds normalized 2/4/8/16/32/64 pyramids for a frame pair directly from
// 64×64 RGB inputs. Averaging and normalization are linear, so direct box
// averages are equivalent to the repeated 2×2 average-pool pyramid apart from
// intermediate FP16 rounding.
kernel void spynet_build_pyramid_pair_fp16(
    device const half *reference [[buffer(0)]],
    device const half *support [[buffer(1)]],
    device half *reference2 [[buffer(2)]],
    device half *support2 [[buffer(3)]],
    device half *reference4 [[buffer(4)]],
    device half *support4 [[buffer(5)]],
    device half *reference8 [[buffer(6)]],
    device half *support8 [[buffer(7)]],
    device half *reference16 [[buffer(8)]],
    device half *support16 [[buffer(9)]],
    device half *reference32 [[buffer(10)]],
    device half *support32 [[buffer(11)]],
    device half *reference64 [[buffer(12)]],
    device half *support64 [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint level = gid.y;
    if (level >= 6u) return;
    uint size = 2u << level;
    uint outputPlane = size * size;
    if (gid.x >= 3u * outputPlane) return;
    uint n = gid.z;
    uint channel = gid.x / outputPlane;
    uint spatial = gid.x % outputPlane;
    uint y = spatial / size;
    uint x = spatial % size;
    uint factor = 64u / size;
    float referenceSum = 0.0f;
    float supportSum = 0.0f;
    for (uint yy = 0; yy < factor; ++yy) {
        for (uint xx = 0; xx < factor; ++xx) {
            uint source = n * 3u * 4096u + channel * 4096u
                + (y * factor + yy) * 64u + x * factor + xx;
            referenceSum += float(reference[source]);
            supportSum += float(support[source]);
        }
    }
    float inverseArea = 1.0f / float(factor * factor);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    half referenceValue = half((referenceSum * inverseArea - mean[channel]) / stddev[channel]);
    half supportValue = half((supportSum * inverseArea - mean[channel]) / stddev[channel]);
    uint destination = n * 3u * outputPlane + gid.x;
    if (level == 0u) { reference2[destination] = referenceValue; support2[destination] = supportValue; }
    else if (level == 1u) { reference4[destination] = referenceValue; support4[destination] = supportValue; }
    else if (level == 2u) { reference8[destination] = referenceValue; support8[destination] = supportValue; }
    else if (level == 3u) { reference16[destination] = referenceValue; support16[destination] = supportValue; }
    else if (level == 4u) { reference32[destination] = referenceValue; support32[destination] = supportValue; }
    else { reference64[destination] = referenceValue; support64[destination] = supportValue; }
}

inline float sample_border_fp16(
    device const half *input,
    uint base,
    uint height,
    uint width,
    float y,
    float x
) {
    y = clamp(y, 0.0f, float(height - 1));
    x = clamp(x, 0.0f, float(width - 1));
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    int y1 = min(y0 + 1, int(height - 1));
    int x1 = min(x0 + 1, int(width - 1));
    float ly = y - float(y0);
    float lx = x - float(x0);
    float v00 = float(input[base + uint(y0) * width + uint(x0)]);
    float v01 = float(input[base + uint(y0) * width + uint(x1)]);
    float v10 = float(input[base + uint(y1) * width + uint(x0)]);
    float v11 = float(input[base + uint(y1) * width + uint(x1)]);
    return v00 * (1.0f - ly) * (1.0f - lx)
         + v01 * (1.0f - ly) * lx
         + v10 * ly * (1.0f - lx)
         + v11 * ly * lx;
}

// Builds SPyNet's 8-channel block input [reference, warped support, flow].
// For levels after the first it also upsamples the previous flow using
// align_corners bilinear interpolation and multiplies it by two.
kernel void spynet_prepare_fp16(
    device const half *reference [[buffer(0)]],
    device const half *support [[buffer(1)]],
    device const half *sourceFlow [[buffer(2)]],
    device half *features [[buffer(3)]],
    device half *baseFlow [[buffer(4)]],
    constant SPyNetPrepareShape &s [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    if (gid >= plane) return;
    uint y = gid / s.width;
    uint x = gid % s.width;
    float flowX = 0.0f;
    float flowY = 0.0f;
    if (s.firstLevel == 0) {
        float sourceX = s.width > 1 ? float(x) * float(s.sourceFlowWidth - 1) / float(s.width - 1) : 0.0f;
        float sourceY = s.height > 1 ? float(y) * float(s.sourceFlowHeight - 1) / float(s.height - 1) : 0.0f;
        uint sourcePlane = s.sourceFlowWidth * s.sourceFlowHeight;
        flowX = 2.0f * sample_border_fp16(sourceFlow, 0, s.sourceFlowHeight, s.sourceFlowWidth, sourceY, sourceX);
        flowY = 2.0f * sample_border_fp16(sourceFlow, sourcePlane, s.sourceFlowHeight, s.sourceFlowWidth, sourceY, sourceX);
    }
    baseFlow[gid] = half(flowX);
    baseFlow[plane + gid] = half(flowY);
    for (uint channel = 0; channel < 3; ++channel) {
        features[channel * plane + gid] = reference[channel * plane + gid];
        float warped = sample_border_fp16(
            support,
            channel * plane,
            s.height,
            s.width,
            float(y) + flowY,
            float(x) + flowX
        );
        features[(channel + 3) * plane + gid] = half(warped);
    }
    features[6 * plane + gid] = half(flowX);
    features[7 * plane + gid] = half(flowY);
}

kernel void spynet_add_flow_fp16(
    device const half *baseFlow [[buffer(0)]],
    device const half *residual [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) output[gid] = baseFlow[gid] + residual[gid];
}

struct SPyNetPaddedShape {
    uint width;
    uint height;
    uint rowStride;
    uint sourceFlowWidth;
    uint sourceFlowHeight;
    uint sourceFlowRowStride;
    uint firstLevel;
    uint batch;
};

kernel void spynet_prepare_padded_fp16(
    device const half *reference [[buffer(0)]],
    device const half *support [[buffer(1)]],
    device const half *sourceFlow [[buffer(2)]],
    device half *features [[buffer(3)]],
    device half *baseFlow [[buffer(4)]],
    constant SPyNetPaddedShape &s [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    if (gid >= s.batch * plane) return;
    uint n = gid / plane;
    uint spatial = gid % plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;
    float flowX = 0.0f;
    float flowY = 0.0f;
    if (s.firstLevel == 0u) {
        float sourceX = s.width > 1u ? float(x) * float(s.sourceFlowWidth - 1u) / float(s.width - 1u) : 0.0f;
        float sourceY = s.height > 1u ? float(y) * float(s.sourceFlowHeight - 1u) / float(s.height - 1u) : 0.0f;
        int x0 = int(floor(sourceX));
        int y0 = int(floor(sourceY));
        int x1 = min(x0 + 1, int(s.sourceFlowWidth - 1u));
        int y1 = min(y0 + 1, int(s.sourceFlowHeight - 1u));
        float lx = sourceX - float(x0);
        float ly = sourceY - float(y0);
        uint sourceStoragePlane = s.sourceFlowRowStride * s.sourceFlowHeight;
        uint sourceBatchBase = n * 2u * sourceStoragePlane;
        for (uint channel = 0u; channel < 2u; ++channel) {
            uint base = sourceBatchBase + channel * sourceStoragePlane;
            float v00 = float(sourceFlow[base + uint(y0) * s.sourceFlowRowStride + uint(x0)]);
            float v01 = float(sourceFlow[base + uint(y0) * s.sourceFlowRowStride + uint(x1)]);
            float v10 = float(sourceFlow[base + uint(y1) * s.sourceFlowRowStride + uint(x0)]);
            float v11 = float(sourceFlow[base + uint(y1) * s.sourceFlowRowStride + uint(x1)]);
            float value = 2.0f * (v00 * (1.0f - ly) * (1.0f - lx)
                + v01 * (1.0f - ly) * lx + v10 * ly * (1.0f - lx) + v11 * ly * lx);
            if (channel == 0u) flowX = value; else flowY = value;
        }
    }
    uint storagePlane = s.rowStride * s.height;
    uint featureBatchBase = n * 8u * storagePlane;
    uint flowBatchBase = n * 2u * storagePlane;
    uint sourceBatchBase = n * 3u * plane;
    uint destination = y * s.rowStride + x;
    baseFlow[flowBatchBase + destination] = half(flowX);
    baseFlow[flowBatchBase + storagePlane + destination] = half(flowY);
    for (uint channel = 0u; channel < 3u; ++channel) {
        features[featureBatchBase + channel * storagePlane + destination]
            = reference[sourceBatchBase + channel * plane + spatial];
        float warped = sample_border_fp16(
            support, sourceBatchBase + channel * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        );
        features[featureBatchBase + (channel + 3u) * storagePlane + destination]
            = half(warped);
    }
    features[featureBatchBase + 6u * storagePlane + destination] = half(flowX);
    features[featureBatchBase + 7u * storagePlane + destination] = half(flowY);
}

kernel void spynet_add_flow_padded_fp16(
    device const half *baseFlow [[buffer(0)]],
    device const half *residual [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant SPyNetPaddedShape &s [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    uint batchElements = 2u * plane;
    if (gid >= s.batch * batchElements) return;
    uint n = gid / batchElements;
    uint local = gid % batchElements;
    uint channel = local / plane;
    uint spatial = local % plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;
    uint storagePlane = s.rowStride * s.height;
    uint index = n * 2u * storagePlane + channel * storagePlane
        + y * s.rowStride + x;
    output[index] = baseFlow[index] + residual[index];
}

struct TemporalPrepareShape {
    uint width;
    uint height;
    uint hasSecondOrder;
    uint batch;
};

// BasicVSR++ composes the previous link with the current first-order flow:
// flow_n2 = flow_n1 + warp(previous_flow, flow_n1). The warp uses PyTorch
// grid_sample's bilinear/zero-padding behavior with align_corners enabled.
kernel void accumulate_second_order_flow_fp16(
    device const half *flow1 [[buffer(0)]],
    device const half *previousFlow [[buffer(1)]],
    device half *flow2 [[buffer(2)]],
    constant TemporalPrepareShape &s [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    uint batchPlane = 2u * plane;
    if (gid >= s.batch * batchPlane) return;
    uint n = gid / batchPlane;
    uint local = gid % batchPlane;
    if (s.hasSecondOrder == 0u) {
        flow2[gid] = half(0.0f);
        return;
    }
    uint channel = local / plane;
    uint spatial = local % plane;
    uint flowBase = n * batchPlane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;
    float flowX = float(flow1[flowBase + spatial]);
    float flowY = float(flow1[flowBase + plane + spatial]);
    float previous = sample_fp16(
        previousFlow, flowBase + channel * plane, s.height, s.width,
        float(y) + flowY, float(x) + flowX
    );
    flow2[gid] = half(float(flow1[flowBase + local]) + previous);
}

// Materializes the exact inputs consumed by Jasna's split offset/DCNv2 path:
// conditions = [warp(feat_prop, flow1), feat_current,
//               warp(feat_n2, flow2), flow1, flow2]
// deformInput = [feat_prop, feat_n2].
kernel void assemble_temporal_alignment_fp16(
    device const half *featProp [[buffer(0)]],
    device const half *featCurrent [[buffer(1)]],
    device const half *featN2 [[buffer(2)]],
    device const half *flow1 [[buffer(3)]],
    device const half *flow2 [[buffer(4)]],
    device half *conditions [[buffer(5)]],
    device half *deformInput [[buffer(6)]],
    constant TemporalPrepareShape &s [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    uint conditionPlane = 196u * plane;
    if (gid >= s.batch * conditionPlane) return;
    uint n = gid / conditionPlane;
    uint local = gid % conditionPlane;
    uint channel = local / plane;
    uint spatial = local % plane;
    uint featureBase = n * 64u * plane;
    uint flowBase = n * 2u * plane;
    uint deformBase = n * 128u * plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;

    if (local < 128u * plane) {
        deformInput[deformBase + local] = channel < 64u
            ? featProp[featureBase + local]
            : featN2[featureBase + (channel - 64u) * plane + spatial];
    }
    if (channel < 64u) {
        float flowX = float(flow1[flowBase + spatial]);
        float flowY = float(flow1[flowBase + plane + spatial]);
        conditions[gid] = half(sample_fp16(
            featProp, featureBase + channel * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        ));
    } else if (channel < 128u) {
        conditions[gid] = featCurrent[featureBase + (channel - 64u) * plane + spatial];
    } else if (channel < 192u) {
        float flowX = float(flow2[flowBase + spatial]);
        float flowY = float(flow2[flowBase + plane + spatial]);
        conditions[gid] = half(sample_fp16(
            featN2, featureBase + (channel - 128u) * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        ));
    } else if (channel < 194u) {
        conditions[gid] = flow1[flowBase + (channel - 192u) * plane + spatial];
    } else {
        conditions[gid] = flow2[flowBase + (channel - 194u) * plane + spatial];
    }
}

// Converts conv_offset's [o1, o2, mask] output into TorchVision's interleaved
// (y,x) offsets plus sigmoid mask, including Jasna's first/second-order flows.
kernel void prepare_dcn_offsets_fp16(
    device const half *raw [[buffer(0)]],
    device const half *flow1 [[buffer(1)]],
    device const half *flow2 [[buffer(2)]],
    device half *offset [[buffer(3)]],
    device half *mask [[buffer(4)]],
    constant uint &plane [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint flattenedChannel = gid / plane;
    uint n = flattenedChannel / 432u;
    uint channel = flattenedChannel % 432u;
    uint spatial = gid % plane;
    uint flowBase = n * 2u * plane;
    if (channel < 288) {
        uint localChannel = channel % 144;
        device const half *flow = channel < 144 ? flow1 + flowBase : flow2 + flowBase;
        // flow is [x,y]; Jasna flip(1).repeat(...) produces [y,x,y,x,...].
        uint flowChannel = (localChannel & 1u) == 0u ? 1u : 0u;
        float residue = 10.0f * tanh(float(raw[gid]));
        uint destination = n * 288u * plane + channel * plane + spatial;
        offset[destination] = half(
            residue + float(flow[flowChannel * plane + spatial])
        );
    } else if (channel < 432) {
        float value = float(raw[gid]);
        uint destination = n * 144u * plane + (channel - 288u) * plane + spatial;
        mask[destination] = half(1.0f / (1.0f + exp(-value)));
    }
}

// Forms the first propagation backbone input [current spatial feature,
// deformably aligned feature]. Both inputs use planar C×H×W storage.
kernel void assemble_propagation_backbone_fp16(
    device const half *prefix [[buffer(0)]],
    device const half *aligned [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &plane [[buffer(3)]],
    constant uint &prefixChannels [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint prefixCount = prefixChannels * plane;
    uint channels = prefixChannels + 64u;
    uint batchCount = channels * plane;
    uint n = gid / batchCount;
    uint local = gid % batchCount;
    output[gid] = local < prefixCount
        ? prefix[n * prefixCount + local]
        : aligned[n * 64u * plane + local - prefixCount];
}

// BasicVSR++ keeps deformable alignment as a residual around each propagation
// backbone.
kernel void add_propagation_residual_fp16(
    device const half *aligned [[buffer(0)]],
    device const half *backbone [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) output[gid] = aligned[gid] + backbone[gid];
}

// Reconstruction consumes [spatial, backward_1, forward_1, backward_2,
// forward_2], with 64 planar channels from each source.
kernel void assemble_reconstruction_fp16(
    device const half *spatial [[buffer(0)]],
    device const half *backward1 [[buffer(1)]],
    device const half *forward1 [[buffer(2)]],
    device const half *backward2 [[buffer(3)]],
    device const half *forward2 [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &plane [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint batchCount = 320u * plane;
    uint n = gid / batchCount;
    uint local = gid % batchCount;
    uint sourceIndex = local / (64u * plane);
    uint localIndex = n * 64u * plane + local % (64u * plane);
    if (sourceIndex == 0u) output[gid] = spatial[localIndex];
    else if (sourceIndex == 1u) output[gid] = backward1[localIndex];
    else if (sourceIndex == 2u) output[gid] = forward1[localIndex];
    else if (sourceIndex == 3u) output[gid] = backward2[localIndex];
    else if (sourceIndex == 4u) output[gid] = forward2[localIndex];
}

kernel void add_frame_residual_fp16(
    device const half *predicted [[buffer(0)]],
    device const half *inputFrame [[buffer(1)]],
    device half *restored [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) restored[gid] = predicted[gid] + inputFrame[gid];
}

// Builds each branch backbone input without flattening its previously-produced
// propagation tensors on the CPU. branchIndex 0...3 selects progressively
// [spatial], [spatial,b1], [spatial,b1,f1], [spatial,b1,f1,b2].
kernel void assemble_temporal_backbone_fp16(
    device const half *spatial [[buffer(0)]],
    device const half *backward1 [[buffer(1)]],
    device const half *forward1 [[buffer(2)]],
    device const half *backward2 [[buffer(3)]],
    device const half *aligned [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &plane [[buffer(6)]],
    constant uint &branchIndex [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint prefixChannels = 64u * (branchIndex + 1u);
    uint totalChannels = prefixChannels + 64u;
    uint batchCount = totalChannels * plane;
    uint n = gid / batchCount;
    uint local = gid % batchCount;
    uint channel = local / plane;
    uint spatialIndex = local % plane;
    if (channel >= prefixChannels) {
        output[gid] = aligned[n * 64u * plane + (channel - prefixChannels) * plane + spatialIndex];
        return;
    }
    uint source = channel / 64u;
    uint localIndex = n * 64u * plane + (channel % 64u) * plane + spatialIndex;
    if (source == 0u) output[gid] = spatial[localIndex];
    else if (source == 1u) output[gid] = backward1[localIndex];
    else if (source == 2u) output[gid] = forward1[localIndex];
    else output[gid] = backward2[localIndex];
}

// For branches 1...3, materialize the immediately preceding propagation
// residual while copying it into the next backbone input. Older branch outputs
// have already been materialized by an earlier fused assembly.
kernel void assemble_temporal_backbone_fused_fp16(
    device const half *spatial [[buffer(0)]],
    device const half *backward1 [[buffer(1)]],
    device const half *forward1 [[buffer(2)]],
    device const half *backward2 [[buffer(3)]],
    device const half *previousAligned [[buffer(4)]],
    device const half *previousBackbone [[buffer(5)]],
    device half *previousOutput [[buffer(6)]],
    device const half *aligned [[buffer(7)]],
    device half *output [[buffer(8)]],
    constant uint &plane [[buffer(9)]],
    constant uint &branchIndex [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint prefixChannels = 64u * (branchIndex + 1u);
    uint totalChannels = prefixChannels + 64u;
    uint batchCount = totalChannels * plane;
    uint n = gid / batchCount;
    uint local = gid % batchCount;
    uint channel = local / plane;
    uint spatialIndex = local % plane;
    if (channel >= prefixChannels) {
        output[gid] = aligned[n * 64u * plane + (channel - prefixChannels) * plane + spatialIndex];
        return;
    }
    uint source = channel / 64u;
    uint localIndex = n * 64u * plane + (channel % 64u) * plane + spatialIndex;
    if (source == branchIndex) {
        half value = previousAligned[localIndex] + previousBackbone[localIndex];
        previousOutput[localIndex] = value;
        output[gid] = value;
    } else if (source == 0u) output[gid] = spatial[localIndex];
    else if (source == 1u) output[gid] = backward1[localIndex];
    else if (source == 2u) output[gid] = forward1[localIndex];
    else output[gid] = backward2[localIndex];
}

struct MosaicCompositeParams {
    uint frameWidth;
    uint regionX;
    uint regionY;
    uint regionWidth;
    uint regionHeight;
    uint modelSize;
    uint maskWidth;
    uint maskHeight;
    uint groupX;
    uint groupY;
    uint groupWidth;
    uint coverageMode;
    float detailResidualLimit;
    float maskRecoveryDeltaThreshold;
};

struct MosaicGroupResolveParams {
    uint groupX;
    uint groupY;
    uint groupWidth;
    uint groupHeight;
    float detailResidualLimit;
};

inline float mosaic_sample_plane(
    device const half *values, uint offset, uint size, float x, float y
) {
    float clampedX = clamp(x, 0.0f, float(size - 1u));
    float clampedY = clamp(y, 0.0f, float(size - 1u));
    uint x0 = uint(floor(clampedX));
    uint y0 = uint(floor(clampedY));
    uint x1 = min(x0 + 1u, size - 1u);
    uint y1 = min(y0 + 1u, size - 1u);
    float fx = clampedX - float(x0);
    float fy = clampedY - float(y0);
    float top = mix(
        float(values[offset + y0 * size + x0]),
        float(values[offset + y0 * size + x1]), fx
    );
    float bottom = mix(
        float(values[offset + y1 * size + x0]),
        float(values[offset + y1 * size + x1]), fx
    );
    return mix(top, bottom, fy);
}

inline float mosaic_sample_mask(
    device const uchar *mask,
    constant MosaicCompositeParams &params,
    uint localX,
    uint localY
) {
    float maskX = float(localX) * float(params.maskWidth - 1u)
        / float(max(params.regionWidth - 1u, 1u));
    float maskY = float(localY) * float(params.maskHeight - 1u)
        / float(max(params.regionHeight - 1u, 1u));
    uint x0 = uint(floor(maskX));
    uint y0 = uint(floor(maskY));
    uint x1 = min(x0 + 1u, params.maskWidth - 1u);
    uint y1 = min(y0 + 1u, params.maskHeight - 1u);
    float fx = maskX - float(x0);
    float fy = maskY - float(y0);
    float top = mix(float(mask[y0 * params.maskWidth + x0]),
                    float(mask[y0 * params.maskWidth + x1]), fx);
    float bottom = mix(float(mask[y1 * params.maskWidth + x0]),
                       float(mask[y1 * params.maskWidth + x1]), fx);
    return mix(top, bottom, fy) / 255.0f;
}

kernel void composite_fisheye_mosaic_delta(
    device uchar *bgra [[buffer(0)]],
    device const half *restored [[buffer(1)]],
    device const half *original [[buffer(2)]],
    device const float4 *compositeSamples [[buffer(3)]],
    device const uchar *mask [[buffer(4)]],
    constant MosaicCompositeParams &params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint regionPixels = params.regionWidth * params.regionHeight;
    if (gid >= regionPixels) return;
    uint pixelX = params.regionX + gid % params.regionWidth;
    uint pixelY = params.regionY + gid / params.regionWidth;

    float4 compositeSample = compositeSamples[gid];
    float maskAlpha = mosaic_sample_mask(
        mask, params, gid % params.regionWidth, gid / params.regionWidth
    );
    float modelX = compositeSample.x;
    float modelY = compositeSample.y;

    uint plane = params.modelSize * params.modelSize;
    float3 restoredColor;
    float3 originalColor;
    for (uint rgbChannel = 0u; rgbChannel < 3u; ++rgbChannel) {
        uint offset = rgbChannel * plane;
        restoredColor[rgbChannel] = mosaic_sample_plane(
            restored, offset, params.modelSize, modelX, modelY
        );
        originalColor[rgbChannel] = mosaic_sample_plane(
            original, offset, params.modelSize, modelX, modelY
        );
    }
    if (params.coverageMode == 3u) {
        float3 delta = restoredColor - originalColor;
        float deltaStrength = max(abs(delta.x), max(abs(delta.y), abs(delta.z)));
        float threshold = params.maskRecoveryDeltaThreshold;
        float recoveredMask = smoothstep(threshold, threshold * 2.5f, deltaStrength);
        maskAlpha = max(maskAlpha, recoveredMask);
    }
    float alpha = compositeSample.z * maskAlpha;
    if (alpha <= 0.0f) return;
    uint destination = 4u * (pixelY * params.frameWidth + pixelX);
    for (uint bgraChannel = 0u; bgraChannel < 3u; ++bgraChannel) {
        uint rgbChannel = 2u - bgraChannel;
        float base = float(bgra[destination + bgraChannel]) / 255.0f;
        float detail = clamp(
            base - originalColor[rgbChannel],
            -params.detailResidualLimit,
            params.detailResidualLimit
        );
        float value = restoredColor[rgbChannel] + detail;
        float restoredByte = clamp(value, 0.0f, 1.0f) * 255.0f;
        float blended = float(bgra[destination + bgraChannel]) * (1.0f - alpha)
            + restoredByte * alpha;
        bgra[destination + bgraChannel] = uchar(clamp(floor(blended + 0.5f), 0.0f, 255.0f));
    }
}

kernel void composite_fisheye_mosaic_delta_texture(
    texture2d<float, access::read_write> frame [[texture(0)]],
    device const half *restored [[buffer(0)]],
    device const half *original [[buffer(1)]],
    device const float4 *compositeSamples [[buffer(2)]],
    device const uchar *mask [[buffer(3)]],
    constant MosaicCompositeParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint regionPixels = params.regionWidth * params.regionHeight;
    if (gid >= regionPixels) return;
    uint2 position = uint2(
        params.regionX + gid % params.regionWidth,
        params.regionY + gid / params.regionWidth
    );
    float4 compositeSample = compositeSamples[gid];
    float maskAlpha = mosaic_sample_mask(
        mask, params, gid % params.regionWidth, gid / params.regionWidth
    );

    float4 color = frame.read(position);
    uint plane = params.modelSize * params.modelSize;
    float3 restoredColor;
    float3 originalColor;
    for (uint rgbChannel = 0u; rgbChannel < 3u; ++rgbChannel) {
        uint offset = rgbChannel * plane;
        restoredColor[rgbChannel] = mosaic_sample_plane(
            restored, offset, params.modelSize, compositeSample.x, compositeSample.y
        );
        originalColor[rgbChannel] = mosaic_sample_plane(
            original, offset, params.modelSize, compositeSample.x, compositeSample.y
        );
    }
    if (params.coverageMode == 3u) {
        float3 delta = restoredColor - originalColor;
        float deltaStrength = max(abs(delta.x), max(abs(delta.y), abs(delta.z)));
        float threshold = params.maskRecoveryDeltaThreshold;
        float recoveredMask = smoothstep(threshold, threshold * 2.5f, deltaStrength);
        maskAlpha = max(maskAlpha, recoveredMask);
    }
    float alpha = compositeSample.z * maskAlpha;
    if (alpha <= 0.0f) return;
    for (uint rgbChannel = 0u; rgbChannel < 3u; ++rgbChannel) {
        float detail = clamp(
            color[rgbChannel] - originalColor[rgbChannel],
            -params.detailResidualLimit,
            params.detailResidualLimit
        );
        float value = restoredColor[rgbChannel] + detail;
        color[rgbChannel] = mix(color[rgbChannel], clamp(value, 0.0f, 1.0f), alpha);
    }
    frame.write(color, position);
}

kernel void clear_fisheye_mosaic_group(
    device float4 *accumulator [[buffer(0)]],
    device float *coverage [[buffer(1)]],
    device half4 *restoredAccumulator [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    accumulator[gid] = float4(0.0f);
    coverage[gid] = 0.0f;
    restoredAccumulator[gid] = half4(half(0.0f));
}

kernel void accumulate_fisheye_mosaic_delta(
    device float4 *accumulator [[buffer(0)]],
    device const half *restored [[buffer(1)]],
    device const half *original [[buffer(2)]],
    device const float4 *compositeSamples [[buffer(3)]],
    device const uchar *mask [[buffer(4)]],
    constant MosaicCompositeParams &params [[buffer(5)]],
    device float *coverage [[buffer(6)]],
    device half4 *restoredAccumulator [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint regionPixels = params.regionWidth * params.regionHeight;
    if (gid >= regionPixels) return;
    uint localX = gid % params.regionWidth;
    uint localY = gid / params.regionWidth;
    uint pixelX = params.regionX + localX;
    uint pixelY = params.regionY + localY;
    float4 compositeSample = compositeSamples[gid];
    float maskAlpha = mosaic_sample_mask(mask, params, localX, localY);
    float alpha = compositeSample.z * (params.coverageMode == 2u ? 1.0f : maskAlpha);
    if (alpha <= 0.0f && params.coverageMode != 3u) return;
    uint plane = params.modelSize * params.modelSize;
    float3 delta;
    float3 restoredColor;
    for (uint rgbChannel = 0u; rgbChannel < 3u; ++rgbChannel) {
        uint offset = rgbChannel * plane;
        restoredColor[rgbChannel] = mosaic_sample_plane(
            restored, offset, params.modelSize, compositeSample.x, compositeSample.y
        );
        delta[rgbChannel] = restoredColor[rgbChannel] - mosaic_sample_plane(
            original, offset, params.modelSize, compositeSample.x, compositeSample.y
        );
    }
    if (params.coverageMode == 3u) {
        float deltaStrength = max(abs(delta.x), max(abs(delta.y), abs(delta.z)));
        float threshold = params.maskRecoveryDeltaThreshold;
        float recoveredMask = smoothstep(threshold, threshold * 2.5f, deltaStrength);
        alpha = compositeSample.z * max(maskAlpha, recoveredMask);
    }
    if (alpha <= 0.0f) return;
    uint destination = (pixelY - params.groupY) * params.groupWidth
        + pixelX - params.groupX;
    accumulator[destination] += float4(delta * alpha, alpha);
    restoredAccumulator[destination] += half4(half3(restoredColor * alpha), half(alpha));
    if (params.coverageMode == 1u || params.coverageMode == 3u) {
        coverage[destination] += alpha;
    } else if (params.coverageMode == 2u) {
        // A high-detail crop may fill a hole left by the moving primary mask,
        // but must not stack opacity and reveal its rectangular footprint.
        // Its own deep spatial feather bounds this contribution.
        coverage[destination] = max(coverage[destination], alpha);
    }
}

kernel void resolve_fisheye_mosaic_delta_group_texture(
    texture2d<float, access::read_write> frame [[texture(0)]],
    device const float4 *accumulator [[buffer(0)]],
    device const float *coverage [[buffer(1)]],
    device const half4 *restoredAccumulator [[buffer(2)]],
    constant MosaicGroupResolveParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint groupPixels = params.groupWidth * params.groupHeight;
    if (gid >= groupPixels) return;
    float4 accumulated = accumulator[gid];
    if (accumulated.w <= 0.0f) return;
    uint2 position = uint2(
        params.groupX + gid % params.groupWidth,
        params.groupY + gid / params.groupWidth
    );
    float4 color = frame.read(position);
    // Detail crops refine the normalized delta and may fill primary-mask holes
    // using maximum (rather than additive) feathered coverage.
    float visibleAlpha = min(coverage[gid], 1.0f);
    float3 averageDelta = accumulated.rgb / accumulated.w;
    float3 averageRestored = float3(restoredAccumulator[gid].rgb)
        / max(float(restoredAccumulator[gid].w), 0.0001f);
    float3 averageOriginal = averageRestored - averageDelta;
    float3 detail = clamp(
        color.rgb - averageOriginal,
        -params.detailResidualLimit,
        params.detailResidualLimit
    );
    color.rgb = mix(
        color.rgb,
        clamp(averageRestored + detail, 0.0f, 1.0f),
        visibleAlpha
    );
    frame.write(color, position);
}
"""#
}
