enum MetalShader {
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

// Jasna's propagation body always uses batch 1, 64 output channels and one
// convolution group. One threadgroup owns one output pixel. Its two SIMD
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
    uint outputPlane = s.outputHeight * s.outputWidth;
    if (row >= s.batch * outputPlane) return;
    uint n = row / outputPlane;
    uint spatial = row % outputPlane;
    uint oy = spatial / s.outputWidth;
    uint ox = spatial % s.outputWidth;
    uint channelsPerOffsetGroup = 128 / s.offsetGroups;
    uint offsetBase = n * 2 * s.offsetGroups * 9 * outputPlane;
    uint inputPlane = s.inputHeight * s.inputWidth;
    for (uint sampleIndex = tid; sampleIndex < sampleCount; sampleIndex += threadsPerGroup) {
        uint ic = sampleIndex / 9;
        uint k = sampleIndex % 9;
        uint ky = k / 3;
        uint kx = k % 3;
        uint offsetGroup = ic / channelsPerOffsetGroup;
        uint offsetChannel = 2 * (offsetGroup * 9 + k);
        float offY = float(offset[offsetBase + offsetChannel * outputPlane + spatial]);
        float offX = float(offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]);
        float y = float(int(oy * s.strideHeight + ky * s.dilationHeight) - int(s.padHeight)) + offY;
        float x = float(int(ox * s.strideWidth + kx * s.dilationWidth) - int(s.padWidth)) + offX;
        float sampled = sample_fp16(
            input, (n * 128 + ic) * inputPlane,
            s.inputHeight, s.inputWidth, y, x
        );
        uint maskIndex = n * s.offsetGroups * 9 * outputPlane
            + (offsetGroup * 9 + k) * outputPlane + spatial;
        gathered[row * sampleCount + sampleIndex] = half(
            sampled * float(mask[maskIndex])
        );
    }
}

// Eight SIMD groups cooperatively produce an 8x64 output tile using the
// GPU's 8x8 matrix instructions. This form is usable from both Metal 3 and
// Metal 4 command encoders, unlike an MPSMatrixMultiplication object.
kernel void deform_conv2d_fp16_jasna_simdgroup_gemm(
    device const half *gathered [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *matrixOutput [[buffer(2)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint innerColumns = 128 * 9;
    constexpr uint outputColumns = 64;
    uint row = tile * 8;
    uint column = simdgroupIndex * 8;
    simdgroup_half8x8 matrixA;
    simdgroup_half8x8 matrixB;
    simdgroup_float8x8 matrixC(0.0f);
    for (uint k = 0; k < innerColumns; k += 8) {
        simdgroup_load(matrixA, gathered + row * innerColumns + k, innerColumns);
        simdgroup_load(matrixB, weight + k * outputColumns + column, outputColumns);
        simdgroup_multiply_accumulate(matrixC, matrixA, matrixB, matrixC);
    }
    simdgroup_store(matrixC, matrixOutput + row * outputColumns + column, outputColumns);
}

// Converts the row-major FP32 accumulator back to Jasna's NCHW FP16 tensor
// layout and applies the convolution bias.
kernel void deform_conv2d_fp16_jasna_float_gemm_output(
    device const float *matrixOutput [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant DeformConvShape &s [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outputPlane = s.outputHeight * s.outputWidth;
    uint total = s.batch * outputPlane * 64;
    if (gid >= total) return;
    uint oc = gid % 64;
    uint row = gid / 64;
    uint n = row / outputPlane;
    uint spatial = row % outputPlane;
    output[(n * 64 + oc) * outputPlane + spatial] = half(
        matrixOutput[gid] + float(bias[oc])
    );
}

struct SPyNetPrepareShape {
    uint width;
    uint height;
    uint sourceFlowWidth;
    uint sourceFlowHeight;
    uint firstLevel;
};

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
    uint2 gid [[thread_position_in_grid]]
) {
    uint level = gid.y;
    if (level >= 6u) return;
    uint size = 2u << level;
    uint outputPlane = size * size;
    if (gid.x >= 3u * outputPlane) return;
    uint channel = gid.x / outputPlane;
    uint spatial = gid.x % outputPlane;
    uint y = spatial / size;
    uint x = spatial % size;
    uint factor = 64u / size;
    float referenceSum = 0.0f;
    float supportSum = 0.0f;
    for (uint yy = 0; yy < factor; ++yy) {
        for (uint xx = 0; xx < factor; ++xx) {
            uint source = channel * 4096u + (y * factor + yy) * 64u + x * factor + xx;
            referenceSum += float(reference[source]);
            supportSum += float(support[source]);
        }
    }
    float inverseArea = 1.0f / float(factor * factor);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
    half referenceValue = half((referenceSum * inverseArea - mean[channel]) / stddev[channel]);
    half supportValue = half((supportSum * inverseArea - mean[channel]) / stddev[channel]);
    if (level == 0u) { reference2[gid.x] = referenceValue; support2[gid.x] = supportValue; }
    else if (level == 1u) { reference4[gid.x] = referenceValue; support4[gid.x] = supportValue; }
    else if (level == 2u) { reference8[gid.x] = referenceValue; support8[gid.x] = supportValue; }
    else if (level == 3u) { reference16[gid.x] = referenceValue; support16[gid.x] = supportValue; }
    else if (level == 4u) { reference32[gid.x] = referenceValue; support32[gid.x] = supportValue; }
    else { reference64[gid.x] = referenceValue; support64[gid.x] = supportValue; }
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
    if (gid >= plane) return;
    uint y = gid / s.width;
    uint x = gid % s.width;
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
        for (uint channel = 0u; channel < 2u; ++channel) {
            uint base = channel * sourceStoragePlane;
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
    uint destination = y * s.rowStride + x;
    baseFlow[destination] = half(flowX);
    baseFlow[storagePlane + destination] = half(flowY);
    for (uint channel = 0u; channel < 3u; ++channel) {
        features[channel * storagePlane + destination] = reference[channel * plane + gid];
        float warped = sample_border_fp16(
            support, channel * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        );
        features[(channel + 3u) * storagePlane + destination] = half(warped);
    }
    features[6u * storagePlane + destination] = half(flowX);
    features[7u * storagePlane + destination] = half(flowY);
}

kernel void spynet_add_flow_padded_fp16(
    device const half *baseFlow [[buffer(0)]],
    device const half *residual [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant SPyNetPaddedShape &s [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint plane = s.width * s.height;
    if (gid >= 2u * plane) return;
    uint channel = gid / plane;
    uint spatial = gid % plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;
    uint index = channel * s.rowStride * s.height + y * s.rowStride + x;
    output[index] = baseFlow[index] + residual[index];
}

struct TemporalPrepareShape {
    uint width;
    uint height;
    uint hasSecondOrder;
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
    if (gid >= 2u * plane) return;
    if (s.hasSecondOrder == 0u) {
        flow2[gid] = half(0.0f);
        return;
    }
    uint channel = gid / plane;
    uint spatial = gid % plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;
    float flowX = float(flow1[spatial]);
    float flowY = float(flow1[plane + spatial]);
    float previous = sample_fp16(
        previousFlow, channel * plane, s.height, s.width,
        float(y) + flowY, float(x) + flowX
    );
    flow2[gid] = half(float(flow1[gid]) + previous);
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
    uint channel = gid / plane;
    uint spatial = gid % plane;
    uint y = spatial / s.width;
    uint x = spatial % s.width;

    if (gid < 128u * plane) {
        deformInput[gid] = channel < 64u
            ? featProp[gid]
            : featN2[(channel - 64u) * plane + spatial];
    }
    if (gid >= 196u * plane) return;
    if (channel < 64u) {
        float flowX = float(flow1[spatial]);
        float flowY = float(flow1[plane + spatial]);
        conditions[gid] = half(sample_fp16(
            featProp, channel * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        ));
    } else if (channel < 128u) {
        conditions[gid] = featCurrent[(channel - 64u) * plane + spatial];
    } else if (channel < 192u) {
        float flowX = float(flow2[spatial]);
        float flowY = float(flow2[plane + spatial]);
        conditions[gid] = half(sample_fp16(
            featN2, (channel - 128u) * plane, s.height, s.width,
            float(y) + flowY, float(x) + flowX
        ));
    } else if (channel < 194u) {
        conditions[gid] = flow1[(channel - 192u) * plane + spatial];
    } else {
        conditions[gid] = flow2[(channel - 194u) * plane + spatial];
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
    uint channel = gid / plane;
    uint spatial = gid % plane;
    if (channel < 288) {
        uint localChannel = channel % 144;
        device const half *flow = channel < 144 ? flow1 : flow2;
        // flow is [x,y]; Jasna flip(1).repeat(...) produces [y,x,y,x,...].
        uint flowChannel = (localChannel & 1u) == 0u ? 1u : 0u;
        float residue = 10.0f * tanh(float(raw[gid]));
        offset[gid] = half(residue + float(flow[flowChannel * plane + spatial]));
    } else if (channel < 432) {
        float value = float(raw[gid]);
        mask[(channel - 288) * plane + spatial] = half(1.0f / (1.0f + exp(-value)));
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
    uint count = prefixCount + 64u * plane;
    if (gid >= count) return;
    output[gid] = gid < prefixCount ? prefix[gid] : aligned[gid - prefixCount];
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
    uint sourceIndex = gid / (64u * plane);
    uint localIndex = gid % (64u * plane);
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
    if (gid >= totalChannels * plane) return;
    uint channel = gid / plane;
    uint spatialIndex = gid % plane;
    if (channel >= prefixChannels) {
        output[gid] = aligned[(channel - prefixChannels) * plane + spatialIndex];
        return;
    }
    uint source = channel / 64u;
    uint localIndex = (channel % 64u) * plane + spatialIndex;
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
    if (gid >= totalChannels * plane) return;
    uint channel = gid / plane;
    uint spatialIndex = gid % plane;
    if (channel >= prefixChannels) {
        output[gid] = aligned[(channel - prefixChannels) * plane + spatialIndex];
        return;
    }
    uint source = channel / 64u;
    uint localIndex = (channel % 64u) * plane + spatialIndex;
    if (source == branchIndex) {
        half value = previousAligned[localIndex] + previousBackbone[localIndex];
        previousOutput[localIndex] = value;
        output[gid] = value;
    } else if (source == 0u) output[gid] = spatial[localIndex];
    else if (source == 1u) output[gid] = backward1[localIndex];
    else if (source == 2u) output[gid] = forward1[localIndex];
    else output[gid] = backward2[localIndex];
}
"""#
}
