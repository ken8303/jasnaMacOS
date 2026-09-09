import Foundation
import Metal

final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    fileprivate(set) var submissionCount = 0

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw LabError.invalid("Metal device or queue unavailable; the GPU comparison was not run")
        }
        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sample_generated_pattern(
            texture2d<float, access::sample> source [[texture(0)]],
            device const float2 *coordinates [[buffer(0)]],
            device float4 *output [[buffer(1)]],
            constant uint &count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
            float2 size = float2(source.get_width(), source.get_height());
            output[index] = source.sample(linearSampler, (coordinates[index] + 0.5f) / size);
        }
        """, options: options)
        guard let function = library.makeFunction(name: "sample_generated_pattern") else {
            throw LabError.invalid("Synthetic sampling kernel missing")
        }
        pipeline = try device.makeComputePipelineState(function: function)
    }
}

struct MetalOutput {
    let pixels: [SIMD4<Float>]
    let gpuMS: Double?
}

struct MetalGroupOutput {
    let frames: [[SIMD4<Float>]]
    let gpuMS: Double?
    var hostTiming: MetalHostTiming? = nil
}

// This diagnostic is single-threaded. Uploads and executions never overlap.
final class MetalSampler {
    let context: MetalContext
    let texture: MTLTexture
    let coordinateBuffer: MTLBuffer
    let outputBuffer: MTLBuffer
    let count: Int
    private var hasUploaded = false
    private(set) var uploadCount = 0
    var reusableGPUResourceBytes: Int { texture.allocatedSize + coordinateBuffer.allocatedSize + outputBuffer.allocatedSize }

    init(context: MetalContext, width: Int, height: Int, count: Int) throws {
        guard width > 0 && height > 0 && count > 0 && count <= Int(UInt32.max) else {
            throw LabError.invalid("Invalid synthetic dimensions")
        }
        self.context = context
        self.count = count
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = context.device.makeTexture(descriptor: descriptor),
              let coordinates = context.device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared),
              let output = context.device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
        else { throw LabError.invalid("Synthetic GPU allocation failed") }
        self.texture = texture
        coordinateBuffer = coordinates
        outputBuffer = output
    }

    func upload(_ raster: Raster, coordinates: [SIMD2<Float>]) throws {
        // An invalid replacement must not make stale input usable by accident.
        hasUploaded = false
        guard raster.width == texture.width && raster.height == texture.height,
              raster.bytes.count == raster.width * raster.height * 4, coordinates.count == count,
              coordinates.allSatisfy({ $0.x.isFinite && $0.y.isFinite })
        else { throw LabError.invalid("Synthetic input shape or coordinates invalid") }
        raster.bytes.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, raster.width, raster.height), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: raster.width * 4)
        }
        coordinates.withUnsafeBytes {
            coordinateBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        uploadCount += 1
        hasUploaded = true
    }

    // End-to-end caller includes upload, submission/wait, and output-array copy.
    func sample(_ raster: Raster, coordinates: [SIMD2<Float>]) throws -> MetalOutput {
        try upload(raster, coordinates: coordinates)
        return try sampleResident()
    }

    func sampleResident() throws -> MetalOutput {
        let result = try Self.sampleResidentGroup([self])
        return MetalOutput(pixels: result.frames[0], gpuMS: result.gpuMS)
    }

    // Independent inputs and outputs, one submission, all output arrays copied.
    // No overwritten/discarded intermediate outputs are counted as useful work.
    static func sampleResidentGroup(_ samplers: [MetalSampler], profileHost: Bool = false) throws -> MetalGroupOutput {
        guard let first = samplers.first, samplers.count <= 4,
              samplers.allSatisfy({ $0.context === first.context && $0.hasUploaded }),
              Set(samplers.map { ObjectIdentifier($0) }).count == samplers.count
        else { throw LabError.invalid("Resident group requires 1–4 distinct, uploaded inputs from one context") }
        let context = first.context
        return try autoreleasepool {
            let executionStart = profileHost ? DispatchTime.now().uptimeNanoseconds : nil
            guard let command = context.queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw LabError.invalid("Synthetic command allocation failed")
            }
            encoder.setComputePipelineState(context.pipeline)
            for sampler in samplers { sampler.encode(into: encoder) }
            encoder.endEncoding()
            context.submissionCount += 1
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw LabError.invalid("Synthetic GPU command failed: \(String(describing: command.error))")
            }
            let executionMS = executionStart.map { elapsedMS(since: $0) }
            let copyStart = profileHost ? DispatchTime.now().uptimeNanoseconds : nil
            let frames = samplers.map { sampler in
                Array(UnsafeBufferPointer(
                    start: sampler.outputBuffer.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: sampler.count
                ))
            }
            let copyMS = copyStart.map { elapsedMS(since: $0) }
            let duration = (command.gpuEndTime - command.gpuStartTime) * 1_000
            let validTiming = command.gpuStartTime > 0 && duration.isFinite && duration > 0
            return MetalGroupOutput(frames: frames, gpuMS: validTiming ? duration : nil,
                                    hostTiming: profileHost ? MetalHostTiming(encodeSubmitWaitMS: executionMS!, outputCopyMS: copyMS!) : nil)
        }
    }

    private func encode(into encoder: MTLComputeCommandEncoder) {
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(coordinateBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var outputCount = UInt32(count)
        encoder.setBytes(&outputCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, context.pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
    }
}
