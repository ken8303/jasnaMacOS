import CoreVideo
import Foundation
import Metal

struct MetalMosaicCompositeInput {
    let region: MosaicRegion
    let restored: [Float16]
    let original: [Float16]
    let samples: [MosaicCompositeSample]
}

private func mosaicCompositeInputsByVisiblePriority(
    _ inputs: [MetalMosaicCompositeInput]
) -> [MetalMosaicCompositeInput] {
    // Composite small regions first. If an unusual or older manifest contains
    // nested crops, the broader restoration wins instead of leaving a sharp
    // inner rectangle where the smaller crop overwrote it.
    inputs.sorted {
        let leftArea = $0.region.effectiveBlendWidth * $0.region.effectiveBlendHeight
        let rightArea = $1.region.effectiveBlendWidth * $1.region.effectiveBlendHeight
        return leftArea < rightArea
    }
}

private struct MetalMosaicCompositeParams {
    var frameWidth: UInt32
    var regionX: UInt32
    var regionY: UInt32
    var regionWidth: UInt32
    var regionHeight: UInt32
    var modelSize: UInt32
    var maskWidth: UInt32
    var maskHeight: UInt32
    var groupX: UInt32
    var groupY: UInt32
    var groupWidth: UInt32
    /// 0: no coverage, 1: additive masked coverage, 2: feathered detail maximum,
    /// 3: additive coverage with model-delta mask-hole recovery.
    var coverageMode: UInt32
    var detailResidualLimit: Float
    var maskRecoveryDeltaThreshold: Float
}

private struct MetalMosaicGroupResolveParams {
    var groupX: UInt32
    var groupY: UInt32
    var groupWidth: UInt32
    var groupHeight: UInt32
    var detailResidualLimit: Float
}

@available(macOS 27.0, *)
final class MetalMosaicCompositor: @unchecked Sendable {
    private struct ReusableBufferKey: Hashable {
        let length: Int
        let options: UInt
    }

    private struct ReusableBuffer {
        let key: ReusableBufferKey
        let buffer: MTLBuffer
    }

    private struct CachedSampleBuffer {
        let buffer: MTLBuffer
        var lastUse: UInt64
    }

    private struct SampleBufferKey: Hashable {
        let frameWidth: Int
        let frameHeight: Int
        let regionX: Int
        let regionY: Int
        let regionWidth: Int
        let regionHeight: Int
        let blendX: Int
        let blendY: Int
        let blendWidth: Int
        let blendHeight: Int
        let detailBlendFeather: Int?
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let texturePipeline: MTLComputePipelineState
    private let groupClearPipeline: MTLComputePipelineState
    private let groupAccumulatePipeline: MTLComputePipelineState
    private let groupResolvePipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private let detailResidualLimit: Float
    private let maskRecoveryDeltaThreshold: Float
    private let ordinaryMaskRecoveryEnabled: Bool
    private let reusableBufferLimitBytes: Int
    private let sampleBufferLimitBytes: Int
    let prefersTextureSurfaces: Bool
    private let textureCacheLock = NSLock()
    private let reusableBufferLock = NSLock()
    private let sampleBufferLock = NSLock()
    private let commandBufferLock = NSLock()
    private var pendingCommandBuffer: MTLCommandBuffer?
    private var reusableBuffers = [ReusableBufferKey: [MTLBuffer]]()
    private var reusableBufferBytes = 0
    private var sampleBuffers = [SampleBufferKey: CachedSampleBuffer]()
    private var sampleBufferBytes = 0
    private var sampleBufferUse: UInt64 = 0

    var memoryBudgetDescription: String {
        "reusable buffers \(reusableBufferLimitBytes / 1_048_576) MiB; "
            + "sampling maps \(sampleBufferLimitBytes / 1_048_576) MiB"
    }

    init(
        device: MTLDevice,
        ordinaryMaskRecoveryEnabled override: Bool? = nil
    ) throws {
        self.device = device
        detailResidualLimit = MosaicCompositeQuality.detailResidualLimit()
        maskRecoveryDeltaThreshold = MosaicCompositeQuality.maskRecoveryDeltaThreshold()
        ordinaryMaskRecoveryEnabled = override
            ?? MosaicCompositeQuality.ordinaryMaskRecoveryEnabled()
        let environment = ProcessInfo.processInfo.environment
        let configuredReusableMiB = Int(
            environment["JASNA_METAL_COMPOSITOR_BUFFER_POOL_MB"] ?? ""
        ) ?? 64
        reusableBufferLimitBytes = min(256, max(0, configuredReusableMiB)) * 1_048_576
        let configuredSampleMiB = Int(
            environment["JASNA_METAL_COMPOSITOR_SAMPLE_CACHE_MB"] ?? ""
        ) ?? 128
        sampleBufferLimitBytes = min(512, max(0, configuredSampleMiB)) * 1_048_576
        prefersTextureSurfaces = ProcessInfo.processInfo.environment[
            "JASNA_METAL_TEXTURE_COMPOSITOR"
        ] != "0"
        let library = try MetalResourceCache.shared.shaderLibrary(device: device) {
            try device.makeLibrary(source: MetalShader.source, options: nil)
        }
        guard let function = library.makeFunction(name: "composite_fisheye_mosaic_delta"),
              let textureFunction = library.makeFunction(
                  name: "composite_fisheye_mosaic_delta_texture"
              ),
              let groupClearFunction = library.makeFunction(
                  name: "clear_fisheye_mosaic_group"
              ),
              let groupAccumulateFunction = library.makeFunction(
                  name: "accumulate_fisheye_mosaic_delta"
              ),
              let groupResolveFunction = library.makeFunction(
                  name: "resolve_fisheye_mosaic_delta_group_texture"
              ),
              let queue = device.makeCommandQueue()
        else { throw DeformConvError.metalUnavailable }
        pipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: function
        )
        texturePipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: textureFunction
        )
        groupClearPipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: groupClearFunction
        )
        groupAccumulatePipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: groupAccumulateFunction
        )
        groupResolvePipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: groupResolveFunction
        )
        var optionalTextureCache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(
            nil, nil, device, nil, &optionalTextureCache
        ) == kCVReturnSuccess, let optionalTextureCache else {
            throw DeformConvError.metalUnavailable
        }
        textureCache = optionalTextureCache
        self.queue = queue
    }

    func composite(
        basePixelBuffer: CVPixelBuffer,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput],
        waitUntilCompleted: Bool = true
    ) throws {
        guard CVPixelBufferGetPixelFormatType(basePixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(outputPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(basePixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(basePixelBuffer) == dimensions.height,
              CVPixelBufferGetWidth(outputPixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(outputPixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }
        let requiresGroupedComposite = inputs.contains {
            $0.region.subdivisionGroup != nil
        }
        if prefersTextureSurfaces {
            if try compositeUsingTextures(
                basePixelBuffer: basePixelBuffer,
                outputPixelBuffer: outputPixelBuffer,
                dimensions: dimensions,
                inputs: inputs,
                waitUntilCompleted: waitUntilCompleted
            ) {
                return
            }
        }
        if requiresGroupedComposite {
            throw DeformConvError.commandFailed(
                "subdivided mosaic regions require normalized Metal texture compositing"
            )
        }
        try compositeUsingBufferCopies(
            basePixelBuffer: basePixelBuffer,
            outputPixelBuffer: outputPixelBuffer,
            dimensions: dimensions,
            inputs: inputs
        )
    }

    func compositeInPlace(
        pixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput],
        waitUntilCompleted: Bool = true
    ) throws {
        guard prefersTextureSurfaces,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(pixelBuffer) == dimensions.height,
              try compositeUsingTextures(
                  basePixelBuffer: nil,
                  outputPixelBuffer: pixelBuffer,
                  dimensions: dimensions,
                  inputs: inputs,
                  waitUntilCompleted: waitUntilCompleted
              )
        else {
            throw DeformConvError.commandFailed(
                "in-place mosaic compositing requires a Metal texture surface"
            )
        }
    }

    func copyStereo(
        leftPixelBuffer: CVPixelBuffer,
        rightPixelBuffer: CVPixelBuffer,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        waitUntilCompleted: Bool = true
    ) throws {
        let eyeDimensions = VideoDimensions(
            width: dimensions.width / 2,
            height: dimensions.height
        )
        guard dimensions.width.isMultiple(of: 2),
              CVPixelBufferGetPixelFormatType(leftPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(rightPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(outputPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(leftPixelBuffer) == eyeDimensions.width,
              CVPixelBufferGetHeight(leftPixelBuffer) == eyeDimensions.height,
              CVPixelBufferGetWidth(rightPixelBuffer) == eyeDimensions.width,
              CVPixelBufferGetHeight(rightPixelBuffer) == eyeDimensions.height,
              CVPixelBufferGetWidth(outputPixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(outputPixelBuffer) == dimensions.height,
              let left = makeTexture(
                  pixelBuffer: leftPixelBuffer, dimensions: eyeDimensions
              ),
              let right = makeTexture(
                  pixelBuffer: rightPixelBuffer, dimensions: eyeDimensions
              ),
              let destination = makeTexture(
                  pixelBuffer: outputPixelBuffer, dimensions: dimensions
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeBlitCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        let eyeSize = MTLSize(
            width: eyeDimensions.width, height: eyeDimensions.height, depth: 1
        )
        encoder.copy(
            from: left.texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: .init(x: 0, y: 0, z: 0),
            sourceSize: eyeSize,
            to: destination.texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: .init(x: 0, y: 0, z: 0)
        )
        encoder.copy(
            from: right.texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: .init(x: 0, y: 0, z: 0),
            sourceSize: eyeSize,
            to: destination.texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: .init(x: eyeDimensions.width, y: 0, z: 0)
        )
        encoder.endEncoding()
        try finishCommandBuffer(commandBuffer, waitUntilCompleted: waitUntilCompleted)
        withExtendedLifetime((left.reference, right.reference, destination.reference)) {}
    }

    func compositeStereo(
        leftPixelBuffer: CVPixelBuffer,
        rightPixelBuffer: CVPixelBuffer,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput],
        waitUntilCompleted: Bool = true
    ) throws {
        let eyeDimensions = VideoDimensions(
            width: dimensions.width / 2, height: dimensions.height
        )
        guard prefersTextureSurfaces,
              dimensions.width.isMultiple(of: 2),
              CVPixelBufferGetPixelFormatType(leftPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(rightPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(outputPixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(leftPixelBuffer) == eyeDimensions.width,
              CVPixelBufferGetHeight(leftPixelBuffer) == eyeDimensions.height,
              CVPixelBufferGetWidth(rightPixelBuffer) == eyeDimensions.width,
              CVPixelBufferGetHeight(rightPixelBuffer) == eyeDimensions.height,
              CVPixelBufferGetWidth(outputPixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(outputPixelBuffer) == dimensions.height,
              try compositeUsingTextures(
                  basePixelBuffer: nil,
                  stereoBasePixelBuffers: (leftPixelBuffer, rightPixelBuffer),
                  outputPixelBuffer: outputPixelBuffer,
                  dimensions: dimensions,
                  inputs: inputs,
                  waitUntilCompleted: waitUntilCompleted
              )
        else {
            throw DeformConvError.commandFailed(
                "fused stereo mosaic compositing requires Metal texture surfaces"
            )
        }
    }

    private func compositeUsingBufferCopies(
        basePixelBuffer: CVPixelBuffer,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput]
    ) throws {
        let packedRowBytes = dimensions.width * 4
        var reusableBuffers = [ReusableBuffer]()
        defer { recycleBuffers(reusableBuffers) }
        let frameReusable = try checkoutBuffer(
            length: packedRowBytes * dimensions.height, options: .storageModeShared
        )
        reusableBuffers.append(frameReusable)
        let frameBuffer = frameReusable.buffer

        CVPixelBufferLockBaseAddress(basePixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(basePixelBuffer, .readOnly) }
        guard let source = CVPixelBufferGetBaseAddress(basePixelBuffer) else {
            throw DeformConvError.commandFailed("base pixel buffer has no base address")
        }
        for row in 0..<dimensions.height {
            frameBuffer.contents().advanced(by: row * packedRowBytes).copyMemory(
                from: source.advanced(by: row * CVPixelBufferGetBytesPerRow(basePixelBuffer)),
                byteCount: packedRowBytes
            )
        }

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        encoder.setComputePipelineState(pipeline)
        let modelSize = SideBySideVideoPlan.modelTileSize
        let modelElements = 3 * modelSize * modelSize
        for input in mosaicCompositeInputsByVisiblePriority(inputs) {
            guard input.restored.count == modelElements,
                  input.original.count == modelElements,
                  input.samples.count == input.region.width * input.region.height
            else { throw DeformConvError.invalidShape }
            let restoredReusable = try checkoutSharedBuffer(copying: input.restored)
            reusableBuffers.append(restoredReusable)
            let originalReusable = try checkoutSharedBuffer(copying: input.original)
            reusableBuffers.append(originalReusable)
            let sampleBuffer = try cachedSampleBuffer(
                input: input, dimensions: dimensions
            )
            let mask = input.region.maskData ?? Data([255])
            let maskReusable = try checkoutSharedBuffer(copying: mask)
            reusableBuffers.append(maskReusable)
            var params = MetalMosaicCompositeParams(
                frameWidth: UInt32(dimensions.width),
                regionX: UInt32(input.region.x),
                regionY: UInt32(input.region.y),
                regionWidth: UInt32(input.region.width),
                regionHeight: UInt32(input.region.height),
                modelSize: UInt32(modelSize),
                maskWidth: UInt32(input.region.maskWidth ?? 1),
                maskHeight: UInt32(input.region.maskHeight ?? 1),
                groupX: 0,
                groupY: 0,
                groupWidth: 0,
                coverageMode: ordinaryMaskRecoveryEnabled ? 3 : 1,
                detailResidualLimit: detailResidualLimit,
                maskRecoveryDeltaThreshold: maskRecoveryDeltaThreshold
            )
            encoder.setBuffer(frameBuffer, offset: 0, index: 0)
            encoder.setBuffer(restoredReusable.buffer, offset: 0, index: 1)
            encoder.setBuffer(originalReusable.buffer, offset: 0, index: 2)
            encoder.setBuffer(sampleBuffer, offset: 0, index: 3)
            encoder.setBuffer(maskReusable.buffer, offset: 0, index: 4)
            encoder.setBytes(
                &params, length: MemoryLayout<MetalMosaicCompositeParams>.stride, index: 5
            )
            let count = input.region.width * input.region.height
            let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
            )
            encoder.memoryBarrier(scope: .buffers)
        }
        encoder.endEncoding()
        // Buffer-path readback requires the GPU to finish before the CPU copy.
        try finishCommandBuffer(commandBuffer, waitUntilCompleted: true)

        CVPixelBufferLockBaseAddress(outputPixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(outputPixelBuffer, []) }
        guard let destination = CVPixelBufferGetBaseAddress(outputPixelBuffer) else {
            throw DeformConvError.commandFailed("output pixel buffer has no base address")
        }
        for row in 0..<dimensions.height {
            destination.advanced(by: row * CVPixelBufferGetBytesPerRow(outputPixelBuffer))
                .copyMemory(
                    from: frameBuffer.contents().advanced(by: row * packedRowBytes),
                    byteCount: packedRowBytes
                )
        }
    }

    private func compositeUsingTextures(
        basePixelBuffer: CVPixelBuffer?,
        stereoBasePixelBuffers: (CVPixelBuffer, CVPixelBuffer)? = nil,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput],
        waitUntilCompleted: Bool = true
    ) throws -> Bool {
        let source = basePixelBuffer.flatMap {
            makeTexture(pixelBuffer: $0, dimensions: dimensions)
        }
        let eyeDimensions = VideoDimensions(
            width: dimensions.width / 2, height: dimensions.height
        )
        let stereoSources: (
            (reference: CVMetalTexture, texture: MTLTexture),
            (reference: CVMetalTexture, texture: MTLTexture)
        )?
        if let stereoBasePixelBuffers {
            guard let left = makeTexture(
                pixelBuffer: stereoBasePixelBuffers.0, dimensions: eyeDimensions
            ), let right = makeTexture(
                pixelBuffer: stereoBasePixelBuffers.1, dimensions: eyeDimensions
            ) else { return false }
            stereoSources = (left, right)
        } else {
            stereoSources = nil
        }
        guard (basePixelBuffer == nil || source != nil),
              (stereoBasePixelBuffers == nil || stereoSources != nil),
              !(basePixelBuffer != nil && stereoBasePixelBuffers != nil),
              let destination = makeTexture(
                  pixelBuffer: outputPixelBuffer, dimensions: dimensions
              ),
              let commandBuffer = queue.makeCommandBuffer()
        else { return false }
        if let stereoSources {
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return false }
            let eyeSize = MTLSize(
                width: eyeDimensions.width, height: eyeDimensions.height, depth: 1
            )
            blitEncoder.copy(
                from: stereoSources.0.texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: .init(x: 0, y: 0, z: 0),
                sourceSize: eyeSize,
                to: destination.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: .init(x: 0, y: 0, z: 0)
            )
            blitEncoder.copy(
                from: stereoSources.1.texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: .init(x: 0, y: 0, z: 0),
                sourceSize: eyeSize,
                to: destination.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: .init(x: eyeDimensions.width, y: 0, z: 0)
            )
            blitEncoder.endEncoding()
        } else if let source {
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return false }
            blitEncoder.copy(
                from: source.texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: .init(x: 0, y: 0, z: 0),
                sourceSize: .init(width: dimensions.width, height: dimensions.height, depth: 1),
                to: destination.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: .init(x: 0, y: 0, z: 0)
            )
            blitEncoder.endEncoding()
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
        encoder.setTexture(destination.texture, index: 0)
        let modelSize = SideBySideVideoPlan.modelTileSize
        let modelElements = 3 * modelSize * modelSize
        var reusableBuffers = [ReusableBuffer]()
        defer { recycleBuffers(reusableBuffers) }
        let orderedInputs = mosaicCompositeInputsByVisiblePriority(inputs)
        encoder.setComputePipelineState(texturePipeline)
        for input in orderedInputs where input.region.subdivisionGroup == nil {
            guard input.restored.count == modelElements,
                  input.original.count == modelElements,
                  input.samples.count == input.region.width * input.region.height
            else { throw DeformConvError.invalidShape }
            let restoredReusable = try checkoutSharedBuffer(copying: input.restored)
            reusableBuffers.append(restoredReusable)
            let originalReusable = try checkoutSharedBuffer(copying: input.original)
            reusableBuffers.append(originalReusable)
            let sampleBuffer = try cachedSampleBuffer(
                input: input, dimensions: dimensions
            )
            let mask = input.region.maskData ?? Data([255])
            let maskReusable = try checkoutSharedBuffer(copying: mask)
            reusableBuffers.append(maskReusable)
            var params = MetalMosaicCompositeParams(
                frameWidth: UInt32(dimensions.width),
                regionX: UInt32(input.region.x),
                regionY: UInt32(input.region.y),
                regionWidth: UInt32(input.region.width),
                regionHeight: UInt32(input.region.height),
                modelSize: UInt32(modelSize),
                maskWidth: UInt32(input.region.maskWidth ?? 1),
                maskHeight: UInt32(input.region.maskHeight ?? 1),
                groupX: 0,
                groupY: 0,
                groupWidth: 0,
                coverageMode: ordinaryMaskRecoveryEnabled ? 3 : 1,
                detailResidualLimit: detailResidualLimit,
                maskRecoveryDeltaThreshold: maskRecoveryDeltaThreshold
            )
            encoder.setBuffer(restoredReusable.buffer, offset: 0, index: 0)
            encoder.setBuffer(originalReusable.buffer, offset: 0, index: 1)
            encoder.setBuffer(sampleBuffer, offset: 0, index: 2)
            encoder.setBuffer(maskReusable.buffer, offset: 0, index: 3)
            encoder.setBytes(
                &params, length: MemoryLayout<MetalMosaicCompositeParams>.stride, index: 4
            )
            let count = input.region.width * input.region.height
            let threads = min(texturePipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
            )
            encoder.memoryBarrier(scope: .textures)
        }
        let groupedInputs = Dictionary(grouping: orderedInputs.compactMap { input in
            input.region.subdivisionGroup == nil ? nil : input
        }) { $0.region.subdivisionGroup! }
        for groupID in groupedInputs.keys.sorted() {
            guard let group = groupedInputs[groupID], !group.isEmpty else { continue }
            let groupX = group.map(\.region.x).min()!
            let groupY = group.map(\.region.y).min()!
            let groupRight = group.map { $0.region.x + $0.region.width }.max()!
            let groupBottom = group.map { $0.region.y + $0.region.height }.max()!
            let groupWidth = groupRight - groupX
            let groupHeight = groupBottom - groupY
            let accumulatorLength = groupWidth * groupHeight * MemoryLayout<SIMD4<Float>>.stride
            let accumulatorReusable = try checkoutBuffer(
                length: accumulatorLength, options: .storageModePrivate
            )
            reusableBuffers.append(accumulatorReusable)
            let coverageReusable = try checkoutBuffer(
                length: groupWidth * groupHeight * MemoryLayout<Float>.stride,
                options: .storageModePrivate
            )
            reusableBuffers.append(coverageReusable)
            let restoredAccumulatorReusable = try checkoutBuffer(
                length: groupWidth * groupHeight * MemoryLayout<SIMD4<Float16>>.stride,
                options: .storageModePrivate
            )
            reusableBuffers.append(restoredAccumulatorReusable)
            let accumulator = accumulatorReusable.buffer
            let coverage = coverageReusable.buffer
            let restoredAccumulator = restoredAccumulatorReusable.buffer
            let groupPixels = groupWidth * groupHeight
            encoder.setComputePipelineState(groupClearPipeline)
            encoder.setBuffer(accumulator, offset: 0, index: 0)
            encoder.setBuffer(coverage, offset: 0, index: 1)
            encoder.setBuffer(restoredAccumulator, offset: 0, index: 2)
            let clearThreads = min(
                groupClearPipeline.maxTotalThreadsPerThreadgroup, 256
            )
            encoder.dispatchThreads(
                MTLSize(width: groupPixels, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: clearThreads, height: 1, depth: 1)
            )
            encoder.memoryBarrier(scope: .buffers)
            let hasPrimaryCrop = group.contains { $0.region.detailBlendFeather == nil }
            encoder.setComputePipelineState(groupAccumulatePipeline)
            for input in group {
                guard input.restored.count == modelElements,
                      input.original.count == modelElements,
                      input.samples.count == input.region.width * input.region.height
                else { throw DeformConvError.invalidShape }
                let restoredReusable = try checkoutSharedBuffer(copying: input.restored)
                reusableBuffers.append(restoredReusable)
                let originalReusable = try checkoutSharedBuffer(copying: input.original)
                reusableBuffers.append(originalReusable)
                let sampleBuffer = try cachedSampleBuffer(input: input, dimensions: dimensions)
                let mask = input.region.maskData ?? Data([255])
                let maskReusable = try checkoutSharedBuffer(copying: mask)
                reusableBuffers.append(maskReusable)
                var params = MetalMosaicCompositeParams(
                    frameWidth: UInt32(dimensions.width),
                    regionX: UInt32(input.region.x),
                    regionY: UInt32(input.region.y),
                    regionWidth: UInt32(input.region.width),
                    regionHeight: UInt32(input.region.height),
                    modelSize: UInt32(modelSize),
                    maskWidth: UInt32(input.region.maskWidth ?? 1),
                    maskHeight: UInt32(input.region.maskHeight ?? 1),
                    groupX: UInt32(groupX),
                    groupY: UInt32(groupY),
                    groupWidth: UInt32(groupWidth),
                    coverageMode: input.region.detailBlendFeather == nil
                        ? 3 : (!hasPrimaryCrop ? 1 : 2),
                    detailResidualLimit: detailResidualLimit,
                    maskRecoveryDeltaThreshold: maskRecoveryDeltaThreshold
                )
                encoder.setBuffer(accumulator, offset: 0, index: 0)
                encoder.setBuffer(restoredReusable.buffer, offset: 0, index: 1)
                encoder.setBuffer(originalReusable.buffer, offset: 0, index: 2)
                encoder.setBuffer(sampleBuffer, offset: 0, index: 3)
                encoder.setBuffer(maskReusable.buffer, offset: 0, index: 4)
                encoder.setBytes(
                    &params, length: MemoryLayout<MetalMosaicCompositeParams>.stride, index: 5
                )
                encoder.setBuffer(coverage, offset: 0, index: 6)
                encoder.setBuffer(restoredAccumulator, offset: 0, index: 7)
                let count = input.region.width * input.region.height
                let threads = min(groupAccumulatePipeline.maxTotalThreadsPerThreadgroup, 256)
                encoder.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
                )
                encoder.memoryBarrier(scope: .buffers)
            }
            var resolveParams = MetalMosaicGroupResolveParams(
                groupX: UInt32(groupX),
                groupY: UInt32(groupY),
                groupWidth: UInt32(groupWidth),
                groupHeight: UInt32(groupHeight),
                detailResidualLimit: detailResidualLimit
            )
            encoder.setComputePipelineState(groupResolvePipeline)
            encoder.setTexture(destination.texture, index: 0)
            encoder.setBuffer(accumulator, offset: 0, index: 0)
            encoder.setBuffer(coverage, offset: 0, index: 1)
            encoder.setBuffer(restoredAccumulator, offset: 0, index: 2)
            encoder.setBytes(
                &resolveParams,
                length: MemoryLayout<MetalMosaicGroupResolveParams>.stride,
                index: 3
            )
            let threads = min(groupResolvePipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: groupPixels, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
            )
            encoder.memoryBarrier(scope: .textures)
        }
        encoder.endEncoding()
        try finishCommandBuffer(commandBuffer, waitUntilCompleted: waitUntilCompleted)
        withExtendedLifetime(
            (
                source?.reference,
                stereoSources?.0.reference,
                stereoSources?.1.reference,
                destination.reference,
                reusableBuffers.map(\.buffer)
            )
        ) {}
        return true
    }


    func synchronize() throws {
        commandBufferLock.lock()
        let pending = pendingCommandBuffer
        pendingCommandBuffer = nil
        commandBufferLock.unlock()
        guard let pending else { return }
        pending.waitUntilCompleted()
        if let error = pending.error { throw error }
    }

    private func finishCommandBuffer(
        _ commandBuffer: MTLCommandBuffer,
        waitUntilCompleted: Bool
    ) throws {
        if waitUntilCompleted {
            try synchronize()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
            return
        }
        commandBuffer.commit()
        commandBufferLock.lock()
        let previous = pendingCommandBuffer
        pendingCommandBuffer = commandBuffer
        commandBufferLock.unlock()
        if let previous {
            previous.waitUntilCompleted()
            if let error = previous.error { throw error }
        }
    }

    private func makeTexture(
        pixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions
    ) -> (reference: CVMetalTexture, texture: MTLTexture)? {
        textureCacheLock.lock()
        defer { textureCacheLock.unlock() }
        var optionalReference: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            dimensions.width,
            dimensions.height,
            0,
            &optionalReference
        )
        guard status == kCVReturnSuccess,
              let reference = optionalReference,
              let texture = CVMetalTextureGetTexture(reference)
        else { return nil }
        return (reference, texture)
    }

    private func checkoutBuffer(
        length: Int,
        options: MTLResourceOptions
    ) throws -> ReusableBuffer {
        guard length > 0 else { throw DeformConvError.invalidShape }
        let alignedLength = (length + 4_095) & ~4_095
        let key = ReusableBufferKey(length: alignedLength, options: options.rawValue)
        reusableBufferLock.lock()
        if var available = reusableBuffers[key], let buffer = available.popLast() {
            if available.isEmpty {
                reusableBuffers.removeValue(forKey: key)
            } else {
                reusableBuffers[key] = available
            }
            reusableBufferBytes -= key.length
            reusableBufferLock.unlock()
            return ReusableBuffer(key: key, buffer: buffer)
        }
        reusableBufferLock.unlock()
        guard let buffer = device.makeBuffer(length: alignedLength, options: options) else {
            throw DeformConvError.metalUnavailable
        }
        return ReusableBuffer(key: key, buffer: buffer)
    }

    private func checkoutSharedBuffer<T>(copying values: [T]) throws -> ReusableBuffer {
        try values.withUnsafeBytes { bytes in
            try checkoutSharedBuffer(copying: bytes)
        }
    }

    private func checkoutSharedBuffer(copying data: Data) throws -> ReusableBuffer {
        try data.withUnsafeBytes { bytes in
            try checkoutSharedBuffer(copying: bytes)
        }
    }

    private func checkoutSharedBuffer(
        copying bytes: UnsafeRawBufferPointer
    ) throws -> ReusableBuffer {
        guard let source = bytes.baseAddress, !bytes.isEmpty else {
            throw DeformConvError.invalidShape
        }
        let reusable = try checkoutBuffer(length: bytes.count, options: .storageModeShared)
        reusable.buffer.contents().copyMemory(from: source, byteCount: bytes.count)
        return reusable
    }

    private func recycleBuffers(_ buffers: [ReusableBuffer]) {
        guard reusableBufferLimitBytes > 0 else { return }
        reusableBufferLock.lock()
        defer { reusableBufferLock.unlock() }
        for reusable in buffers where reusable.key.length <= reusableBufferLimitBytes {
            guard reusableBufferBytes <= reusableBufferLimitBytes - reusable.key.length else {
                continue
            }
            reusableBuffers[reusable.key, default: []].append(reusable.buffer)
            reusableBufferBytes += reusable.key.length
        }
    }

    private func cachedSampleBuffer(
        input: MetalMosaicCompositeInput,
        dimensions: VideoDimensions
    ) throws -> MTLBuffer {
        let key = SampleBufferKey(
            frameWidth: dimensions.width,
            frameHeight: dimensions.height,
            regionX: input.region.x,
            regionY: input.region.y,
            regionWidth: input.region.width,
            regionHeight: input.region.height,
            blendX: input.region.effectiveBlendX,
            blendY: input.region.effectiveBlendY,
            blendWidth: input.region.effectiveBlendWidth,
            blendHeight: input.region.effectiveBlendHeight,
            detailBlendFeather: input.region.detailBlendFeather
        )
        sampleBufferLock.lock()
        defer { sampleBufferLock.unlock() }
        sampleBufferUse &+= 1
        if var cached = sampleBuffers[key] {
            cached.lastUse = sampleBufferUse
            sampleBuffers[key] = cached
            return cached.buffer
        }
        guard let buffer = input.samples.withUnsafeBytes({ bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: .storageModeShared
            )
        }) else { throw DeformConvError.metalUnavailable }
        guard sampleBufferLimitBytes > 0, buffer.length <= sampleBufferLimitBytes else {
            return buffer
        }
        sampleBuffers[key] = CachedSampleBuffer(buffer: buffer, lastUse: sampleBufferUse)
        sampleBufferBytes += buffer.length
        while sampleBufferBytes > sampleBufferLimitBytes,
              let oldest = sampleBuffers.min(by: { $0.value.lastUse < $1.value.lastUse })
        {
            sampleBufferBytes -= oldest.value.buffer.length
            sampleBuffers.removeValue(forKey: oldest.key)
        }
        return buffer
    }
}
