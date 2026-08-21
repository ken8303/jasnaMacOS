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
    var contributesCoverage: UInt32
    var detailResidualLimit: Float
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
    let prefersTextureSurfaces: Bool
    private let textureCacheLock = NSLock()
    private let sampleBufferLock = NSLock()
    private var sampleBuffers = [SampleBufferKey: MTLBuffer]()

    init(device: MTLDevice) throws {
        self.device = device
        detailResidualLimit = MosaicCompositeQuality.detailResidualLimit()
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
        inputs: [MetalMosaicCompositeInput]
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
                inputs: inputs
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
        inputs: [MetalMosaicCompositeInput]
    ) throws {
        guard prefersTextureSurfaces,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(pixelBuffer) == dimensions.height,
              try compositeUsingTextures(
                  basePixelBuffer: nil,
                  outputPixelBuffer: pixelBuffer,
                  dimensions: dimensions,
                  inputs: inputs
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
        dimensions: VideoDimensions
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        withExtendedLifetime((left.reference, right.reference, destination.reference)) {}
    }

    func compositeStereo(
        leftPixelBuffer: CVPixelBuffer,
        rightPixelBuffer: CVPixelBuffer,
        outputPixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions,
        inputs: [MetalMosaicCompositeInput]
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
                  inputs: inputs
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
        guard let frameBuffer = device.makeBuffer(
            length: packedRowBytes * dimensions.height, options: .storageModeShared
        ) else { throw DeformConvError.metalUnavailable }

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
        var heldBuffers = [MTLBuffer]()
        for input in mosaicCompositeInputsByVisiblePriority(inputs) {
            guard input.restored.count == modelElements,
                  input.original.count == modelElements,
                  input.samples.count == input.region.width * input.region.height
            else { throw DeformConvError.invalidShape }
            let restoredBuffer = input.restored.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            let originalBuffer = input.original.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            let sampleBuffer = try cachedSampleBuffer(
                input: input, dimensions: dimensions
            )
            let mask = input.region.maskData ?? Data([255])
            let maskBuffer = mask.withUnsafeBytes {
                device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared
                )
            }
            guard let restoredBuffer, let originalBuffer, let maskBuffer else {
                throw DeformConvError.metalUnavailable
            }
            heldBuffers += [restoredBuffer, originalBuffer, maskBuffer]
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
                contributesCoverage: 1,
                detailResidualLimit: detailResidualLimit
            )
            encoder.setBuffer(frameBuffer, offset: 0, index: 0)
            encoder.setBuffer(restoredBuffer, offset: 0, index: 1)
            encoder.setBuffer(originalBuffer, offset: 0, index: 2)
            encoder.setBuffer(sampleBuffer, offset: 0, index: 3)
            encoder.setBuffer(maskBuffer, offset: 0, index: 4)
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        _ = heldBuffers

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
        inputs: [MetalMosaicCompositeInput]
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
        var heldBuffers = [MTLBuffer]()
        let orderedInputs = mosaicCompositeInputsByVisiblePriority(inputs)
        encoder.setComputePipelineState(texturePipeline)
        for input in orderedInputs where input.region.subdivisionGroup == nil {
            guard input.restored.count == modelElements,
                  input.original.count == modelElements,
                  input.samples.count == input.region.width * input.region.height
            else { throw DeformConvError.invalidShape }
            let restoredBuffer = input.restored.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            let originalBuffer = input.original.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            let sampleBuffer = try cachedSampleBuffer(
                input: input, dimensions: dimensions
            )
            let mask = input.region.maskData ?? Data([255])
            let maskBuffer = mask.withUnsafeBytes {
                device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared
                )
            }
            guard let restoredBuffer, let originalBuffer, let maskBuffer else {
                throw DeformConvError.metalUnavailable
            }
            heldBuffers += [restoredBuffer, originalBuffer, maskBuffer]
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
                contributesCoverage: 1,
                detailResidualLimit: detailResidualLimit
            )
            encoder.setBuffer(restoredBuffer, offset: 0, index: 0)
            encoder.setBuffer(originalBuffer, offset: 0, index: 1)
            encoder.setBuffer(sampleBuffer, offset: 0, index: 2)
            encoder.setBuffer(maskBuffer, offset: 0, index: 3)
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
            guard let accumulator = device.makeBuffer(
                length: accumulatorLength, options: .storageModePrivate
            ), let coverage = device.makeBuffer(
                length: groupWidth * groupHeight * MemoryLayout<Float>.stride,
                options: .storageModePrivate
            ), let restoredAccumulator = device.makeBuffer(
                length: groupWidth * groupHeight * MemoryLayout<SIMD4<Float16>>.stride,
                options: .storageModePrivate
            ) else { throw DeformConvError.metalUnavailable }
            heldBuffers += [accumulator, coverage, restoredAccumulator]
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
                let restoredBuffer = input.restored.withUnsafeBytes {
                    device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared
                    )
                }
                let originalBuffer = input.original.withUnsafeBytes {
                    device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared
                    )
                }
                let sampleBuffer = try cachedSampleBuffer(input: input, dimensions: dimensions)
                let mask = input.region.maskData ?? Data([255])
                let maskBuffer = mask.withUnsafeBytes {
                    device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared
                    )
                }
                guard let restoredBuffer, let originalBuffer, let maskBuffer else {
                    throw DeformConvError.metalUnavailable
                }
                heldBuffers += [restoredBuffer, originalBuffer, maskBuffer]
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
                    contributesCoverage: input.region.detailBlendFeather == nil
                        || !hasPrimaryCrop ? 1 : 0,
                    detailResidualLimit: detailResidualLimit
                )
                encoder.setBuffer(accumulator, offset: 0, index: 0)
                encoder.setBuffer(restoredBuffer, offset: 0, index: 1)
                encoder.setBuffer(originalBuffer, offset: 0, index: 2)
                encoder.setBuffer(sampleBuffer, offset: 0, index: 3)
                encoder.setBuffer(maskBuffer, offset: 0, index: 4)
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        withExtendedLifetime(
            (
                source?.reference,
                stereoSources?.0.reference,
                stereoSources?.1.reference,
                destination.reference,
                heldBuffers
            )
        ) {}
        return true
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
        if let buffer = sampleBuffers[key] { return buffer }
        guard let buffer = input.samples.withUnsafeBytes({ bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: .storageModeShared
            )
        }) else { throw DeformConvError.metalUnavailable }
        sampleBuffers[key] = buffer
        return buffer
    }
}
