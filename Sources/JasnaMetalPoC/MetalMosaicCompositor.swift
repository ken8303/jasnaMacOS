import CoreVideo
import Foundation
import Metal

struct MetalMosaicCompositeInput {
    let region: MosaicRegion
    let restored: [Float16]
    let original: [Float16]
    let samples: [MosaicCompositeSample]
}

private struct MetalMosaicCompositeParams {
    var frameWidth: UInt32
    var regionX: UInt32
    var regionY: UInt32
    var regionWidth: UInt32
    var regionHeight: UInt32
    var modelSize: UInt32
}

@available(macOS 27.0, *)
final class MetalMosaicCompositor: @unchecked Sendable {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device
        let library = try MetalResourceCache.shared.shaderLibrary(device: device) {
            try device.makeLibrary(source: MetalShader.source, options: nil)
        }
        guard let function = library.makeFunction(name: "composite_fisheye_mosaic_delta"),
              let queue = device.makeCommandQueue()
        else { throw DeformConvError.metalUnavailable }
        pipeline = try MetalResourceCache.shared.computePipeline(
            device: device, function: function
        )
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
        for input in inputs {
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
            let sampleBuffer = input.samples.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            guard let restoredBuffer, let originalBuffer, let sampleBuffer else {
                throw DeformConvError.metalUnavailable
            }
            heldBuffers += [restoredBuffer, originalBuffer, sampleBuffer]
            var params = MetalMosaicCompositeParams(
                frameWidth: UInt32(dimensions.width),
                regionX: UInt32(input.region.x),
                regionY: UInt32(input.region.y),
                regionWidth: UInt32(input.region.width),
                regionHeight: UInt32(input.region.height),
                modelSize: UInt32(modelSize)
            )
            encoder.setBuffer(frameBuffer, offset: 0, index: 0)
            encoder.setBuffer(restoredBuffer, offset: 0, index: 1)
            encoder.setBuffer(originalBuffer, offset: 0, index: 2)
            encoder.setBuffer(sampleBuffer, offset: 0, index: 3)
            encoder.setBytes(
                &params, length: MemoryLayout<MetalMosaicCompositeParams>.stride, index: 4
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
}
