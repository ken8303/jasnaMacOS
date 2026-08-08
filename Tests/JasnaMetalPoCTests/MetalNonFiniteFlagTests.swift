import Metal
import Testing
@testable import JasnaMetalPoC

@Test func metalNonFiniteFlagDetectsInfinity() throws {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue()
    else { return }
    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let function = library.makeFunction(name: "flag_non_finite_fp16") else {
        throw DeformConvError.shaderResourceMissing
    }
    let pipeline = try device.makeComputePipelineState(function: function)

    func flagged(_ values: [Float16]) throws -> UInt32 {
        guard !values.isEmpty else { throw DeformConvError.invalidShape }
        let valueBuffer = values.withUnsafeBytes { bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
            )
        }
        var count = UInt32(values.count)
        guard let valueBuffer,
              let flagBuffer = device.makeBuffer(length: 4, options: .storageModeShared),
              let countBuffer = device.makeBuffer(
                  bytes: &count, length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        flagBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(valueBuffer, offset: 0, index: 0)
        encoder.setBuffer(flagBuffer, offset: 0, index: 1)
        encoder.setBuffer(countBuffer, offset: 0, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: values.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(values.count, pipeline.threadExecutionWidth), height: 1, depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        return flagBuffer.contents().load(as: UInt32.self)
    }

    #expect(try flagged([0, 1, -2, 0.5]) == 0)
    #expect(try flagged([0, .infinity, 1]) == 1)
    #expect(try flagged([0, .nan, 1]) == 1)
}
