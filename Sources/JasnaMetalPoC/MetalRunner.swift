import Foundation
import Metal

private struct MetalShape {
    var batch: UInt32
    var inputChannels: UInt32
    var inputHeight: UInt32
    var inputWidth: UInt32
    var outputChannels: UInt32
    var outputHeight: UInt32
    var outputWidth: UInt32
    var kernelHeight: UInt32
    var kernelWidth: UInt32
    var padHeight: UInt32
    var padWidth: UInt32
    var strideHeight: UInt32
    var strideWidth: UInt32
    var dilationHeight: UInt32
    var dilationWidth: UInt32
    var groups: UInt32
    var offsetGroups: UInt32
    var hasMask: UInt32

    init(_ s: DeformConvShape) {
        batch = UInt32(s.batch)
        inputChannels = UInt32(s.inputChannels)
        inputHeight = UInt32(s.inputHeight)
        inputWidth = UInt32(s.inputWidth)
        outputChannels = UInt32(s.outputChannels)
        outputHeight = UInt32(s.outputHeight)
        outputWidth = UInt32(s.outputWidth)
        kernelHeight = UInt32(s.kernelHeight)
        kernelWidth = UInt32(s.kernelWidth)
        padHeight = UInt32(s.padHeight)
        padWidth = UInt32(s.padWidth)
        strideHeight = UInt32(s.strideHeight)
        strideWidth = UInt32(s.strideWidth)
        dilationHeight = UInt32(s.dilationHeight)
        dilationWidth = UInt32(s.dilationWidth)
        groups = UInt32(s.groups)
        offsetGroups = UInt32(s.offsetGroups)
        hasMask = 1
    }
}

private struct SPyNetPrepareShape {
    var width: UInt32
    var height: UInt32
    var sourceFlowWidth: UInt32
    var sourceFlowHeight: UInt32
    var firstLevel: UInt32
}

private struct TemporalPrepareShape {
    var width: UInt32
    var height: UInt32
    var hasSecondOrder: UInt32
}

struct TemporalPreparationResult {
    let gpuMilliseconds: Double
    let conditions: [Float16]
    let deformInput: [Float16]
    let secondOrderFlow: [Float16]
}

struct BenchmarkResult {
    let baselineStatistics: BenchmarkStatistics
    let simdStatistics: BenchmarkStatistics
    let tiledStatistics: BenchmarkStatistics
    let gemmStatistics: BenchmarkStatistics
    let baselineMedianMilliseconds: Double
    let baselineMinimumMilliseconds: Double
    let baselineMaximumMilliseconds: Double
    let simdMedianMilliseconds: Double
    let simdMinimumMilliseconds: Double
    let simdMaximumMilliseconds: Double
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let simdgroupGEMMMedianMilliseconds: Double
    let simdgroupGEMMMinimumMilliseconds: Double
    let simdgroupGEMMMaximumMilliseconds: Double
    let iterations: Int
    let checksum: Double
    let maxDifferenceFromBaseline: Float
    let simdgroupGEMMMaxDifferenceFromBaseline: Float
}

final class MetalDeformConv {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let fp32Pipeline: MTLComputePipelineState
    private let fp16Pipeline: MTLComputePipelineState
    private let fp16JasnaSIMDPipeline: MTLComputePipelineState
    private let fp16JasnaTiledPipeline: MTLComputePipelineState
    private let fp16JasnaGatherPipeline: MTLComputePipelineState
    private let fp16JasnaSIMDGroupGEMMPipeline: MTLComputePipelineState
    private let fp16JasnaFloatGEMMOutputPipeline: MTLComputePipelineState
    private let spynetPreparePipeline: MTLComputePipelineState
    private let dcnOffsetPipeline: MTLComputePipelineState
    private let secondOrderFlowPipeline: MTLComputePipelineState
    private let temporalAlignmentPipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw DeformConvError.metalUnavailable }
        self.device = device
        self.queue = queue

        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: MetalShader.source, options: options)
        guard let fp32 = library.makeFunction(name: "deform_conv2d_fp32"),
              let fp16 = library.makeFunction(name: "deform_conv2d_fp16"),
              let fp16JasnaSIMD = library.makeFunction(name: "deform_conv2d_fp16_jasna_simd"),
              let fp16JasnaTiled = library.makeFunction(name: "deform_conv2d_fp16_jasna_tiled"),
              let fp16JasnaGather = library.makeFunction(name: "deform_conv2d_fp16_jasna_gather"),
              let fp16JasnaSIMDGroupGEMM = library.makeFunction(name: "deform_conv2d_fp16_jasna_simdgroup_gemm"),
              let fp16JasnaFloatGEMMOutput = library.makeFunction(name: "deform_conv2d_fp16_jasna_float_gemm_output"),
              let spynetPrepare = library.makeFunction(name: "spynet_prepare_fp16"),
              let dcnOffset = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
              let secondOrderFlow = library.makeFunction(name: "accumulate_second_order_flow_fp16"),
              let temporalAlignment = library.makeFunction(name: "assemble_temporal_alignment_fp16")
        else { throw DeformConvError.shaderResourceMissing }
        fp32Pipeline = try device.makeComputePipelineState(function: fp32)
        fp16Pipeline = try device.makeComputePipelineState(function: fp16)
        fp16JasnaSIMDPipeline = try device.makeComputePipelineState(function: fp16JasnaSIMD)
        fp16JasnaTiledPipeline = try device.makeComputePipelineState(function: fp16JasnaTiled)
        fp16JasnaGatherPipeline = try device.makeComputePipelineState(function: fp16JasnaGather)
        fp16JasnaSIMDGroupGEMMPipeline = try device.makeComputePipelineState(function: fp16JasnaSIMDGroupGEMM)
        fp16JasnaFloatGEMMOutputPipeline = try device.makeComputePipelineState(function: fp16JasnaFloatGEMMOutput)
        spynetPreparePipeline = try device.makeComputePipelineState(function: spynetPrepare)
        dcnOffsetPipeline = try device.makeComputePipelineState(function: dcnOffset)
        secondOrderFlowPipeline = try device.makeComputePipelineState(function: secondOrderFlow)
        temporalAlignmentPipeline = try device.makeComputePipelineState(function: temporalAlignment)
    }

    func canCreateMetalMLTensor() -> Bool {
        var sizes = [64, 64, 128, 1]
        guard let extents = sizes.withUnsafeMutableBufferPointer({
            MTLTensorExtents(__rank: $0.count, values: $0.baseAddress)
        }) else { return false }
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = extents
        descriptor.dataType = .float16
        descriptor.usage = [.compute, .machineLearning]
        descriptor.storageMode = .private
        return (try? device.makeTensor(descriptor: descriptor)) != nil
    }

    func runFloat32(
        shape: DeformConvShape,
        input: [Float],
        offset: [Float],
        mask: [Float],
        weight: [Float],
        bias: [Float]
    ) throws -> [Float] {
        try shape.validate()
        let inputBuffer = try makeBuffer(input)
        let offsetBuffer = try makeBuffer(offset)
        let maskBuffer = try makeBuffer(mask)
        let weightBuffer = try makeBuffer(weight)
        let biasBuffer = try makeBuffer(bias)
        guard let outputBuffer = device.makeBuffer(
            length: shape.outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else { throw DeformConvError.metalUnavailable }

        _ = try execute(
            pipeline: fp32Pipeline,
            shape: shape,
            buffers: [inputBuffer, offsetBuffer, maskBuffer, weightBuffer, biasBuffer, outputBuffer]
        )
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: shape.outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: shape.outputCount))
    }

    func benchmarkJasnaShape(
        iterations: Int = 20,
        weightSet: DeformConvWeightSet? = nil
    ) throws -> BenchmarkResult {
        let shape = DeformConvShape(
            batch: 1,
            inputChannels: 128,
            inputHeight: 64,
            inputWidth: 64,
            outputChannels: 64,
            outputHeight: 64,
            outputWidth: 64,
            kernelHeight: 3,
            kernelWidth: 3,
            padHeight: 1,
            padWidth: 1,
            strideHeight: 1,
            strideWidth: 1,
            dilationHeight: 1,
            dilationWidth: 1,
            groups: 1,
            offsetGroups: 16
        )
        var rng = SeededGenerator(seed: 0x4A41_534E_41)
        let input = rng.floats(count: shape.inputCount, range: -1...1).map(Float16.init)
        let offset = rng.floats(count: shape.offsetCount, range: -0.25...0.25).map(Float16.init)
        let mask = rng.floats(count: shape.maskCount, range: 0.25...0.75).map(Float16.init)
        var weight = [Float16](repeating: 0, count: shape.weightCount)
        var packedWeight = [Float16](repeating: 0, count: shape.weightCount)
        let bias: [Float16]
        if let weightSet {
            let values = weightSet.data.withUnsafeBytes { bytes in
                Array(bytes.bindMemory(to: UInt16.self)).map(Float16.init(bitPattern:))
            }
            packedWeight = Array(values[..<shape.weightCount])
            bias = Array(values[shape.weightCount..<(shape.weightCount + shape.outputChannels)])
            for ic in 0..<shape.inputChannels {
                for k in 0..<shape.kernelArea {
                    for oc in 0..<shape.outputChannels {
                        let packed = (ic * shape.kernelArea + k) * shape.outputChannels + oc
                        let standard = (oc * shape.inputChannels + ic) * shape.kernelArea + k
                        weight[standard] = packedWeight[packed]
                    }
                }
            }
        } else {
            weight = rng.floats(count: shape.weightCount, range: -0.02...0.02).map(Float16.init)
            bias = rng.floats(count: shape.outputChannels, range: -0.02...0.02).map(Float16.init)
            for oc in 0..<shape.outputChannels {
                for ic in 0..<shape.inputChannels {
                    for k in 0..<shape.kernelArea {
                        let source = (oc * shape.inputChannels + ic) * shape.kernelArea + k
                        let destination = (ic * shape.kernelArea + k) * shape.outputChannels + oc
                        packedWeight[destination] = weight[source]
                    }
                }
            }
        }

        let inputBuffer = try makeBuffer(input)
        let offsetBuffer = try makeBuffer(offset)
        let maskBuffer = try makeBuffer(mask)
        let weightBuffer = try makeBuffer(weight)
        let packedWeightBuffer = try makeBuffer(packedWeight)
        let biasBuffer = try makeBuffer(bias)
        guard let outputBuffer = device.makeBuffer(
            length: shape.outputCount * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ), let tiledOutputBuffer = device.makeBuffer(
            length: shape.outputCount * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ), let baselineOutputBuffer = device.makeBuffer(
            length: shape.outputCount * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ), let gatheredBuffer = device.makeBuffer(
            length: shape.batch * shape.outputHeight * shape.outputWidth
                * shape.inputChannels * shape.kernelArea * MemoryLayout<Float16>.stride,
            options: .storageModePrivate
        ), let simdgroupMatrixOutputBuffer = device.makeBuffer(
            length: shape.batch * shape.outputHeight * shape.outputWidth
                * shape.outputChannels * MemoryLayout<Float>.stride,
            options: .storageModePrivate
        ), let simdgroupOutputBuffer = device.makeBuffer(
            length: shape.outputCount * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else { throw DeformConvError.metalUnavailable }
        let buffers = [inputBuffer, offsetBuffer, maskBuffer, packedWeightBuffer, biasBuffer, outputBuffer]
        let tiledBuffers = [inputBuffer, offsetBuffer, maskBuffer, packedWeightBuffer, biasBuffer, tiledOutputBuffer]
        let baselineBuffers = [inputBuffer, offsetBuffer, maskBuffer, weightBuffer, biasBuffer, baselineOutputBuffer]
        let simdgroupGEMMBuffers = [
            inputBuffer, offsetBuffer, maskBuffer, packedWeightBuffer, biasBuffer,
            gatheredBuffer, simdgroupMatrixOutputBuffer, simdgroupOutputBuffer,
        ]

        for _ in 0..<2 {
            _ = try execute(pipeline: fp16Pipeline, shape: shape, buffers: baselineBuffers)
        }
        var baselineSamples = [Double]()
        for _ in 0..<max(iterations, 1) {
            baselineSamples.append(try execute(pipeline: fp16Pipeline, shape: shape, buffers: baselineBuffers))
        }
        baselineSamples.sort()
        for _ in 0..<2 {
            _ = try executeJasnaSIMD(shape: shape, buffers: buffers)
        }
        var simdSamples = [Double]()
        for _ in 0..<max(iterations, 1) {
            simdSamples.append(try executeJasnaSIMD(shape: shape, buffers: buffers))
        }
        simdSamples.sort()
        for _ in 0..<2 {
            _ = try executeJasnaTiled(shape: shape, buffers: tiledBuffers)
        }
        var samples = [Double]()
        for _ in 0..<max(iterations, 1) {
            samples.append(try executeJasnaTiled(shape: shape, buffers: tiledBuffers))
        }
        samples.sort()
        for _ in 0..<2 {
            _ = try executeJasnaSIMDGroupGEMM(shape: shape, buffers: simdgroupGEMMBuffers)
        }
        var simdgroupGEMMSamples = [Double]()
        for _ in 0..<max(iterations, 1) {
            simdgroupGEMMSamples.append(
                try executeJasnaSIMDGroupGEMM(shape: shape, buffers: simdgroupGEMMBuffers)
            )
        }
        simdgroupGEMMSamples.sort()
        let output = tiledOutputBuffer.contents().bindMemory(to: Float16.self, capacity: shape.outputCount)
        let simdgroupOutput = simdgroupOutputBuffer.contents().bindMemory(
            to: Float16.self, capacity: shape.outputCount
        )
        let baselineOutput = baselineOutputBuffer.contents().bindMemory(to: Float16.self, capacity: shape.outputCount)
        var checksum = 0.0
        var maxDifference: Float = 0
        var simdgroupGEMMMaxDifference: Float = 0
        for index in stride(from: 0, to: shape.outputCount, by: 257) {
            checksum += Double(Float(output[index]))
        }
        for index in 0..<shape.outputCount {
            maxDifference = max(maxDifference, abs(Float(output[index]) - Float(baselineOutput[index])))
            simdgroupGEMMMaxDifference = max(
                simdgroupGEMMMaxDifference,
                abs(Float(simdgroupOutput[index]) - Float(baselineOutput[index]))
            )
        }
        guard maxDifference <= 0.001 else {
            throw DeformConvError.commandFailed(
                "tiled FP16 result differs from baseline by \(maxDifference)"
            )
        }
        guard simdgroupGEMMMaxDifference <= 0.002 else {
            throw DeformConvError.commandFailed(
                "SIMD-group GEMM FP16 result differs from baseline by \(simdgroupGEMMMaxDifference)"
            )
        }
        guard let baselineStatistics = BenchmarkStatistics(baselineSamples),
              let simdStatistics = BenchmarkStatistics(simdSamples),
              let tiledStatistics = BenchmarkStatistics(samples),
              let gemmStatistics = BenchmarkStatistics(simdgroupGEMMSamples)
        else {
            throw DeformConvError.commandFailed("invalid DCNv2 benchmark samples")
        }
        return BenchmarkResult(
            baselineStatistics: baselineStatistics,
            simdStatistics: simdStatistics,
            tiledStatistics: tiledStatistics,
            gemmStatistics: gemmStatistics,
            baselineMedianMilliseconds: baselineStatistics.median,
            baselineMinimumMilliseconds: baselineStatistics.minimum,
            baselineMaximumMilliseconds: baselineStatistics.maximum,
            simdMedianMilliseconds: simdStatistics.median,
            simdMinimumMilliseconds: simdStatistics.minimum,
            simdMaximumMilliseconds: simdStatistics.maximum,
            medianMilliseconds: tiledStatistics.median,
            minimumMilliseconds: tiledStatistics.minimum,
            maximumMilliseconds: tiledStatistics.maximum,
            simdgroupGEMMMedianMilliseconds: gemmStatistics.median,
            simdgroupGEMMMinimumMilliseconds: gemmStatistics.minimum,
            simdgroupGEMMMaximumMilliseconds: gemmStatistics.maximum,
            iterations: gemmStatistics.samples.count,
            checksum: checksum,
            maxDifferenceFromBaseline: maxDifference,
            simdgroupGEMMMaxDifferenceFromBaseline: simdgroupGEMMMaxDifference
        )
    }

    func runSPyNetPrepare(
        width: Int,
        height: Int,
        reference: [Float16],
        support: [Float16],
        sourceFlowWidth: Int,
        sourceFlowHeight: Int,
        sourceFlow: [Float16],
        firstLevel: Bool
    ) throws -> (features: [Float16], baseFlow: [Float16]) {
        let plane = width * height
        guard width > 0, height > 0,
              reference.count == 3 * plane,
              support.count == 3 * plane,
              sourceFlow.count == 2 * sourceFlowWidth * sourceFlowHeight
        else { throw DeformConvError.invalidShape }
        let referenceBuffer = try makeBuffer(reference)
        let supportBuffer = try makeBuffer(support)
        let flowBuffer = try makeBuffer(sourceFlow)
        guard let featureBuffer = device.makeBuffer(length: 8 * plane * 2, options: .storageModeShared),
              let baseFlowBuffer = device.makeBuffer(length: 2 * plane * 2, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        encoder.setComputePipelineState(spynetPreparePipeline)
        encoder.setBuffer(referenceBuffer, offset: 0, index: 0)
        encoder.setBuffer(supportBuffer, offset: 0, index: 1)
        encoder.setBuffer(flowBuffer, offset: 0, index: 2)
        encoder.setBuffer(featureBuffer, offset: 0, index: 3)
        encoder.setBuffer(baseFlowBuffer, offset: 0, index: 4)
        var shape = SPyNetPrepareShape(
            width: UInt32(width), height: UInt32(height),
            sourceFlowWidth: UInt32(sourceFlowWidth), sourceFlowHeight: UInt32(sourceFlowHeight),
            firstLevel: firstLevel ? 1 : 0
        )
        encoder.setBytes(&shape, length: MemoryLayout<SPyNetPrepareShape>.stride, index: 5)
        let threads = min(spynetPreparePipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let featurePointer = featureBuffer.contents().bindMemory(to: Float16.self, capacity: 8 * plane)
        let flowPointer = baseFlowBuffer.contents().bindMemory(to: Float16.self, capacity: 2 * plane)
        return (
            Array(UnsafeBufferPointer(start: featurePointer, count: 8 * plane)),
            Array(UnsafeBufferPointer(start: flowPointer, count: 2 * plane))
        )
    }

    func runDCNOffsetTransform(
        plane: Int,
        raw: [Float16],
        flow1: [Float16],
        flow2: [Float16]
    ) throws -> (offset: [Float16], mask: [Float16]) {
        guard plane > 0, raw.count == 432 * plane,
              flow1.count == 2 * plane, flow2.count == 2 * plane
        else { throw DeformConvError.invalidShape }
        let rawBuffer = try makeBuffer(raw)
        let flow1Buffer = try makeBuffer(flow1)
        let flow2Buffer = try makeBuffer(flow2)
        guard let offsetBuffer = device.makeBuffer(length: 288 * plane * 2, options: .storageModeShared),
              let maskBuffer = device.makeBuffer(length: 144 * plane * 2, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        encoder.setComputePipelineState(dcnOffsetPipeline)
        encoder.setBuffer(rawBuffer, offset: 0, index: 0)
        encoder.setBuffer(flow1Buffer, offset: 0, index: 1)
        encoder.setBuffer(flow2Buffer, offset: 0, index: 2)
        encoder.setBuffer(offsetBuffer, offset: 0, index: 3)
        encoder.setBuffer(maskBuffer, offset: 0, index: 4)
        var planeValue = UInt32(plane)
        encoder.setBytes(&planeValue, length: 4, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: 432 * plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: dcnOffsetPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let offsets = offsetBuffer.contents().bindMemory(to: Float16.self, capacity: 288 * plane)
        let masks = maskBuffer.contents().bindMemory(to: Float16.self, capacity: 144 * plane)
        return (
            Array(UnsafeBufferPointer(start: offsets, count: 288 * plane)),
            Array(UnsafeBufferPointer(start: masks, count: 144 * plane))
        )
    }

    func runTemporalPreparation(
        width: Int,
        height: Int,
        featProp: [Float16],
        featCurrent: [Float16],
        featN2: [Float16],
        flow1: [Float16],
        previousFlow: [Float16],
        hasSecondOrder: Bool
    ) throws -> TemporalPreparationResult {
        let plane = width * height
        guard width > 0, height > 0,
              featProp.count == 64 * plane,
              featCurrent.count == 64 * plane,
              featN2.count == 64 * plane,
              flow1.count == 2 * plane,
              previousFlow.count == 2 * plane
        else { throw DeformConvError.invalidShape }
        let featPropBuffer = try makeBuffer(featProp)
        let featCurrentBuffer = try makeBuffer(featCurrent)
        let featN2Buffer = try makeBuffer(featN2)
        let flow1Buffer = try makeBuffer(flow1)
        let previousFlowBuffer = try makeBuffer(previousFlow)
        guard let flow2Buffer = device.makeBuffer(length: 2 * plane * 2, options: .storageModeShared),
              let conditionBuffer = device.makeBuffer(length: 196 * plane * 2, options: .storageModeShared),
              let deformInputBuffer = device.makeBuffer(length: 128 * plane * 2, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let flowEncoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        var shape = TemporalPrepareShape(
            width: UInt32(width), height: UInt32(height),
            hasSecondOrder: hasSecondOrder ? 1 : 0
        )
        flowEncoder.setComputePipelineState(secondOrderFlowPipeline)
        flowEncoder.setBuffer(flow1Buffer, offset: 0, index: 0)
        flowEncoder.setBuffer(previousFlowBuffer, offset: 0, index: 1)
        flowEncoder.setBuffer(flow2Buffer, offset: 0, index: 2)
        flowEncoder.setBytes(&shape, length: MemoryLayout<TemporalPrepareShape>.stride, index: 3)
        flowEncoder.dispatchThreads(
            MTLSize(width: 2 * plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: secondOrderFlowPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        flowEncoder.endEncoding()
        guard let assemblyEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        assemblyEncoder.setComputePipelineState(temporalAlignmentPipeline)
        for (index, buffer) in [
            featPropBuffer, featCurrentBuffer, featN2Buffer, flow1Buffer,
            flow2Buffer, conditionBuffer, deformInputBuffer,
        ].enumerated() {
            assemblyEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        assemblyEncoder.setBytes(&shape, length: MemoryLayout<TemporalPrepareShape>.stride, index: 7)
        assemblyEncoder.dispatchThreads(
            MTLSize(width: 196 * plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: temporalAlignmentPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        assemblyEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let gpuMilliseconds = max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000
        let conditions = conditionBuffer.contents().bindMemory(to: Float16.self, capacity: 196 * plane)
        let deformInput = deformInputBuffer.contents().bindMemory(to: Float16.self, capacity: 128 * plane)
        let flow2 = flow2Buffer.contents().bindMemory(to: Float16.self, capacity: 2 * plane)
        return TemporalPreparationResult(
            gpuMilliseconds: gpuMilliseconds,
            conditions: Array(UnsafeBufferPointer(start: conditions, count: 196 * plane)),
            deformInput: Array(UnsafeBufferPointer(start: deformInput, count: 128 * plane)),
            secondOrderFlow: Array(UnsafeBufferPointer(start: flow2, count: 2 * plane))
        )
    }

    private func makeBuffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty else { throw DeformConvError.invalidShape }
        let length = values.count * MemoryLayout<T>.stride
        return try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
            else { throw DeformConvError.metalUnavailable }
            return buffer
        }
    }

    private func execute(
        pipeline: MTLComputePipelineState,
        shape: DeformConvShape,
        buffers: [MTLBuffer]
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var metalShape = MetalShape(shape)
        encoder.setBytes(&metalShape, length: MemoryLayout<MetalShape>.stride, index: 6)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: shape.outputCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        let start = ContinuousClock.now
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wall = start.duration(to: .now)
        if commandBuffer.status == .error {
            throw DeformConvError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown error")
        }
        let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuSeconds > 0 {
            return gpuSeconds * 1_000
        }
        return Double(wall.components.seconds) * 1_000
            + Double(wall.components.attoseconds) / 1.0e15
    }

    private func executeJasnaSIMD(shape: DeformConvShape, buffers: [MTLBuffer]) throws -> Double {
        guard shape.outputChannels == 64,
              shape.groups == 1,
              shape.offsetGroups > 0,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.invalidShape }
        encoder.setComputePipelineState(fp16JasnaSIMDPipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var metalShape = MetalShape(shape)
        encoder.setBytes(&metalShape, length: MemoryLayout<MetalShape>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: shape.batch * shape.outputHeight * shape.outputWidth, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: fp16JasnaSIMDPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        let start = ContinuousClock.now
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wall = start.duration(to: .now)
        if commandBuffer.status == .error {
            throw DeformConvError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown error")
        }
        let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuSeconds > 0 { return gpuSeconds * 1_000 }
        return Double(wall.components.seconds) * 1_000
            + Double(wall.components.attoseconds) / 1.0e15
    }

    private func executeJasnaTiled(shape: DeformConvShape, buffers: [MTLBuffer]) throws -> Double {
        guard shape.inputChannels == 128,
              shape.outputChannels == 64,
              shape.kernelHeight == 3,
              shape.kernelWidth == 3,
              shape.groups == 1,
              shape.offsetGroups > 0,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.invalidShape }
        encoder.setComputePipelineState(fp16JasnaTiledPipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var metalShape = MetalShape(shape)
        encoder.setBytes(&metalShape, length: MemoryLayout<MetalShape>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: shape.batch * shape.outputHeight * shape.outputWidth, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        let start = ContinuousClock.now
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wall = start.duration(to: .now)
        if commandBuffer.status == .error {
            throw DeformConvError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown error")
        }
        let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuSeconds > 0 { return gpuSeconds * 1_000 }
        return Double(wall.components.seconds) * 1_000
            + Double(wall.components.attoseconds) / 1.0e15
    }

    private func executeJasnaSIMDGroupGEMM(
        shape: DeformConvShape,
        buffers: [MTLBuffer]
    ) throws -> Double {
        guard shape.batch == 1,
              shape.inputChannels == 128,
              shape.outputChannels == 64,
              shape.kernelHeight == 3,
              shape.kernelWidth == 3,
              shape.groups == 1,
              shape.offsetGroups > 0,
              buffers.count == 8,
              let commandBuffer = queue.makeCommandBuffer(),
              let gatherEncoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DeformConvError.invalidShape }

        let rows = shape.outputHeight * shape.outputWidth
        gatherEncoder.setComputePipelineState(fp16JasnaGatherPipeline)
        gatherEncoder.setBuffer(buffers[0], offset: 0, index: 0)
        gatherEncoder.setBuffer(buffers[1], offset: 0, index: 1)
        gatherEncoder.setBuffer(buffers[2], offset: 0, index: 2)
        gatherEncoder.setBuffer(buffers[5], offset: 0, index: 3)
        var metalShape = MetalShape(shape)
        gatherEncoder.setBytes(&metalShape, length: MemoryLayout<MetalShape>.stride, index: 4)
        gatherEncoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        gatherEncoder.endEncoding()

        guard let gemmEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        gemmEncoder.setComputePipelineState(fp16JasnaSIMDGroupGEMMPipeline)
        gemmEncoder.setBuffer(buffers[5], offset: 0, index: 0)
        gemmEncoder.setBuffer(buffers[3], offset: 0, index: 1)
        gemmEncoder.setBuffer(buffers[6], offset: 0, index: 2)
        gemmEncoder.dispatchThreadgroups(
            MTLSize(width: rows / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 8 * fp16JasnaSIMDGroupGEMMPipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
        )
        gemmEncoder.endEncoding()

        guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        outputEncoder.setComputePipelineState(fp16JasnaFloatGEMMOutputPipeline)
        outputEncoder.setBuffer(buffers[6], offset: 0, index: 0)
        outputEncoder.setBuffer(buffers[4], offset: 0, index: 1)
        outputEncoder.setBuffer(buffers[7], offset: 0, index: 2)
        outputEncoder.setBytes(&metalShape, length: MemoryLayout<MetalShape>.stride, index: 3)
        outputEncoder.dispatchThreads(
            MTLSize(width: shape.outputCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: fp16JasnaFloatGEMMOutputPipeline.threadExecutionWidth, height: 1, depth: 1
            )
        )
        outputEncoder.endEncoding()

        let start = ContinuousClock.now
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wall = start.duration(to: .now)
        if commandBuffer.status == .error {
            throw DeformConvError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "unknown error"
            )
        }
        let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuSeconds > 0 { return gpuSeconds * 1_000 }
        return Double(wall.components.seconds) * 1_000
            + Double(wall.components.attoseconds) / 1.0e15
    }
}
