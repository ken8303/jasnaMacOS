import CoreML
import Foundation
import Metal

struct CoreMLSPyNetInteropResult: Sendable {
    let compileAndLoadMilliseconds: Double
    let statistics: BenchmarkStatistics
    let repeatMaximumError: Float
    let metalMLMaximumError: Float
    let oracleMaximumError: Float
    let backwardChecksum: Double
    let forwardChecksum: Double
}

private struct CoreMLSPyNetPrepareShape {
    var width: UInt32
    var height: UInt32
    var sourceFlowWidth: UInt32
    var sourceFlowHeight: UInt32
    var firstLevel: UInt32
}

private struct CoreMLSPyNetLevel {
    let size: Int
    let model: MLModel
    let featureBuffers: [MTLBuffer]
    let residualBuffers: [MTLBuffer]
    let baseFlowBuffers: [MTLBuffer]
    let outputFlowBuffers: [MTLBuffer]
    let shapeBuffer: MTLBuffer
    let countBuffer: MTLBuffer
    let providers: [MLDictionaryFeatureProvider]
    let predictionOptions: [MLPredictionOptions]
    let arrays: [MLMultiArray]
}

private func coreMLSPyNetMilliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}

private func coreMLSPyNetBuffer(device: MTLDevice, elements: Int) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
        throw DeformConvError.metalUnavailable
    }
    return buffer
}

private func coreMLSPyNetConstant<T>(device: MTLDevice, value: inout T) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: 256, options: .storageModeShared) else {
        throw DeformConvError.metalUnavailable
    }
    withUnsafeBytes(of: &value) { bytes in
        buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
    }
    return buffer
}

private func coreMLSPyNetArray(
    buffer: MTLBuffer,
    channels: Int,
    size: Int
) throws -> MLMultiArray {
    try MLMultiArray(
        dataPointer: buffer.contents(),
        shape: [1, channels, size, size].map(NSNumber.init(value:)),
        dataType: .float16,
        strides: [channels * size * size, size * size, size, 1].map(NSNumber.init(value:)),
        deallocator: { _ in }
    )
}

private func completeCoreMLSPyNetCommand(_ commandBuffer: MTLCommandBuffer) throws {
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }
}

private func coreMLSPyNetValues(_ buffer: MTLBuffer, count: Int) -> [Float16] {
    let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

private func coreMLSPyNetMaximumDifference(_ lhs: [Float16], _ rhs: [Float16]) throws -> Float {
    guard lhs.count == rhs.count else { throw DeformConvError.invalidShape }
    var maximum: Float = 0
    for index in lhs.indices {
        maximum = max(maximum, abs(Float(lhs[index]) - Float(rhs[index])))
    }
    return maximum
}

private func coreMLSPyNetOracle(url: URL, count: Int) throws -> [Float16] {
    let data = try Data(contentsOf: url)
    guard data.count == count * 2 else {
        throw DeformConvError.commandFailed("invalid SPyNet oracle size")
    }
    return data.withUnsafeBytes {
        Array($0.bindMemory(to: UInt16.self)).map(Float16.init(bitPattern:))
    }
}

func benchmarkCoreMLSPyNetInterop(
    device: MTLDevice,
    coreMLDirectory: URL,
    oracleURL: URL,
    metalMLBackward: [Float16],
    metalMLForward: [Float16],
    iterations: Int = 7
) throws -> CoreMLSPyNetInteropResult {
    let sizes = [2, 4, 8, 16, 32, 64]
    let frameElements = 3 * 64 * 64
    guard iterations > 0,
          metalMLBackward.count == 2 * 64 * 64,
          metalMLForward.count == 2 * 64 * 64
    else { throw DeformConvError.invalidShape }

    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let pyramidFunction = library.makeFunction(name: "spynet_build_pyramid_pair_fp16"),
          let prepareFunction = library.makeFunction(name: "spynet_prepare_fp16"),
          let addFunction = library.makeFunction(name: "spynet_add_flow_fp16"),
          let commandQueue = device.makeCommandQueue()
    else { throw DeformConvError.shaderResourceMissing }
    let pyramidPipeline = try device.makeComputePipelineState(function: pyramidFunction)
    let preparePipeline = try device.makeComputePipelineState(function: prepareFunction)
    let addPipeline = try device.makeComputePipelineState(function: addFunction)

    let referenceFrame = try coreMLSPyNetBuffer(device: device, elements: frameElements)
    let supportFrame = try coreMLSPyNetBuffer(device: device, elements: frameElements)
    let referencePyramid = try sizes.map {
        try coreMLSPyNetBuffer(device: device, elements: 3 * $0 * $0)
    }
    let supportPyramid = try sizes.map {
        try coreMLSPyNetBuffer(device: device, elements: 3 * $0 * $0)
    }
    let zeroFlow = try coreMLSPyNetBuffer(device: device, elements: 2 * 2 * 2)

    let loadStart = DispatchTime.now().uptimeNanoseconds
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    var compiledURLs = [URL]()
    var models = [MLModel]()
    for level in sizes.indices {
        let packageURL = coreMLDirectory.appendingPathComponent(
            "spynet_level_\(level).mlpackage"
        )
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw DeformConvError.commandFailed("missing Core ML package: \(packageURL.path)")
        }
        let compiledURL = try MLModel.compileModel(at: packageURL)
        compiledURLs.append(compiledURL)
        models.append(try MLModel(contentsOf: compiledURL, configuration: configuration))
    }
    defer {
        for url in compiledURLs { try? FileManager.default.removeItem(at: url) }
    }
    let compileAndLoadMilliseconds = coreMLSPyNetMilliseconds(since: loadStart)

    var levels = [CoreMLSPyNetLevel]()
    for (level, size) in sizes.enumerated() {
        let model = models[level]
        guard model.modelDescription.inputDescriptionsByName["features"]?.type == .multiArray,
              model.modelDescription.outputDescriptionsByName["output"]?.type == .multiArray
        else {
            throw DeformConvError.commandFailed("unexpected SPyNet Core ML interface")
        }
        let featureBuffers = try (0..<2).map { _ in
            try coreMLSPyNetBuffer(device: device, elements: 8 * size * size)
        }
        let residualBuffers = try (0..<2).map { _ in
            try coreMLSPyNetBuffer(device: device, elements: 2 * size * size)
        }
        let baseFlowBuffers = try (0..<2).map { _ in
            try coreMLSPyNetBuffer(device: device, elements: 2 * size * size)
        }
        let outputFlowBuffers = try (0..<2).map { _ in
            try coreMLSPyNetBuffer(device: device, elements: 2 * size * size)
        }
        var shape = CoreMLSPyNetPrepareShape(
            width: UInt32(size),
            height: UInt32(size),
            sourceFlowWidth: UInt32(level == 0 ? 2 : sizes[level - 1]),
            sourceFlowHeight: UInt32(level == 0 ? 2 : sizes[level - 1]),
            firstLevel: level == 0 ? 1 : 0
        )
        var count = UInt32(2 * size * size)
        let shapeBuffer = try coreMLSPyNetConstant(device: device, value: &shape)
        let countBuffer = try coreMLSPyNetConstant(device: device, value: &count)
        var providers = [MLDictionaryFeatureProvider]()
        var predictionOptions = [MLPredictionOptions]()
        var heldArrays = [MLMultiArray]()
        for direction in 0..<2 {
            let input = try coreMLSPyNetArray(
                buffer: featureBuffers[direction], channels: 8, size: size
            )
            let output = try coreMLSPyNetArray(
                buffer: residualBuffers[direction], channels: 2, size: size
            )
            providers.append(try MLDictionaryFeatureProvider(dictionary: [
                "features": MLFeatureValue(multiArray: input),
            ]))
            let options = MLPredictionOptions()
            options.outputBackings = ["output": output]
            predictionOptions.append(options)
            heldArrays += [input, output]
        }
        levels.append(CoreMLSPyNetLevel(
            size: size,
            model: model,
            featureBuffers: featureBuffers,
            residualBuffers: residualBuffers,
            baseFlowBuffers: baseFlowBuffers,
            outputFlowBuffers: outputFlowBuffers,
            shapeBuffer: shapeBuffer,
            countBuffer: countBuffer,
            providers: providers,
            predictionOptions: predictionOptions,
            arrays: heldArrays
        ))
    }

    let allBuffers = [referenceFrame, supportFrame, zeroFlow]
        + referencePyramid + supportPyramid
        + levels.flatMap {
            $0.featureBuffers + $0.residualBuffers + $0.baseFlowBuffers + $0.outputFlowBuffers
        }
    func initializeBuffers() {
        for buffer in allBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        let reference = referenceFrame.contents().bindMemory(
            to: Float16.self, capacity: frameElements
        )
        let support = supportFrame.contents().bindMemory(
            to: Float16.self, capacity: frameElements
        )
        for index in 0..<frameElements {
            reference[index] = Float16(Float((index * 29 + 17) % 1021) / 1020)
            support[index] = Float16(Float((index * 43 + 31) % 1019) / 1018)
        }
    }

    func encodePrepare(
        _ encoder: MTLComputeCommandEncoder,
        level: Int
    ) {
        let runtime = levels[level]
        for direction in 0..<2 {
            let previousFlow = level == 0
                ? zeroFlow : levels[level - 1].outputFlowBuffers[direction]
            let reference = direction == 0
                ? referencePyramid[level] : supportPyramid[level]
            let support = direction == 0
                ? supportPyramid[level] : referencePyramid[level]
            encoder.setComputePipelineState(preparePipeline)
            encoder.setBuffer(reference, offset: 0, index: 0)
            encoder.setBuffer(support, offset: 0, index: 1)
            encoder.setBuffer(previousFlow, offset: 0, index: 2)
            encoder.setBuffer(runtime.featureBuffers[direction], offset: 0, index: 3)
            encoder.setBuffer(runtime.baseFlowBuffers[direction], offset: 0, index: 4)
            encoder.setBuffer(runtime.shapeBuffer, offset: 0, index: 5)
            encoder.dispatchThreads(
                MTLSize(width: runtime.size * runtime.size, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: preparePipeline.threadExecutionWidth, height: 1, depth: 1
                )
            )
        }
    }

    func encodeAdd(_ encoder: MTLComputeCommandEncoder, level: Int) {
        let runtime = levels[level]
        for direction in 0..<2 {
            encoder.setComputePipelineState(addPipeline)
            encoder.setBuffer(runtime.baseFlowBuffers[direction], offset: 0, index: 0)
            encoder.setBuffer(runtime.residualBuffers[direction], offset: 0, index: 1)
            encoder.setBuffer(runtime.outputFlowBuffers[direction], offset: 0, index: 2)
            encoder.setBuffer(runtime.countBuffer, offset: 0, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: 2 * runtime.size * runtime.size, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: addPipeline.threadExecutionWidth, height: 1, depth: 1
                )
            )
        }
    }

    func execute() throws -> (wall: Double, outputs: [[Float16]]) {
        initializeBuffers()
        let start = DispatchTime.now().uptimeNanoseconds
        guard let firstCommand = commandQueue.makeCommandBuffer(),
              let pyramid = firstCommand.makeComputeCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        pyramid.setComputePipelineState(pyramidPipeline)
        let pyramidBuffers = [referenceFrame, supportFrame]
            + sizes.indices.flatMap { [referencePyramid[$0], supportPyramid[$0]] }
        for (index, buffer) in pyramidBuffers.enumerated() {
            pyramid.setBuffer(buffer, offset: 0, index: index)
        }
        pyramid.dispatchThreads(
            MTLSize(width: 3 * 64 * 64, height: 6, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pyramidPipeline.threadExecutionWidth * 4, 256), height: 1, depth: 1
            )
        )
        pyramid.endEncoding()
        guard let firstPrepare = firstCommand.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        encodePrepare(firstPrepare, level: 0)
        firstPrepare.endEncoding()
        try completeCoreMLSPyNetCommand(firstCommand)

        for level in levels.indices {
            let runtime = levels[level]
            for direction in 0..<2 {
                _ = try runtime.model.prediction(
                    from: runtime.providers[direction],
                    options: runtime.predictionOptions[direction]
                )
            }
            guard let command = commandQueue.makeCommandBuffer(),
                  let add = command.makeComputeCommandEncoder()
            else { throw DeformConvError.metalUnavailable }
            encodeAdd(add, level: level)
            add.endEncoding()
            if level + 1 < levels.count {
                guard let prepare = command.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                encodePrepare(prepare, level: level + 1)
                prepare.endEncoding()
            }
            try completeCoreMLSPyNetCommand(command)
        }
        let wall = coreMLSPyNetMilliseconds(since: start)
        let outputCount = 2 * 64 * 64
        return (
            wall,
            [
                coreMLSPyNetValues(levels[5].outputFlowBuffers[0], count: outputCount),
                coreMLSPyNetValues(levels[5].outputFlowBuffers[1], count: outputCount),
            ]
        )
    }

    _ = try execute()
    _ = try execute()
    let first = try execute()
    var samples = [first.wall]
    var last = first.outputs
    for _ in 1..<iterations {
        let run = try execute()
        samples.append(run.wall)
        last = run.outputs
    }
    guard let statistics = BenchmarkStatistics(samples) else {
        throw DeformConvError.commandFailed("invalid Core ML SPyNet samples")
    }
    let backwardOracle = try coreMLSPyNetOracle(
        url: oracleURL.appendingPathComponent("backward.f16"), count: last[0].count
    )
    let forwardOracle = try coreMLSPyNetOracle(
        url: oracleURL.appendingPathComponent("forward.f16"), count: last[1].count
    )
    let repeatError = max(
        try coreMLSPyNetMaximumDifference(first.outputs[0], last[0]),
        try coreMLSPyNetMaximumDifference(first.outputs[1], last[1])
    )
    let metalError = max(
        try coreMLSPyNetMaximumDifference(metalMLBackward, last[0]),
        try coreMLSPyNetMaximumDifference(metalMLForward, last[1])
    )
    let oracleError = max(
        try coreMLSPyNetMaximumDifference(backwardOracle, last[0]),
        try coreMLSPyNetMaximumDifference(forwardOracle, last[1])
    )
    guard repeatError <= 0.001, metalError <= 0.05, oracleError <= 0.05 else {
        throw DeformConvError.commandFailed(
            "Core ML SPyNet validation failed "
                + "(repeat=\(repeatError), metal=\(metalError), oracle=\(oracleError))"
        )
    }
    let checksums = last.map { values in
        values.indices.filter { $0.isMultiple(of: 257) }.reduce(0.0) {
            $0 + Double(values[$1])
        }
    }
    _ = levels.flatMap(\.arrays)
    return CoreMLSPyNetInteropResult(
        compileAndLoadMilliseconds: compileAndLoadMilliseconds,
        statistics: statistics,
        repeatMaximumError: repeatError,
        metalMLMaximumError: metalError,
        oracleMaximumError: oracleError,
        backwardChecksum: checksums[0],
        forwardChecksum: checksums[1]
    )
}
