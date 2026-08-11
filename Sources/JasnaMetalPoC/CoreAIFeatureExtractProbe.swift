import CoreAI
import Foundation
import Metal

struct CoreAIFeatureExtractResult: Sendable {
    let specializationAndLoadMilliseconds: Double
    let statistics: BenchmarkStatistics
    let streamedStatistics: BenchmarkStatistics
    let streamedBatchSize: Int
    let maximumError: Float
    let meanError: Double
    let checksum: Double
    let outputShape: [Int]
}

private func loadFloat32File(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    guard data.count.isMultiple(of: MemoryLayout<Float>.stride) else {
        throw DeformConvError.commandFailed("invalid Float32 file: \(url.path)")
    }
    return data.withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
}

@available(macOS 27.0, *)
func probeCoreAIFeatureExtract(
    modelURL: URL,
    inputURL: URL,
    referenceURL: URL,
    iterations: Int = 7
) async throws -> CoreAIFeatureExtractResult {
    guard iterations > 0 else { throw DeformConvError.invalidShape }
    let inputValues = try loadFloat32File(inputURL)
    let reference = try loadFloat32File(referenceURL)
    guard inputValues.count == 1 * 3 * 256 * 256,
          reference.count == 1 * 64 * 64 * 64
    else {
        throw DeformConvError.commandFailed("unexpected Core AI fixture shape")
    }

    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try await AIModel(
        contentsOf: modelURL,
        options: SpecializationOptions(preferredComputeUnitKind: .gpu)
    )
    guard let function = try model.loadFunction(named: "main") else {
        throw DeformConvError.commandFailed("Core AI model has no main function")
    }
    let loadEnd = DispatchTime.now().uptimeNanoseconds

    guard let inputValueDescriptor = function.descriptor.inputDescriptor(of: "frames"),
          case .ndArray(let inputDescriptor) = inputValueDescriptor,
          inputDescriptor.shape == [1, 3, 256, 256],
          inputDescriptor.scalarType == .float32,
          let outputValueDescriptor = function.descriptor.outputDescriptor(of: "features"),
          case .ndArray(let outputDescriptor) = outputValueDescriptor,
          outputDescriptor.shape == [1, 64, 64, 64],
          outputDescriptor.scalarType == .float32
    else {
        throw DeformConvError.commandFailed("unexpected Core AI function signature")
    }

    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue(),
          let inputBuffer = device.makeBuffer(
              length: inputValues.count * MemoryLayout<Float>.stride,
              options: .storageModeShared
          ),
          let outputBuffer = device.makeBuffer(
              length: reference.count * MemoryLayout<Float>.stride,
              options: .storageModeShared
          )
    else {
        throw DeformConvError.commandFailed("unable to allocate Core AI Metal buffers")
    }
    inputValues.withUnsafeBytes { bytes in
        inputBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
    }
    let readableOutputBuffer = outputBuffer
    let stream = ComputeStream(commandQueue: commandQueue)
    let input = InferenceFunction.AsyncValue(
        unsafeBuffer: inputBuffer,
        scalarType: .float32,
        shape: [1, 3, 256, 256]
    )
    var output = InferenceFunction.AsyncMutableValue(
        unsafeBuffer: outputBuffer,
        scalarType: .float32,
        shape: [1, 64, 64, 64]
    )

    func encodeOnce() throws {
        var outputViews = InferenceFunction.AsyncMutableViews()
        outputViews.insert(&output, for: "features")
        _ = try function.encode(
            inputs: ["frames": input],
            outputViews: consume outputViews,
            to: stream
        )
    }

    func runOnce() async throws {
        try encodeOnce()
        await stream.currentWorkCompleted()
    }

    try await runOnce()
    var samples = [Double]()
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        try await runOnce()
        let end = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(end - start) / 1_000_000)
    }
    guard let statistics = BenchmarkStatistics(samples) else {
        throw DeformConvError.commandFailed("unable to summarize Core AI benchmark")
    }
    let streamedBatchSize = 30
    var streamedSamples = [Double]()
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<streamedBatchSize {
            try encodeOnce()
        }
        await stream.currentWorkCompleted()
        let end = DispatchTime.now().uptimeNanoseconds
        streamedSamples.append(
            Double(end - start) / 1_000_000 / Double(streamedBatchSize)
        )
    }
    guard let streamedStatistics = BenchmarkStatistics(streamedSamples) else {
        throw DeformConvError.commandFailed("unable to summarize Core AI stream benchmark")
    }
    let outputValues = Array(
        UnsafeBufferPointer(
            start: readableOutputBuffer.contents().bindMemory(to: Float.self, capacity: reference.count),
            count: reference.count
        )
    )

    var maximumError: Float = 0
    var errorSum = 0.0
    var checksum = 0.0
    for (expected, actual) in zip(reference, outputValues) {
        let error = abs(expected - actual)
        maximumError = max(maximumError, error)
        errorSum += Double(error)
        checksum += Double(actual)
    }
    guard maximumError <= 0.02 else {
        throw DeformConvError.commandFailed(
            "Core AI maximum error \(maximumError) exceeds 0.02"
        )
    }
    return CoreAIFeatureExtractResult(
        specializationAndLoadMilliseconds: Double(loadEnd - loadStart) / 1_000_000,
        statistics: statistics,
        streamedStatistics: streamedStatistics,
        streamedBatchSize: streamedBatchSize,
        maximumError: maximumError,
        meanError: errorSum / Double(outputValues.count),
        checksum: checksum,
        outputShape: outputDescriptor.shape
    )
}
