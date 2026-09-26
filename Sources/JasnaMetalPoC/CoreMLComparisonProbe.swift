import CoreML
import Foundation

struct CoreMLPolicyBenchmark: Sendable {
    let policy: String
    let loadMilliseconds: Double
    let firstPredictionMilliseconds: Double
    let statistics: BenchmarkStatistics
    let maximumDifferenceFromAll: Float
    let checksum: Double
}

struct CoreMLPackageBenchmark: Sendable {
    let package: String
    let compileMilliseconds: Double
    let policies: [CoreMLPolicyBenchmark]
}

private struct CoreMLComputePolicy {
    let name: String
    let units: MLComputeUnits
}

private let coreMLComparisonPackages = [
    "feature_extract",
    "spynet_level_5",
    "backbone_backward_1",
]

private func elapsedMilliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}

private func makeMultiArray(
    description: MLFeatureDescription,
    featureName: String
) throws -> MLMultiArray {
    guard description.type == .multiArray,
          let constraint = description.multiArrayConstraint
    else {
        throw DeformConvError.commandFailed(
            "Core ML feature \(featureName) is not a fixed multi-array"
        )
    }
    return try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
}

private func fillDeterministicInput(_ array: MLMultiArray) {
    switch array.dataType {
    case .float16:
        let values = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for index in 0..<array.count {
            values[index] = Float16(Float((index * 37) % 251) / 125.0 - 1.0)
        }
    case .float32:
        let values = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for index in 0..<array.count {
            values[index] = Float((index * 37) % 251) / 125.0 - 1.0
        }
    case .double:
        let values = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        for index in 0..<array.count {
            values[index] = Double((index * 37) % 251) / 125.0 - 1.0
        }
    default:
        for index in 0..<array.count {
            array[index] = NSNumber(value: Float((index * 37) % 251) / 125.0 - 1.0)
        }
    }
}

private func floatValues(_ array: MLMultiArray) -> [Float] {
    switch array.dataType {
    case .float16:
        let values = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        return (0..<array.count).map { Float(values[$0]) }
    case .float32:
        let values = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        return Array(UnsafeBufferPointer(start: values, count: array.count))
    case .double:
        let values = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        return (0..<array.count).map { Float(values[$0]) }
    default:
        return (0..<array.count).map { array[$0].floatValue }
    }
}

private func maximumDifference(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
    guard lhs.count == rhs.count else { throw DeformConvError.invalidShape }
    var maximum: Float = 0
    for index in lhs.indices {
        maximum = max(maximum, abs(lhs[index] - rhs[index]))
    }
    return maximum
}

private func benchmarkCoreMLPackage(
    packageURL: URL,
    iterations: Int
) throws -> CoreMLPackageBenchmark {
    let compileStart = DispatchTime.now().uptimeNanoseconds
    let compiledURL = try MLModel.compileModel(at: packageURL)
    let compileMilliseconds = elapsedMilliseconds(since: compileStart)
    defer { try? FileManager.default.removeItem(at: compiledURL) }

    let policies = [
        CoreMLComputePolicy(name: "all", units: .all),
        CoreMLComputePolicy(name: "cpu+gpu", units: .cpuAndGPU),
        CoreMLComputePolicy(name: "cpu+neural-engine", units: .cpuAndNeuralEngine),
    ]
    var referenceOutput: [Float]?
    var results = [CoreMLPolicyBenchmark]()

    for policy in policies {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = policy.units
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        let loadMilliseconds = elapsedMilliseconds(since: loadStart)

        guard model.modelDescription.inputDescriptionsByName.count == 1,
              let inputDescription = model.modelDescription.inputDescriptionsByName.first,
              model.modelDescription.outputDescriptionsByName.count == 1,
              let outputDescription = model.modelDescription.outputDescriptionsByName.first
        else {
            throw DeformConvError.commandFailed(
                "Core ML comparison requires one tensor input and one tensor output"
            )
        }
        let input = try makeMultiArray(
            description: inputDescription.value,
            featureName: inputDescription.key
        )
        fillDeterministicInput(input)
        let output = try makeMultiArray(
            description: outputDescription.value,
            featureName: outputDescription.key
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputDescription.key: MLFeatureValue(multiArray: input),
        ])
        let options = MLPredictionOptions()
        options.outputBackings = [outputDescription.key: output]

        func predict() throws -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try model.prediction(from: provider, options: options)
            return elapsedMilliseconds(since: start)
        }

        let firstPredictionMilliseconds = try predict()
        _ = try predict()
        var samples = [Double]()
        for _ in 0..<iterations {
            samples.append(try predict())
        }
        guard let statistics = BenchmarkStatistics(samples) else {
            throw DeformConvError.commandFailed("invalid Core ML benchmark samples")
        }
        let values = floatValues(output)
        guard values.allSatisfy(\.isFinite) else {
            throw DeformConvError.commandFailed("Core ML produced a non-finite output")
        }
        let difference: Float
        if let referenceOutput {
            difference = try maximumDifference(values, referenceOutput)
        } else {
            referenceOutput = values
            difference = 0
        }
        results.append(CoreMLPolicyBenchmark(
            policy: policy.name,
            loadMilliseconds: loadMilliseconds,
            firstPredictionMilliseconds: firstPredictionMilliseconds,
            statistics: statistics,
            maximumDifferenceFromAll: difference,
            checksum: values.reduce(0) { $0 + Double($1) }
        ))
    }

    return CoreMLPackageBenchmark(
        package: packageURL.deletingPathExtension().lastPathComponent,
        compileMilliseconds: compileMilliseconds,
        policies: results
    )
}

func benchmarkCoreMLComparison(
    coreMLDirectory: URL,
    iterations: Int = 10
) throws -> [CoreMLPackageBenchmark] {
    guard iterations > 0 else { throw DeformConvError.invalidShape }
    return try coreMLComparisonPackages.map { package in
        let url = coreMLDirectory.appendingPathComponent("\(package).mlpackage")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DeformConvError.commandFailed("missing Core ML package: \(url.path)")
        }
        return try benchmarkCoreMLPackage(packageURL: url, iterations: iterations)
    }
}

func coreMLComparisonPackageNames() -> [String] {
    coreMLComparisonPackages
}
