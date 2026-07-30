import Foundation

struct FullModelOracleMetrics {
    let elementCount: Int
    let maximumAbsoluteError: Float
    let meanAbsoluteError: Double
    let percentile99AbsoluteError: Float
    let rootMeanSquaredError: Double
    let psnr: Double
    let fp16MaximumError: Float
}

func calculateFullModelOracleMetrics(
    restored: [Float16],
    fp32Oracle: [Float],
    fp16Oracle: [Float16]
) throws -> FullModelOracleMetrics {
    let elementCount = restored.count
    guard elementCount > 0,
          fp32Oracle.count == elementCount,
          fp16Oracle.count == elementCount
    else { throw DeformConvError.invalidShape }

    var errors = [Float]()
    errors.reserveCapacity(elementCount)
    var absoluteSum: Double = 0
    var squaredSum: Double = 0
    var maximumError: Float = 0
    var fp16MaximumError: Float = 0
    for index in 0..<elementCount {
        let value = Float(restored[index])
        let error = abs(value - fp32Oracle[index])
        guard value.isFinite, error.isFinite else {
            throw DeformConvError.commandFailed("non-finite full-model comparison value")
        }
        errors.append(error)
        maximumError = max(maximumError, error)
        absoluteSum += Double(error)
        squaredSum += Double(error) * Double(error)
        fp16MaximumError = max(
            fp16MaximumError, abs(value - Float(fp16Oracle[index]))
        )
    }
    errors.sort()
    let p99Index = Int(Double(errors.count - 1) * 0.99)
    let mean = absoluteSum / Double(elementCount)
    let mse = squaredSum / Double(elementCount)
    let rmse = sqrt(mse)
    let psnr = mse == 0 ? .infinity : 10 * log10(1 / mse)
    return FullModelOracleMetrics(
        elementCount: elementCount,
        maximumAbsoluteError: maximumError,
        meanAbsoluteError: mean,
        percentile99AbsoluteError: errors[p99Index],
        rootMeanSquaredError: rmse,
        psnr: psnr,
        fp16MaximumError: fp16MaximumError
    )
}

func compareFullModelOracle(
    restoredFrames: [[Float16]],
    oracleURL: URL
) throws -> FullModelOracleMetrics {
    let frameElements = 3 * 256 * 256
    let elementCount = restoredFrames.count * frameElements
    guard restoredFrames.count >= 3,
          restoredFrames.allSatisfy({ $0.count == frameElements })
    else { throw DeformConvError.invalidShape }

    let fp32Data = try Data(contentsOf: oracleURL.appendingPathComponent("restored.f32"))
    let fp16Data = try Data(contentsOf: oracleURL.appendingPathComponent("restored.f16"))
    guard fp32Data.count == elementCount * MemoryLayout<Float>.size,
          fp16Data.count == elementCount * MemoryLayout<Float16>.size
    else {
        throw DeformConvError.commandFailed("invalid full-model oracle size")
    }
    let fp32Oracle: [Float] = fp32Data.withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float.self))
    }
    let fp16Oracle: [Float16] = fp16Data.withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: UInt16.self)).map(Float16.init(bitPattern:))
    }

    return try calculateFullModelOracleMetrics(
        restored: restoredFrames.flatMap { $0 },
        fp32Oracle: fp32Oracle,
        fp16Oracle: fp16Oracle
    )
}
