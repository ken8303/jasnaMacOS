import Foundation

struct TemporalPreparationProbeResult {
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let iterations: Int
    let maximumMagnitude: Float
    let repeatMaximumError: Float
    let firstOrderZeroMaximumError: Float
    let checksum: Double
}

func verifyTemporalPreparation(
    runner: MetalDeformConv,
    backwardFlow: [Float16],
    forwardFlow: [Float16]
) throws -> TemporalPreparationProbeResult {
    let width = 64, height = 64, plane = width * height
    guard backwardFlow.count == 2 * plane, forwardFlow.count == 2 * plane else {
        throw DeformConvError.invalidShape
    }
    func features(seed: Int) -> [Float16] {
        (0..<(64 * plane)).map { index in
            Float16((Float((index * 37 + seed) % 509) / 508 - 0.5) * 0.5)
        }
    }
    let featProp = features(seed: 17)
    let featCurrent = features(seed: 31)
    let featN2 = features(seed: 47)
    let zeroFeature = [Float16](repeating: 0, count: 64 * plane)

    func execute(secondOrder: Bool) throws -> (Double, [TemporalPreparationResult]) {
        let backward = try runner.runTemporalPreparation(
            width: width, height: height,
            featProp: featProp, featCurrent: featCurrent,
            featN2: secondOrder ? featN2 : zeroFeature,
            flow1: backwardFlow, previousFlow: backwardFlow,
            hasSecondOrder: secondOrder
        )
        let forward = try runner.runTemporalPreparation(
            width: width, height: height,
            featProp: featProp, featCurrent: featCurrent,
            featN2: secondOrder ? featN2 : zeroFeature,
            flow1: forwardFlow, previousFlow: forwardFlow,
            hasSecondOrder: secondOrder
        )
        return (backward.gpuMilliseconds + forward.gpuMilliseconds, [backward, forward])
    }

    _ = try execute(secondOrder: true)
    _ = try execute(secondOrder: true)
    let (firstTime, first) = try execute(secondOrder: true)
    var samples = [firstTime]
    var last = first
    for _ in 1..<7 {
        let (milliseconds, output) = try execute(secondOrder: true)
        samples.append(milliseconds)
        last = output
    }
    let (_, firstOrderOnly) = try execute(secondOrder: false)
    samples.sort()

    var maximumMagnitude: Float = 0
    var repeatMaximumError: Float = 0
    var firstOrderZeroMaximumError: Float = 0
    var checksum = 0.0
    for direction in 0..<2 {
        let current = last[direction]
        for index in current.conditions.indices {
            let value = Float(current.conditions[index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("temporal preparation produced a non-finite condition")
            }
            maximumMagnitude = max(maximumMagnitude, abs(value))
            repeatMaximumError = max(
                repeatMaximumError,
                abs(Float(first[direction].conditions[index]) - value)
            )
            if index.isMultiple(of: 4099) { checksum += Double(value) }
        }
        for value in firstOrderOnly[direction].secondOrderFlow {
            firstOrderZeroMaximumError = max(firstOrderZeroMaximumError, abs(Float(value)))
        }
    }
    guard maximumMagnitude > 0.001,
          repeatMaximumError <= 0.001,
          firstOrderZeroMaximumError == 0
    else {
        throw DeformConvError.commandFailed(
            "temporal preparation validation failed (max=\(maximumMagnitude), "
                + "repeat=\(repeatMaximumError), zero=\(firstOrderZeroMaximumError))"
        )
    }
    return TemporalPreparationProbeResult(
        medianMilliseconds: samples[samples.count / 2],
        minimumMilliseconds: samples[0],
        maximumMilliseconds: samples[samples.count - 1],
        iterations: samples.count,
        maximumMagnitude: maximumMagnitude,
        repeatMaximumError: repeatMaximumError,
        firstOrderZeroMaximumError: firstOrderZeroMaximumError,
        checksum: checksum
    )
}
