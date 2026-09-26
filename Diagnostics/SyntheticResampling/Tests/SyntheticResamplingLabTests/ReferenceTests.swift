import Foundation
import Testing
@testable import SyntheticResamplingLab

@Test func knownCentreAndClampedCorners() throws {
    let source = Raster(width: 2, height: 2, bytes: [
        0, 0, 0, 255, 255, 0, 0, 255,
        0, 255, 0, 255, 255, 255, 255, 255,
    ])
    let coordinates: [SIMD2<Float>] = [SIMD2(0.5, 0.5), SIMD2(-100, -100), SIMD2(100, 100)]
    let expected: [SIMD4<Float>] = [SIMD4(0.5, 0.5, 0.25, 1), SIMD4(0, 0, 0, 1), SIMD4(1, 1, 1, 1)]
    #expect(try maximumError(referenceSample(source, coordinates: coordinates), expected) == 0)
    #expect(try maximumError(cpuSample(source, coordinates: coordinates), expected) == 0)
}

@Test func identityPreservesDistinctChannels() throws {
    let source = Raster.generated(width: 31, height: 19, frame: 3)
    let coordinates = (0..<(31 * 19)).map { SIMD2<Float>(Float($0 % 31), Float($0 / 31)) }
    let actual = cpuSample(source, coordinates: coordinates)
    for index in actual.indices {
        for channel in 0..<4 { #expect(abs(actual[index][channel] - Float(source.bytes[index * 4 + channel]) / 255) < 1e-7) }
    }
}

@Test func motionAndFractionalCoordinatesAreNotStatic() {
    let first = Raster.generated(width: 257, height: 193, frame: 0)
    let second = Raster.generated(width: 257, height: 193, frame: 1)
    #expect(first.bytes != second.bytes)
    let a = movingCoordinates(source: first, width: 96, height: 80, frame: 0)
    let b = movingCoordinates(source: second, width: 96, height: 80, frame: 1)
    #expect(a != b)
    #expect(a.contains { $0.x < 0 || $0.y < 0 })
}

@Test func errorGateRejectsNonfiniteAndMismatchedData() {
    #expect(throws: LabError.self) { try maximumError([], []) }
    #expect(throws: LabError.self) { try maximumError([SIMD4<Float>(repeating: .nan)], [.zero]) }
}

@Test func timingReportsDispersionAndRejectsBadSamples() throws {
    let summary = try TimingSummary([10, 0, 5, 15, 20])
    #expect(summary.samples == 5)
    #expect(summary.medianMS == 10)
    #expect(summary.p10MS == 2)
    #expect(summary.p90MS == 18)
    #expect(throws: LabError.self) { try TimingSummary([]) }
    #expect(throws: LabError.self) { try TimingSummary([.nan]) }
}

@Test func allModesHaveBalancedMeasuredPositions() {
    let modeCount = SamplingMode.allCases.count
    #expect(modeCount == 6)
    #expect(24 % modeCount == 0)
    for mode in SamplingMode.allCases {
        for position in 0..<modeCount {
            #expect((4..<28).filter { samplingOrder(round: $0)[position] == mode }.count == 24 / modeCount)
        }
    }
    for round in 0..<28 { #expect(Set(samplingOrder(round: round)) == Set(SamplingMode.allCases)) }
}

@Test func amortizedTimingsDivideByActualOutputCount() throws {
    let result = try perImageTimings([4, 8, 12], imagesPerRound: 4)
    #expect(result.rawMS == [1, 2, 3])
    #expect(result.samples == 3)
    #expect(result.medianMS == 2)
    #expect(throws: LabError.self) { try perImageTimings([4], imagesPerRound: 0) }
    #expect(throws: LabError.self) { try perImageTimings([4], imagesPerRound: -1) }
    #expect(throws: LabError.self) { try perImageTimings([.nan], imagesPerRound: 4) }
}

@Test func reusePlansCountUploadsSubmissionsAndEveryOutput() throws {
    for uses in [1, 2, 4, 8] {
        let plan = try ReusePlan(usesPerInput: uses)
        #expect(plan.outputsPerTrial == 4 * uses)
        #expect(plan.uploads(for: .cpu) == 0)
        #expect(plan.uploads(for: .upload) == 4 * uses)
        #expect(plan.uploads(for: .resident) == 4)
        #expect(plan.uploads(for: .grouped) == 4)
        #expect(plan.submissions(for: .cpu) == 0)
        #expect(plan.submissions(for: .upload) == 4 * uses)
        #expect(plan.submissions(for: .resident) == 4 * uses)
        #expect(plan.submissions(for: .grouped) == uses)
        for mode in [SamplingMode.cpuSIMD, .cpuParallel] {
            #expect(mode.isCPU)
            #expect(plan.uploads(for: mode) == 0)
            #expect(plan.submissions(for: mode) == 0)
        }
        let timing = try perImageTimings([Double(plan.outputsPerTrial) * 2], imagesPerRound: plan.outputsPerTrial)
        #expect(timing.medianMS == 2)
    }
}

@Test func reusePlansRejectUnsupportedOrEmptyWorkloads() {
    for uses in [-1, 0, 3, 16, Int.max] {
        #expect(throws: LabError.self) { try ReusePlan(usesPerInput: uses) }
    }
}

@Test func checksumConsumesEveryOutputAndRejectsInvalidShapes() throws {
    var frames: [[SIMD4<Float>]] = (1...4).map { [SIMD4(Float($0), 0, 0, 1)] }
    #expect(try consumeFourOutputs(frames, pixelsPerFrame: 1) == 10)
    frames[3][0].x = 10
    #expect(try consumeFourOutputs(frames, pixelsPerFrame: 1) == 16)
    #expect(throws: LabError.self) { try consumeFourOutputs([], pixelsPerFrame: 1) }
    #expect(throws: LabError.self) { try consumeFourOutputs(frames, pixelsPerFrame: 0) }
    #expect(throws: LabError.self) { try consumeFourOutputs(frames, pixelsPerFrame: 2) }
    frames[3][0].x = .nan
    #expect(throws: LabError.self) { try consumeFourOutputs(frames, pixelsPerFrame: 1) }
}
