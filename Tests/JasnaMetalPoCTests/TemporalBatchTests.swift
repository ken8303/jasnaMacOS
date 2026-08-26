import Metal
import Testing
@testable import JasnaMetalPoC

@available(macOS 27.0, *)
@Test func batchTwoFailureRetriesBothCropsIndividuallyInOrder() throws {
    struct InjectedBatchFailure: Error {}

    let work = [0, 1].map { index in
        SideBySideRestoration.PreparedRegionRestoration(
            regionIndex: index,
            localStart: index,
            activeFrameCount: 3,
            restoredFrameOffset: 0,
            inputFrames: [[Float16(index)], [Float16(index + 1)], [Float16(index + 2)]],
            context: "test crop \(index)"
        )
    }
    var batchAttempts = 0
    var individualAttempts = 0
    var reportedFailures = 0

    let restored = try SideBySideRestoration.restorePreparedRegionsWithBatchFallback(
        work: work,
        batchRestore: { _ in
            batchAttempts += 1
            throw InjectedBatchFailure()
        },
        individualRestore: { individualWork in
            individualAttempts += 1
            return individualWork.map { item in
                SideBySideRestoration.CompletedRegionRestoration(
                    prepared: item,
                    frames: item.inputFrames,
                    gpuMilliseconds: Double(item.regionIndex + 1),
                    wallMilliseconds: Double(item.regionIndex + 2)
                )
            }
        },
        onBatchFailure: { error in
            #expect(error is InjectedBatchFailure)
            reportedFailures += 1
        }
    )

    #expect(batchAttempts == 1)
    #expect(individualAttempts == 1)
    #expect(reportedFailures == 1)
    #expect(restored.map(\.prepared.regionIndex) == [0, 1])
    #expect(restored.map(\.frames) == work.map(\.inputFrames))
}

@available(macOS 27.0, *)
@Test func batchTwoCircuitBreakerDisablesOnlyOnce() {
    let breaker = RestorationBatchCircuitBreaker()

    #expect(!breaker.isDisabled)
    #expect(breaker.disable())
    #expect(breaker.isDisabled)
    #expect(!breaker.disable())
}

@Test func retainedProductionGraphCoversNormalAndWarmupWindowsOnly() {
    func eligible(_ frames: Int) -> Bool {
        productionGraphReuseEligible(
            frameCount: frames,
            warmupCount: 0,
            measurementCount: 1,
            collectDiagnostics: false,
            hasFlowOracle: false,
            hasStagedPropagation: false,
            hasStagedRestoration: false
        )
    }

    #expect(eligible(30))
    #expect(eligible(35))
    #expect(!eligible(29))
    #expect(!eligible(34))
    #expect(!eligible(36))
    #expect(!productionGraphReuseEligible(
        frameCount: 35,
        warmupCount: 1,
        measurementCount: 1,
        collectDiagnostics: false,
        hasFlowOracle: false,
        hasStagedPropagation: false,
        hasStagedRestoration: false
    ))
}

@available(macOS 27.0, *)
@Test func modelCropReuseSummaryRequiresExactGeometryAndTemporalSchedule() {
    let regions = [
        MosaicRegion(
            startFrame: 0, endFrame: 30,
            x: 0, y: 0, width: 100, height: 100, confidence: 1
        ),
        MosaicRegion(
            startFrame: 0, endFrame: 30,
            x: 0, y: 0, width: 100, height: 100, confidence: 0.8,
            blendX: 10, blendY: 10, blendWidth: 80, blendHeight: 80
        ),
        MosaicRegion(
            startFrame: 0, endFrame: 30,
            x: 10, y: 10, width: 90, height: 90, confidence: 1
        ),
        MosaicRegion(
            startFrame: 5, endFrame: 30,
            x: 0, y: 0, width: 50, height: 50, confidence: 1
        ),
    ]

    let summary = SideBySideRestoration.modelCropReuseSummary(
        regions: regions,
        windowStartFrame: 0,
        outputCount: 30,
        temporalWarmupFrames: 0
    )

    #expect(summary.cropCount == 4)
    #expect(summary.uniqueExactCropCount == 3)
    #expect(summary.exactDuplicateCount == 1)
    #expect(summary.highOverlapPairCount == 2)
    #expect(summary.containedPairCount == 2)
}

@available(macOS 27.0, *)
@Test func malformedBatchOutputThrowsBeforeTensorSlicing() throws {
    let work = [0, 1].map { index in
        SideBySideRestoration.PreparedRegionRestoration(
            regionIndex: index,
            localStart: 0,
            activeFrameCount: 1,
            restoredFrameOffset: 0,
            inputFrames: [[Float16(index)]],
            context: "malformed crop \(index)"
        )
    }

    #expect(throws: (any Error).self) {
        try SideBySideRestoration.splitBatchedRegionFrames(
            [[Float16(1)]],
            work: work,
            gpuMilliseconds: 1,
            wallMilliseconds: 1
        )
    }
}

@available(macOS 27.0, *)
@Test func batchOptimizedRegionsGroupCompatibleTemporalLengthsStably() {
    let ranges = [(0, 30), (4, 20), (0, 30), (8, 20), (0, 30)]
    let regions = ranges.enumerated().map { index, range in
        MosaicRegion(
            startFrame: range.0,
            endFrame: range.1,
            x: index,
            y: 0,
            width: 256,
            height: 256,
            confidence: 1
        )
    }

    let ordered = SideBySideRestoration.batchOptimizedRegions(
        regions,
        windowStartFrame: 0,
        outputCount: 30,
        batch2Enabled: true
    )

    #expect(ordered.map(\.x) == [0, 2, 3, 1, 4])
    #expect(
        SideBySideRestoration.batchOptimizedRegions(
            regions,
            windowStartFrame: 0,
            outputCount: 30,
            batch2Enabled: false
        ).map(\.x) == [0, 1, 2, 3, 4]
    )
}

@available(macOS 27.0, *)
@Test func batchOptimizedRegionsDoNotSplitAnEvenGroupAfterAnOddGroup() {
    let ranges = [(0, 10), (0, 20), (0, 20)]
    let regions = ranges.enumerated().map { index, range in
        MosaicRegion(
            startFrame: range.0,
            endFrame: range.1,
            x: index,
            y: 0,
            width: 256,
            height: 256,
            confidence: 1
        )
    }

    let ordered = SideBySideRestoration.batchOptimizedRegions(
        regions,
        windowStartFrame: 0,
        outputCount: 30,
        batch2Enabled: true
    )

    #expect(ordered.map(\.x) == [1, 2, 0])
}

@Test func temporalPreparationBatchMatchesIndependentRuns() throws {
    guard #available(macOS 27.0, *), MTLCreateSystemDefaultDevice() != nil else { return }
    let runner = try MetalDeformConv()
    let width = 2
    let height = 2
    let plane = width * height

    func values(count: Int, scale: Float, offset: Float) -> [Float16] {
        (0..<count).map { Float16(Float($0 % 11) * scale + offset) }
    }
    let first = (
        prop: values(count: 64 * plane, scale: 0.01, offset: -0.2),
        current: values(count: 64 * plane, scale: 0.02, offset: -0.1),
        n2: values(count: 64 * plane, scale: -0.01, offset: 0.3),
        flow: values(count: 2 * plane, scale: 0.01, offset: -0.02),
        previous: values(count: 2 * plane, scale: -0.015, offset: 0.04)
    )
    let second = (
        prop: values(count: 64 * plane, scale: -0.02, offset: 0.4),
        current: values(count: 64 * plane, scale: 0.015, offset: -0.3),
        n2: values(count: 64 * plane, scale: 0.005, offset: 0.1),
        flow: values(count: 2 * plane, scale: -0.02, offset: 0.03),
        previous: values(count: 2 * plane, scale: 0.01, offset: -0.05)
    )
    let firstResult = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: first.prop, featCurrent: first.current, featN2: first.n2,
        flow1: first.flow, previousFlow: first.previous, hasSecondOrder: true
    )
    let secondResult = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: second.prop, featCurrent: second.current, featN2: second.n2,
        flow1: second.flow, previousFlow: second.previous, hasSecondOrder: true
    )
    let batched = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: first.prop + second.prop,
        featCurrent: first.current + second.current,
        featN2: first.n2 + second.n2,
        flow1: first.flow + second.flow,
        previousFlow: first.previous + second.previous,
        hasSecondOrder: true,
        batch: 2
    )
    #expect(batched.conditions == firstResult.conditions + secondResult.conditions)
    #expect(batched.deformInput == firstResult.deformInput + secondResult.deformInput)
    #expect(
        batched.secondOrderFlow
            == firstResult.secondOrderFlow + secondResult.secondOrderFlow
    )
}

@Test func dcnOffsetTransformBatchMatchesIndependentRuns() throws {
    guard #available(macOS 27.0, *), MTLCreateSystemDefaultDevice() != nil else { return }
    let runner = try MetalDeformConv()
    let plane = 4
    func values(count: Int, scale: Float, offset: Float) -> [Float16] {
        (0..<count).map { Float16(Float($0 % 17) * scale + offset) }
    }
    let firstRaw = values(count: 432 * plane, scale: 0.002, offset: -0.1)
    let secondRaw = values(count: 432 * plane, scale: -0.003, offset: 0.2)
    let firstFlow1 = values(count: 2 * plane, scale: 0.01, offset: -0.03)
    let secondFlow1 = values(count: 2 * plane, scale: -0.02, offset: 0.05)
    let firstFlow2 = values(count: 2 * plane, scale: -0.01, offset: 0.04)
    let secondFlow2 = values(count: 2 * plane, scale: 0.015, offset: -0.02)
    let first = try runner.runDCNOffsetTransform(
        plane: plane, raw: firstRaw, flow1: firstFlow1, flow2: firstFlow2
    )
    let second = try runner.runDCNOffsetTransform(
        plane: plane, raw: secondRaw, flow1: secondFlow1, flow2: secondFlow2
    )
    let batched = try runner.runDCNOffsetTransform(
        plane: plane,
        raw: firstRaw + secondRaw,
        flow1: firstFlow1 + secondFlow1,
        flow2: firstFlow2 + secondFlow2,
        batch: 2
    )
    #expect(batched.offset == first.offset + second.offset)
    #expect(batched.mask == first.mask + second.mask)
}
