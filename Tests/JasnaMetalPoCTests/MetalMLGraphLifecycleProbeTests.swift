import Testing
@testable import JasnaMetalPoC

@Test
func graphLifecyclePlanMatchesProductionShapeTransition() {
    #expect(MetalMLGraphLifecycleProbe.plan == [
        MetalMLGraphLifecycleStep(frameCount: 30, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 30, expectedCacheHit: true),
        MetalMLGraphLifecycleStep(frameCount: 17, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 17, expectedCacheHit: true),
        MetalMLGraphLifecycleStep(frameCount: 35, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 35, expectedCacheHit: true),
    ])
}

@Test
func graphLifecycleRepetitionCountIsBoundedAndDefaultsToEight() {
    #expect(MetalMLGraphLifecycleProbe.repetitionCount(environment: [:]) == 8)
    #expect(MetalMLGraphLifecycleProbe.repetitionCount(
        environment: ["JASNA_GRAPH_LIFECYCLE_REPEATS": "4"]
    ) == 4)
    #expect(MetalMLGraphLifecycleProbe.repetitionCount(
        environment: ["JASNA_GRAPH_LIFECYCLE_REPEATS": "100"]
    ) == 32)
    #expect(MetalMLGraphLifecycleProbe.repetitionCount(
        environment: ["JASNA_GRAPH_LIFECYCLE_REPEATS": "0"]
    ) == 8)
    #expect(MetalMLGraphLifecycleProbe.repetitionCount(
        environment: ["JASNA_GRAPH_LIFECYCLE_REPEATS": "invalid"]
    ) == 8)
}

@Test
func graphLifecycleValidationAcceptsStableRetainedOutputs() throws {
    let samples = [
        MetalMLGraphLifecycleSample(
            step: 1, frameCount: 30, cacheHit: false, lookupMilliseconds: 0.1,
            setupMilliseconds: 200_000, gpuMilliseconds: 600, wallMilliseconds: 200_700,
            outputHash: "thirty-a", maximumErrorFromPreviousSameShape: nil
        ),
        MetalMLGraphLifecycleSample(
            step: 2, frameCount: 30, cacheHit: true, lookupMilliseconds: 0.1,
            setupMilliseconds: 0, gpuMilliseconds: 600, wallMilliseconds: 700,
            outputHash: "thirty-b", maximumErrorFromPreviousSameShape: 0.0005
        ),
        MetalMLGraphLifecycleSample(
            step: 3, frameCount: 17, cacheHit: false, lookupMilliseconds: 0.1,
            setupMilliseconds: 500, gpuMilliseconds: 350, wallMilliseconds: 900,
            outputHash: "seventeen-a", maximumErrorFromPreviousSameShape: nil
        ),
        MetalMLGraphLifecycleSample(
            step: 4, frameCount: 17, cacheHit: true, lookupMilliseconds: 0.1,
            setupMilliseconds: 0, gpuMilliseconds: 350, wallMilliseconds: 400,
            outputHash: "seventeen-b", maximumErrorFromPreviousSameShape: 0
        ),
        MetalMLGraphLifecycleSample(
            step: 5, frameCount: 35, cacheHit: false, lookupMilliseconds: 0.1,
            setupMilliseconds: 210_000, gpuMilliseconds: 700, wallMilliseconds: 210_800,
            outputHash: "thirty-five-a", maximumErrorFromPreviousSameShape: nil
        ),
        MetalMLGraphLifecycleSample(
            step: 6, frameCount: 35, cacheHit: true, lookupMilliseconds: 0.1,
            setupMilliseconds: 0, gpuMilliseconds: 700, wallMilliseconds: 800,
            outputHash: "thirty-five-b", maximumErrorFromPreviousSameShape: 0
        ),
    ]
    try MetalMLGraphLifecycleProbe.validate(samples)
}

@Test
func graphLifecycleValidationRejectsCacheOrOutputMismatch() {
    var samples = MetalMLGraphLifecycleProbe.plan.enumerated().map { index, step in
        MetalMLGraphLifecycleSample(
            step: index + 1,
            frameCount: step.frameCount,
            cacheHit: step.expectedCacheHit,
            lookupMilliseconds: 0,
            setupMilliseconds: step.expectedCacheHit ? 0 : 1,
            gpuMilliseconds: 1,
            wallMilliseconds: 2,
            outputHash: "same",
            maximumErrorFromPreviousSameShape: step.expectedCacheHit ? 0 : nil
        )
    }
    samples[1] = MetalMLGraphLifecycleSample(
        step: 2, frameCount: 30, cacheHit: false, lookupMilliseconds: 0,
        setupMilliseconds: 1, gpuMilliseconds: 1, wallMilliseconds: 2,
        outputHash: "different", maximumErrorFromPreviousSameShape: 0
    )
    #expect(throws: DeformConvError.self) {
        try MetalMLGraphLifecycleProbe.validate(samples)
    }
}

@Test
func graphLifecycleMaximumErrorMeasuresFP16Difference() throws {
    let error = try MetalMLGraphLifecycleProbe.maximumError(
        [[Float16(1), Float16(1)]],
        [[Float16(1), Float16(1.0005)]]
    )
    #expect(error > 0)
    #expect(error <= 0.001)
}

@Test
func graphComponentTimingPlausibilityRejectsInvalidCounters() {
    let valid = FusedGraphComponentTimings(
        featureExtraction: 10, spynet: 20, backward1: 100, forward1: 100,
        backward2: 100, forward2: 100, reconstruction: 20
    )
    #expect(valid.total == 450)
    #expect(valid.isPlausible(totalGPUMilliseconds: 500))

    let invalid = FusedGraphComponentTimings(
        featureExtraction: 10, spynet: 20, backward1: 100, forward1: 100,
        backward2: 100, forward2: 100, reconstruction: -1
    )
    #expect(!invalid.isPlausible(totalGPUMilliseconds: 500))
    #expect(!valid.isPlausible(totalGPUMilliseconds: 400))
}

@Test
func graphPropagationTimingPlausibilityRejectsInvalidCounters() {
    let valid = FusedGraphPropagationTimings(
        offsetNetwork: 80, tensorPreparation: 20, dcnTransform: 15, dcnGather: 60,
        dcnGEMM: 100, backboneNetwork: 250, residual: 15
    )
    #expect(valid.total == 540)
    #expect(valid.isPlausible(totalPropagationMilliseconds: 550))

    let invalid = FusedGraphPropagationTimings(
        offsetNetwork: 80, tensorPreparation: 20, dcnTransform: 15, dcnGather: -1,
        dcnGEMM: 100, backboneNetwork: 250, residual: 15
    )
    #expect(!invalid.isPlausible(totalPropagationMilliseconds: 550))
    #expect(!valid.isPlausible(totalPropagationMilliseconds: 450))

    let coalesced = FusedGraphPropagationTimings(
        offsetNetwork: 80, tensorPreparation: 20, dcnTransform: 175, dcnGather: 0,
        dcnGEMM: 0, backboneNetwork: 250, residual: 15
    )
    #expect(!coalesced.isPlausible(totalPropagationMilliseconds: 550))
}

@Test
func graphBranchPropagationTimingPlausibilityRejectsInvalidCounters() {
    let valid = FusedGraphBranchPropagationTimings(
        name: "backward_1", offsetNetwork: 20, tensorPreparation: 5,
        dcnTransform: 4, dcnGather: 15, dcnGEMM: 25,
        backboneNetwork: 60, residual: 3
    )
    #expect(valid.total == 132)
    #expect(valid.isPlausible(totalBranchMilliseconds: 135))

    let invalid = FusedGraphBranchPropagationTimings(
        name: "", offsetNetwork: 20, tensorPreparation: 5,
        dcnTransform: 4, dcnGather: 15, dcnGEMM: 25,
        backboneNetwork: 60, residual: 3
    )
    #expect(!invalid.isPlausible(totalBranchMilliseconds: 135))
    #expect(!valid.isPlausible(totalBranchMilliseconds: 100))
}

@Test
func graphOffsetLocalityValidationRejectsInvalidFractions() {
    let valid = FusedGraphBranchOffsetLocality(
        name: "backward_1", sampleCount: 1_000,
        meanMagnitude: 1.5, maximumMagnitude: 9,
        fractionAbove2: 0.4, fractionAbove4: 0.2, fractionAbove8: 0.01,
        outOfBoundsFraction: 0.05, meanNeighborDelta: 0.3
    )
    #expect(valid.isPlausible)
    let invalid = FusedGraphBranchOffsetLocality(
        name: "backward_1", sampleCount: 1_000,
        meanMagnitude: 1.5, maximumMagnitude: 9,
        fractionAbove2: 0.1, fractionAbove4: 0.2, fractionAbove8: 0.01,
        outOfBoundsFraction: 0.05, meanNeighborDelta: 0.3
    )
    #expect(!invalid.isPlausible)
}

@Test
func offsetLocalityRequiresDetailedTelemetry() {
    let enabled = ["JASNA_DCN_OFFSET_LOCALITY": "1"]
    #expect(!dcnOffsetLocalityEnabled(
        detailedTelemetry: false, environment: enabled
    ))
    #expect(dcnOffsetLocalityEnabled(
        detailedTelemetry: true, environment: enabled
    ))
    #expect(!dcnOffsetLocalityEnabled(
        detailedTelemetry: true, environment: [:]
    ))
}

@Test
func channelLastGatherIsExperimentalAndOptIn() {
    #expect(!dcnChannelLastGatherEnabled(environment: [:]))
    #expect(dcnChannelLastGatherEnabled(
        environment: ["JASNA_DCN_CHANNEL_LAST_GATHER": "1"]
    ))
    #expect(!dcnChannelLastGatherEnabled(
        environment: ["JASNA_DCN_CHANNEL_LAST_GATHER": "0"]
    ))
}

@Test
func graphWarmupCoversFirstFamilyAndSlowLaterSpecialization() {
    #expect(coldGraphWarmupRequired(
        setupMilliseconds: 594,
        reusableProductionGraph: true,
        isFirstGraphForFamily: true,
        environment: [:]
    ))
    #expect(coldGraphWarmupRequired(
        setupMilliseconds: 208_610,
        reusableProductionGraph: true,
        environment: [:]
    ))
    #expect(!coldGraphWarmupRequired(
        setupMilliseconds: 594,
        reusableProductionGraph: true,
        environment: [:]
    ))
    #expect(!coldGraphWarmupRequired(
        setupMilliseconds: 208_610,
        reusableProductionGraph: false,
        isFirstGraphForFamily: true,
        environment: [:]
    ))
    #expect(!coldGraphWarmupRequired(
        setupMilliseconds: 594,
        reusableProductionGraph: true,
        isFirstGraphForFamily: true,
        environment: ["JASNA_INITIAL_GRAPH_WARMUP": "0"]
    ))
    #expect(!coldGraphWarmupRequired(
        setupMilliseconds: 208_610,
        reusableProductionGraph: true,
        environment: ["JASNA_COLD_GRAPH_WARMUP_THRESHOLD_MS": "0"]
    ))
}
