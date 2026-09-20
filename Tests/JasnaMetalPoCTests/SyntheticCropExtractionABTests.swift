import Testing
@testable import JasnaMetalPoC

@Suite("Synthetic application crop extraction A/B")
struct SyntheticCropExtractionABTests {
    @Test func workCountsBothConsumersWithoutSkippingOutputs() {
        let control = SyntheticCropExtractionABPlan.work(for: .extractTwice)
        let candidate = SyntheticCropExtractionABPlan.work(for: .reuseOnce)
        #expect(SyntheticCropExtractionABPlan.jobsPerTrial == 8)
        #expect(control.jobs == 8 && candidate.jobs == 8)
        #expect(control.extractionCalls == 16)
        #expect(candidate.extractionCalls == 8)
        #expect(control.consumerPasses == 16 && candidate.consumerPasses == 16)
        #expect(control.outputElementsConsumed == candidate.outputElementsConsumed)
        #expect(control.outputElementsConsumed == 16 * SideBySideRestoration.tileElements)
    }

    @Test func measuredPairOrderIsBalanced() {
        let rounds = SyntheticCropExtractionABPlan.warmupPairs..<(
            SyntheticCropExtractionABPlan.warmupPairs
                + SyntheticCropExtractionABPlan.measuredPairs
        )
        #expect(rounds.count == 48)
        #expect(rounds.filter { SyntheticCropExtractionABPlan.candidateFirst(round: $0) }.count == 24)
        #expect(rounds.filter { !SyntheticCropExtractionABPlan.candidateFirst(round: $0) }.count == 24)
    }

    @Test func kernelPairOrderIsBalanced() {
        let rounds = SyntheticCropExtractionABPlan.kernelWarmupPairs..<(
            SyntheticCropExtractionABPlan.kernelWarmupPairs
                + SyntheticCropExtractionABPlan.kernelMeasuredPairs
        )
        #expect(rounds.count == 48)
        #expect(rounds.filter { SyntheticCropExtractionABPlan.candidateFirst(round: $0) }.count == 24)
        #expect(rounds.filter { !SyntheticCropExtractionABPlan.candidateFirst(round: $0) }.count == 24)
        #expect(SyntheticCropSamplingKernel.allCases == [.scalar, .parallel])
    }

    @Test func applicationShapeRemainsBelowSyntheticMetalBoundary() {
        let modelPixels = SideBySideVideoPlan.modelTileSize
            * SideBySideVideoPlan.modelTileSize
        #expect(modelPixels == 65_536)
        #expect(modelPixels < 512 * 512)
        #expect(SyntheticCropExtractionABPlan.sourceWidth == 4_096)
        #expect(SyntheticCropExtractionABPlan.sourceHeight == 4_096)
    }

    @Test func delayedWindowReportsItsCompleteMemoryAndWorkBudget() {
        let jobs = SyntheticCropExtractionABPlan.windowJobsPerTrial
        let control = SyntheticCropExtractionABPlan.work(
            for: .extractTwice, jobs: jobs
        )
        let candidate = SyntheticCropExtractionABPlan.work(
            for: .reuseOnce, jobs: jobs
        )
        #expect(jobs == 60)
        #expect(control.extractionCalls == 120)
        #expect(candidate.extractionCalls == 60)
        #expect(control.consumerPasses == 120 && candidate.consumerPasses == 120)
        #expect(jobs * SideBySideRestoration.tileBytes == 23_592_960)
    }
}
