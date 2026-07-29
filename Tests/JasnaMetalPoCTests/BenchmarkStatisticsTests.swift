import Foundation
import Testing
@testable import JasnaMetalPoC

@Test func benchmarkStatisticsSummarizeEvenSampleSet() throws {
    let statistics = try #require(BenchmarkStatistics([4, 1, 3, 2]))
    #expect(statistics.samples == [1, 2, 3, 4])
    #expect(statistics.minimum == 1)
    #expect(statistics.maximum == 4)
    #expect(statistics.median == 2.5)
    #expect(statistics.mean == 2.5)
    #expect(abs(statistics.standardDeviation - sqrt(1.25)) < 1e-12)
    #expect(abs(statistics.percentile10 - 1.3) < 1e-12)
    #expect(abs(statistics.percentile90 - 3.7) < 1e-12)
}

@Test func benchmarkStatisticsRejectEmptyAndNonfiniteSamples() {
    #expect(BenchmarkStatistics([]) == nil)
    #expect(BenchmarkStatistics([1, .infinity]) == nil)
}
