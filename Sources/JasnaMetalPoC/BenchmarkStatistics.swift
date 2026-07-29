import Foundation

struct BenchmarkStatistics: Sendable {
    let samples: [Double]
    let minimum: Double
    let maximum: Double
    let median: Double
    let mean: Double
    let standardDeviation: Double
    let percentile10: Double
    let percentile90: Double

    init?(_ values: [Double]) {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { return nil }
        let sorted = values.sorted()
        let meanValue = sorted.reduce(0, +) / Double(sorted.count)
        samples = sorted
        minimum = sorted[0]
        maximum = sorted[sorted.count - 1]
        if sorted.count.isMultiple(of: 2) {
            let upper = sorted.count / 2
            median = (sorted[upper - 1] + sorted[upper]) / 2
        } else {
            median = sorted[sorted.count / 2]
        }
        mean = meanValue
        let variance = sorted.reduce(0) { partial, sample in
            let difference = sample - meanValue
            return partial + difference * difference
        } / Double(sorted.count)
        standardDeviation = sqrt(variance)
        percentile10 = Self.percentile(0.10, sorted: sorted)
        percentile90 = Self.percentile(0.90, sorted: sorted)
    }

    private static func percentile(_ fraction: Double, sorted: [Double]) -> Double {
        guard sorted.count > 1 else { return sorted[0] }
        let position = fraction * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = Int(position.rounded(.up))
        let weight = position - Double(lower)
        return sorted[lower] * (1 - weight) + sorted[upper] * weight
    }
}
