import Foundation
import Testing
@testable import JasnaMetalPoC

@Test func fullModelOracleMetricsSummarizeKnownErrors() throws {
    let restored: [Float16] = [0, 0.25, 0.5, 1]
    let fp32Oracle: [Float] = [0, 0.2, 0.6, 1]
    let fp16Oracle = fp32Oracle.map(Float16.init)
    let metrics = try calculateFullModelOracleMetrics(
        restored: restored, fp32Oracle: fp32Oracle, fp16Oracle: fp16Oracle
    )
    #expect(metrics.elementCount == 4)
    #expect(abs(metrics.maximumAbsoluteError - 0.1) < 0.0001)
    #expect(metrics.maximumErrorIndex == 2)
    #expect(abs(metrics.meanAbsoluteError - 0.0375) < 0.0001)
    #expect(abs(metrics.rootMeanSquaredError - sqrt(0.003125)) < 0.0001)
    #expect(metrics.psnr > 25 && metrics.psnr < 26)
}

@Test func fullModelOracleMetricsRejectMismatchedArrays() {
    #expect(throws: DeformConvError.self) {
        _ = try calculateFullModelOracleMetrics(
            restored: [0], fp32Oracle: [], fp16Oracle: [0]
        )
    }
}
