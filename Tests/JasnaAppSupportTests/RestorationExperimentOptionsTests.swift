import Testing
@testable import JasnaAppSupport

struct RestorationExperimentOptionsTests {
    @Test func stableDetectorRemainsFP32Default() {
        let mode = DetectorExecutionMode(rawValue: "invalid") ?? .stableFP32
        #expect(mode == .stableFP32)
        #expect(mode.processEnvironment["JASNA_DETECT_DEVICE"] == "mlx")
        #expect(mode.processEnvironment["JASNA_MLX_DETECTOR_MODE"] == "stable-fp32")
    }

    @Test func everyDetectorOptionHasADistinctRuntimeValue() {
        let values = DetectorExecutionMode.allCases.map(\.rawValue)
        #expect(Set(values).count == values.count)
        #expect(values == ["stable-fp32", "compiled-fp32", "float16", "compiled-float16"])
    }

    @Test func referenceRestorationKernelRemainsTheSafeDefault() {
        let kernel = MLXRestorationKernel(rawValue: "invalid") ?? .reference
        #expect(kernel == .reference)
        #expect(kernel.processEnvironment["JASNA_MLX_RESTORATION_KERNEL"] == "reference")
        #expect(MLXRestorationKernel.fused.processEnvironment["JASNA_MLX_RESTORATION_KERNEL"] == "fused")
    }
}
