public enum DetectorExecutionMode: String, CaseIterable, Identifiable, Sendable {
    case stableFP32 = "stable-fp32"
    case compiledFP32 = "compiled-fp32"
    case float16 = "float16"
    case compiledFloat16 = "compiled-float16"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .stableFP32: "Stable FP32"
        case .compiledFP32: "Compiled FP32"
        case .float16: "FP16"
        case .compiledFloat16: "Compiled FP16"
        }
    }

    public var detail: String {
        switch self {
        case .stableFP32: "Current stable MLX detector"
        case .compiledFP32: "Experimental compiled graph with FP32 weights"
        case .float16: "Experimental FP16 weights with lower memory use"
        case .compiledFloat16: "Experimental compiled graph with FP16 weights"
        }
    }

    public var processEnvironment: [String: String] {
        [
            "JASNA_DETECT_DEVICE": "mlx",
            "JASNA_MLX_DETECTOR_MODE": rawValue,
        ]
    }
}
