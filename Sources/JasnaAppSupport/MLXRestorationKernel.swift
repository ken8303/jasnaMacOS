public enum MLXRestorationKernel: String, CaseIterable, Identifiable, Sendable {
    case reference
    case fused

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .reference: "Reference"
        case .fused: "Fused Metal (test)"
        }
    }

    public var detail: String {
        switch self {
        case .reference: "Existing MLX deformable-convolution implementation"
        case .fused: "Experimental single-pass Metal sampling kernel"
        }
    }

    public var processEnvironment: [String: String] {
        ["JASNA_MLX_RESTORATION_KERNEL": rawValue]
    }
}
