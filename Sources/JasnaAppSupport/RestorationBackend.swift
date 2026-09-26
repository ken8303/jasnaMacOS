public enum RestorationBackend: String, CaseIterable, Identifiable, Sendable {
    case metal
    case mlx

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .metal: "Metal ML"
        case .mlx: "MLX"
        }
    }

    public var detail: String {
        switch self {
        case .metal: "Validated restoration engine"
        case .mlx: "Experimental MLX detection and v1.2 restoration"
        }
    }

    public var processEnvironment: [String: String] {
        ["JASNA_RESTORATION_BACKEND": rawValue]
    }
}
