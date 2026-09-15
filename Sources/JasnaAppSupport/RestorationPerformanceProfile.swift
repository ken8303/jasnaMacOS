import Foundation

public enum RestorationPerformanceProfile: String, CaseIterable, Identifiable, Sendable {
    case fast
    case balanced

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .fast: "Fast"
        case .balanced: "Balanced"
        }
    }

    public var detail: String {
        switch self {
        case .fast:
            "Uses eight Metal windows, detector batch 2, and keeps up to 512 MiB of "
                + "restored crops in memory for handoff while still writing disk checkpoints "
                + "so Stop mid-window can resume. Recommended for a 16 GB or larger Mac."
        case .balanced:
            "Uses detector batch 1 and disk-backed crop handoff to reduce memory pressure."
        }
    }

    public var logDescription: String {
        switch self {
        case .fast:
            "fast: BasicVSR++ batch 2, eight Metal windows, detector batch 2, "
                + "512 MiB in-memory crop handoff with disk checkpoints, "
                + "overlapped crop preparation"
        case .balanced:
            "balanced: BasicVSR++ batch 2, two Metal windows, detector batch 1, "
                + "disk-backed crop handoff"
        }
    }

    public var processEnvironment: [String: String] {
        switch self {
        case .fast:
            [
                "JASNA_PERFORMANCE_PROFILE": rawValue,
                "JASNA_METAL_WINDOWS_PER_PROCESS": "8",
                "JASNA_DETECT_BATCH_SIZE": "2",
                "JASNA_IN_MEMORY_CROP_CACHE": "1",
                "JASNA_IN_MEMORY_CACHE_LIMIT_MB": "512",
                "JASNA_STEREO_WRITER_DEPTH": "2",
                "JASNA_REGION_PREPARE_DEPTH": "2",
            ]
        case .balanced:
            [
                "JASNA_PERFORMANCE_PROFILE": rawValue,
                "JASNA_METAL_WINDOWS_PER_PROCESS": "2",
                "JASNA_DETECT_BATCH_SIZE": "1",
                "JASNA_IN_MEMORY_CROP_CACHE": "0",
                "JASNA_IN_MEMORY_CACHE_LIMIT_MB": "128",
                "JASNA_STEREO_WRITER_DEPTH": "1",
                "JASNA_REGION_PREPARE_DEPTH": "1",
            ]
        }
    }
}
