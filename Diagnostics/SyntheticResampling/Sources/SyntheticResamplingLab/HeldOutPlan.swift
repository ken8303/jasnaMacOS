struct HeldOutSchedule: Encodable, Equatable {
    let phase: Int
    let bankParity: Int
    let orderParity: Int

    init(round: Int) {
        precondition(round >= 0)
        phase = round % 4
        bankParity = (round / 4) % 2
        orderParity = (round / 8) % 2
    }
}

enum HeldOutPlan {
    static let frameOffsets = [24, 40]
    static let warmupTrials = 16
    static let measuredTrials = 48
    static let validationScenarios = 16

    static func phases() throws -> [MixedPlan] {
        // Four new shapes; repeated 512x512 controls exercise the frozen rule.
        let sizes = [(192, 128), (512, 512), (320, 256), (640, 512),
                     (448, 384), (192, 128), (512, 512), (640, 512)]
        let uses = [1, 2, 4, 8]
        return try (0..<4).map { phase in
            MixedPlan(name: "held-out-reuse-phase-\(phase)", jobs: try sizes.enumerated().map { index, size in
                try MixedJob(size.0, size.1, uses: uses[(index + phase) % uses.count])
            })
        }
    }
}
