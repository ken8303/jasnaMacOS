enum BoundaryPolicy: String, CaseIterable, Encodable, Hashable, MixedRoutingPolicy {
    case cpu = "cpu-only-parallel4"
    case metal = "metal-only-group4"
    case control = "control-frozen-hybrid"
    case candidate = "candidate-area262144-reuse4"

    func backend(for job: MixedJob) -> MixedBackend {
        switch self {
        case .cpu: return .cpu
        case .metal: return .metal
        case .control: return MixedPolicy.hybrid.backend(for: job)
        case .candidate:
            let existing = MixedPolicy.hybrid.backend(for: job) == .metal
            let newAreaReuse = job.shape.pixels >= BoundaryHoldoutPlan.minimumMetalPixels
                && job.usesPerInput >= 4
            return existing || newAreaReuse ? .metal : .cpu
        }
    }
}

enum BoundaryHoldoutPlan {
    static let frameOffsets = [56, 72]
    static let warmupTrials = 16
    static let measuredTrials = 64
    static let validationScenarios = 16
    static let minimumMetalPixels = 512 * 512

    static func phases() throws -> [MixedPlan] {
        // Two independently shaped outputs straddle the predeclared pixel-area
        // boundary. Each repeated job cycles through all four reuse counts.
        let sizes = [(576, 448), (704, 560), (576, 448), (704, 560),
                     (576, 448), (704, 560), (576, 448), (704, 560)]
        let uses = [1, 2, 4, 8]
        return try (0..<4).map { phase in
            MixedPlan(name: "boundary-holdout-phase-\(phase)", jobs: try sizes.enumerated().map { index, size in
                try MixedJob(size.0, size.1, uses: uses[(index + phase) % uses.count])
            })
        }
    }

    static func policyOrder(round: Int) -> [BoundaryPolicy] {
        precondition(round >= 0)
        let policies = BoundaryPolicy.allCases
        // Balance every policy position while putting candidate before and
        // after control equally often over the 64 measured trials.
        switch (round / validationScenarios) % policies.count {
        case 0: return policies
        case 1: return policies.reversed()
        case 2: return Array(policies[2...]) + Array(policies[..<2])
        default: return [policies[1], policies[0], policies[3], policies[2]]
        }
    }
}
