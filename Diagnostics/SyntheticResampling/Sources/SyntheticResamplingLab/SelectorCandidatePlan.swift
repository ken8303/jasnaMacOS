enum SelectorPolicy: String, CaseIterable, Encodable, Hashable, MixedRoutingPolicy {
    case cpu = "cpu-only-parallel4"
    case metal = "metal-only-group4"
    case control = "control-frozen-hybrid"
    case candidate = "candidate-640x512-reuse4"

    func backend(for job: MixedJob) -> MixedBackend {
        switch self {
        case .cpu: return .cpu
        case .metal: return .metal
        case .control: return MixedPolicy.hybrid.backend(for: job)
        case .candidate:
            let existing = MixedPolicy.hybrid.backend(for: job) == .metal
            let newLargeReuse = job.shape.width == 640 && job.shape.height == 512 && job.usesPerInput >= 4
            return existing || newLargeReuse ? .metal : .cpu
        }
    }
}

enum SelectorCandidatePlan {
    static let warmupTrials = 16
    static let measuredTrials = 64

    static func policyOrder(round: Int) -> [SelectorPolicy] {
        precondition(round >= 0)
        let policies = SelectorPolicy.allCases
        // Two forward and two reversed Latin rows balance every policy position
        // while also putting candidate before/after control equally often.
        switch (round / HeldOutPlan.validationScenarios) % policies.count {
        case 0: return policies
        case 1: return policies.reversed()
        case 2: return Array(policies[2...]) + Array(policies[..<2])
        default: return [policies[1], policies[0], policies[3], policies[2]]
        }
    }
}
