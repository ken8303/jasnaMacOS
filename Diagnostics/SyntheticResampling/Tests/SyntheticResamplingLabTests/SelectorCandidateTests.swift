import Testing
@testable import SyntheticResamplingLab

@Suite("Large reuse selector candidate")
struct SelectorCandidateTests {
    @Test func candidateExtendsOnlyTheDeclaredLargeReuseCase() throws {
        for uses in [1, 2, 4, 8] {
            let large = try MixedJob(640, 512, uses: uses)
            #expect(SelectorPolicy.control.backend(for: large) == .cpu)
            #expect(SelectorPolicy.candidate.backend(for: large) == (uses >= 4 ? .metal : .cpu))
            let existing = try MixedJob(512, 512, uses: uses)
            #expect(SelectorPolicy.candidate.backend(for: existing) == MixedPolicy.hybrid.backend(for: existing))
            for size in [(639, 512), (640, 511), (1_024, 1_024), (320, 256)] {
                let other = try MixedJob(size.0, size.1, uses: uses)
                #expect(SelectorPolicy.candidate.backend(for: other) == MixedPolicy.hybrid.backend(for: other))
            }
        }
    }

    @Test func candidateWorkChangesOnlyFourAndEightUsePhases() throws {
        let plans = try HeldOutPlan.phases()
        let control = plans.map { $0.expectedSelectorWork(policy: .control, round: 0) }
        let candidate = plans.map { $0.expectedSelectorWork(policy: .candidate, round: 0) }
        #expect(control.map(\.metalJobs) == [1, 2, 1, 0])
        #expect(candidate.map(\.metalJobs) == [3, 2, 1, 2])
        #expect(candidate.map(\.uploads) == [12, 8, 4, 8])
        #expect(candidate.map(\.submissions) == [20, 12, 8, 8])
        #expect(candidate[1] == control[1] && candidate[2] == control[2])
        for phase in plans.indices {
            #expect(candidate[phase].outputs == control[phase].outputs)
            #expect(candidate[phase].outputPixels == control[phase].outputPixels)
        }
    }

    @Test func fourPoliciesAreJointlyBalancedWithHeldOutInputs() {
        let rounds = SelectorCandidatePlan.warmupTrials..<(SelectorCandidatePlan.warmupTrials + SelectorCandidatePlan.measuredTrials)
        #expect(rounds.count == 64)
        for policy in SelectorPolicy.allCases {
            var combinations = Set<String>()
            for round in rounds {
                let schedule = HeldOutSchedule(round: round)
                let position = SelectorCandidatePlan.policyOrder(round: round).firstIndex(of: policy)!
                let key = "\(position)/\(schedule.phase)/\(schedule.bankParity)/\(schedule.orderParity)"
                #expect(combinations.insert(key).inserted)
            }
            #expect(combinations.count == 4 * 4 * 2 * 2)
        }
        let candidateBeforeControl = rounds.filter { round in
            let order = SelectorCandidatePlan.policyOrder(round: round)
            return order.firstIndex(of: .candidate)! < order.firstIndex(of: .control)!
        }
        #expect(candidateBeforeControl.count == 32)
    }

    @Test func standardPolicySetRemainsUnchanged() {
        #expect(MixedPolicy.allCases == [.cpu, .metal, .hybrid])
        #expect(SelectorPolicy.allCases == [.cpu, .metal, .control, .candidate])
        for round in 0..<80 {
            #expect(Set(SelectorCandidatePlan.policyOrder(round: round)) == Set(SelectorPolicy.allCases))
        }
    }

    @Test func focusedSelectorScopeIsExplicitAndExclusive() throws {
        let option = try LabOptions(arguments: ["--report", "selector.json", "--selector-candidate-only"])
        #expect(option.scope == .selectorCandidateOnly)
        for flag in ["--mixed-only", "--held-out-only", "--job-profile-only", "--selector-candidate-only"] {
            #expect(throws: LabError.self) {
                try LabOptions(arguments: ["--report", "selector.json", "--selector-candidate-only", flag])
            }
        }
    }
}
