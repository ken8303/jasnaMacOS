import Testing
@testable import SyntheticResamplingLab

@Suite("Pixel-area selector boundary holdout")
struct BoundaryHoldoutTests {
    @Test func candidateUsesAreaAndReuseRatherThanOneExactShape() throws {
        for uses in [1, 2, 4, 8] {
            let below = try MixedJob(576, 448, uses: uses)
            let above = try MixedJob(704, 560, uses: uses)
            #expect(BoundaryPolicy.candidate.backend(for: below) == .cpu)
            #expect(BoundaryPolicy.candidate.backend(for: above) == (uses >= 4 ? .metal : .cpu))

            let existing = try MixedJob(512, 512, uses: uses)
            #expect(BoundaryPolicy.candidate.backend(for: existing) == MixedPolicy.hybrid.backend(for: existing))

            let differentAbove = try MixedJob(640, 410, uses: uses)
            let differentBelow = try MixedJob(800, 327, uses: uses)
            #expect(BoundaryPolicy.candidate.backend(for: differentAbove) == (uses >= 4 ? .metal : .cpu))
            #expect(BoundaryPolicy.candidate.backend(for: differentBelow) == .cpu)
        }
    }

    @Test func planUsesOnlyTwoUnseenShapesAndCyclesEveryReuseCount() throws {
        let plans = try BoundaryHoldoutPlan.phases()
        #expect(plans.count == 4)
        let expectedShapes = Set([try MixedShape(576, 448), try MixedShape(704, 560)])
        let priorShapes = Set(try HeldOutPlan.phases().flatMap { $0.jobs.map(\.shape) })
        for plan in plans {
            #expect(plan.jobs.count == 8)
            #expect(Set(plan.jobs.map(\.shape)) == expectedShapes)
            #expect(plan.expectedBoundaryWork(policy: .cpu, round: 0).outputs == 120)
            #expect(plan.expectedBoundaryWork(policy: .cpu, round: 0).outputPixels > 0)
        }
        for jobIndex in 0..<8 {
            #expect(Set(plans.map { $0.jobs[jobIndex].usesPerInput }) == Set([1, 2, 4, 8]))
        }
        #expect(expectedShapes.isDisjoint(with: priorShapes))
    }

    @Test func candidateWorkMatchesPredeclaredPhaseBudget() throws {
        let plans = try BoundaryHoldoutPlan.phases()
        let candidate = plans.map { $0.expectedBoundaryWork(policy: .candidate, round: 0) }
        let control = plans.map { $0.expectedBoundaryWork(policy: .control, round: 0) }
        #expect(candidate.map(\.metalJobs) == [2, 2, 2, 2])
        #expect(candidate.map(\.uploads) == [8, 8, 8, 8])
        #expect(candidate.map(\.submissions) == [16, 8, 16, 8])
        #expect(control.map(\.cpuJobs) == [8, 8, 8, 8])
        #expect(control.map(\.metalJobs) == [0, 0, 0, 0])
        #expect(control.map(\.uploads) == [0, 0, 0, 0])
        #expect(control.map(\.submissions) == [0, 0, 0, 0])
        for phase in plans.indices {
            #expect(candidate[phase].outputs == 120)
            #expect(candidate[phase].outputs == control[phase].outputs)
            #expect(candidate[phase].outputPixels == control[phase].outputPixels)
        }
    }

    @Test func fourPoliciesAreJointlyBalancedWithEqualCandidatePrecedence() {
        let rounds = BoundaryHoldoutPlan.warmupTrials..<(BoundaryHoldoutPlan.warmupTrials + BoundaryHoldoutPlan.measuredTrials)
        #expect(rounds.count == 64)
        for policy in BoundaryPolicy.allCases {
            var combinations = Set<String>()
            for round in rounds {
                let schedule = HeldOutSchedule(round: round)
                let position = BoundaryHoldoutPlan.policyOrder(round: round).firstIndex(of: policy)!
                let key = "\(position)/\(schedule.phase)/\(schedule.bankParity)/\(schedule.orderParity)"
                #expect(combinations.insert(key).inserted)
            }
            #expect(combinations.count == 4 * 4 * 2 * 2)
        }
        let candidateBeforeControl = rounds.filter { round in
            let order = BoundaryHoldoutPlan.policyOrder(round: round)
            return order.firstIndex(of: .candidate)! < order.firstIndex(of: .control)!
        }
        #expect(candidateBeforeControl.count == 32)
    }

    @Test func originalPolicySetsRemainUnchanged() {
        #expect(MixedPolicy.allCases == [.cpu, .metal, .hybrid])
        #expect(SelectorPolicy.allCases == [.cpu, .metal, .control, .candidate])
        #expect(BoundaryPolicy.allCases == [.cpu, .metal, .control, .candidate])
    }

    @Test func focusedBoundaryScopeIsExplicitAndExclusive() throws {
        let option = try LabOptions(arguments: ["--report", "boundary.json", "--boundary-holdout-only"])
        #expect(option.scope == .boundaryHoldoutOnly)
        for flag in ["--mixed-only", "--held-out-only", "--job-profile-only",
                     "--selector-candidate-only", "--boundary-holdout-only"] {
            #expect(throws: LabError.self) {
                try LabOptions(arguments: ["--report", "boundary.json", "--boundary-holdout-only", flag])
            }
        }
    }
}
