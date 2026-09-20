import Testing
@testable import SyntheticResamplingLab

@Suite("Held-out changing reuse")
struct HeldOutPlanTests {
    @Test func newShapesRemainOnFrozenCPUFallback() throws {
        let old = Set(try MixedPlan.fixtures().flatMap { $0.jobs.map(\.shape) })
        let plans = try HeldOutPlan.phases()
        let shapes = Set(plans.flatMap { $0.jobs.map(\.shape) })
        let control = try MixedShape(512, 512)
        #expect(shapes.intersection(old) == [control])
        #expect(shapes.subtracting(old).count == 4)
        for plan in plans {
            for job in plan.jobs where job.shape != control {
                #expect(MixedPolicy.hybrid.backend(for: job) == .cpu)
            }
        }
        #expect(HeldOutPlan.frameOffsets == [24, 40])
    }

    @Test func everyJobCyclesReuseWithoutOmittingOutputs() throws {
        let plans = try HeldOutPlan.phases()
        #expect(plans.count == 4)
        for index in 0..<8 {
            #expect(Set(plans.map { $0.jobs[index].usesPerInput }) == [1, 2, 4, 8])
            #expect(Set(plans.map { $0.jobs[index].shape }).count == 1)
        }
        for plan in plans {
            #expect(plan.jobs.count == 8)
            for policy in MixedPolicy.allCases {
                let work = plan.expectedWork(policy: policy, round: 0)
                #expect(work.outputs == 120 && work.jobs == 8)
                #expect(work.outputPixels == plan.jobs.reduce(0) { $0 + $1.outputs * $1.shape.pixels })
                #expect(work == plan.expectedWork(policy: policy, round: 1))
            }
        }
        #expect(Set(plans.map { $0.expectedWork(policy: .cpu, round: 0).outputPixels }).count > 1)
    }

    @Test func knownSizeControlsSwitchBackendAsReuseChanges() throws {
        let plans = try HeldOutPlan.phases()
        let work = plans.map { $0.expectedWork(policy: .hybrid, round: 0) }
        #expect(work.map(\.metalJobs) == [1, 2, 1, 0])
        #expect(work.map(\.uploads) == [4, 8, 4, 0])
        #expect(work.map(\.submissions) == [4, 12, 8, 0])
        #expect(work.map(\.metalOutputs) == [16, 48, 32, 0])
        #expect(work.allSatisfy { $0.cpuOutputs + $0.metalOutputs == 120 })
        for plan in plans {
            let metal = plan.expectedWork(policy: .metal, round: 0)
            #expect(metal.uploads == 32 && metal.submissions == 30)
            #expect(plan.expectedWork(policy: .cpu, round: 0).submissions == 0)
        }
    }

    @Test func measuredScheduleBalancesEveryJointCombination() {
        let rounds = HeldOutPlan.warmupTrials..<(HeldOutPlan.warmupTrials + HeldOutPlan.measuredTrials)
        #expect(rounds.count == 48)
        for policy in MixedPolicy.allCases {
            for position in 0..<3 {
                for phase in 0..<4 {
                    for bank in 0...1 {
                        for order in 0...1 {
                            let matching = rounds.filter { round in
                                let schedule = HeldOutSchedule(round: round)
                                return mixedPolicyOrder(round: round)[position] == policy && schedule.phase == phase
                                    && schedule.bankParity == bank && schedule.orderParity == order
                            }
                            #expect(matching.count == 1)
                        }
                    }
                }
            }
        }
    }

    @Test func validationAndWarmupCoverAllInputAndOrderPairs() {
        #expect(HeldOutPlan.warmupTrials == HeldOutPlan.validationScenarios)
        for phase in 0..<4 {
            for bank in 0...1 {
                for order in 0...1 {
                    let matching = (0..<HeldOutPlan.validationScenarios).filter {
                        let schedule = HeldOutSchedule(round: $0)
                        return schedule.phase == phase && schedule.bankParity == bank && schedule.orderParity == order
                    }
                    #expect(matching.count == 1)
                }
            }
        }
    }

    @Test func focusedScopesAreExplicitAndMutuallyExclusive() throws {
        let options = try LabOptions(arguments: ["--report", "held-out.json", "--held-out-only"])
        #expect(options.scope == .heldOutOnly && options.heldOutOnly && !options.mixedOnly)
        #expect(try LabOptions(arguments: ["--report", "all.json"]).scope == .all)
        #expect(try LabOptions(arguments: ["--report", "mixed.json", "--mixed-only"]).scope == .mixedOnly)
        for arguments in [["--report", "out", "--mixed-only", "--held-out-only"],
                          ["--report", "out", "--held-out-only", "--mixed-only"],
                          ["--held-out-only", "--report", "out"]] {
            #expect(throws: LabError.self) { try LabOptions(arguments: arguments) }
        }
    }
}
