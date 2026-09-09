import Testing
@testable import SyntheticResamplingLab

@Suite("Mixed synthetic workload")
struct MixedPlanTests {
    @Test func fixedRuleDoesNotLearnFromTimingsOrExtrapolateSizes() throws {
        for uses in [1, 2, 4, 8] {
            let large = try MixedJob(512, 512, uses: uses)
            #expect(MixedPolicy.hybrid.backend(for: large) == (uses >= 4 ? .metal : .cpu))
            for size in [(96, 80), (256, 256), (384, 384), (512, 511), (1_024, 1_024)] {
                let other = try MixedJob(size.0, size.1, uses: uses)
                #expect(MixedPolicy.hybrid.backend(for: other) == .cpu)
                #expect(MixedPolicy.cpu.backend(for: other) == .cpu)
                #expect(MixedPolicy.metal.backend(for: other) == .metal)
            }
        }
    }

    @Test func plansCountEveryOutputAndActualUploadBudget() throws {
        let plans = try MixedPlan.fixtures()
        #expect(plans.count == 2)
        let reuse = plans[0], fresh = plans[1]
        #expect(reuse.jobs.count == 8 && fresh.jobs.count == 8)
        let cpu = reuse.expectedWork(policy: .cpu, round: 0)
        #expect(cpu.outputs == 144 && cpu.cpuJobs == 8 && cpu.cpuOutputs == 144)
        #expect(cpu.uploads == 0 && cpu.submissions == 0 && cpu.metalOutputs == 0)
        let metal = reuse.expectedWork(policy: .metal, round: 0)
        #expect(metal.outputs == 144 && metal.metalJobs == 8 && metal.metalOutputs == 144)
        #expect(metal.uploads == 32 && metal.submissions == 36 && metal.cpuOutputs == 0)
        let hybrid = reuse.expectedWork(policy: .hybrid, round: 0)
        #expect(hybrid.cpuJobs == 6 && hybrid.metalJobs == 2)
        #expect(hybrid.cpuOutputs == 96 && hybrid.metalOutputs == 48)
        #expect(hybrid.uploads == 8 && hybrid.submissions == 12 && hybrid.backendSwitches == 3)
        #expect(cpu.outputPixels == metal.outputPixels && metal.outputPixels == hybrid.outputPixels)
        #expect(cpu.outputPixels == reuse.jobs.reduce(0) { $0 + $1.outputs * $1.shape.pixels })
        for policy in MixedPolicy.allCases {
            let work = fresh.expectedWork(policy: policy, round: 0)
            #expect(work.outputs == 32)
            #expect(work.submissions == (policy == .metal ? 8 : 0))
            #expect(work.uploads == (policy == .metal ? 32 : 0))
            #expect(work.backendSwitches == 0)
        }
    }

    @Test func forwardReversePreserveJobsAndWork() throws {
        for plan in try MixedPlan.fixtures() {
            let forward = mixedJobOrder(count: plan.jobs.count, round: 0)
            let reverse = mixedJobOrder(count: plan.jobs.count, round: 1)
            #expect(forward == Array(plan.jobs.indices))
            #expect(reverse == Array(forward.reversed()))
            #expect(Set(forward) == Set(reverse))
            for policy in MixedPolicy.allCases {
                #expect(plan.expectedWork(policy: policy, round: 0) == plan.expectedWork(policy: policy, round: 1))
            }
        }
    }

    @Test func policyPositionAndBankOrderAreJointlyBalanced() {
        for policy in MixedPolicy.allCases {
            for position in 0..<3 {
                #expect((4..<28).filter { mixedPolicyOrder(round: $0)[position] == policy }.count == 8)
                for parity in 0...1 {
                    #expect((4..<28).filter { mixedPolicyOrder(round: $0)[position] == policy && $0 % 2 == parity }.count == 4)
                }
            }
        }
        for round in 0..<28 {
            #expect(Set(mixedPolicyOrder(round: round)) == Set(MixedPolicy.allCases))
            for job in 0..<8 {
                #expect(mixedBankIndex(round: round, job: job) != mixedBankIndex(round: round + 1, job: job))
                #expect(mixedBankIndex(round: round, job: job) == (round + job) % 2)
            }
        }
    }

    @Test func malformedPlansAreRejectedBeforeAllocation() {
        for size in [(0, 1), (1, -1), (Int.max, 4), (1_025, 4)] {
            #expect(throws: LabError.self) { try MixedJob(size.0, size.1, uses: 1) }
        }
        for uses in [0, 3, 16, Int.max] {
            #expect(throws: LabError.self) { try MixedJob(512, 512, uses: uses) }
        }
    }

    @Test func focusedOptionsKeepExistingInvocationCompatible() throws {
        let all = try LabOptions(arguments: ["--report", "new report.json"])
        #expect(!all.mixedOnly && all.reportPath == "new report.json")
        let mixed = try LabOptions(arguments: ["--report", "new.json", "--mixed-only"])
        #expect(mixed.mixedOnly && mixed.reportPath == "new.json")
        for arguments in [[], ["--report"], ["--report", ""], ["--report", "--mixed-only"],
                          ["--report", "file", "--unknown"], ["--report", "file", "--mixed-only", "extra"]] {
            #expect(throws: LabError.self) { try LabOptions(arguments: arguments) }
        }
    }
}
