import Foundation
import Testing
@testable import SyntheticResamplingLab

@Suite("Opt-in job profiling")
struct JobTimingTests {
    @Test func hostStagesExcludeOverlappingGPUTimestamps() throws {
        let job = try JobTiming(jobIndex: 3, backend: .metal, uses: 8, wallMS: 10,
                                uploadMS: 2, cpuComputeAllocateMS: nil, metalEncodeSubmitWaitMS: 5,
                                metalOutputCopyMS: 1, gpuMS: 4, gpuTimestampSamples: 8)
        #expect(job.otherHostMS == 2)
        #expect(job.cpuComputeAllocateMS == nil && job.gpuMS == 4)
    }

    @Test func CPUAllocationIsNotReportedAsSeparateCopy() throws {
        let job = try JobTiming(jobIndex: 3, backend: .cpu, uses: 8, wallMS: 10,
                                uploadMS: nil, cpuComputeAllocateMS: 9, metalEncodeSubmitWaitMS: nil,
                                metalOutputCopyMS: nil, gpuMS: nil, gpuTimestampSamples: 0)
        #expect(job.otherHostMS == 1)
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(job)) as! [String: Any]
        #expect(encoded["uploadMS"] == nil && encoded["metalOutputCopyMS"] == nil && encoded["gpuMS"] == nil)
    }

    @Test func incompleteAndInvalidStagesFail() {
        #expect(throws: LabError.self) {
            try JobTiming(jobIndex: 0, backend: .cpu, uses: 1, wallMS: 1,
                          uploadMS: 0, cpuComputeAllocateMS: 1, metalEncodeSubmitWaitMS: nil,
                          metalOutputCopyMS: nil, gpuMS: nil, gpuTimestampSamples: 0)
        }
        for cpu in [Double.nan, .infinity, -1, 2] {
            #expect(throws: LabError.self) {
                try JobTiming(jobIndex: 0, backend: .cpu, uses: 1, wallMS: 1,
                              uploadMS: nil, cpuComputeAllocateMS: cpu, metalEncodeSubmitWaitMS: nil,
                              metalOutputCopyMS: nil, gpuMS: nil, gpuTimestampSamples: 0)
            }
        }
        #expect(throws: LabError.self) {
            try JobTiming(jobIndex: 0, backend: .metal, uses: 1, wallMS: 1,
                          uploadMS: nil, cpuComputeAllocateMS: nil, metalEncodeSubmitWaitMS: 0,
                          metalOutputCopyMS: 0, gpuMS: nil, gpuTimestampSamples: 0)
        }
    }

    @Test func partialGPUTimestampsAreOmittedNotScaledUp() throws {
        let job = try JobTiming(jobIndex: 0, backend: .metal, uses: 8, wallMS: 10,
                                uploadMS: 2, cpuComputeAllocateMS: nil, metalEncodeSubmitWaitMS: 5,
                                metalOutputCopyMS: 1, gpuMS: nil, gpuTimestampSamples: 7)
        #expect(job.gpuMS == nil && job.gpuTimestampSamples == 7)
        #expect(throws: LabError.self) {
            try JobTiming(jobIndex: 0, backend: .metal, uses: 8, wallMS: 10,
                          uploadMS: 2, cpuComputeAllocateMS: nil, metalEncodeSubmitWaitMS: 5,
                          metalOutputCopyMS: 1, gpuMS: 4, gpuTimestampSamples: 7)
        }
    }

    @Test func enclosingClocksRejectOverlap() throws {
        #expect(try unassignedHostMS(total: 10, parts: [2, 5, 1]) == 2)
        #expect(throws: LabError.self) { try unassignedHostMS(total: 10, parts: [2, 5, 4]) }
        #expect(throws: LabError.self) { try unassignedHostMS(total: .nan, parts: []) }
        #expect(throws: LabError.self) { try unassignedHostMS(total: 10, parts: [-1]) }
    }

    @Test func signedPairedDifferencesRetainFasterProfiledRuns() throws {
        let summary = try PairedTimingDifference(profiled: [2, 1, 4], plain: [3, 3, 1])
        #expect(summary.rawMS == [-1, -2, 3] && summary.medianMS == -1)
        #expect(abs(summary.p10MS - (-1.8)) < 1e-10)
        #expect(abs(summary.p90MS - 2.2) < 1e-10)
        #expect(throws: LabError.self) { try PairedTimingDifference(profiled: [], plain: []) }
        #expect(throws: LabError.self) { try PairedTimingDifference(profiled: [1], plain: [1, 2]) }
        #expect(throws: LabError.self) { try PairedTimingDifference(profiled: [-1], plain: [1]) }
    }

    @Test func withinPairOrderIsJointlyBalancedWithAllOtherFactors() {
        let rounds = JobProfilePlan.warmupPairs..<(JobProfilePlan.warmupPairs + JobProfilePlan.measuredPairs)
        #expect(rounds.count == 96)
        for policy in MixedPolicy.allCases {
            var combinations = Set<String>()
            for round in rounds {
                let s = HeldOutSchedule(round: round), position = mixedPolicyOrder(round: round).firstIndex(of: policy)!
                let key = "\(position)/\(s.phase)/\(s.bankParity)/\(s.orderParity)/\(JobProfilePlan.profiledFirst(round: round))"
                #expect(combinations.insert(key).inserted)
            }
            #expect(combinations.count == 3 * 4 * 2 * 2 * 2)
        }
    }

    @Test func newScopeIsExplicitAndCannotBeCombined() throws {
        #expect(try LabOptions(arguments: ["--report", "profile.json", "--job-profile-only"]).scope == .jobProfileOnly)
        for flag in ["--mixed-only", "--held-out-only", "--job-profile-only"] {
            #expect(throws: LabError.self) {
                try LabOptions(arguments: ["--report", "profile.json", "--job-profile-only", flag])
            }
        }
        #expect(try LabOptions(arguments: ["--report", "out.json"]).scope == .all)
    }
}
