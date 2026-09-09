import Foundation

struct HeldOutTrial: Encodable {
    let round: Int
    let schedule: HeldOutSchedule
    let wallMS: Double
    let gpuMS: Double?
    let work: MixedWork
    let jobChecksums: [Double]
}

struct HeldOutPhaseMeasurement: Encodable {
    let phase: Int
    let jobs: [MixedJob]
    let backendsByJob: [MixedBackend]
    let expectedWork: MixedWork
    let trialWall: TimingSummary
    let perOutputWall: TimingSummary
    let maxCPUReferenceError: Double?
    let maxMetalReferenceError: Double?
}

struct HeldOutMeasurement: Encodable {
    let policy: MixedPolicy
    let trialWall: TimingSummary
    let perOutputWall: TimingSummary
    let gpuTimedTrials: Int
    let phases: [HeldOutPhaseMeasurement]
    let trials: [HeldOutTrial]
}

struct HeldOutReport: Encodable {
    let sourceSize = "1024x1024"
    let frameOffsets = HeldOutPlan.frameOffsets
    let warmupTrials = HeldOutPlan.warmupTrials
    let measuredTrials = HeldOutPlan.measuredTrials
    let validationScenarios = HeldOutPlan.validationScenarios
    let setupMS: Double
    let resources: MixedResourceUsage
    let measurements: [HeldOutMeasurement]
    let policyScope = "The existing rule is frozen: Metal only at exactly 512x512 with >=4 known uses/input, parallel CPU otherwise. Four new shapes therefore always take its CPU fallback. 512x512 controls change CPU/Metal routing as reuse changes. This tests limitations; it does not learn or extend a size threshold."
    let scheduleScope = "Eight jobs, four outputs/use; each job cycles through 1/2/4/8 uses in four phases. Every trial returns 120 arrays, but pixel work differs by phase. 48 measured trials balance three policy positions, four reuse phases, two input-bank parities and two job orders: each joint combination once per policy. Sixteen warmups cover all phase/bank/order scenarios. Per-phase timings have 12 samples, aggregate timings 48."
    let timingScope = "Whole trials include schedule/plan selection, routing, pool lookup, worker scheduling, fresh uploads for each Metal job, submission/wait, CPU-readable output allocation/copy, checksums and actual workload counting. Known reuse is provided metadata, not inferred. No uploads or GPU work on CPU routes; no reuse across jobs or CPU/GPU overlap. Setup/pools/fixtures/oracles and full-pixel validation excluded. Amortized mixed output times are not single-output latency or video FPS."
    let qualityScope = "Before and after timing, every pixel on every use is checked in all 16 phase/bank/order combinations for all three policies. CPU max error <1e-6; Metal <1/255. Every measured trial verifies its actual arrays/pixels/uploads/submissions/routes and exact per-job checksums. Unused-backend errors and absent GPU timestamps are omitted, not zero."
    let limitationScope = "New dimensions and source-frame offsets, but the same generated ramp/checkerboard family. Not an independent real-image dataset or a restoration-quality test. This does not validate quality across CPU/Metal transitions in an application. All resources remain resident during comparison; byte counts are payload/resource accounting, not peak RSS or a production memory limit."
}

private struct HeldOutSamples {
    var trials = [HeldOutTrial]()
    var cpuErrors = [Double](repeating: 0, count: 4)
    var metalErrors = [Double](repeating: 0, count: 4)
}

func benchmarkHeldOut(context: MetalContext) throws -> HeldOutReport {
    let plans = try HeldOutPlan.phases()
    let shapes = Set(plans.flatMap { $0.jobs.map(\.shape) }).sorted { $0.pixels < $1.pixels }
    let setup = try timed { try MixedHarness(context: context, shapes: shapes, frameOffsets: HeldOutPlan.frameOffsets) }
    let harness = setup.value
    print(String(format: "Held-out fixture/oracle/pool setup and initial checks: %.3f ms (excluded from timings)", setup.milliseconds))
    var records = Dictionary(uniqueKeysWithValues: MixedPolicy.allCases.map { ($0, HeldOutSamples()) })

    func validatePixels() throws {
        for policy in MixedPolicy.allCases {
            for index in 0..<HeldOutPlan.validationScenarios {
                let schedule = HeldOutSchedule(round: index), plan = plans[schedule.phase]
                let output = try harness.trial(plan: plan, policy: policy, round: schedule.orderParity,
                                               bankRound: schedule.bankParity, verifyPixels: true)
                try harness.verify(output, plan: plan, policy: policy, round: schedule.orderParity, bankRound: schedule.bankParity)
                records[policy]!.cpuErrors[schedule.phase] = max(records[policy]!.cpuErrors[schedule.phase], output.cpuError)
                records[policy]!.metalErrors[schedule.phase] = max(records[policy]!.metalErrors[schedule.phase], output.metalError)
            }
        }
    }
    try validatePixels()
    print("PASS held-out pre-timing pixels/workloads: 16 schedules × 3 policies")
    for round in 0..<(HeldOutPlan.warmupTrials + HeldOutPlan.measuredTrials) {
        for policy in mixedPolicyOrder(round: round) {
            let result = try timed {
                // Include schedule and plan selection in the measurement, not
                // just the CPU/GPU execution selected from that metadata.
                let schedule = HeldOutSchedule(round: round)
                return try harness.trial(plan: plans[schedule.phase], policy: policy, round: schedule.orderParity,
                                         bankRound: schedule.bankParity, verifyPixels: false)
            }
            let schedule = HeldOutSchedule(round: round)
            try harness.verify(result.value, plan: plans[schedule.phase], policy: policy,
                               round: schedule.orderParity, bankRound: schedule.bankParity)
            if round >= HeldOutPlan.warmupTrials {
                records[policy]!.trials.append(HeldOutTrial(round: round, schedule: schedule, wallMS: result.milliseconds,
                                                          gpuMS: result.value.gpuMS, work: result.value.work,
                                                          jobChecksums: result.value.checksums))
            }
        }
    }
    try validatePixels()
    print("PASS held-out post-timing pixels/workloads: 16 schedules × 3 policies")
    let measurements = try MixedPolicy.allCases.map { policy in
        let record = records[policy]!
        let phases = try plans.indices.map { phase in
            let plan = plans[phase], expected = plan.expectedWork(policy: policy, round: 0)
            let trials = record.trials.filter { $0.schedule.phase == phase }
            try require(trials.count == 12, "Held-out phase sampling is unbalanced")
            return HeldOutPhaseMeasurement(
                phase: phase, jobs: plan.jobs, backendsByJob: plan.jobs.map { policy.backend(for: $0) }, expectedWork: expected,
                trialWall: try TimingSummary(trials.map(\.wallMS)),
                perOutputWall: try TimingSummary(trials.map { $0.wallMS / Double($0.work.outputs) }),
                maxCPUReferenceError: expected.cpuJobs == 0 ? nil : record.cpuErrors[phase],
                maxMetalReferenceError: expected.metalJobs == 0 ? nil : record.metalErrors[phase]
            )
        }
        try require(record.trials.count == HeldOutPlan.measuredTrials, "Held-out trial sampling is incomplete")
        return HeldOutMeasurement(policy: policy, trialWall: try TimingSummary(record.trials.map(\.wallMS)),
                                  perOutputWall: try TimingSummary(record.trials.map { $0.wallMS / Double($0.work.outputs) }),
                                  gpuTimedTrials: record.trials.filter { $0.gpuMS != nil }.count, phases: phases, trials: record.trials)
    }
    print("PASS held-out changing-reuse workload: 48 measured trials/policy, 120 output arrays/trial")
    for measurement in measurements {
        print(String(format: "  %@: whole trial %.3f ms [P10 %.3f–P90 %.3f]; amortized %.3f ms/output",
                     measurement.policy.rawValue, measurement.trialWall.medianMS, measurement.trialWall.p10MS,
                     measurement.trialWall.p90MS, measurement.perOutputWall.medianMS))
        for phase in measurement.phases {
            print(String(format: "    reuse phase %d: %.3f ms [%.3f–%.3f]; CPU/Metal jobs %d/%d; uploads/submissions %d/%d",
                         phase.phase, phase.trialWall.medianMS, phase.trialWall.p10MS, phase.trialWall.p90MS,
                         phase.expectedWork.cpuJobs, phase.expectedWork.metalJobs,
                         phase.expectedWork.uploads, phase.expectedWork.submissions))
        }
    }
    return HeldOutReport(setupMS: setup.milliseconds, resources: harness.resourceUsage, measurements: measurements)
}
