import Foundation

struct SelectorTrial: Encodable {
    let round: Int
    let schedule: HeldOutSchedule
    let wallMS: Double
    let gpuMS: Double?
    let work: MixedWork
    let jobChecksums: [Double]
}

struct SelectorPhase: Encodable {
    let phase: Int
    let jobs: [MixedJob]
    let backendsByJob: [MixedBackend]
    let expectedWork: MixedWork
    let trialWall: TimingSummary
    let maxCPUReferenceError: Double?
    let maxMetalReferenceError: Double?
}

struct SelectorMeasurement: Encodable {
    let policy: SelectorPolicy
    let trialWall: TimingSummary
    let phases: [SelectorPhase]
    let trials: [SelectorTrial]
}

struct SelectorComparison: Encodable {
    let candidateMinusControl: PairedTimingDifference
    let candidateFasterSamples: Int
    let samples: Int
    let phases: [SelectorPhaseComparison]
}

struct SelectorPhaseComparison: Encodable {
    let phase: Int
    let candidateMinusControl: PairedTimingDifference
    let candidateFasterSamples: Int
    let samples: Int
}

struct SelectorCandidateReport: Encodable {
    let sourceSize = "1024x1024"
    let frameOffsets = HeldOutPlan.frameOffsets
    let warmupTrials = SelectorCandidatePlan.warmupTrials
    let measuredTrials = SelectorCandidatePlan.measuredTrials
    let validationScenarios = HeldOutPlan.validationScenarios
    let setupMS: Double
    let resources: MixedResourceUsage
    let measurements: [SelectorMeasurement]
    let comparison: SelectorComparison
    let candidateRule = "In this diagnostic only: retain the frozen 512x512 >=4-use Metal route and additionally route exactly 640x512 to Metal at >=4 uses. One/two-use 640x512 jobs remain on CPU."
    let scheduleScope = "Four policies share identical jobs/data/order per trial. Sixteen warmups cover all phase/bank/order scenarios. Sixty-four measured trials balance four policy positions x four phases x two banks x two job orders exactly once per policy. Each phase has 16 samples."
    let timingScope = "Whole serial trials include routing, lookup, CPU worker scheduling or fresh four-input Metal upload, submission/wait, CPU-readable output allocation/copy, checksums and workload counters. No CPU/GPU overlap or cross-job reuse. Setup and full-pixel validation excluded. Candidate-minus-control uses same-round signed pairs; negative is faster."
    let qualityScope = "Before and after timing, all pixels on every use are checked for 16 phase/bank/order scenarios x four policies. Every timed route, array/pixel/upload/submission count, and exact backend-specific checksum is verified. CPU error <1e-6; Metal <1/255."
    let limitationScope = "Generated ramp/checkerboard patterns only. The candidate is not an application policy, a universal threshold, a model benchmark, or a video-speed/quality result. Existing standard diagnostic modes and the frozen selector remain unchanged."
}

private struct SelectorSamples {
    var trials = [SelectorTrial]()
    var cpuErrors = [Double](repeating: 0, count: 4)
    var metalErrors = [Double](repeating: 0, count: 4)
}

func benchmarkSelectorCandidate(context: MetalContext) throws -> SelectorCandidateReport {
    let plans = try HeldOutPlan.phases()
    let shapes = Set(plans.flatMap { $0.jobs.map(\.shape) }).sorted { $0.pixels < $1.pixels }
    let setup = try timed { try MixedHarness(context: context, shapes: shapes, frameOffsets: HeldOutPlan.frameOffsets) }
    let harness = setup.value
    print(String(format: "Selector-candidate fixture/oracle/pool setup: %.3f ms (excluded from timings)", setup.milliseconds))
    var records = Dictionary(uniqueKeysWithValues: SelectorPolicy.allCases.map { ($0, SelectorSamples()) })

    func validatePixels() throws {
        for policy in SelectorPolicy.allCases {
            for index in 0..<HeldOutPlan.validationScenarios {
                let schedule = HeldOutSchedule(round: index), plan = plans[schedule.phase]
                let output = try harness.trial(plan: plan, policy: policy, round: schedule.orderParity,
                                               bankRound: schedule.bankParity, verifyPixels: true)
                try harness.verify(output, plan: plan, policy: policy, round: schedule.orderParity,
                                   bankRound: schedule.bankParity)
                records[policy]!.cpuErrors[schedule.phase] = max(records[policy]!.cpuErrors[schedule.phase], output.cpuError)
                records[policy]!.metalErrors[schedule.phase] = max(records[policy]!.metalErrors[schedule.phase], output.metalError)
            }
        }
    }

    try validatePixels()
    print("PASS selector-candidate pre-timing pixels/workloads: 16 schedules × 4 policies")
    for round in 0..<(SelectorCandidatePlan.warmupTrials + SelectorCandidatePlan.measuredTrials) {
        let schedule = HeldOutSchedule(round: round), plan = plans[schedule.phase]
        for policy in SelectorCandidatePlan.policyOrder(round: round) {
            let result = try timed {
                let selected = HeldOutSchedule(round: round)
                return try harness.trial(plan: plans[selected.phase], policy: policy, round: selected.orderParity,
                                         bankRound: selected.bankParity, verifyPixels: false)
            }
            try harness.verify(result.value, plan: plan, policy: policy, round: schedule.orderParity,
                               bankRound: schedule.bankParity)
            if round >= SelectorCandidatePlan.warmupTrials {
                records[policy]!.trials.append(SelectorTrial(
                    round: round, schedule: schedule, wallMS: result.milliseconds, gpuMS: result.value.gpuMS,
                    work: result.value.work, jobChecksums: result.value.checksums))
            }
        }
    }
    try validatePixels()
    print("PASS selector-candidate post-timing pixels/workloads: 16 schedules × 4 policies")

    let measurements = try SelectorPolicy.allCases.map { policy in
        let record = records[policy]!
        try require(record.trials.count == SelectorCandidatePlan.measuredTrials,
                    "Selector-candidate trial sampling is incomplete")
        let phases = try plans.indices.map { phase in
            let plan = plans[phase], trials = record.trials.filter { $0.schedule.phase == phase }
            let expected = plan.expectedSelectorWork(policy: policy, round: 0)
            try require(trials.count == 16, "Selector-candidate phase sampling is unbalanced")
            return SelectorPhase(
                phase: phase, jobs: plan.jobs, backendsByJob: plan.jobs.map { policy.backend(for: $0) },
                expectedWork: expected, trialWall: try TimingSummary(trials.map(\.wallMS)),
                maxCPUReferenceError: expected.cpuJobs == 0 ? nil : record.cpuErrors[phase],
                maxMetalReferenceError: expected.metalJobs == 0 ? nil : record.metalErrors[phase])
        }
        return SelectorMeasurement(policy: policy, trialWall: try TimingSummary(record.trials.map(\.wallMS)),
                                   phases: phases, trials: record.trials)
    }

    let control = records[.control]!.trials, candidate = records[.candidate]!.trials
    try require(control.map(\.round) == candidate.map(\.round), "Selector comparison pairs are not aligned")
    func difference(_ control: [SelectorTrial], _ candidate: [SelectorTrial]) throws -> PairedTimingDifference {
        try PairedTimingDifference(profiled: candidate.map(\.wallMS), plain: control.map(\.wallMS))
    }
    let phaseComparisons = try plans.indices.map { phase in
        let baseline = control.filter { $0.schedule.phase == phase }
        let proposed = candidate.filter { $0.schedule.phase == phase }
        return SelectorPhaseComparison(
            phase: phase, candidateMinusControl: try difference(baseline, proposed),
            candidateFasterSamples: zip(proposed, baseline).filter { $0.wallMS < $1.wallMS }.count,
            samples: baseline.count)
    }
    let comparison = SelectorComparison(
        candidateMinusControl: try difference(control, candidate),
        candidateFasterSamples: zip(candidate, control).filter { $0.wallMS < $1.wallMS }.count,
        samples: control.count, phases: phaseComparisons)

    print("PASS selector candidate: 64 measured trials/policy; 16 samples/reuse phase")
    for measurement in measurements {
        print(String(format: "  %@: whole-trial median %.3f ms [P10 %.3f–P90 %.3f]",
                     measurement.policy.rawValue, measurement.trialWall.medianMS,
                     measurement.trialWall.p10MS, measurement.trialWall.p90MS))
        for phase in measurement.phases {
            print(String(format: "    phase %d: %.3f ms; CPU/Metal jobs %d/%d; uploads/submissions %d/%d",
                         phase.phase, phase.trialWall.medianMS, phase.expectedWork.cpuJobs,
                         phase.expectedWork.metalJobs, phase.expectedWork.uploads, phase.expectedWork.submissions))
        }
    }
    print(String(format: "  candidate-control paired difference: %+.3f ms [P10 %+.3f–P90 %+.3f]; candidate faster %d/%d trials",
                 comparison.candidateMinusControl.medianMS, comparison.candidateMinusControl.p10MS,
                 comparison.candidateMinusControl.p90MS, comparison.candidateFasterSamples, comparison.samples))
    return SelectorCandidateReport(setupMS: setup.milliseconds, resources: harness.resourceUsage,
                                   measurements: measurements, comparison: comparison)
}
