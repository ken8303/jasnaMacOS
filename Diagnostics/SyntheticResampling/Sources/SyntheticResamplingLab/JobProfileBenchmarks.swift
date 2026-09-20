import Foundation

struct JobProfilePair: Encodable {
    let round: Int
    let schedule: HeldOutSchedule
    let profiledFirst: Bool
    let plainWallMS: Double
    let profiledWallMS: Double
    let outsideJobsMS: Double
    let work: MixedWork
    let jobChecksums: [Double]
    let jobs: [JobTiming] // Execution order; jobIndex remains the stable identity.
}

struct JobStageSummary: Encodable {
    let jobIndex: Int
    let job: MixedJob
    let backend: MixedBackend
    let wall: TimingSummary
    let upload: TimingSummary?
    let cpuComputeAllocate: TimingSummary?
    let metalEncodeSubmitWait: TimingSummary?
    let metalOutputCopy: TimingSummary?
    let otherHost: TimingSummary
    let gpu: TimingSummary?
    let gpuTimedJobs: Int

    init(jobIndex: Int, job: MixedJob, backend: MixedBackend, records: [JobTiming]) throws {
        try require(records.count == JobProfilePlan.measuredPairs / 4 && records.allSatisfy {
            $0.jobIndex == jobIndex && $0.backend == backend && $0.uses == job.usesPerInput
        }, "Per-job phase sampling is incomplete")
        func summary(_ key: KeyPath<JobTiming, Double?>, present: Bool) throws -> TimingSummary? {
            let values = records.compactMap { $0[keyPath: key] }
            try require(values.count == (present ? records.count : 0), "Missing or inapplicable job stage")
            return values.isEmpty ? nil : try TimingSummary(values)
        }
        self.jobIndex = jobIndex; self.job = job; self.backend = backend
        wall = try TimingSummary(records.map(\.wallMS))
        upload = try summary(\.uploadMS, present: backend == .metal)
        cpuComputeAllocate = try summary(\.cpuComputeAllocateMS, present: backend == .cpu)
        metalEncodeSubmitWait = try summary(\.metalEncodeSubmitWaitMS, present: backend == .metal)
        metalOutputCopy = try summary(\.metalOutputCopyMS, present: backend == .metal)
        otherHost = try TimingSummary(records.map(\.otherHostMS))
        let gpuValues = records.compactMap(\.gpuMS)
        gpuTimedJobs = gpuValues.count
        gpu = gpuValues.isEmpty ? nil : try TimingSummary(gpuValues)
    }
}

struct JobProfilePhase: Encodable {
    let phase: Int
    let plainWall: TimingSummary
    let profiledWall: TimingSummary
    let pairedDifference: PairedTimingDifference
    let maxCPUReferenceError: Double?
    let maxMetalReferenceError: Double?
    let jobs: [JobStageSummary]
}

struct JobProfileMeasurement: Encodable {
    let policy: MixedPolicy
    let plainWall: TimingSummary
    let profiledWall: TimingSummary
    let pairedDifference: PairedTimingDifference
    let phases: [JobProfilePhase]
    let pairs: [JobProfilePair]
}

struct JobProfileReport: Encodable {
    let sourceSize = "1024x1024"
    let frameOffsets = HeldOutPlan.frameOffsets
    let warmupPairs = JobProfilePlan.warmupPairs
    let measuredPairs = JobProfilePlan.measuredPairs
    let validationScenarios = HeldOutPlan.validationScenarios
    let profileParityChecks: Int
    let setupMS: Double
    let resources: MixedResourceUsage
    let measurements: [JobProfileMeasurement]
    let policyScope = "Unchanged experimental rule: Metal only for exactly 512x512 with >=4 uses/input. Profiling does not change routing, kernels, coordinates, copies, or the application."
    let scheduleScope = "96 measured pairs/policy, 24 pairs/phase, following 16 warmup pairs. Each pair runs plain and profiled trials on identical data/order/work, with a fresh upload at each Metal job. Three policy positions x four phases x two banks x two job orders x two within-pair orders are jointly balanced once per policy. Jobs 3 and 7 both produce 640x512 outputs, retained separately; zero-based job indices."
    let timingScope = "Opt-in job clocks split upload (including validation), CPU compute+allocation+worker scheduling, Metal host encode/submit/wait, and copying Metal output buffers into CPU arrays. Other-host remainder includes lookup/routing, checksums, counters, output lifetime cleanup, and instrumentation; it is not pure routing cost. Outside-jobs remainder includes trial bookkeeping and recording/validating profiles. All host stages are nonoverlapping; GPU timestamps overlap encode/submit/wait and must NOT be added to host totals. Per-stage medians need not sum to the median whole job. No CPU/GPU overlap or cross-job resident reuse. Setup and full-pixel validation excluded."
    let overheadScope = "Raw paired differences are profiled-minus-plain whole-trial wall times, including profiling work and runtime/scheduling/cache noise. Negative differences are retained; this is not a calibrated timer cost or a correction subtracted from stage timings. Plain partners also execute the disabled instrumentation branches. No cross-version speedup claim."
    let qualityScope = "Identity/motion/resident gates plus pre/post full-pixel checks in all 16 schedules x 3 policies. Profiled sampling is also bit-identical to plain sampling for every shape/bank/backend before and after timing. Each timed pair verifies both actual workloads and exact per-job checksums, and identical plain/profiled results. CPU <1e-6, Metal <1/255. GPU durations require every submission timestamp in that job; missing values are omitted."
    let limitationScope = "Generated ramp/checkerboard data only, not a restoration or real-video benchmark. All comparison GPU pools coexist even during CPU routes; resource/payload accounting is not peak RSS. Known reuse counts are supplied, not predicted. No application defaults change."
}

private struct JobProfileSamples {
    var pairs = [JobProfilePair]()
    var cpuErrors = [Double](repeating: 0, count: 4)
    var metalErrors = [Double](repeating: 0, count: 4)
}

func benchmarkJobProfiles(context: MetalContext) throws -> JobProfileReport {
    let plans = try HeldOutPlan.phases()
    let shapes = Set(plans.flatMap { $0.jobs.map(\.shape) }).sorted { $0.pixels < $1.pixels }
    let setup = try timed { try MixedHarness(context: context, shapes: shapes, frameOffsets: HeldOutPlan.frameOffsets) }
    let harness = setup.value
    print(String(format: "Job-profile fixture/oracle/pool setup: %.3f ms (excluded from timings)", setup.milliseconds))
    var records = Dictionary(uniqueKeysWithValues: MixedPolicy.allCases.map { ($0, JobProfileSamples()) })
    var parityChecks = 0

    func validatePixels() throws {
        parityChecks += try harness.validateProfileParity()
        for policy in MixedPolicy.allCases {
            for index in 0..<HeldOutPlan.validationScenarios {
                let schedule = HeldOutSchedule(round: index), plan = plans[schedule.phase]
                let output = try harness.trial(plan: plan, policy: policy, round: schedule.orderParity,
                                               bankRound: schedule.bankParity, verifyPixels: true)
                try harness.verify(output, plan: plan, policy: policy, round: schedule.orderParity, bankRound: schedule.bankParity)
                try require(output.jobTimings == nil, "Plain validation unexpectedly collected job profiles")
                records[policy]!.cpuErrors[schedule.phase] = max(records[policy]!.cpuErrors[schedule.phase], output.cpuError)
                records[policy]!.metalErrors[schedule.phase] = max(records[policy]!.metalErrors[schedule.phase], output.metalError)
            }
        }
    }
    try validatePixels()
    print("PASS job-profile pre-timing pixels, instrumentation parity, and all 16 schedules × 3 policies")
    for round in 0..<(JobProfilePlan.warmupPairs + JobProfilePlan.measuredPairs) {
        let schedule = HeldOutSchedule(round: round), plan = plans[schedule.phase]
        let profileFirst = JobProfilePlan.profiledFirst(round: round)
        for policy in mixedPolicyOrder(round: round) {
            var plain: (value: MixedObservation, milliseconds: Double)?
            var profiled: (value: MixedObservation, milliseconds: Double)?
            for enabled in profileFirst ? [true, false] : [false, true] {
                let result = try timed {
                    let selected = HeldOutSchedule(round: round)
                    return try harness.trial(plan: plans[selected.phase], policy: policy, round: selected.orderParity,
                                             bankRound: selected.bankParity, verifyPixels: false, profileJobs: enabled)
                }
                // Validate after BOTH partners to avoid contaminating the second
                // partner with additional verification work between them.
                if enabled { profiled = result } else { plain = result }
            }
            let baseline = plain!, instrumented = profiled!
            for observation in [baseline.value, instrumented.value] {
                try harness.verify(observation, plan: plan, policy: policy, round: schedule.orderParity, bankRound: schedule.bankParity)
            }
            try require(baseline.value.jobTimings == nil && instrumented.value.jobTimings != nil,
                        "Trial profiling was not opt-in")
            try require(baseline.value.work == instrumented.value.work && baseline.value.checksums == instrumented.value.checksums,
                        "Profiler changed work or output checksums")
            let jobs = instrumented.value.jobTimings!
            let outsideJobs = try unassignedHostMS(total: instrumented.milliseconds, parts: jobs.map(\.wallMS))
            if round >= JobProfilePlan.warmupPairs {
                records[policy]!.pairs.append(JobProfilePair(
                    round: round, schedule: schedule, profiledFirst: profileFirst,
                    plainWallMS: baseline.milliseconds, profiledWallMS: instrumented.milliseconds,
                    outsideJobsMS: outsideJobs, work: baseline.value.work, jobChecksums: baseline.value.checksums, jobs: jobs))
            }
        }
    }
    try validatePixels()
    print("PASS job-profile post-timing pixels, instrumentation parity, and all 16 schedules × 3 policies")
    let measurements = try MixedPolicy.allCases.map { policy in
        let record = records[policy]!
        try require(record.pairs.count == JobProfilePlan.measuredPairs, "Job-profile pairs incomplete")
        let phases = try plans.indices.map { phase in
            let plan = plans[phase], pairs = record.pairs.filter { $0.schedule.phase == phase }
            let work = plan.expectedWork(policy: policy, round: 0)
            let jobs = try plan.jobs.indices.map { index in
                try JobStageSummary(jobIndex: index, job: plan.jobs[index], backend: policy.backend(for: plan.jobs[index]),
                                    records: pairs.map { $0.jobs.first { $0.jobIndex == index }! })
            }
            return JobProfilePhase(phase: phase, plainWall: try TimingSummary(pairs.map(\.plainWallMS)),
                                   profiledWall: try TimingSummary(pairs.map(\.profiledWallMS)),
                                   pairedDifference: try PairedTimingDifference(profiled: pairs.map(\.profiledWallMS), plain: pairs.map(\.plainWallMS)),
                                   maxCPUReferenceError: work.cpuJobs == 0 ? nil : record.cpuErrors[phase],
                                   maxMetalReferenceError: work.metalJobs == 0 ? nil : record.metalErrors[phase], jobs: jobs)
        }
        return JobProfileMeasurement(policy: policy, plainWall: try TimingSummary(record.pairs.map(\.plainWallMS)),
                                     profiledWall: try TimingSummary(record.pairs.map(\.profiledWallMS)),
                                     pairedDifference: try PairedTimingDifference(profiled: record.pairs.map(\.profiledWallMS), plain: record.pairs.map(\.plainWallMS)),
                                     phases: phases, pairs: record.pairs)
    }
    print("PASS per-job profiling: 96 measured pairs/policy; 24 samples/job/reuse phase")
    for measurement in measurements {
        let difference = measurement.pairedDifference
        print(String(format: "  %@: plain %.3f ms; profiled %.3f ms; paired difference %+.3f ms [P10 %+.3f–P90 %+.3f]",
                     measurement.policy.rawValue, measurement.plainWall.medianMS, measurement.profiledWall.medianMS,
                     difference.medianMS, difference.p10MS, difference.p90MS))
        // Keep console output focused; JSON contains every job/phase and raw pair.
        for job in measurement.phases[0].jobs where job.job.shape.width == 640 {
            if job.backend == .cpu {
                print(String(format: "    job %d, 640x512 ×8: wall %.3f ms; CPU compute/allocation %.3f ms; other %.3f ms",
                             job.jobIndex, job.wall.medianMS, job.cpuComputeAllocate!.medianMS, job.otherHost.medianMS))
            } else {
                print(String(format: "    job %d, 640x512 ×8: wall %.3f ms; upload %.3f ms; encode/submit/wait %.3f ms; output copy %.3f ms; other %.3f ms",
                             job.jobIndex, job.wall.medianMS, job.upload!.medianMS, job.metalEncodeSubmitWait!.medianMS,
                             job.metalOutputCopy!.medianMS, job.otherHost.medianMS))
            }
        }
    }
    return JobProfileReport(profileParityChecks: parityChecks, setupMS: setup.milliseconds,
                            resources: harness.resourceUsage, measurements: measurements)
}
