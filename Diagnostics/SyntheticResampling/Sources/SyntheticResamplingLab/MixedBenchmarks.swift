import Foundation

private struct MixedFixture {
    let coordinates: [[SIMD2<Float>]]
    let reference: [[SIMD4<Float>]]
    var checksums: [MixedBackend: Double] = [:]
}

private struct MixedBank {
    let sources: [Raster]
    var fixtures: [MixedShape: MixedFixture]
}

struct MixedObservation {
    let work: MixedWork
    let checksums: [Double]
    let gpuMS: Double?
    let cpuError: Double
    let metalError: Double
    let jobTimings: [JobTiming]?
}

struct MixedResourceUsage: Encodable {
    let reusableGPUResourceBytes: Int
    let hostSourcePayloadBytes: Int
    let hostCoordinatePayloadBytes: Int
    let hostOraclePayloadBytes: Int
    let largestFourOutputPayloadBytes: Int
}

struct MixedMeasurement: Encodable {
    let policy: MixedPolicy
    let backendsByJob: [MixedBackend]
    let trialWall: TimingSummary
    let perOutputWall: TimingSummary
    let trialGPU: TimingSummary?
    let gpuTimestampSamples: Int
    let expectedWork: MixedWork
    let observedWork: [MixedWork]
    let observedJobChecksums: [[Double]]
    let maxCPUReferenceError: Double?
    let maxMetalReferenceError: Double?
}

struct MixedBenchmark: Encodable {
    let name: String
    let jobs: [MixedJob]
    let warmupTrials = 4
    let measuredTrials = 24
    let measurements: [MixedMeasurement]
}

struct MixedReport: Encodable {
    let sourceSize = "1024x1024"
    let fixtureBanks = 2
    let setupMS: Double
    let reusableGPUResourceBytes: Int
    let hostSourcePayloadBytes: Int
    let hostCoordinatePayloadBytes: Int
    let hostOraclePayloadBytes: Int
    let largestFourOutputPayloadBytes: Int
    let benchmarks: [MixedBenchmark]
    let policyRule = "Fixed experimental M4 rule: Metal group4 only for exactly 512x512 outputs with at least four known uses per input; parallel CPU otherwise. No online fitting or application policy."
    let timingScope = "Whole mixed trial timed: policy selection, resource lookup, synchronous CPU worker scheduling, fresh uploads at the start of every Metal job, GPU submissions/waits, CPU-readable output allocation/copy, checksums and workload counters. No CPU/GPU overlap, no skipped outputs, no reuse across jobs. Pipeline/resource allocation, fixture/oracle generation and full-pixel validation excluded. Per-output figures amortize a mixture of sizes and are not individual-call latency or video FPS."
    let orderScope = "Three-policy order rotates over 24 measured trials; each policy occupies each position eight times. Jobs alternate forward/reverse order. Bank=(trial+jobIndex)%2 selects one of two generated banks. All policies see identical data and job order in each trial. Every policy-position/bank-order pair occurs four times."
    let qualityScope = "Every pixel on every use is checked before and after timing, in both job orders and both bank assignments, with CPU error below 1e-6 and Metal below 1/255. Every timed job checksum and actual upload/submission/output workload must match its validated expectation. Hybrid CPU and Metal outputs need not be bit-identical; their separate error budgets remain in force."
    let memoryScope = "One four-slot GPU pool per distinct output shape, shared by all policies; CPU policy performs no uploads but these comparison pools remain allocated. Host input/oracle payloads are shared across jobs. Only one four-output batch per use is retained. Counts exclude temporary validation arrays, thread/driver/runtime overhead, and process peak RSS."
}

private struct MixedSamples {
    var wall = [Double](), gpu = [Double]()
    var work = [MixedWork](), checksums = [[Double]]()
    var cpuError = 0.0, metalError = 0.0
}

final class MixedHarness {
    let context: MetalContext
    private let pools: [MixedShape: [MetalSampler]]
    private var banks: [MixedBank]

    init(context: MetalContext, shapes: [MixedShape], frameOffsets: [Int] = [0, 8]) throws {
        try require(!shapes.isEmpty && Set(shapes).count == shapes.count,
                    "Mixed pools require distinct, nonempty shapes")
        try require(frameOffsets.count == 2 && Set(frameOffsets).count == 2 &&
                    frameOffsets.allSatisfy({ (0...1_000).contains($0) }),
                    "Mixed fixtures require two distinct, bounded frame offsets")
        self.context = context
        pools = try Dictionary(uniqueKeysWithValues: shapes.map { shape in
            (shape, try (0..<4).map { _ in
                try MetalSampler(context: context, width: 1_024, height: 1_024, count: shape.pixels)
            })
        })
        banks = frameOffsets.map { offset in
            let sources = (0..<4).map { Raster.generated(width: 1_024, height: 1_024, frame: offset + $0) }
            let fixtures = Dictionary(uniqueKeysWithValues: shapes.map { shape in
                let coordinates = sources.indices.map {
                    movingCoordinates(source: sources[$0], width: shape.width, height: shape.height, frame: offset + $0)
                }
                let reference = sources.indices.map { referenceSample(sources[$0], coordinates: coordinates[$0]) }
                return (shape, MixedFixture(coordinates: coordinates, reference: reference))
            })
            return MixedBank(sources: sources, fixtures: fixtures)
        }
        // Establish independent per-shape/bank/backend expectations before any
        // timed trial. Retain small checksums, not another full GPU output bank.
        for shape in shapes {
            for bankIndex in banks.indices {
                for backend in [MixedBackend.cpu, .metal] {
                    let checksum = try autoreleasepool {
                        let bank = banks[bankIndex], fixture = bank.fixtures[shape]!, slots = pools[shape]!
                        if backend == .metal { try upload(bank: bank, fixture: fixture, slots: slots) }
                        let output = try sample(backend: backend, bank: bank, fixture: fixture, slots: slots)
                        _ = try pixelError(output.frames, reference: fixture.reference, backend: backend)
                        return try consumeFourOutputs(output.frames, pixelsPerFrame: shape.pixels)
                    }
                    banks[bankIndex].fixtures[shape]!.checksums[backend] = checksum
                }
            }
            for backend in [MixedBackend.cpu, .metal] {
                try require(banks[0].fixtures[shape]!.checksums[backend] != banks[1].fixtures[shape]!.checksums[backend],
                            "Mixed input banks must have distinct checksums")
            }
        }
    }

    private func upload(bank: MixedBank, fixture: MixedFixture, slots: [MetalSampler]) throws {
        for index in bank.sources.indices { try slots[index].upload(bank.sources[index], coordinates: fixture.coordinates[index]) }
    }

    private func sample(backend: MixedBackend, bank: MixedBank, fixture: MixedFixture,
                        slots: [MetalSampler], profileHost: Bool = false) throws -> MetalGroupOutput {
        if backend == .metal { return try MetalSampler.sampleResidentGroup(slots, profileHost: profileHost) }
        return MetalGroupOutput(frames: try bank.sources.indices.map {
            try optimizedCPUSample(bank.sources[$0], coordinates: fixture.coordinates[$0], workers: 4)
        }, gpuMS: nil)
    }

    private func pixelError(_ frames: [[SIMD4<Float>]], reference: [[SIMD4<Float>]],
                            backend: MixedBackend) throws -> Double {
        try require(frames.count == 4 && reference.count == 4, "Mixed validation output count changed")
        var error = 0.0
        for index in frames.indices { error = max(error, try maximumError(frames[index], reference[index])) }
        try require(error < (backend == .cpu ? 0.000_001 : 1.0 / 255), "Mixed \(backend.rawValue) pixel gate failed: \(error)")
        return error
    }

    private var uploads: Int { pools.values.reduce(0) { $0 + $1.reduce(0) { $0 + $1.uploadCount } } }

    func validateProfileParity() throws -> Int {
        var checks = 0
        for shape in pools.keys.sorted(by: { $0.pixels < $1.pixels }) {
            for bank in banks {
                for backend in [MixedBackend.cpu, .metal] {
                    try autoreleasepool {
                        let fixture = bank.fixtures[shape]!, slots = pools[shape]!
                        if backend == .metal { try upload(bank: bank, fixture: fixture, slots: slots) }
                        let plain = try sample(backend: backend, bank: bank, fixture: fixture, slots: slots)
                        let profiled = try sample(backend: backend, bank: bank, fixture: fixture, slots: slots, profileHost: true)
                        try require(plain.hostTiming == nil && (profiled.hostTiming != nil) == (backend == .metal),
                                    "Sampler instrumentation was not opt-in")
                        for index in 0..<4 {
                            try require(try maximumError(plain.frames[index], profiled.frames[index]) == 0,
                                        "Profiler changed output pixels")
                        }
                        _ = try pixelError(profiled.frames, reference: fixture.reference, backend: backend)
                    }
                    checks += 1
                }
            }
        }
        return checks
    }

    // Separate bank/order controls permit a balanced held-out schedule. Omitting
    // bankRound preserves the original mixed experiment's exact input schedule.
    func trial<Policy: MixedRoutingPolicy>(plan: MixedPlan, policy: Policy, round: Int, bankRound: Int? = nil,
               verifyPixels: Bool, profileJobs: Bool = false) throws -> MixedObservation {
        // Full-pixel checks would dominate a job profile. Run them separately.
        try require(!(profileJobs && verifyPixels), "Pixel validation is outside job profiling")
        let beforeUploads = uploads, beforeSubmissions = context.submissionCount
        var work = MixedWork(), checksums = [Double](repeating: 0, count: plan.jobs.count)
        var previous: MixedBackend?, gpuTimes = [Double](), cpuError = 0.0, metalError = 0.0
        var jobTimings: [JobTiming]? = profileJobs ? [] : nil
        for jobIndex in mixedJobOrder(count: plan.jobs.count, round: round) {
            let jobStart = profileJobs ? DispatchTime.now().uptimeNanoseconds : nil
            var uploadMS = 0.0, cpuMS = 0.0, executeMS = 0.0, copyMS = 0.0, jobGPU = 0.0, gpuSamples = 0
            let job = plan.jobs[jobIndex]
            // Routing and resource lookup are inside the whole-trial stopwatch.
            let backend = policy.backend(for: job)
            let bank = banks[mixedBankIndex(round: bankRound ?? round, job: jobIndex)]
            let fixture = bank.fixtures[job.shape]!, slots = pools[job.shape]!
            if backend == .metal {
                // Always refresh all four inputs, even if a prior job/trial used
                // this pool. Only repetitions WITHIN this job get resident reuse.
                let uploadStart = profileJobs ? DispatchTime.now().uptimeNanoseconds : nil
                try upload(bank: bank, fixture: fixture, slots: slots)
                if let uploadStart { uploadMS = elapsedMS(since: uploadStart) }
                work.metalJobs += 1
            } else { work.cpuJobs += 1 }
            if let previous, previous != backend { work.backendSwitches += 1 }
            previous = backend
            work.jobs += 1
            for _ in 0..<job.usesPerInput {
                let cpuStart = profileJobs && backend == .cpu ? DispatchTime.now().uptimeNanoseconds : nil
                let output = try sample(backend: backend, bank: bank, fixture: fixture, slots: slots, profileHost: profileJobs)
                if let cpuStart { cpuMS += elapsedMS(since: cpuStart) }
                if let host = output.hostTiming {
                    executeMS += host.encodeSubmitWaitMS
                    copyMS += host.outputCopyMS
                }
                if profileJobs, let gpuMS = output.gpuMS { jobGPU += gpuMS; gpuSamples += 1 }
                checksums[jobIndex] += try consumeFourOutputs(output.frames, pixelsPerFrame: job.shape.pixels)
                work.outputs += output.frames.count
                work.outputPixels += output.frames.reduce(0) { $0 + $1.count }
                if backend == .cpu { work.cpuOutputs += output.frames.count }
                else { work.metalOutputs += output.frames.count }
                if let gpuMS = output.gpuMS { gpuTimes.append(gpuMS) }
                if verifyPixels {
                    let error = try pixelError(output.frames, reference: fixture.reference, backend: backend)
                    if backend == .cpu { cpuError = max(cpuError, error) }
                    else { metalError = max(metalError, error) }
                }
            }
            if let jobStart {
                let wallMS = elapsedMS(since: jobStart)
                jobTimings!.append(try JobTiming(
                    jobIndex: jobIndex, backend: backend, uses: job.usesPerInput, wallMS: wallMS,
                    uploadMS: backend == .metal ? uploadMS : nil,
                    cpuComputeAllocateMS: backend == .cpu ? cpuMS : nil,
                    metalEncodeSubmitWaitMS: backend == .metal ? executeMS : nil,
                    metalOutputCopyMS: backend == .metal ? copyMS : nil,
                    gpuMS: gpuSamples == job.usesPerInput ? jobGPU : nil, gpuTimestampSamples: gpuSamples))
            }
        }
        work.uploads = uploads - beforeUploads
        work.submissions = context.submissionCount - beforeSubmissions
        return MixedObservation(work: work, checksums: checksums,
                                gpuMS: !gpuTimes.isEmpty && gpuTimes.count == work.submissions ? gpuTimes.reduce(0, +) : nil,
                                cpuError: cpuError, metalError: metalError, jobTimings: jobTimings)
    }

    func verify<Policy: MixedRoutingPolicy>(_ observation: MixedObservation, plan: MixedPlan, policy: Policy, round: Int,
                bankRound: Int? = nil) throws {
        try require(observation.work == plan.expectedWork(routing: policy, round: round), "Mixed trial workload changed")
        let expectedChecksums = plan.jobs.enumerated().map { index, job in
            let checksum = banks[mixedBankIndex(round: bankRound ?? round, job: index)].fixtures[job.shape]!.checksums[policy.backend(for: job)]!
            // Same accumulation order as the trial, checked exactly against the
            // selected backend. Full-pixel gates complement this small checksum.
            return (0..<job.usesPerInput).reduce(0.0) { sum, _ in sum + checksum }
        }
        try require(observation.checksums == expectedChecksums, "Mixed trial has stale or incorrect job outputs")
        if let timings = observation.jobTimings {
            try require(timings.map(\.jobIndex) == mixedJobOrder(count: plan.jobs.count, round: round),
                        "Profile omitted or reordered jobs")
            for timing in timings {
                let job = plan.jobs[timing.jobIndex]
                try require(timing.backend == policy.backend(for: job) && timing.uses == job.usesPerInput,
                            "Profile routing or reuse changed")
            }
        }
    }

    var resourceUsage: MixedResourceUsage {
        let allFixtures = banks.flatMap { $0.fixtures.values }
        return MixedResourceUsage(
            reusableGPUResourceBytes: pools.values.flatMap { $0 }.reduce(0) { $0 + $1.reusableGPUResourceBytes },
            hostSourcePayloadBytes: banks.flatMap(\.sources).reduce(0) { $0 + $1.bytes.count },
            hostCoordinatePayloadBytes: allFixtures.flatMap(\.coordinates).reduce(0) { $0 + $1.count * MemoryLayout<SIMD2<Float>>.stride },
            hostOraclePayloadBytes: allFixtures.flatMap(\.reference).reduce(0) { $0 + $1.count * MemoryLayout<SIMD4<Float>>.stride },
            largestFourOutputPayloadBytes: 4 * pools.keys.map(\.pixels).max()! * MemoryLayout<SIMD4<Float>>.stride
        )
    }

    func benchmark(_ plan: MixedPlan) throws -> MixedBenchmark {
        var records = Dictionary(uniqueKeysWithValues: MixedPolicy.allCases.map { ($0, MixedSamples()) })
        func validateAllPixels() throws {
            for policy in MixedPolicy.allCases {
                for parity in 0...1 {
                    let result = try trial(plan: plan, policy: policy, round: parity, verifyPixels: true)
                    try verify(result, plan: plan, policy: policy, round: parity)
                    records[policy]!.cpuError = max(records[policy]!.cpuError, result.cpuError)
                    records[policy]!.metalError = max(records[policy]!.metalError, result.metalError)
                }
            }
        }
        try validateAllPixels()
        for round in 0..<28 {
            for policy in mixedPolicyOrder(round: round) {
                let result = try timed { try trial(plan: plan, policy: policy, round: round, verifyPixels: false) }
                try verify(result.value, plan: plan, policy: policy, round: round)
                if round >= 4 {
                    records[policy]!.wall.append(result.milliseconds)
                    if let gpuMS = result.value.gpuMS { records[policy]!.gpu.append(gpuMS) }
                    records[policy]!.work.append(result.value.work)
                    records[policy]!.checksums.append(result.value.checksums)
                }
            }
        }
        try validateAllPixels()
        let measurements = try MixedPolicy.allCases.map { policy in
            let record = records[policy]!, expected = plan.expectedWork(policy: policy, round: 0)
            return MixedMeasurement(policy: policy, backendsByJob: plan.jobs.map { policy.backend(for: $0) },
                                    trialWall: try TimingSummary(record.wall),
                                    perOutputWall: try perImageTimings(record.wall, imagesPerRound: expected.outputs),
                                    trialGPU: record.gpu.isEmpty ? nil : try TimingSummary(record.gpu),
                                    gpuTimestampSamples: record.gpu.count, expectedWork: expected,
                                    observedWork: record.work, observedJobChecksums: record.checksums,
                                    maxCPUReferenceError: expected.cpuJobs == 0 ? nil : record.cpuError,
                                    maxMetalReferenceError: expected.metalJobs == 0 ? nil : record.metalError)
        }
        return MixedBenchmark(name: plan.name, jobs: plan.jobs, measurements: measurements)
    }
}

func benchmarkMixed(context: MetalContext) throws -> MixedReport {
    let plans = try MixedPlan.fixtures()
    let shapes = Set(plans.flatMap { $0.jobs.map(\.shape) }).sorted { $0.pixels < $1.pixels }
    let setup = try timed { try MixedHarness(context: context, shapes: shapes) }
    let harness = setup.value
    print(String(format: "Mixed fixture/oracle/pool setup and initial checks: %.3f ms (excluded from timings)", setup.milliseconds))
    var results = [MixedBenchmark]()
    for plan in plans {
        let result = try autoreleasepool { try harness.benchmark(plan) }
        results.append(result)
        print("PASS mixed workload \(result.name): \(result.jobs.count) jobs, \(result.measurements[0].expectedWork.outputs) output arrays/trial")
        for measurement in result.measurements {
            print(String(format: "  %@: whole trial %.3f ms [P10 %.3f–P90 %.3f]; amortized %.3f ms/output; CPU/Metal jobs %d/%d; uploads/submissions %d/%d",
                         measurement.policy.rawValue, measurement.trialWall.medianMS, measurement.trialWall.p10MS,
                         measurement.trialWall.p90MS, measurement.perOutputWall.medianMS,
                         measurement.expectedWork.cpuJobs, measurement.expectedWork.metalJobs,
                         measurement.expectedWork.uploads, measurement.expectedWork.submissions))
        }
    }
    let resources = harness.resourceUsage
    return MixedReport(setupMS: setup.milliseconds,
                       reusableGPUResourceBytes: resources.reusableGPUResourceBytes,
                       hostSourcePayloadBytes: resources.hostSourcePayloadBytes,
                       hostCoordinatePayloadBytes: resources.hostCoordinatePayloadBytes,
                       hostOraclePayloadBytes: resources.hostOraclePayloadBytes,
                       largestFourOutputPayloadBytes: resources.largestFourOutputPayloadBytes,
                       benchmarks: results)
}
