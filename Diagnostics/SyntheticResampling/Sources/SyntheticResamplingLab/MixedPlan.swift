import Foundation

struct MixedShape: Hashable, Encodable {
    let width: Int
    let height: Int
    var pixels: Int { width * height }

    init(_ width: Int, _ height: Int) throws {
        guard (1...1_024).contains(width), (1...1_024).contains(height) else {
            throw LabError.invalid("Mixed synthetic dimensions must be within 1...1024")
        }
        self.width = width
        self.height = height
    }
}

struct MixedJob: Encodable {
    let shape: MixedShape
    let usesPerInput: Int
    let inputs = 4
    var outputs: Int { inputs * usesPerInput }

    init(_ width: Int, _ height: Int, uses: Int) throws {
        guard [1, 2, 4, 8].contains(uses) else { throw LabError.invalid("Invalid mixed-job reuse count") }
        shape = try MixedShape(width, height)
        usesPerInput = uses
    }
}

enum MixedBackend: String, Encodable, Hashable { case cpu, metal }

protocol MixedRoutingPolicy {
    func backend(for job: MixedJob) -> MixedBackend
}

enum MixedPolicy: String, CaseIterable, Encodable, Hashable, MixedRoutingPolicy {
    case cpu = "cpu-only-parallel4"
    case metal = "metal-only-group4"
    case hybrid = "experimental-hybrid"

    // Fixed before this experiment, based on the earlier isolated M4 results.
    // No online fitting, future timing lookup, or unmeasured-size extrapolation.
    func backend(for job: MixedJob) -> MixedBackend {
        switch self {
        case .cpu: .cpu
        case .metal: .metal
        case .hybrid:
            job.shape.width == 512 && job.shape.height == 512 && job.usesPerInput >= 4 ? .metal : .cpu
        }
    }
}

func mixedPolicyOrder(round: Int) -> [MixedPolicy] {
    precondition(round >= 0)
    let modes = MixedPolicy.allCases
    return (0..<modes.count).map { modes[($0 + round % modes.count) % modes.count] }
}

func mixedJobOrder(count: Int, round: Int) -> [Int] {
    precondition(count > 0 && round >= 0)
    return round % 2 == 0 ? Array(0..<count) : Array((0..<count).reversed())
}

func mixedBankIndex(round: Int, job: Int) -> Int {
    precondition(round >= 0 && job >= 0)
    return (round % 2 + job % 2) % 2
}

struct MixedPlan {
    let name: String
    let jobs: [MixedJob]

    static func fixtures() throws -> [MixedPlan] {
        let jobs = try [MixedJob(96, 80, uses: 1), MixedJob(512, 512, uses: 8),
                        MixedJob(256, 256, uses: 2), MixedJob(384, 384, uses: 4),
                        MixedJob(512, 512, uses: 1), MixedJob(256, 256, uses: 8),
                        MixedJob(96, 80, uses: 8), MixedJob(512, 512, uses: 4)]
        return [MixedPlan(name: "mixed-reuse", jobs: jobs),
                MixedPlan(name: "fresh-every-job", jobs: try jobs.map {
                    try MixedJob($0.shape.width, $0.shape.height, uses: 1)
                })]
    }

    func expectedWork(policy: MixedPolicy, round: Int) -> MixedWork {
        expectedWork(routing: policy, round: round)
    }

    func expectedSelectorWork(policy: SelectorPolicy, round: Int) -> MixedWork {
        expectedWork(routing: policy, round: round)
    }

    func expectedBoundaryWork(policy: BoundaryPolicy, round: Int) -> MixedWork {
        expectedWork(routing: policy, round: round)
    }

    func expectedWork<Policy: MixedRoutingPolicy>(routing policy: Policy, round: Int) -> MixedWork {
        var work = MixedWork(), previous: MixedBackend?
        for index in mixedJobOrder(count: jobs.count, round: round) {
            let job = jobs[index], backend = policy.backend(for: job)
            work.jobs += 1
            work.outputs += job.outputs
            work.outputPixels += job.outputs * job.shape.pixels
            if backend == .cpu {
                work.cpuJobs += 1
                work.cpuOutputs += job.outputs
            } else {
                work.metalJobs += 1
                work.metalOutputs += job.outputs
                work.uploads += job.inputs
                work.submissions += job.usesPerInput
            }
            if let previous, previous != backend { work.backendSwitches += 1 }
            previous = backend
        }
        return work
    }
}

struct MixedWork: Encodable, Equatable {
    var jobs = 0, outputs = 0, outputPixels = 0
    var cpuJobs = 0, metalJobs = 0, cpuOutputs = 0, metalOutputs = 0
    var uploads = 0, submissions = 0, backendSwitches = 0
}
