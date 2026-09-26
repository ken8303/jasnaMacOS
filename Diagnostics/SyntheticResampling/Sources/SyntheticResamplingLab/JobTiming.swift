import Foundation

struct MetalHostTiming {
    let encodeSubmitWaitMS: Double
    let outputCopyMS: Double
}

// GPU timestamps overlap encode/submit/wait; they are deliberately NOT part of
// the additive host-stage accounting. CPU output allocation is part of compute.
struct JobTiming: Encodable {
    let jobIndex: Int
    let backend: MixedBackend
    let uses: Int
    let wallMS: Double
    let uploadMS: Double?
    let cpuComputeAllocateMS: Double?
    let metalEncodeSubmitWaitMS: Double?
    let metalOutputCopyMS: Double?
    let otherHostMS: Double
    let gpuMS: Double?
    let gpuTimestampSamples: Int

    init(jobIndex: Int, backend: MixedBackend, uses: Int, wallMS: Double,
         uploadMS: Double?, cpuComputeAllocateMS: Double?, metalEncodeSubmitWaitMS: Double?,
         metalOutputCopyMS: Double?, gpuMS: Double?, gpuTimestampSamples: Int) throws {
        try require(jobIndex >= 0 && [1, 2, 4, 8].contains(uses), "Invalid profiled job")
        let stages = [uploadMS, cpuComputeAllocateMS, metalEncodeSubmitWaitMS, metalOutputCopyMS]
        try require(stages.compactMap { $0 }.allSatisfy { $0.isFinite && $0 >= 0 }, "Invalid job stage timing")
        if backend == .cpu {
            try require(cpuComputeAllocateMS != nil && uploadMS == nil && metalEncodeSubmitWaitMS == nil &&
                        metalOutputCopyMS == nil && gpuMS == nil && gpuTimestampSamples == 0,
                        "CPU profile cannot contain GPU stages")
        } else {
            try require(uploadMS != nil && cpuComputeAllocateMS == nil && metalEncodeSubmitWaitMS != nil &&
                        metalOutputCopyMS != nil && (0...uses).contains(gpuTimestampSamples),
                        "Metal profile stages are incomplete")
            try require((gpuMS != nil) == (gpuTimestampSamples == uses), "Partial GPU timestamps cannot form a job total")
        }
        if let gpuMS { try require(gpuMS.isFinite && gpuMS > 0, "Invalid GPU timestamp duration") }
        otherHostMS = try unassignedHostMS(total: wallMS, parts: stages.compactMap { $0 })
        self.jobIndex = jobIndex; self.backend = backend; self.uses = uses; self.wallMS = wallMS
        self.uploadMS = uploadMS; self.cpuComputeAllocateMS = cpuComputeAllocateMS
        self.metalEncodeSubmitWaitMS = metalEncodeSubmitWaitMS; self.metalOutputCopyMS = metalOutputCopyMS
        self.gpuMS = gpuMS; self.gpuTimestampSamples = gpuTimestampSamples
    }
}

func unassignedHostMS(total: Double, parts: [Double]) throws -> Double {
    try require(total.isFinite && total >= 0 && parts.allSatisfy { $0.isFinite && $0 >= 0 },
                "Invalid host timing accounting")
    let remainder = total - parts.reduce(0, +)
    // One nanosecond allowance for floating-point conversion/summation, not an
    // allowance to hide overlapping stages or subtract the GPU time twice.
    try require(remainder >= -0.000_001, "Host stages exceed the enclosing stopwatch")
    return max(0, remainder)
}

func elapsedMS(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}

enum JobProfilePlan {
    static let warmupPairs = 16
    static let measuredPairs = 96

    static func profiledFirst(round: Int) -> Bool {
        precondition(round >= 0)
        return (round / 16) % 2 == 0
    }
}

// Signed differences need their own summary: a noisy profiled run can be faster
// than its plain partner. Do not clamp negatives or label this a pure timer cost.
struct PairedTimingDifference: Encodable {
    let rawMS: [Double]
    let samples: Int
    let medianMS: Double
    let p10MS: Double
    let p90MS: Double

    init(profiled: [Double], plain: [Double]) throws {
        try require(!profiled.isEmpty && profiled.count == plain.count &&
                    (profiled + plain).allSatisfy { $0.isFinite && $0 >= 0 }, "Invalid paired timing samples")
        rawMS = zip(profiled, plain).map { $0 - $1 }
        samples = rawMS.count
        // Reuse the established percentile definition, translated to nonnegative
        // values; preserve the original signed raw samples in the report.
        let offset = min(0, rawMS.min()!)
        let shifted = try TimingSummary(rawMS.map { $0 - offset })
        medianMS = shifted.medianMS + offset
        p10MS = shifted.p10MS + offset
        p90MS = shifted.p90MS + offset
    }
}
