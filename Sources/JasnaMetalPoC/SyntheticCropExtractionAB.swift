import CoreVideo
import Foundation

enum SyntheticCropExtractionPolicy: String, CaseIterable, Encodable {
    case extractTwice = "control-extract-twice"
    case reuseOnce = "candidate-extract-once-reuse"
}

enum SyntheticCropSamplingKernel: String, CaseIterable, Encodable {
    case scalar = "control-scalar"
    case parallel = "candidate-parallel4"
}

struct SyntheticCropExtractionWork: Equatable, Encodable {
    let jobs: Int
    let extractionCalls: Int
    let consumerPasses: Int
    let outputElementsConsumed: Int
}

enum SyntheticCropExtractionABPlan {
    static let sourceWidth = 4_096
    static let sourceHeight = 4_096
    static let frameCount = 4
    static let regionCount = 2
    static let warmupPairs = 8
    static let measuredPairs = 48
    static let windowFrameCount = 30
    static let windowWarmupPairs = 4
    static let windowMeasuredPairs = 24
    static let kernelWarmupPairs = 8
    static let kernelMeasuredPairs = 48

    static var jobsPerTrial: Int { frameCount * regionCount }
    static var windowJobsPerTrial: Int { windowFrameCount * regionCount }

    static func work(
        for policy: SyntheticCropExtractionPolicy,
        jobs: Int = jobsPerTrial
    ) -> SyntheticCropExtractionWork {
        precondition(jobs > 0)
        return SyntheticCropExtractionWork(
            jobs: jobs,
            extractionCalls: jobs * (policy == .extractTwice ? 2 : 1),
            consumerPasses: jobs * 2,
            outputElementsConsumed: jobs * 2 * SideBySideRestoration.tileElements
        )
    }

    static func candidateFirst(round: Int) -> Bool {
        precondition(round >= 0)
        return round.isMultiple(of: 2)
    }
}

private struct SyntheticCropTimingSummary: Encodable {
    let samples: Int
    let rawMS: [Double]
    let medianMS: Double
    let p10MS: Double
    let p90MS: Double
    let meanMS: Double
    let standardDeviationMS: Double

    init(_ values: [Double]) throws {
        guard let statistics = BenchmarkStatistics(values) else {
            throw DeformConvError.commandFailed("invalid synthetic crop timing samples")
        }
        samples = values.count
        rawMS = values
        medianMS = statistics.median
        p10MS = statistics.percentile10
        p90MS = statistics.percentile90
        meanMS = statistics.mean
        standardDeviationMS = statistics.standardDeviation
    }
}

private struct SyntheticCropPairedSummary: Encodable {
    let samples: Int
    let rawCandidateMinusControlMS: [Double]
    let medianMS: Double
    let p10MS: Double
    let p90MS: Double
    let candidateFasterSamples: Int

    init(candidate: [Double], control: [Double]) throws {
        guard candidate.count == control.count, !candidate.isEmpty else {
            throw DeformConvError.commandFailed("synthetic crop timing pairs are incomplete")
        }
        let differences = zip(candidate, control).map { $0 - $1 }
        guard let statistics = BenchmarkStatistics(differences) else {
            throw DeformConvError.commandFailed("invalid synthetic crop paired timings")
        }
        samples = differences.count
        rawCandidateMinusControlMS = differences
        medianMS = statistics.median
        p10MS = statistics.percentile10
        p90MS = statistics.percentile90
        candidateFasterSamples = differences.filter { $0 < 0 }.count
    }
}

private struct SyntheticCropExtractionABReport: Encodable {
    let schemaVersion = 3
    let status = "PASS"
    let createdAt = ISO8601DateFormatter().string(from: Date())
    let sourceSize = "4096x4096"
    let modelOutputSize = "256x256x3-fp16"
    let projection = "fisheye"
    let frames = SyntheticCropExtractionABPlan.frameCount
    let regionsPerFrame = SyntheticCropExtractionABPlan.regionCount
    let jobsPerTrial = SyntheticCropExtractionABPlan.jobsPerTrial
    let warmupPairs = SyntheticCropExtractionABPlan.warmupPairs
    let measuredPairs = SyntheticCropExtractionABPlan.measuredPairs
    let controlWork = SyntheticCropExtractionABPlan.work(for: .extractTwice)
    let candidateWork = SyntheticCropExtractionABPlan.work(for: .reuseOnce)
    let windowLogicalFrames = SyntheticCropExtractionABPlan.windowFrameCount
    let windowJobsPerTrial = SyntheticCropExtractionABPlan.windowJobsPerTrial
    let windowWarmupPairs = SyntheticCropExtractionABPlan.windowWarmupPairs
    let windowMeasuredPairs = SyntheticCropExtractionABPlan.windowMeasuredPairs
    let windowControlWork = SyntheticCropExtractionABPlan.work(
        for: .extractTwice, jobs: SyntheticCropExtractionABPlan.windowJobsPerTrial
    )
    let windowCandidateWork = SyntheticCropExtractionABPlan.work(
        for: .reuseOnce, jobs: SyntheticCropExtractionABPlan.windowJobsPerTrial
    )
    let sourcePayloadBytes: Int
    let maximumRetainedCropBytes = SideBySideRestoration.tileBytes
    let windowRetainedCropBytes = SyntheticCropExtractionABPlan.windowJobsPerTrial
        * SideBySideRestoration.tileBytes
    let validationComparisons: Int
    let control: SyntheticCropTimingSummary
    let candidate: SyntheticCropTimingSummary
    let paired: SyntheticCropPairedSummary
    let windowControl: SyntheticCropTimingSummary
    let windowCandidate: SyntheticCropTimingSummary
    let windowPaired: SyntheticCropPairedSummary
    let kernelControl: SyntheticCropTimingSummary
    let kernelCandidate: SyntheticCropTimingSummary
    let kernelPaired: SyntheticCropPairedSummary
    let fixtureChecksums: [UInt64]
    let scope = "Generated 4096x4096 BGRA eye frames through the application MosaicCropSamplingMap and its production Float16 extraction method. No media, detector, model, restoration graph, compositor, encoder, or NPU."
    let candidateScope = "The candidate retains extracted 256x256x3 Float16 crops for two consumers instead of extracting an identical source/map pair twice. It is measured only; no restoration default or cache lifecycle changes."
    let kernelCandidateScope = "The parallel candidate divides each 256x256 fisheye crop into four disjoint CPU ranges and initializes the planar FP16 output without a redundant zero fill. It retains no crop data and remains diagnostic-only until exact output and a stable release-build gain are demonstrated."
    let timingScope = "Immediate trials process four generated frames x two fisheye regions, with eight warmup and 48 measured adjacent pairs. Delayed-window trials process 30 logical frames x two regions, cycle four physical generated buffers, retain all 60 candidate crops between consumer phases, and use four warmup plus 24 measured pairs. Control extracts twice/job; candidate once/job. Both checksum-consume every full array twice. Policy order alternates. Fixture/map construction and pre/post exact validation are excluded."
    let limitationScope = "This isolates duplicate crop extraction and candidate retention only. It does not include model execution time, decoded-buffer lifetime/pressure, concurrent eyes, restoration-cache resume behavior, compositing, encoding, or end-to-end video throughput."
}

private struct SyntheticCropObservation {
    let work: SyntheticCropExtractionWork
    let hashes: [UInt64]
}

private func syntheticCropPixelBuffer(frame: Int) throws -> CVPixelBuffer {
    let attributes: CFDictionary = [
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ] as CFDictionary
    var optionalBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        nil,
        SyntheticCropExtractionABPlan.sourceWidth,
        SyntheticCropExtractionABPlan.sourceHeight,
        kCVPixelFormatType_32BGRA,
        attributes,
        &optionalBuffer
    )
    guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
        throw DeformConvError.commandFailed("failed creating synthetic crop source")
    }
    guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
        throw DeformConvError.commandFailed("failed locking synthetic crop source")
    }
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        throw DeformConvError.commandFailed("synthetic crop source has no address")
    }
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<SyntheticCropExtractionABPlan.sourceHeight {
        let row = bytes.advanced(by: y * rowBytes)
        for x in 0..<SyntheticCropExtractionABPlan.sourceWidth {
            let pixel = row.advanced(by: x * 4)
            pixel[0] = UInt8(truncatingIfNeeded: x * 3 + y * 5 + frame * 17)
            pixel[1] = UInt8(truncatingIfNeeded: x * 7 + y * 2 + frame * 29)
            pixel[2] = UInt8(truncatingIfNeeded: x * 11 + y * 13 + frame * 31)
            pixel[3] = 255
        }
    }
    return buffer
}

private func syntheticCropMaps() -> [MosaicCropSamplingMap] {
    let regions = [
        MosaicRegion(
            startFrame: 0, endFrame: 30,
            x: 512, y: 2_048, width: 512, height: 512, confidence: 1
        ),
        MosaicRegion(
            startFrame: 0, endFrame: 30,
            x: 2_300, y: 2_600, width: 800, height: 600, confidence: 1
        ),
    ]
    return regions.map {
        MosaicCropSamplingMap(
            region: $0,
            eyeWidth: SyntheticCropExtractionABPlan.sourceWidth,
            eyeHeight: SyntheticCropExtractionABPlan.sourceHeight,
            projection: .fisheye
        )
    }
}

@inline(never)
private func consumeSyntheticCrop(_ values: [Float16]) throws -> UInt64 {
    guard values.count == SideBySideRestoration.tileElements else {
        throw DeformConvError.invalidShape
    }
    var hash: UInt64 = 0xcbf29ce484222325
    for value in values {
        hash ^= UInt64(value.bitPattern)
        hash &*= 0x100000001b3
    }
    return hash
}

private func runSyntheticCropPolicy(
    _ policy: SyntheticCropExtractionPolicy,
    frames: [CVPixelBuffer],
    maps: [MosaicCropSamplingMap]
) throws -> SyntheticCropObservation {
    var hashes = [UInt64]()
    hashes.reserveCapacity(SyntheticCropExtractionABPlan.jobsPerTrial * 2)
    for frame in frames {
        for map in maps {
            switch policy {
            case .extractTwice:
                hashes.append(try consumeSyntheticCrop(map.extractPlanarRGB(from: frame)))
                hashes.append(try consumeSyntheticCrop(map.extractPlanarRGB(from: frame)))
            case .reuseOnce:
                let values = try map.extractPlanarRGB(from: frame)
                hashes.append(try consumeSyntheticCrop(values))
                hashes.append(try consumeSyntheticCrop(values))
            }
        }
    }
    return SyntheticCropObservation(
        work: SyntheticCropExtractionABPlan.work(for: policy), hashes: hashes
    )
}

private func runSyntheticCropKernel(
    _ kernel: SyntheticCropSamplingKernel,
    frames: [CVPixelBuffer],
    maps: [MosaicCropSamplingMap]
) throws -> [UInt64] {
    var hashes = [UInt64]()
    hashes.reserveCapacity(SyntheticCropExtractionABPlan.jobsPerTrial)
    for frame in frames {
        for map in maps {
            let values: [Float16]
            switch kernel {
            case .scalar:
                values = try map.extractPlanarRGBSerial(from: frame)
            case .parallel:
                values = try map.extractPlanarRGBParallel(from: frame)
            }
            hashes.append(try consumeSyntheticCrop(values))
        }
    }
    return hashes
}

private func runSyntheticCropWindowPolicy(
    _ policy: SyntheticCropExtractionPolicy,
    frames: [CVPixelBuffer],
    maps: [MosaicCropSamplingMap]
) throws -> SyntheticCropObservation {
    let jobs = SyntheticCropExtractionABPlan.windowJobsPerTrial
    var firstPass = [UInt64]()
    firstPass.reserveCapacity(jobs)
    var retained = [[Float16]]()
    if policy == .reuseOnce { retained.reserveCapacity(jobs) }
    for logicalFrame in 0..<SyntheticCropExtractionABPlan.windowFrameCount {
        let frame = frames[logicalFrame % frames.count]
        for map in maps {
            let values = try map.extractPlanarRGB(from: frame)
            firstPass.append(try consumeSyntheticCrop(values))
            if policy == .reuseOnce { retained.append(values) }
        }
    }
    var secondPass = [UInt64]()
    secondPass.reserveCapacity(jobs)
    switch policy {
    case .extractTwice:
        for logicalFrame in 0..<SyntheticCropExtractionABPlan.windowFrameCount {
            let frame = frames[logicalFrame % frames.count]
            for map in maps {
                secondPass.append(try consumeSyntheticCrop(map.extractPlanarRGB(from: frame)))
            }
        }
    case .reuseOnce:
        guard retained.count == jobs else { throw DeformConvError.invalidShape }
        for values in retained { secondPass.append(try consumeSyntheticCrop(values)) }
    }
    return SyntheticCropObservation(
        work: SyntheticCropExtractionABPlan.work(for: policy, jobs: jobs),
        hashes: firstPass + secondPass
    )
}

private func syntheticCropElapsedMS<T>(_ body: () throws -> T) rethrows -> (T, Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let value = try body()
    return (value, Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
}

func runSyntheticCropExtractionAB(reportURL: URL) throws {
    #if DEBUG
    throw DeformConvError.commandFailed(
        "synthetic crop extraction timing requires a release build"
    )
    #else
    guard !FileManager.default.fileExists(atPath: reportURL.path) else {
        throw DeformConvError.commandFailed("refusing to overwrite synthetic crop report")
    }
    let frames = try (0..<SyntheticCropExtractionABPlan.frameCount).map {
        try syntheticCropPixelBuffer(frame: $0)
    }
    let maps = syntheticCropMaps()
    let expected = try runSyntheticCropPolicy(.extractTwice, frames: frames, maps: maps)
    let initialCandidate = try runSyntheticCropPolicy(.reuseOnce, frames: frames, maps: maps)
    guard expected.hashes == initialCandidate.hashes else {
        throw DeformConvError.commandFailed("synthetic crop candidate changed output pixels")
    }
    let expectedWindow = try runSyntheticCropWindowPolicy(
        .extractTwice, frames: frames, maps: maps
    )
    let initialWindowCandidate = try runSyntheticCropWindowPolicy(
        .reuseOnce, frames: frames, maps: maps
    )
    guard expectedWindow.hashes == initialWindowCandidate.hashes else {
        throw DeformConvError.commandFailed(
            "synthetic delayed-window crop candidate changed output pixels"
        )
    }
    let expectedKernel = try runSyntheticCropKernel(.scalar, frames: frames, maps: maps)
    let initialKernelCandidate = try runSyntheticCropKernel(
        .parallel, frames: frames, maps: maps
    )
    guard expectedKernel == initialKernelCandidate else {
        throw DeformConvError.commandFailed(
            "synthetic parallel crop candidate changed output pixels"
        )
    }

    var controlMS = [Double](), candidateMS = [Double]()
    controlMS.reserveCapacity(SyntheticCropExtractionABPlan.measuredPairs)
    candidateMS.reserveCapacity(SyntheticCropExtractionABPlan.measuredPairs)
    let totalPairs = SyntheticCropExtractionABPlan.warmupPairs
        + SyntheticCropExtractionABPlan.measuredPairs
    for round in 0..<totalPairs {
        let order: [SyntheticCropExtractionPolicy] = SyntheticCropExtractionABPlan
            .candidateFirst(round: round) ? [.reuseOnce, .extractTwice] : [.extractTwice, .reuseOnce]
        for policy in order {
            let result = try syntheticCropElapsedMS {
                try runSyntheticCropPolicy(policy, frames: frames, maps: maps)
            }
            guard result.0.work == SyntheticCropExtractionABPlan.work(for: policy),
                  result.0.hashes == expected.hashes
            else {
                throw DeformConvError.commandFailed("synthetic crop timed work changed")
            }
            if round >= SyntheticCropExtractionABPlan.warmupPairs {
                if policy == .extractTwice { controlMS.append(result.1) }
                else { candidateMS.append(result.1) }
            }
        }
    }
    let finalControl = try runSyntheticCropPolicy(.extractTwice, frames: frames, maps: maps)
    let finalCandidate = try runSyntheticCropPolicy(.reuseOnce, frames: frames, maps: maps)
    guard finalControl.hashes == expected.hashes,
          finalCandidate.hashes == expected.hashes,
          controlMS.count == SyntheticCropExtractionABPlan.measuredPairs,
          candidateMS.count == SyntheticCropExtractionABPlan.measuredPairs
    else { throw DeformConvError.commandFailed("synthetic crop final validation failed") }

    var windowControlMS = [Double](), windowCandidateMS = [Double]()
    let windowPairs = SyntheticCropExtractionABPlan.windowWarmupPairs
        + SyntheticCropExtractionABPlan.windowMeasuredPairs
    for round in 0..<windowPairs {
        let order: [SyntheticCropExtractionPolicy] = SyntheticCropExtractionABPlan
            .candidateFirst(round: round) ? [.reuseOnce, .extractTwice] : [.extractTwice, .reuseOnce]
        for policy in order {
            let result = try syntheticCropElapsedMS {
                try runSyntheticCropWindowPolicy(policy, frames: frames, maps: maps)
            }
            guard result.0.work == SyntheticCropExtractionABPlan.work(
                for: policy, jobs: SyntheticCropExtractionABPlan.windowJobsPerTrial
            ), result.0.hashes == expectedWindow.hashes else {
                throw DeformConvError.commandFailed(
                    "synthetic delayed-window crop timed work changed"
                )
            }
            if round >= SyntheticCropExtractionABPlan.windowWarmupPairs {
                if policy == .extractTwice { windowControlMS.append(result.1) }
                else { windowCandidateMS.append(result.1) }
            }
        }
    }
    let finalWindowControl = try runSyntheticCropWindowPolicy(
        .extractTwice, frames: frames, maps: maps
    )
    let finalWindowCandidate = try runSyntheticCropWindowPolicy(
        .reuseOnce, frames: frames, maps: maps
    )
    guard finalWindowControl.hashes == expectedWindow.hashes,
          finalWindowCandidate.hashes == expectedWindow.hashes,
          windowControlMS.count == SyntheticCropExtractionABPlan.windowMeasuredPairs,
          windowCandidateMS.count == SyntheticCropExtractionABPlan.windowMeasuredPairs
    else {
        throw DeformConvError.commandFailed(
            "synthetic delayed-window crop final validation failed"
        )
    }

    var kernelControlMS = [Double](), kernelCandidateMS = [Double]()
    let kernelPairs = SyntheticCropExtractionABPlan.kernelWarmupPairs
        + SyntheticCropExtractionABPlan.kernelMeasuredPairs
    for round in 0..<kernelPairs {
        let order: [SyntheticCropSamplingKernel] = SyntheticCropExtractionABPlan
            .candidateFirst(round: round) ? [.parallel, .scalar] : [.scalar, .parallel]
        for kernel in order {
            let result = try syntheticCropElapsedMS {
                try runSyntheticCropKernel(kernel, frames: frames, maps: maps)
            }
            guard result.0 == expectedKernel else {
                throw DeformConvError.commandFailed(
                    "synthetic parallel crop timed work changed output pixels"
                )
            }
            if round >= SyntheticCropExtractionABPlan.kernelWarmupPairs {
                if kernel == .scalar { kernelControlMS.append(result.1) }
                else { kernelCandidateMS.append(result.1) }
            }
        }
    }
    guard try runSyntheticCropKernel(.scalar, frames: frames, maps: maps) == expectedKernel,
          try runSyntheticCropKernel(.parallel, frames: frames, maps: maps) == expectedKernel,
          kernelControlMS.count == SyntheticCropExtractionABPlan.kernelMeasuredPairs,
          kernelCandidateMS.count == SyntheticCropExtractionABPlan.kernelMeasuredPairs
    else {
        throw DeformConvError.commandFailed(
            "synthetic parallel crop final validation failed"
        )
    }

    let report = try SyntheticCropExtractionABReport(
        sourcePayloadBytes: frames.reduce(0) {
            $0 + CVPixelBufferGetBytesPerRow($1) * CVPixelBufferGetHeight($1)
        },
        validationComparisons: (SyntheticCropExtractionABPlan.jobsPerTrial
            + SyntheticCropExtractionABPlan.windowJobsPerTrial) * 4,
        control: SyntheticCropTimingSummary(controlMS),
        candidate: SyntheticCropTimingSummary(candidateMS),
        paired: SyntheticCropPairedSummary(candidate: candidateMS, control: controlMS),
        windowControl: SyntheticCropTimingSummary(windowControlMS),
        windowCandidate: SyntheticCropTimingSummary(windowCandidateMS),
        windowPaired: SyntheticCropPairedSummary(
            candidate: windowCandidateMS, control: windowControlMS
        ),
        kernelControl: SyntheticCropTimingSummary(kernelControlMS),
        kernelCandidate: SyntheticCropTimingSummary(kernelCandidateMS),
        kernelPaired: SyntheticCropPairedSummary(
            candidate: kernelCandidateMS, control: kernelControlMS
        ),
        fixtureChecksums: stride(from: 0, to: expected.hashes.count, by: 2).map {
            expected.hashes[$0]
        }
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: reportURL, options: .withoutOverwriting)
    print("Synthetic application crop-extraction A/B: PASS")
    print("Production path: 4 generated 4096×4096 eye frames × 2 fisheye maps → 256×256×3 FP16")
    print(String(format: "Control extract twice: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.control.medianMS, report.control.p10MS, report.control.p90MS))
    print(String(format: "Candidate reuse once: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.candidate.medianMS, report.candidate.p10MS, report.candidate.p90MS))
    print(String(format: "Paired candidate−control: %+.3f ms [P10 %+.3f–P90 %+.3f]; faster %d/%d",
                 report.paired.medianMS, report.paired.p10MS, report.paired.p90MS,
                 report.paired.candidateFasterSamples, report.paired.samples))
    print(String(format: "Delayed 30-frame control: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.windowControl.medianMS, report.windowControl.p10MS,
                 report.windowControl.p90MS))
    print(String(format: "Delayed 30-frame reuse: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.windowCandidate.medianMS, report.windowCandidate.p10MS,
                 report.windowCandidate.p90MS))
    print(String(format: "Delayed paired candidate−control: %+.3f ms [P10 %+.3f–P90 %+.3f]; faster %d/%d; retained %.2f MiB/eye",
                 report.windowPaired.medianMS, report.windowPaired.p10MS,
                 report.windowPaired.p90MS, report.windowPaired.candidateFasterSamples,
                 report.windowPaired.samples,
                 Double(report.windowRetainedCropBytes) / 1_048_576))
    print(String(format: "Scalar sampler once: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.kernelControl.medianMS, report.kernelControl.p10MS,
                 report.kernelControl.p90MS))
    print(String(format: "Parallel×4 sampler once: %.3f ms [P10 %.3f–P90 %.3f]",
                 report.kernelCandidate.medianMS, report.kernelCandidate.p10MS,
                 report.kernelCandidate.p90MS))
    print(String(format: "Parallel×4 paired candidate−control: %+.3f ms [P10 %+.3f–P90 %+.3f]; faster %d/%d; exact FP16 output",
                 report.kernelPaired.medianMS, report.kernelPaired.p10MS,
                 report.kernelPaired.p90MS, report.kernelPaired.candidateFasterSamples,
                 report.kernelPaired.samples))
    print("Report: \(reportURL.path)")
    #endif
}
