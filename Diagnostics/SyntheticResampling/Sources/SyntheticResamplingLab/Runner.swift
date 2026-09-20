import Foundation

struct Validation: Codable {
    let name: String
    let frames: Int
    let sampledPixels: Int
    let maxCPUError: Double
    let maxSIMDCPUError: Double
    let maxParallelCPUError: Double
    let maxMetalError: Double
    let maxTemporalError: Double
    let cpuTolerance: Double
    let metalTolerance: Double
    let temporalTolerance: Double
}

struct Report: Encodable {
    let schemaVersion = 9
    var status = "RUNNING"
    var device: String?
    var pipelineSetupMS: Double?
    var validations = [Validation]()
    var residentSafetyChecks = [String]()
    var benchmarks = [Benchmark]()
    var reuseBenchmarks = [ReuseBenchmark]()
    var requestedBenchmarks = "all"
    var mixedBenchmarks: MixedReport?
    var heldOutBenchmarks: HeldOutReport?
    var jobProfileBenchmarks: JobProfileReport?
    var selectorCandidateBenchmarks: SelectorCandidateReport?
    var boundaryHoldoutBenchmarks: BoundaryHoldoutReport?
    var error: String?
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let createdAt = ISO8601DateFormatter().string(from: Date())
    let activeLogicalProcessors = ProcessInfo.processInfo.activeProcessorCount
    let parallelCPUChunks = 4
    let scope = "Standalone generated RGBA patterns only; no restoration code, models, video, detector or NPU."
    let timingScope = "Steady-state benchmarks: release build, four outputs per round, four warm-up rounds, 24 measured rounds, rotating six-mode order. Per-image times are amortized batch times, not individual-call latency. Scalar Swift and SIMD serial/four-chunk CPU, not Accelerate. All Metal modes include submission, wait and output-array copies; only upload mode includes per-image uploads. Resident initial uploads are reported separately. Fixture/oracle/pipeline/resource construction excluded."
    let cpuScope = "SIMD CPU uses direct RGBA8 gathers and four-lane Float arithmetic without changing coordinate or output formats. Both SIMD modes include finite-input checks and output allocation. Four-chunk mode includes synchronous GCD scheduling; it requests four chunks, not affinity to four cores. No preconverted image or interpolation-plan cache. Scalar CPU is the original reference baseline; no claim that these are the best possible CPU implementations."
    let reuseTimingScope = "Upload-inclusive reuse benchmarks: 1/2/4/8 uses of each of four inputs, two alternating generated input banks, four warm-up trials and 24 measured trials per mode/case. Each resident trial uploads all four inputs inside its stopwatch before any use. Upload-per-call mode uploads before every output. CPU input arrays are already available. All outputs are copied and checksum-consumed on every use. Timed counters and checksums are verified outside the stopwatch; full-pixel comparisons run before/after timing. Per-output time is whole-trial time divided by 4*uses. Buffers/pipeline allocation and fixture generation are excluded; this is not a cold-process benchmark."
    let memoryScope = "Reusable Metal resource allocations and host array payloads, not process peak RSS or driver/compiler overhead."
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw LabError.invalid(message) }
}

func validateIdentity(context: MetalContext) throws -> Validation {
    let source = Raster.generated(width: 31, height: 19, frame: 0)
    var coordinates = (0..<(source.width * source.height)).map {
        SIMD2<Float>(Float($0 % source.width), Float($0 / source.width))
    }
    // Include corners and far out-of-bounds positions to test clamp-to-edge.
    for y: Float in [-100, -0.5, 0, 18, 18.5, 100] {
        for x: Float in [-100, -0.5, 0, 30, 30.5, 100] { coordinates.append(SIMD2(x, y)) }
    }
    let sampler = try MetalSampler(context: context, width: source.width, height: source.height, count: coordinates.count)
    let expected = referenceSample(source, coordinates: coordinates)
    let cpuError = try maximumError(cpuSample(source, coordinates: coordinates), expected)
    let simdError = try maximumError(try optimizedCPUSample(source, coordinates: coordinates), expected)
    let parallelError = try maximumError(try optimizedCPUSample(source, coordinates: coordinates, workers: 4), expected)
    let actual = try sampler.sample(source, coordinates: coordinates).pixels
    let gpuError = try maximumError(actual, expected)
    try require(max(cpuError, simdError, parallelError) < 0.000_001 && gpuError < 0.000_01,
                "Identity/edge alignment failed: CPU \(cpuError)/\(simdError)/\(parallelError), Metal \(gpuError)")
    let repeatError = try maximumError(try sampler.sample(source, coordinates: coordinates).pixels, actual)
    try require(repeatError == 0, "Identical GPU input was not repeatable")
    return Validation(name: "identity-and-clamped-edges", frames: 1, sampledPixels: coordinates.count,
                      maxCPUError: cpuError, maxSIMDCPUError: simdError, maxParallelCPUError: parallelError,
                      maxMetalError: gpuError, maxTemporalError: 0,
                      cpuTolerance: 0.000_001, metalTolerance: 0.000_01, temporalTolerance: 0)
}

func validateMotion(context: MetalContext) throws -> Validation {
    let width = 96, height = 80, frames = 24
    let sampler = try MetalSampler(context: context, width: 257, height: 193, count: width * height)
    var cpuError = 0.0, gpuError = 0.0, temporalError = 0.0
    var simdError = 0.0, parallelError = 0.0
    var previousGPU = [SIMD4<Float>](), previousReference = [SIMD4<Float>]()
    for frame in 0..<frames {
        let source = Raster.generated(width: 257, height: 193, frame: frame)
        let coordinates = movingCoordinates(source: source, width: width, height: height, frame: frame)
        let expected = referenceSample(source, coordinates: coordinates)
        cpuError = max(cpuError, try maximumError(cpuSample(source, coordinates: coordinates), expected))
        simdError = max(simdError, try maximumError(try optimizedCPUSample(source, coordinates: coordinates), expected))
        parallelError = max(parallelError, try maximumError(try optimizedCPUSample(source, coordinates: coordinates, workers: 4), expected))
        let gpu = try sampler.sample(source, coordinates: coordinates).pixels
        gpuError = max(gpuError, try maximumError(gpu, expected))
        if !previousGPU.isEmpty {
            for index in gpu.indices {
                for channel in 0..<4 {
                    temporalError = max(temporalError, Double(abs(
                        (gpu[index][channel] - previousGPU[index][channel])
                            - (expected[index][channel] - previousReference[index][channel])
                    )))
                }
            }
        }
        previousGPU = gpu
        previousReference = expected
    }
    // Explicit image-domain budgets, not an assumption of bit-exact hardware filtering.
    try require(max(cpuError, simdError, parallelError) < 0.000_001,
                "CPU disagrees with Double oracle: \(cpuError)/\(simdError)/\(parallelError)")
    try require(gpuError <= 1.0 / 255 && temporalError <= 2.0 / 255,
                "Moving-pattern error exceeded one spatial/two temporal 8-bit levels: \(gpuError), \(temporalError)")
    return Validation(name: "moving-checkerboard-and-fractional-grid", frames: frames,
                      sampledPixels: frames * width * height, maxCPUError: cpuError,
                      maxSIMDCPUError: simdError, maxParallelCPUError: parallelError,
                      maxMetalError: gpuError, maxTemporalError: temporalError,
                      cpuTolerance: 0.000_001, metalTolerance: 1.0 / 255, temporalTolerance: 2.0 / 255)
}

@main struct SyntheticResamplingLab {
    static func main() {
        var report = Report()
        var reportURL: URL?
        var exitStatus: Int32 = 0
        do {
            let options = try LabOptions(arguments: Array(CommandLine.arguments.dropFirst()))
            report.requestedBenchmarks = options.scope.rawValue
            reportURL = URL(fileURLWithPath: options.reportPath)
            try require(!FileManager.default.fileExists(atPath: reportURL!.path), "Refusing to overwrite an existing report")
            #if DEBUG
            throw LabError.invalid("Use a release build for timing: swift run -c release")
            #else
            let setup = try timed { try MetalContext() }
            let context = setup.value
            report.device = context.device.name
            report.pipelineSetupMS = setup.milliseconds
            print("Synthetic resampling on \(context.device.name); pipeline setup \(String(format: "%.3f", setup.milliseconds)) ms (excluded from timings)")
            report.validations.append(try validateIdentity(context: context))
            report.validations.append(try validateMotion(context: context))
            report.residentSafetyChecks = try validateResidentSafety(context: context)
            for validation in report.validations {
                print("PASS \(validation.name): Metal error \(validation.maxMetalError), temporal error \(validation.maxTemporalError)")
                print("  CPU reference errors: scalar \(validation.maxCPUError), SIMD \(validation.maxSIMDCPUError), SIMD×4 \(validation.maxParallelCPUError)")
            }
            print("PASS resident safety: \(report.residentSafetyChecks.count) checks")
            if options.scope == .all {
                print("Per-image amortized medians; 4 output images/round, 24 measured rounds. All modes return CPU-readable output arrays.")
                for size in [(257, 193, 96, 80), (1_024, 1_024, 96, 80), (1_024, 1_024, 128, 128),
                             (1_024, 1_024, 256, 256), (1_024, 1_024, 384, 384), (1_024, 1_024, 512, 512)] {
                    let result = try autoreleasepool {
                        try benchmark(context: context, sourceWidth: size.0, sourceHeight: size.1, width: size.2, height: size.3)
                    }
                    report.benchmarks.append(result)
                    let times = result.measurements.map {
                        String(format: "%@ %.3f ms", $0.mode.label, $0.perImageWall.medianMS)
                    }.joined(separator: "; ")
                    print("\(result.sourceSize) → \(result.outputSize): \(times)")
                    print(String(format: "  Metal allocations: upload slot %.2f MiB, four resident slots %.2f MiB; resident uploads %d → %d",
                                 Double(result.singleUploadGPUResourceBytes) / 1_048_576,
                                 Double(result.residentGPUResourceBytes) / 1_048_576,
                                 result.residentUploadCountBefore, result.residentUploadCountAfter))
                }
                print("Upload-inclusive reuse: fresh uploads each trial; 4 inputs × 1/2/4/8 uses, two alternating input banks.")
                for size in [(96, 80), (256, 256), (512, 512)] {
                    for uses in [1, 2, 4, 8] {
                        let result = try autoreleasepool {
                            try reuseBenchmark(context: context, width: size.0, height: size.1, usesPerInput: uses)
                        }
                        report.reuseBenchmarks.append(result)
                        let times = result.measurements.map {
                            String(format: "%@ %.3f ms", $0.mode.label, $0.perOutputWall.medianMS)
                        }.joined(separator: "; ")
                        print("1024x1024 → \(result.outputSize), \(uses) use(s)/input: \(times) (upload included)")
                    }
                }
            }
            if options.scope == .all || options.mixedOnly {
                print("Mixed-size experiment: fixed CPU, fixed Metal, and experimental hybrid; uploads and selection included.")
                if options.mixedOnly { print("Focused scope: fixed-size and held-out timings skipped; identity/motion/resident safety gates retained.") }
                report.mixedBenchmarks = try autoreleasepool { try benchmarkMixed(context: context) }
            }
            if options.scope == .all || options.heldOutOnly {
                print("Held-out experiment: four new sizes, 512x512 controls, changing 1/2/4/8 uses; selection rule unchanged.")
                if options.heldOutOnly { print("Focused scope: earlier fixed-size/mixed timings skipped; identity/motion/resident safety gates retained.") }
                report.heldOutBenchmarks = try autoreleasepool { try benchmarkHeldOut(context: context) }
            }
            if options.scope == .jobProfileOnly {
                print("Focused per-job instrumentation: upload, execution/wait, output copy, and paired overhead; routing unchanged.")
                print("Earlier timing suites skipped; identity/motion/resident safety gates retained.")
                report.jobProfileBenchmarks = try autoreleasepool { try benchmarkJobProfiles(context: context) }
            }
            if options.scope == .selectorCandidateOnly {
                print("Focused selector candidate: unchanged control plus 640x512 Metal routing at four/eight uses.")
                print("Standard timing suites skipped; identity/motion/resident safety gates retained.")
                report.selectorCandidateBenchmarks = try autoreleasepool { try benchmarkSelectorCandidate(context: context) }
            }
            if options.scope == .boundaryHoldoutOnly {
                print("Focused boundary holdout: unseen rectangular sizes below/above a predeclared pixel-area candidate.")
                print("Standard timing suites skipped; identity/motion/resident safety gates retained.")
                report.boundaryHoldoutBenchmarks = try autoreleasepool { try benchmarkBoundaryHoldout(context: context) }
            }
            report.status = "PASS"
            #endif
        } catch {
            report.status = "FAIL"
            report.error = String(describing: error)
            print("FAIL: \(error)")
            exitStatus = 1
        }
        if let reportURL {
            do {
                // Existing reports must survive even an argument/validation failure.
                try require(!FileManager.default.fileExists(atPath: reportURL.path), "Report already exists")
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: reportURL, options: .withoutOverwriting)
                print("Report: \(reportURL.path)")
            } catch { print("Cannot save report: \(error)"); exitStatus = 1 }
        }
        exit(exitStatus)
    }
}
