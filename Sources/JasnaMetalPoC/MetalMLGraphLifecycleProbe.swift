import Foundation
import Metal

struct MetalMLGraphLifecycleStep: Codable, Equatable, Sendable {
    let frameCount: Int
    let expectedCacheHit: Bool
}

struct MetalMLGraphLifecycleSample: Codable, Equatable, Sendable {
    let step: Int
    let frameCount: Int
    let cacheHit: Bool
    let lookupMilliseconds: Double
    let setupMilliseconds: Double
    let gpuMilliseconds: Double
    let wallMilliseconds: Double
    let outputHash: String
    let maximumErrorFromPreviousSameShape: Float?
    var componentTimings: FusedGraphComponentTimings? = nil
    var propagationTimings: FusedGraphPropagationTimings? = nil
}

struct MetalMLGraphLifecycleReport: Codable, Sendable {
    let schemaVersion: Int
    let createdAt: String
    let device: String
    let batch: Int
    let generatedInputOnly: Bool
    let samples: [MetalMLGraphLifecycleSample]
    let retained35FrameSeries: MetalMLGraphLifecycleSeries?
}

struct MetalMLGraphLifecycleSeries: Codable, Equatable, Sendable {
    let repetitions: Int
    let gpu: BenchmarkStatisticsSnapshot
    let wall: BenchmarkStatisticsSnapshot
    let hostOverhead: BenchmarkStatisticsSnapshot
    let outputHash: String
    let maximumRepeatError: Float
    let componentMedians: FusedGraphComponentTimings?
    let propagationMedians: FusedGraphPropagationTimings?
}

struct BenchmarkStatisticsSnapshot: Codable, Equatable, Sendable {
    let minimum: Double
    let maximum: Double
    let median: Double
    let mean: Double
    let standardDeviation: Double
    let percentile10: Double
    let percentile90: Double

    init(_ statistics: BenchmarkStatistics) {
        minimum = statistics.minimum
        maximum = statistics.maximum
        median = statistics.median
        mean = statistics.mean
        standardDeviation = statistics.standardDeviation
        percentile10 = statistics.percentile10
        percentile90 = statistics.percentile90
    }
}

enum MetalMLGraphLifecycleProbe {
    static func repetitionCount(environment: [String: String] = ProcessInfo.processInfo.environment) -> Int {
        guard let value = environment["JASNA_GRAPH_LIFECYCLE_REPEATS"],
              let count = Int(value), count > 0
        else { return 8 }
        return min(count, 32)
    }

    // Full production windows have 30 frames or 35 inputs with temporal warm-up.
    // A 17-frame pair exercises a representative partial sparse-region shape.
    // Each shape repeats immediately to prove bounded retained-graph reuse.
    static let plan = [
        MetalMLGraphLifecycleStep(frameCount: 30, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 30, expectedCacheHit: true),
        MetalMLGraphLifecycleStep(frameCount: 17, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 17, expectedCacheHit: true),
        MetalMLGraphLifecycleStep(frameCount: 35, expectedCacheHit: false),
        MetalMLGraphLifecycleStep(frameCount: 35, expectedCacheHit: true),
    ]

    static func validate(_ samples: [MetalMLGraphLifecycleSample]) throws {
        guard samples.count == plan.count else {
            throw DeformConvError.commandFailed("graph lifecycle returned an incomplete sample set")
        }
        for (index, pair) in zip(plan.indices, zip(plan, samples)) {
            let (expected, sample) = pair
            guard sample.step == index + 1,
                  sample.frameCount == expected.frameCount,
                  sample.cacheHit == expected.expectedCacheHit,
                  sample.lookupMilliseconds >= 0,
                  sample.setupMilliseconds >= 0,
                  sample.gpuMilliseconds >= 0,
                  sample.wallMilliseconds >= sample.gpuMilliseconds,
                  !sample.outputHash.isEmpty
            else {
                throw DeformConvError.commandFailed(
                    "graph lifecycle sample \(index + 1) did not match the production plan"
                )
            }
            if sample.cacheHit, sample.setupMilliseconds != 0 {
                throw DeformConvError.commandFailed(
                    "retained graph unexpectedly reported setup work"
                )
            }
        }
        for sample in samples {
            if sample.cacheHit {
                guard let error = sample.maximumErrorFromPreviousSameShape,
                      error <= 0.001 else {
                    throw DeformConvError.commandFailed(
                        "retained graph output exceeded the FP16 repeat tolerance"
                    )
                }
            } else if sample.maximumErrorFromPreviousSameShape != nil {
                throw DeformConvError.commandFailed(
                    "first shape execution unexpectedly reported a repeat error"
                )
            }
        }
    }

    static func outputHash(_ frames: [[Float16]]) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for value in frames.joined() {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes in
                for byte in bytes {
                    hash ^= UInt64(byte)
                    hash &*= 1_099_511_628_211
                }
            }
        }
        return String(format: "%016llx", hash)
    }

    static func maximumError(_ first: [[Float16]], _ second: [[Float16]]) throws -> Float {
        guard first.count == second.count,
              zip(first, second).allSatisfy({ $0.count == $1.count })
        else { throw DeformConvError.invalidShape }
        var maximum: Float = 0
        for (firstFrame, secondFrame) in zip(first, second) {
            for (firstValue, secondValue) in zip(firstFrame, secondFrame) {
                maximum = max(maximum, abs(Float(firstValue) - Float(secondValue)))
            }
        }
        return maximum
    }
}

@available(macOS 27.0, *)
func runMetalMLGraphLifecycle(commandLine: JasnaCommandLine, index: Int) throws {
    guard index == 1, commandLine.arguments.count == 6,
          let batch = Int(commandLine[index + 4]), batch > 0
    else {
        throw DeformConvError.commandFailed(
            "--metal-ml-graph-lifecycle requires MetalML, DeformConv, report, and batch"
        )
    }
    let modelsURL = URL(fileURLWithPath: commandLine[index + 1], isDirectory: true)
    let weightsURL = URL(fileURLWithPath: commandLine[index + 2], isDirectory: true)
    let reportURL = URL(fileURLWithPath: commandLine[index + 3])
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw DeformConvError.metalUnavailable
    }

    SideBySideRestoration.report("Metal ML graph lifecycle device: \(device.name)")
    SideBySideRestoration.report(
        "Generated inputs only; no video decode, detection, compositor, or encoder"
    )
    SideBySideRestoration.report(
        "Production lifecycle: 30-frame, partial 17-frame, then 35-frame "
            + "build/reuse; batch \(batch)"
    )

    var samples = [MetalMLGraphLifecycleSample]()
    var previousOutputs = [Int: [[Float16]]]()
    for (stepIndex, step) in MetalMLGraphLifecycleProbe.plan.enumerated() {
        SideBySideRestoration.report(
            "Graph lifecycle step \(stepIndex + 1)/\(MetalMLGraphLifecycleProbe.plan.count): "
                + "\(step.frameCount) frames, expected cache "
                + (step.expectedCacheHit ? "hit" : "miss")
        )
        let frames = (0..<step.frameCount).map { frame in
            (0..<batch).flatMap { sample in
                makeJasnaSyntheticFrame(index: frame + sample * step.frameCount)
            }
        }
        let started = ContinuousClock.now
        let result = try verifyFusedFourPassRecurrence(
            device: device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            backwardFlows: [],
            forwardFlows: [],
            inputFrames: frames,
            stagedBranchFrames: [],
            stagedRestoredFrames: [],
            warmupCount: 0,
            measurementCount: 1,
            collectDiagnostics: false,
            batch: batch
        )
        let wallMilliseconds = SideBySideRestoration.elapsedMilliseconds(since: started)
        let repeatError = try previousOutputs[step.frameCount].map {
            try MetalMLGraphLifecycleProbe.maximumError($0, result.restoredFrames)
        }
        previousOutputs[step.frameCount] = result.restoredFrames
        let sample = MetalMLGraphLifecycleSample(
            step: stepIndex + 1,
            frameCount: step.frameCount,
            cacheHit: result.graphCacheHit,
            lookupMilliseconds: result.graphLookupMilliseconds,
            setupMilliseconds: result.graphSetupMilliseconds,
            gpuMilliseconds: result.statistics.median,
            wallMilliseconds: wallMilliseconds,
            outputHash: MetalMLGraphLifecycleProbe.outputHash(result.restoredFrames),
            maximumErrorFromPreviousSameShape: repeatError,
            componentTimings: result.componentTimings,
            propagationTimings: result.propagationTimings
        )
        samples.append(sample)
        SideBySideRestoration.report(
            "Graph lifecycle step \(sample.step): cache \(sample.cacheHit ? "hit" : "miss"), "
                + "lookup \(String(format: "%.3f", sample.lookupMilliseconds)) ms, setup "
                + "\(String(format: "%.3f", sample.setupMilliseconds)) ms, GPU "
                + "\(String(format: "%.3f", sample.gpuMilliseconds)) ms, wall "
                + "\(String(format: "%.3f", sample.wallMilliseconds)) ms, hash "
                + sample.outputHash + ", repeat error "
                + (sample.maximumErrorFromPreviousSameShape.map { String($0) } ?? "first use")
        )
    }
    try MetalMLGraphLifecycleProbe.validate(samples)

    let repetitionCount = MetalMLGraphLifecycleProbe.repetitionCount()
    SideBySideRestoration.report(
        "Retained 35-frame submission series: \(repetitionCount) measured execution(s)"
    )
    let seriesFrames = (0..<35).map { frame in
        (0..<batch).flatMap { sample in
            makeJasnaSyntheticFrame(index: frame + sample * 35)
        }
    }
    var seriesGPUMilliseconds = [Double]()
    var seriesWallMilliseconds = [Double]()
    var seriesHostOverheadMilliseconds = [Double]()
    var seriesReference: [[Float16]]?
    var seriesMaximumError: Float = 0
    var seriesHash = ""
    var seriesComponents = [FusedGraphComponentTimings]()
    var seriesPropagation = [FusedGraphPropagationTimings]()
    for repetition in 0..<repetitionCount {
        let started = ContinuousClock.now
        let result = try verifyFusedFourPassRecurrence(
            device: device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            backwardFlows: [],
            forwardFlows: [],
            inputFrames: seriesFrames,
            stagedBranchFrames: [],
            stagedRestoredFrames: [],
            warmupCount: 0,
            measurementCount: 1,
            collectDiagnostics: false,
            batch: batch
        )
        let wallMilliseconds = SideBySideRestoration.elapsedMilliseconds(since: started)
        guard result.graphCacheHit else {
            throw DeformConvError.commandFailed(
                "retained 35-frame submission series unexpectedly missed the graph cache"
            )
        }
        let gpuMilliseconds = result.statistics.median
        seriesGPUMilliseconds.append(gpuMilliseconds)
        seriesWallMilliseconds.append(wallMilliseconds)
        seriesHostOverheadMilliseconds.append(max(0, wallMilliseconds - gpuMilliseconds))
        if let components = result.componentTimings { seriesComponents.append(components) }
        if let propagation = result.propagationTimings {
            seriesPropagation.append(propagation)
        }
        if let reference = seriesReference {
            seriesMaximumError = max(
                seriesMaximumError,
                try MetalMLGraphLifecycleProbe.maximumError(reference, result.restoredFrames)
            )
        } else {
            seriesReference = result.restoredFrames
            seriesHash = MetalMLGraphLifecycleProbe.outputHash(result.restoredFrames)
        }
        SideBySideRestoration.report(
            "Retained 35-frame execution \(repetition + 1)/\(repetitionCount): GPU "
                + "\(String(format: "%.3f", gpuMilliseconds)) ms, wall "
                + "\(String(format: "%.3f", wallMilliseconds)) ms"
        )
    }
    guard let gpuStatistics = BenchmarkStatistics(seriesGPUMilliseconds),
          let wallStatistics = BenchmarkStatistics(seriesWallMilliseconds),
          let hostStatistics = BenchmarkStatistics(seriesHostOverheadMilliseconds),
          seriesMaximumError <= 0.001
    else {
        throw DeformConvError.commandFailed(
            "retained 35-frame submission series failed statistics or repeat validation"
        )
    }
    func componentMedian(_ keyPath: KeyPath<FusedGraphComponentTimings, Double>) -> Double {
        BenchmarkStatistics(seriesComponents.map { $0[keyPath: keyPath] })?.median ?? 0
    }
    let componentMedians: FusedGraphComponentTimings? = seriesComponents.isEmpty ? nil
        : FusedGraphComponentTimings(
            featureExtraction: componentMedian(\.featureExtraction),
            spynet: componentMedian(\.spynet),
            backward1: componentMedian(\.backward1),
            forward1: componentMedian(\.forward1),
            backward2: componentMedian(\.backward2),
            forward2: componentMedian(\.forward2),
            reconstruction: componentMedian(\.reconstruction)
        )
    func propagationMedian(
        _ keyPath: KeyPath<FusedGraphPropagationTimings, Double>
    ) -> Double {
        BenchmarkStatistics(seriesPropagation.map { $0[keyPath: keyPath] })?.median ?? 0
    }
    let propagationMedians: FusedGraphPropagationTimings? = seriesPropagation.isEmpty ? nil
        : FusedGraphPropagationTimings(
            offsetNetwork: propagationMedian(\.offsetNetwork),
            tensorPreparation: propagationMedian(\.tensorPreparation),
            dcnTransform: propagationMedian(\.dcnTransform),
            dcnGather: propagationMedian(\.dcnGather),
            dcnGEMM: propagationMedian(\.dcnGEMM),
            backboneNetwork: propagationMedian(\.backboneNetwork),
            residual: propagationMedian(\.residual),
            branches: (0..<4).compactMap { branchIndex in
                let samples = seriesPropagation.compactMap { timings in
                    timings.branches.indices.contains(branchIndex)
                        ? timings.branches[branchIndex] : nil
                }
                guard samples.count == seriesPropagation.count,
                      let name = samples.first?.name
                else { return nil }
                func median(
                    _ keyPath: KeyPath<FusedGraphBranchPropagationTimings, Double>
                ) -> Double {
                    BenchmarkStatistics(samples.map { $0[keyPath: keyPath] })?.median ?? 0
                }
                return FusedGraphBranchPropagationTimings(
                    name: name,
                    offsetNetwork: median(\.offsetNetwork),
                    tensorPreparation: median(\.tensorPreparation),
                    dcnTransform: median(\.dcnTransform),
                    dcnGather: median(\.dcnGather),
                    dcnGEMM: median(\.dcnGEMM),
                    backboneNetwork: median(\.backboneNetwork),
                    residual: median(\.residual)
                )
            },
            offsetLocality: (0..<4).compactMap { branchIndex in
                let samples = seriesPropagation.compactMap { timings in
                    timings.offsetLocality.indices.contains(branchIndex)
                        ? timings.offsetLocality[branchIndex] : nil
                }
                guard samples.count == seriesPropagation.count,
                      let first = samples.first
                else { return nil }
                func median(
                    _ keyPath: KeyPath<FusedGraphBranchOffsetLocality, Double>
                ) -> Double {
                    BenchmarkStatistics(samples.map { $0[keyPath: keyPath] })?.median ?? 0
                }
                return FusedGraphBranchOffsetLocality(
                    name: first.name,
                    sampleCount: first.sampleCount,
                    meanMagnitude: median(\.meanMagnitude),
                    maximumMagnitude: median(\.maximumMagnitude),
                    fractionAbove2: median(\.fractionAbove2),
                    fractionAbove4: median(\.fractionAbove4),
                    fractionAbove8: median(\.fractionAbove8),
                    outOfBoundsFraction: median(\.outOfBoundsFraction),
                    meanNeighborDelta: median(\.meanNeighborDelta)
                )
            }
        )
    let retainedSeries = MetalMLGraphLifecycleSeries(
        repetitions: repetitionCount,
        gpu: BenchmarkStatisticsSnapshot(gpuStatistics),
        wall: BenchmarkStatisticsSnapshot(wallStatistics),
        hostOverhead: BenchmarkStatisticsSnapshot(hostStatistics),
        outputHash: seriesHash,
        maximumRepeatError: seriesMaximumError,
        componentMedians: componentMedians,
        propagationMedians: propagationMedians
    )
    SideBySideRestoration.report(
        "Retained 35-frame series: GPU median "
            + "\(String(format: "%.3f", gpuStatistics.median)) ms "
            + "[P10 \(String(format: "%.3f", gpuStatistics.percentile10))–P90 "
            + "\(String(format: "%.3f", gpuStatistics.percentile90))], wall median "
            + "\(String(format: "%.3f", wallStatistics.median)) ms, host overhead median "
            + "\(String(format: "%.3f", hostStatistics.median)) ms, repeat error "
            + "\(seriesMaximumError)"
    )
    if let componentMedians {
        SideBySideRestoration.report(
            "Retained 35-frame component medians: feature "
                + "\(String(format: "%.3f", componentMedians.featureExtraction)) ms, SPyNet "
                + "\(String(format: "%.3f", componentMedians.spynet)) ms, backward_1 "
                + "\(String(format: "%.3f", componentMedians.backward1)) ms, forward_1 "
                + "\(String(format: "%.3f", componentMedians.forward1)) ms, backward_2 "
                + "\(String(format: "%.3f", componentMedians.backward2)) ms, forward_2 "
                + "\(String(format: "%.3f", componentMedians.forward2)) ms, reconstruction "
                + "\(String(format: "%.3f", componentMedians.reconstruction)) ms"
        )
    }
    if let propagationMedians {
        SideBySideRestoration.report(
            "Retained 35-frame propagation medians: offset "
                + "\(String(format: "%.3f", propagationMedians.offsetNetwork)) ms, prepare/assemble "
                + "\(String(format: "%.3f", propagationMedians.tensorPreparation)) ms, DCN transform "
                + "\(String(format: "%.3f", propagationMedians.dcnTransform)) ms, DCN gather "
                + "\(String(format: "%.3f", propagationMedians.dcnGather)) ms, DCN GEMM "
                + "\(String(format: "%.3f", propagationMedians.dcnGEMM)) ms, backbone "
                + "\(String(format: "%.3f", propagationMedians.backboneNetwork)) ms, residual "
                + "\(String(format: "%.3f", propagationMedians.residual)) ms"
        )
        for branch in propagationMedians.branches {
            SideBySideRestoration.report(
                "Retained 35-frame \(branch.name) package medians: offset "
                    + "\(String(format: "%.3f", branch.offsetNetwork)) ms, prepare/assemble "
                    + "\(String(format: "%.3f", branch.tensorPreparation)) ms, DCNv2 "
                    + "\(String(format: "%.3f", branch.dcnTransform + branch.dcnGather + branch.dcnGEMM)) ms, backbone "
                    + "\(String(format: "%.3f", branch.backboneNetwork)) ms, residual "
                    + "\(String(format: "%.3f", branch.residual)) ms"
            )
        }
        for locality in propagationMedians.offsetLocality {
            SideBySideRestoration.report(
                "Retained 35-frame \(locality.name) offset locality median: mean/max "
                    + "\(String(format: "%.3f", locality.meanMagnitude))/"
                    + "\(String(format: "%.3f", locality.maximumMagnitude)) px, >2/4/8 px "
                    + "\(String(format: "%.2f", 100 * locality.fractionAbove2))/"
                    + "\(String(format: "%.2f", 100 * locality.fractionAbove4))/"
                    + "\(String(format: "%.2f", 100 * locality.fractionAbove8))%, outside "
                    + "\(String(format: "%.2f", 100 * locality.outOfBoundsFraction))%, neighbor delta "
                    + "\(String(format: "%.3f", locality.meanNeighborDelta)) px"
            )
        }
    }

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let report = MetalMLGraphLifecycleReport(
        schemaVersion: 3,
        createdAt: formatter.string(from: Date()),
        device: device.name,
        batch: batch,
        generatedInputOnly: true,
        samples: samples,
        retained35FrameSeries: retainedSeries
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try FileManager.default.createDirectory(
        at: reportURL.deletingLastPathComponent(), withIntermediateDirectories: true
    )
    try encoder.encode(report).write(to: reportURL, options: .atomic)
    SideBySideRestoration.report("Metal ML graph lifecycle: PASS")
    SideBySideRestoration.report("Report: \(reportURL.path)")
}
