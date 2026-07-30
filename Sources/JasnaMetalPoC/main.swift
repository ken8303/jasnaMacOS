import Foundation

@available(macOS 27.0, *)
private func verifySyntheticClipFlows(
    runner: MetalDeformConv,
    modelsURL: URL,
    oracleURL: URL
) throws -> (SPyNetPairResult, SPyNetPairResult) {
    let flowFrames = try (0..<3).map { frameIndex in
        try runner.runBicubicDownsampleQuarter(makeJasnaSyntheticFrame(index: frameIndex))
    }
    let first = try verifySPyNetPair(
        device: runner.device,
        modelsURL: modelsURL,
        oracleURL: oracleURL,
        validateOracle: false,
        referenceInput: flowFrames[0],
        supportInput: flowFrames[1]
    )
    let second = try verifySPyNetPair(
        device: runner.device,
        modelsURL: modelsURL,
        oracleURL: oracleURL,
        inputVariant: 1,
        validateOracle: false,
        referenceInput: flowFrames[1],
        supportInput: flowFrames[2]
    )
    return (first, second)
}

private struct VariableSyntheticClipFlows {
    let pairs: [SPyNetPairResult]
    let backward: [[Float16]]
    let forward: [[Float16]]
}

@available(macOS 27.0, *)
private func verifyVariableSyntheticClipFlows(
    runner: MetalDeformConv,
    modelsURL: URL,
    oracleURL: URL,
    frameCount: Int
) throws -> VariableSyntheticClipFlows {
    guard frameCount >= 3 else { throw DeformConvError.invalidShape }
    let flowFrames = try (0..<frameCount).map { frameIndex in
        try runner.runBicubicDownsampleQuarter(makeJasnaSyntheticFrame(index: frameIndex))
    }
    var pairs = [SPyNetPairResult]()
    for pair in 0..<(frameCount - 1) {
        pairs.append(try verifySPyNetPair(
            device: runner.device,
            modelsURL: modelsURL,
            oracleURL: oracleURL,
            inputVariant: pair,
            validateOracle: false,
            referenceInput: flowFrames[pair],
            supportInput: flowFrames[pair + 1]
        ))
    }
    return VariableSyntheticClipFlows(
        pairs: pairs,
        backward: pairs.map(\.backwardFlow),
        forward: pairs.map(\.forwardFlow)
    )
}

private func correctnessTest(runner: MetalDeformConv) throws {
    let shape = DeformConvShape(
        batch: 1,
        inputChannels: 2,
        inputHeight: 5,
        inputWidth: 5,
        outputChannels: 3,
        outputHeight: 5,
        outputWidth: 5,
        kernelHeight: 3,
        kernelWidth: 3,
        padHeight: 1,
        padWidth: 1,
        strideHeight: 1,
        strideWidth: 1,
        dilationHeight: 1,
        dilationWidth: 1,
        groups: 1,
        offsetGroups: 1
    )
    var rng = SeededGenerator(seed: 0xC0FFEE)
    let input = rng.floats(count: shape.inputCount, range: -1...1)
    let offset = rng.floats(count: shape.offsetCount, range: -0.4...0.4)
    let mask = rng.floats(count: shape.maskCount, range: 0.1...0.9)
    let weight = rng.floats(count: shape.weightCount, range: -0.2...0.2)
    let bias = rng.floats(count: shape.outputChannels, range: -0.1...0.1)
    let expected = try DeformConvCPU.run(
        shape: shape, input: input, offset: offset, mask: mask, weight: weight, bias: bias
    )
    let actual = try runner.runFloat32(
        shape: shape, input: input, offset: offset, mask: mask, weight: weight, bias: bias
    )
    var maxAbsoluteError: Float = 0
    var maxRelativeError: Float = 0
    for (reference, candidate) in zip(expected, actual) {
        let absolute = abs(reference - candidate)
        maxAbsoluteError = max(maxAbsoluteError, absolute)
        maxRelativeError = max(maxRelativeError, absolute / max(abs(reference), 1e-6))
    }
    guard maxAbsoluteError < 2e-5 else {
        throw DeformConvError.commandFailed("correctness error \(maxAbsoluteError) exceeds tolerance")
    }
    print("Correctness: PASS (max abs \(String(format: "%.3g", maxAbsoluteError)), max rel \(String(format: "%.3g", maxRelativeError)))")

    func sampleBorder(_ values: [Float16], base: Int, height: Int, width: Int, y: Float, x: Float) -> Float {
        let clampedY = min(max(y, 0), Float(height - 1))
        let clampedX = min(max(x, 0), Float(width - 1))
        let y0 = Int(floor(clampedY)), x0 = Int(floor(clampedX))
        let y1 = min(y0 + 1, height - 1), x1 = min(x0 + 1, width - 1)
        let ly = clampedY - Float(y0), lx = clampedX - Float(x0)
        func value(_ yy: Int, _ xx: Int) -> Float { Float(values[base + yy * width + xx]) }
        return value(y0, x0) * (1 - ly) * (1 - lx)
            + value(y0, x1) * (1 - ly) * lx
            + value(y1, x0) * ly * (1 - lx)
            + value(y1, x1) * ly * lx
    }

    let warpWidth = 4, warpHeight = 4, warpPlane = warpWidth * warpHeight
    let flowWidth = 2, flowHeight = 2, flowPlane = flowWidth * flowHeight
    let warpReference = (0..<(3 * warpPlane)).map { Float16(Float($0) / 31 - 0.5) }
    let warpSupport = (0..<(3 * warpPlane)).map { Float16(Float(($0 * 7) % 23) / 17 - 0.6) }
    let sourceFlow: [Float16] = [-0.2, 0.1, 0.3, -0.1, 0.15, -0.25, 0.05, 0.2]
    let prepared = try runner.runSPyNetPrepare(
        width: warpWidth, height: warpHeight,
        reference: warpReference, support: warpSupport,
        sourceFlowWidth: flowWidth, sourceFlowHeight: flowHeight,
        sourceFlow: sourceFlow, firstLevel: false
    )
    var expectedFeatures = [Float16](repeating: 0, count: 8 * warpPlane)
    var expectedFlow = [Float16](repeating: 0, count: 2 * warpPlane)
    for y in 0..<warpHeight {
        for x in 0..<warpWidth {
            let index = y * warpWidth + x
            let sourceX = Float(x) * Float(flowWidth - 1) / Float(warpWidth - 1)
            let sourceY = Float(y) * Float(flowHeight - 1) / Float(warpHeight - 1)
            let flowX = 2 * sampleBorder(sourceFlow, base: 0, height: flowHeight, width: flowWidth, y: sourceY, x: sourceX)
            let flowY = 2 * sampleBorder(sourceFlow, base: flowPlane, height: flowHeight, width: flowWidth, y: sourceY, x: sourceX)
            expectedFlow[index] = Float16(flowX)
            expectedFlow[warpPlane + index] = Float16(flowY)
            for channel in 0..<3 {
                expectedFeatures[channel * warpPlane + index] = warpReference[channel * warpPlane + index]
                expectedFeatures[(channel + 3) * warpPlane + index] = Float16(sampleBorder(
                    warpSupport, base: channel * warpPlane, height: warpHeight, width: warpWidth,
                    y: Float(y) + flowY, x: Float(x) + flowX
                ))
            }
            expectedFeatures[6 * warpPlane + index] = Float16(flowX)
            expectedFeatures[7 * warpPlane + index] = Float16(flowY)
        }
    }
    let warpError = zip(expectedFeatures + expectedFlow, prepared.features + prepared.baseFlow)
        .map { abs(Float($0) - Float($1)) }.max() ?? 0
    guard warpError < 0.002 else {
        throw DeformConvError.commandFailed("SPyNet prepare error \(warpError) exceeds tolerance")
    }
    print("SPyNet warp/upsample: PASS (max abs \(String(format: "%.3g", warpError)))")

    let fullFrame = makeJasnaSyntheticFrame(index: 0)
    let downsampled = try runner.runBicubicDownsampleQuarter(fullFrame)
    let downsampleReference = try bicubicDownsampleQuarterReference(fullFrame)
    let downsampleError = zip(downsampled, downsampleReference)
        .map { abs(Float($0) - Float($1)) }.max() ?? 0
    guard downsampleError <= 0.0005 else {
        throw DeformConvError.commandFailed(
            "Jasna bicubic downsample error \(downsampleError) exceeds tolerance"
        )
    }
    print("Jasna 256→64 bicubic flow input: PASS (max abs \(String(format: "%.3g", downsampleError)))")

    let transformPlane = 7
    let rawOffsets = (0..<(432 * transformPlane)).map {
        Float16(Float(($0 * 13) % 101) / 50 - 1)
    }
    let transformFlow1 = (0..<(2 * transformPlane)).map { Float16(Float($0) / 20 - 0.25) }
    let transformFlow2 = (0..<(2 * transformPlane)).map { Float16(0.3 - Float($0) / 30) }
    let transformed = try runner.runDCNOffsetTransform(
        plane: transformPlane, raw: rawOffsets, flow1: transformFlow1, flow2: transformFlow2
    )
    var transformError: Float = 0
    for channel in 0..<288 {
        for spatial in 0..<transformPlane {
            let index = channel * transformPlane + spatial
            let localChannel = channel % 144
            let flow = channel < 144 ? transformFlow1 : transformFlow2
            let flowChannel = localChannel.isMultiple(of: 2) ? 1 : 0
            let expected = 10 * tanh(Float(rawOffsets[index]))
                + Float(flow[flowChannel * transformPlane + spatial])
            transformError = max(transformError, abs(expected - Float(transformed.offset[index])))
        }
    }
    for channel in 0..<144 {
        for spatial in 0..<transformPlane {
            let rawIndex = (channel + 288) * transformPlane + spatial
            let expected = 1 / (1 + exp(-Float(rawOffsets[rawIndex])))
            transformError = max(
                transformError,
                abs(expected - Float(transformed.mask[channel * transformPlane + spatial]))
            )
        }
    }
    guard transformError < 0.005 else {
        throw DeformConvError.commandFailed("DCNv2 offset transform error \(transformError)")
    }
    print("DCNv2 offset/mask transform: PASS (max abs \(String(format: "%.3g", transformError)))")

    func sampleZero(_ values: [Float16], base: Int, height: Int, width: Int, y: Float, x: Float) -> Float {
        let y0 = Int(floor(y)), x0 = Int(floor(x))
        let y1 = y0 + 1, x1 = x0 + 1
        let ly = y - Float(y0), lx = x - Float(x0)
        func value(_ yy: Int, _ xx: Int) -> Float {
            guard yy >= 0, yy < height, xx >= 0, xx < width else { return 0 }
            return Float(values[base + yy * width + xx])
        }
        return value(y0, x0) * (1 - ly) * (1 - lx)
            + value(y0, x1) * (1 - ly) * lx
            + value(y1, x0) * ly * (1 - lx)
            + value(y1, x1) * ly * lx
    }

    let temporalWidth = 3, temporalHeight = 3, temporalPlane = 9
    let featProp = (0..<(64 * temporalPlane)).map {
        Float16(Float(($0 * 11) % 71) / 50 - 0.7)
    }
    let featCurrent = (0..<(64 * temporalPlane)).map {
        Float16(Float(($0 * 17) % 83) / 60 - 0.6)
    }
    let featN2 = (0..<(64 * temporalPlane)).map {
        Float16(Float(($0 * 23) % 97) / 70 - 0.65)
    }
    let flow1 = (0..<(2 * temporalPlane)).map {
        Float16(Float(($0 * 7) % 19) / 20 - 0.4)
    }
    let previousFlow = (0..<(2 * temporalPlane)).map {
        Float16(Float(($0 * 5) % 17) / 25 - 0.3)
    }
    let temporal = try runner.runTemporalPreparation(
        width: temporalWidth, height: temporalHeight,
        featProp: featProp, featCurrent: featCurrent, featN2: featN2,
        flow1: flow1, previousFlow: previousFlow, hasSecondOrder: true
    )
    var expectedFlow2 = [Float16](repeating: 0, count: 2 * temporalPlane)
    for channel in 0..<2 {
        for spatial in 0..<temporalPlane {
            let y = spatial / temporalWidth, x = spatial % temporalWidth
            let warped = sampleZero(
                previousFlow, base: channel * temporalPlane,
                height: temporalHeight, width: temporalWidth,
                y: Float(y) + Float(flow1[temporalPlane + spatial]),
                x: Float(x) + Float(flow1[spatial])
            )
            expectedFlow2[channel * temporalPlane + spatial] = Float16(
                Float(flow1[channel * temporalPlane + spatial]) + warped
            )
        }
    }
    var expectedConditions = [Float16](repeating: 0, count: 196 * temporalPlane)
    for channel in 0..<196 {
        for spatial in 0..<temporalPlane {
            let y = spatial / temporalWidth, x = spatial % temporalWidth
            let value: Float
            if channel < 64 {
                value = sampleZero(
                    featProp, base: channel * temporalPlane,
                    height: temporalHeight, width: temporalWidth,
                    y: Float(y) + Float(flow1[temporalPlane + spatial]),
                    x: Float(x) + Float(flow1[spatial])
                )
            } else if channel < 128 {
                value = Float(featCurrent[(channel - 64) * temporalPlane + spatial])
            } else if channel < 192 {
                value = sampleZero(
                    featN2, base: (channel - 128) * temporalPlane,
                    height: temporalHeight, width: temporalWidth,
                    y: Float(y) + Float(expectedFlow2[temporalPlane + spatial]),
                    x: Float(x) + Float(expectedFlow2[spatial])
                )
            } else if channel < 194 {
                value = Float(flow1[(channel - 192) * temporalPlane + spatial])
            } else {
                value = Float(expectedFlow2[(channel - 194) * temporalPlane + spatial])
            }
            expectedConditions[channel * temporalPlane + spatial] = Float16(value)
        }
    }
    let expectedDeformInput = featProp + featN2
    let temporalError = zip(
        expectedFlow2 + expectedConditions + expectedDeformInput,
        temporal.secondOrderFlow + temporal.conditions + temporal.deformInput
    ).map { abs(Float($0) - Float($1)) }.max() ?? 0
    guard temporalError < 0.004 else {
        throw DeformConvError.commandFailed("temporal preparation error \(temporalError)")
    }
    let firstOrderOnly = try runner.runTemporalPreparation(
        width: temporalWidth, height: temporalHeight,
        featProp: featProp, featCurrent: featCurrent,
        featN2: [Float16](repeating: 0, count: 64 * temporalPlane),
        flow1: flow1, previousFlow: previousFlow, hasSecondOrder: false
    )
    let firstOrderZeroError = firstOrderOnly.secondOrderFlow
        .map { abs(Float($0)) }.max() ?? 0
    guard firstOrderZeroError == 0 else {
        throw DeformConvError.commandFailed("first-order temporal flow was not zeroed")
    }
    print("Temporal condition/second-order flow: PASS (max abs \(String(format: "%.3g", temporalError)))")
}

private func benchmark(runner: MetalDeformConv, weightSet: DeformConvWeightSet? = nil) throws {
    let result = try runner.benchmarkJasnaShape(weightSet: weightSet)
    if let weightSet { print("Checkpoint weights: \(weightSet.direction)") }
    print("Jasna DCNv2 shape: 1×128×64×64 → 1×64×64×64, FP16")
    print("Baseline median: \(String(format: "%.3f", result.baselineMedianMilliseconds)) ms")
    print("Baseline range:  \(String(format: "%.3f", result.baselineMinimumMilliseconds))–\(String(format: "%.3f", result.baselineMaximumMilliseconds)) ms")
    print("Baseline P10–P90/stddev: \(String(format: "%.3f–%.3f / %.3f", result.baselineStatistics.percentile10, result.baselineStatistics.percentile90, result.baselineStatistics.standardDeviation)) ms")
    print("SIMD median:     \(String(format: "%.3f", result.simdMedianMilliseconds)) ms")
    print("SIMD range:      \(String(format: "%.3f", result.simdMinimumMilliseconds))–\(String(format: "%.3f", result.simdMaximumMilliseconds)) ms")
    print("SIMD P10–P90/stddev:     \(String(format: "%.3f–%.3f / %.3f", result.simdStatistics.percentile10, result.simdStatistics.percentile90, result.simdStatistics.standardDeviation)) ms")
    print("Tiled median:    \(String(format: "%.3f", result.medianMilliseconds)) ms")
    print("Tiled range:     \(String(format: "%.3f", result.minimumMilliseconds))–\(String(format: "%.3f", result.maximumMilliseconds)) ms")
    print("Tiled P10–P90/stddev:    \(String(format: "%.3f–%.3f / %.3f", result.tiledStatistics.percentile10, result.tiledStatistics.percentile90, result.tiledStatistics.standardDeviation)) ms")
    print("Gather+GEMM:     \(String(format: "%.3f", result.simdgroupGEMMMedianMilliseconds)) ms")
    print("GEMM range:      \(String(format: "%.3f", result.simdgroupGEMMMinimumMilliseconds))–\(String(format: "%.3f", result.simdgroupGEMMMaximumMilliseconds)) ms")
    print("GEMM P10–P90/stddev:     \(String(format: "%.3f–%.3f / %.3f", result.gemmStatistics.percentile10, result.gemmStatistics.percentile90, result.gemmStatistics.standardDeviation)) ms")
    print("Tiled speedup:   \(String(format: "%.2f", result.baselineMedianMilliseconds / result.medianMilliseconds))×")
    print("GEMM speedup:    \(String(format: "%.2f", result.baselineMedianMilliseconds / result.simdgroupGEMMMedianMilliseconds))×")
    print("Iterations:      \(result.iterations)")
    print("Checksum:        \(String(format: "%.6f", result.checksum))")
    print("Max FP16 delta:  \(result.maxDifferenceFromBaseline)")
    print("GEMM FP16 delta: \(result.simdgroupGEMMMaxDifferenceFromBaseline)")
}

do {
    let runner = try MetalDeformConv()
    print("Metal device: \(runner.device.name)")
    print("Metal ML tensor family available: \(runner.device.supportsFamily(.apple7) ? "yes" : "no")")
    print("Metal ML tensor allocation: \(runner.canCreateMetalMLTensor() ? "PASS" : "FAIL")")
    let arguments = Set(CommandLine.arguments.dropFirst())
    if arguments.isEmpty || arguments.contains("--self-test") {
        try correctnessTest(runner: runner)
    }
    if arguments.isEmpty || arguments.contains("--benchmark") {
        try benchmark(runner: runner)
    }
    if let planIndex = CommandLine.arguments.firstIndex(of: "--plan-sbs-video") {
        guard CommandLine.arguments.indices.contains(planIndex + 4),
              let width = Int(CommandLine.arguments[planIndex + 1]),
              let height = Int(CommandLine.arguments[planIndex + 2]),
              let sourceFPS = Double(CommandLine.arguments[planIndex + 3]),
              let duration = Double(CommandLine.arguments[planIndex + 4])
        else {
            throw DeformConvError.commandFailed(
                "--plan-sbs-video requires width, height, source FPS, and duration seconds"
            )
        }
        let plan = try SideBySideVideoPlan(
            width: width,
            height: height,
            sourceFramesPerSecond: sourceFPS,
            durationSeconds: duration
        )
        print("Side-by-side video plan: PASS")
        print("Input canvas:       \(plan.dimensions.width)×\(plan.dimensions.height)")
        print("Per-eye canvas:     \(plan.eyeDimensions.width)×\(plan.eyeDimensions.height)")
        print("Model tile/overlap: \(SideBySideVideoPlan.modelTileSize) / \(plan.overlap) pixels")
        print("Tiles per eye/frame: \(plan.tilesPerEye)")
        print("Total tiles/frame:   \(plan.tiles.count)")
        print("Output rate:          \(String(format: "%.0f", plan.frameRate.outputFramesPerSecond)) FPS")
        print("Output frames:        \(plan.frameRate.outputFrameCount)")
        print("30-frame windows:     \(plan.temporalWindowCount)")
        print("Model graph runs:     \(plan.modelGraphExecutions)")
        print("Output BGRA frame:    \(String(format: "%.2f", Double(plan.outputBGRABytesPerFrame) / 1_048_576)) MiB")
    }
    if let realBenchmarkIndex = CommandLine.arguments.firstIndex(of: "--benchmark-real-weights") {
        guard CommandLine.arguments.indices.contains(realBenchmarkIndex + 1) else {
            throw DeformConvError.commandFailed("--benchmark-real-weights requires a directory")
        }
        let directoryURL = URL(fileURLWithPath: CommandLine.arguments[realBenchmarkIndex + 1], isDirectory: true)
        for set in try loadDeformConvWeightSets(directoryURL: directoryURL) {
            try benchmark(runner: runner, weightSet: set)
        }
    }
    if let probeIndex = CommandLine.arguments.firstIndex(of: "--metal-ml-probe") {
        guard CommandLine.arguments.indices.contains(probeIndex + 1) else {
            throw DeformConvError.commandFailed("--metal-ml-probe requires a .mtlpackage path")
        }
        let packageURL = URL(fileURLWithPath: CommandLine.arguments[probeIndex + 1])
        let info = try compileMetalMLPackage(device: runner.device, packageURL: packageURL)
        print("Metal ML pipeline: PASS")
        print("Bindings:          \(info.bindingDescriptions.count)")
        for binding in info.bindingDescriptions {
            print("  \(binding)")
        }
        print("Intermediate heap: \(info.intermediatesHeapSize) bytes")
    }
    if let benchmarkIndex = CommandLine.arguments.firstIndex(of: "--metal-ml-benchmark") {
        guard CommandLine.arguments.indices.contains(benchmarkIndex + 1) else {
            throw DeformConvError.commandFailed("--metal-ml-benchmark requires the feature_extract.mtlpackage path")
        }
        let packageURL = URL(fileURLWithPath: CommandLine.arguments[benchmarkIndex + 1])
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("buffer-backed Metal ML benchmark requires macOS 27")
        }
        let result = try benchmarkMetalMLPackage(device: runner.device, packageURL: packageURL)
        print("Metal ML package: \(packageURL.lastPathComponent)")
        print("Median:     \(String(format: "%.3f", result.medianMilliseconds)) ms")
        print("Best:       \(String(format: "%.3f", result.minimumMilliseconds)) ms")
        print("Iterations: \(result.iterations)")
    }
    if let interopIndex = CommandLine.arguments.firstIndex(of: "--metal-ml-interop") {
        guard CommandLine.arguments.indices.contains(interopIndex + 1) else {
            throw DeformConvError.commandFailed("--metal-ml-interop requires a .mtlpackage path")
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("Metal ML/compute interop requires macOS 27")
        }
        let packageURL = URL(fileURLWithPath: CommandLine.arguments[interopIndex + 1])
        let result = try verifyMetalMLComputeInterop(device: runner.device, packageURL: packageURL)
        print("Single-timeline Metal ML → custom compute: PASS")
        print("Elements checked: \(result.elementCount)")
        print("Maximum error:    \(result.maximumError)")
        print("Model output max: \(result.baselineMaximum)")
        print("Model checksum:   \(String(format: "%.6f", result.baselineChecksum))")
    }
    if let propagationIndex = CommandLine.arguments.firstIndex(of: "--propagation-smoke") {
        guard CommandLine.arguments.indices.contains(propagationIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--propagation-smoke requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("integrated propagation requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[propagationIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[propagationIndex + 2], isDirectory: true
        )
        let result = try verifyPropagationDirection(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            direction: "backward_1",
            backboneInputChannels: 128
        )
        print("Single-command-buffer backward_1 propagation: PASS")
        print("Chain: Metal ML offset → offset transform → checkpoint DCNv2 → Metal ML backbone → residual")
        print("Elements checked: \(result.elementCount)")
        print("GPU timeline:     \(String(format: "%.3f", result.gpuMilliseconds)) ms")
        print("Output maximum:   \(result.maximumMagnitude)")
        print("Repeat max error: \(result.repeatMaximumError)")
        print("Output checksum:  \(String(format: "%.6f", result.checksum))")
    }
    if let suiteIndex = CommandLine.arguments.firstIndex(of: "--propagation-suite") {
        guard CommandLine.arguments.indices.contains(suiteIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--propagation-suite requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("integrated propagation requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[suiteIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[suiteIndex + 2], isDirectory: true
        )
        let branches = [
            ("backward_1", 128), ("forward_1", 192),
            ("backward_2", 256), ("forward_2", 320),
        ]
        var totalMilliseconds = 0.0
        print("Four-direction single-command-buffer propagation suite:")
        for (direction, channels) in branches {
            let result = try verifyPropagationDirection(
                device: runner.device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                direction: direction,
                backboneInputChannels: channels
            )
            totalMilliseconds += result.gpuMilliseconds
            print(
                "\(direction): PASS, \(channels) channels, "
                    + "\(String(format: "%.3f", result.gpuMilliseconds)) ms, "
                    + "repeat error \(result.repeatMaximumError), "
                    + "checksum \(String(format: "%.6f", result.checksum))"
            )
        }
        print("Four-pass total: \(String(format: "%.3f", totalMilliseconds)) ms")
    }
    if let frameIndex = CommandLine.arguments.firstIndex(of: "--reconstruct-frame") {
        guard CommandLine.arguments.indices.contains(frameIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--reconstruct-frame requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("frame reconstruction requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[frameIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[frameIndex + 2], isDirectory: true
        )
        let branches = [
            ("backward_1", 128), ("forward_1", 192),
            ("backward_2", 256), ("forward_2", 320),
        ]
        var propagationResults = [String: PropagationSmokeResult]()
        for (direction, channels) in branches {
            propagationResults[direction] = try verifyPropagationDirection(
                device: runner.device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                direction: direction,
                backboneInputChannels: channels
            )
        }
        let result = try verifyReconstructedFrame(
            device: runner.device,
            modelsURL: modelsURL,
            propagationResults: propagationResults
        )
        print("Complete 256×256 restored-frame tensor: PASS")
        print("Chain: feature extract + four propagation outputs → upsample/reconstruction → frame residual")
        print("Elements checked:        \(result.elementCount)")
        print("Reconstruction timeline: \(String(format: "%.3f", result.reconstructionMilliseconds)) ms")
        print("Estimated hybrid total:  \(String(format: "%.3f", result.estimatedPipelineMilliseconds)) ms")
        print("Output maximum:          \(result.maximumMagnitude)")
        print("Residual max error:      \(result.residualMaximumError)")
        print("Repeat max error:        \(result.repeatMaximumError)")
        print("Output checksum:         \(String(format: "%.6f", result.checksum))")
    }
    if let graphIndex = CommandLine.arguments.firstIndex(of: "--zero-copy-frame") {
        guard CommandLine.arguments.indices.contains(graphIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--zero-copy-frame requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("zero-copy frame graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 2], isDirectory: true
        )
        let result = try verifyZeroCopyFrameGraph(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL
        )
        print("Zero-copy single-command-buffer frame graph: PASS")
        print("Chain: feature extract → four dependent propagation branches → reconstruction → frame residual")
        print("Elements checked:   \(result.elementCount)")
        print("GPU median:         \(String(format: "%.3f", result.gpuMilliseconds)) ms")
        print("GPU best/worst:     \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:            \(result.iterations)")
        print("Theoretical median: \(String(format: "%.1f", 1_000 / result.gpuMilliseconds)) FPS")
        print("Resident buffers:   \(String(format: "%.2f", Double(result.allocatedBytes) / 1_048_576)) MiB")
        print("Output maximum:     \(result.maximumMagnitude)")
        print("Residual max error: \(result.residualMaximumError)")
        print("Repeat max error:   \(result.repeatMaximumError)")
        print("Output checksum:    \(String(format: "%.6f", result.checksum))")
    }
    if let graphIndex = CommandLine.arguments.firstIndex(of: "--zero-copy-frame-grouped") {
        guard CommandLine.arguments.indices.contains(graphIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--zero-copy-frame-grouped requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("zero-copy frame graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 2], isDirectory: true
        )
        let result = try verifyZeroCopyFrameGraph(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            groupOffsetPredictions: true
        )
        print("Grouped-offset zero-copy frame graph: PASS")
        print("Chain: feature/offset ML group → four dependent propagation bodies → reconstruction")
        print("Elements checked:   \(result.elementCount)")
        print("GPU median:         \(String(format: "%.3f", result.gpuMilliseconds)) ms")
        print("GPU best/worst:     \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:            \(result.iterations)")
        print("Theoretical median: \(String(format: "%.1f", 1_000 / result.gpuMilliseconds)) FPS")
        print("Repeat max error:   \(result.repeatMaximumError)")
        print("Output checksum:    \(String(format: "%.6f", result.checksum))")
    }
    if let graphIndex = CommandLine.arguments.firstIndex(of: "--zero-copy-frame-staged") {
        guard CommandLine.arguments.indices.contains(graphIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--zero-copy-frame-staged requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("zero-copy frame graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 2], isDirectory: true
        )
        let result = try verifyZeroCopyFrameGraph(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            groupOffsetPredictions: true,
            groupAlignments: true
        )
        print("Staged zero-copy frame graph: PASS")
        print("Chain: feature/offset ML → grouped DCNv2 alignments → dependent backbones → reconstruction")
        print("Elements checked:   \(result.elementCount)")
        print("GPU median:         \(String(format: "%.3f", result.gpuMilliseconds)) ms")
        print("GPU best/worst:     \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:            \(result.iterations)")
        print("Theoretical median: \(String(format: "%.1f", 1_000 / result.gpuMilliseconds)) FPS")
        print("Repeat max error:   \(result.repeatMaximumError)")
        print("Output checksum:    \(String(format: "%.6f", result.checksum))")
    }
    if let graphIndex = CommandLine.arguments.firstIndex(of: "--zero-copy-frame-fused") {
        guard CommandLine.arguments.indices.contains(graphIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--zero-copy-frame-fused requires the MetalML and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("zero-copy frame graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[graphIndex + 2], isDirectory: true
        )
        let result = try verifyZeroCopyFrameGraph(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            groupOffsetPredictions: true,
            groupAlignments: true,
            fusePropagationResiduals: true
        )
        print("Fused zero-copy frame graph: PASS")
        print("Chain: grouped offsets/alignments → fused residual assemblies → reconstruction")
        print("Elements checked:   \(result.elementCount)")
        print("GPU median:         \(String(format: "%.3f", result.gpuMilliseconds)) ms")
        print("GPU best/worst:     \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:            \(result.iterations)")
        print("Theoretical median: \(String(format: "%.1f", 1_000 / result.gpuMilliseconds)) FPS")
        print("Repeat max error:   \(result.repeatMaximumError)")
        print("Output checksum:    \(String(format: "%.6f", result.checksum))")
    }
    if let spynetIndex = CommandLine.arguments.firstIndex(of: "--spynet-pair") {
        guard CommandLine.arguments.indices.contains(spynetIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--spynet-pair requires the MetalML and SPyNetOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("SPyNet pair graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[spynetIndex + 1], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[spynetIndex + 2], isDirectory: true
        )
        let result = try verifySPyNetPair(
            device: runner.device, modelsURL: modelsURL, oracleURL: oracleURL
        )
        print("Bidirectional six-level SPyNet graph: PASS")
        print("Chain: normalized pyramids → warp/upsample → 12 Metal ML blocks → residual flow adds")
        print("Flow elements checked: \(result.elementCount)")
        print("GPU median:            \(String(format: "%.3f", result.medianMilliseconds)) ms")
        print("GPU best/worst:        \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:               \(result.iterations)")
        print("Backward/forward max:  \(result.backwardMaximum) / \(result.forwardMaximum)")
        print("Repeat max error:      \(result.repeatMaximumError)")
        print("PyTorch max error:     \(result.oracleMaximumError)")
        print("Flow checksums:        \(String(format: "%.6f", result.backwardChecksum)) / \(String(format: "%.6f", result.forwardChecksum))")
    }
    if let combinedIndex = CommandLine.arguments.firstIndex(of: "--frame-with-spynet") {
        guard CommandLine.arguments.indices.contains(combinedIndex + 3) else {
            throw DeformConvError.commandFailed(
                "--frame-with-spynet requires MetalML, DeformConv, and SPyNetOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("SPyNet frame graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[combinedIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[combinedIndex + 2], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[combinedIndex + 3], isDirectory: true
        )
        let flowResult = try verifySPyNetPair(
            device: runner.device, modelsURL: modelsURL, oracleURL: oracleURL
        )
        let frameResult = try verifyZeroCopyFrameGraph(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            groupOffsetPredictions: true,
            groupAlignments: true,
            firstOrderFlows: [
                "backward": flowResult.backwardFlow,
                "forward": flowResult.forwardFlow,
            ]
        )
        let combinedMilliseconds = flowResult.medianMilliseconds + frameResult.gpuMilliseconds
        print("SPyNet-fed staged frame graph: PASS")
        print("Flow source: checkpoint-validated backward/forward 64×64 fields; second order zero for two frames")
        print("SPyNet median:   \(String(format: "%.3f", flowResult.medianMilliseconds)) ms")
        print("Frame median:    \(String(format: "%.3f", frameResult.gpuMilliseconds)) ms")
        print("Combined median: \(String(format: "%.3f", combinedMilliseconds)) ms")
        print("Estimated rate:  \(String(format: "%.1f", 1_000 / combinedMilliseconds)) FPS")
        print("Repeat max error: \(frameResult.repeatMaximumError)")
        print("Frame checksum:   \(String(format: "%.6f", frameResult.checksum))")
    }
    if let temporalIndex = CommandLine.arguments.firstIndex(of: "--temporal-inputs") {
        guard CommandLine.arguments.indices.contains(temporalIndex + 2) else {
            throw DeformConvError.commandFailed(
                "--temporal-inputs requires the MetalML and SPyNetOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("temporal input probe requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[temporalIndex + 1], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[temporalIndex + 2], isDirectory: true
        )
        let flowResult = try verifySPyNetPair(
            device: runner.device, modelsURL: modelsURL, oracleURL: oracleURL
        )
        let result = try verifyTemporalPreparation(
            runner: runner,
            backwardFlow: flowResult.backwardFlow,
            forwardFlow: flowResult.forwardFlow
        )
        print("Full-size temporal recurrence inputs: PASS")
        print("Chain: learned first-order flows → composed second-order flows → warped 196ch conditions + 128ch DCN inputs")
        print("Bidirectional GPU median:     \(String(format: "%.3f", result.medianMilliseconds)) ms")
        print("GPU best/worst:               \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:                      \(result.iterations)")
        print("Maximum magnitude:            \(result.maximumMagnitude)")
        print("Repeat max error:             \(result.repeatMaximumError)")
        print("First-step second-order zero: \(result.firstOrderZeroMaximumError)")
        print("Condition checksum:           \(String(format: "%.6f", result.checksum))")
    }
    if let recurrenceIndex = CommandLine.arguments.firstIndex(of: "--three-frame-recurrence") {
        guard CommandLine.arguments.indices.contains(recurrenceIndex + 3) else {
            throw DeformConvError.commandFailed(
                "--three-frame-recurrence requires MetalML, DeformConv, and SPyNetOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("three-frame recurrence requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[recurrenceIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[recurrenceIndex + 2], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[recurrenceIndex + 3], isDirectory: true
        )
        let (firstFlowResult, secondFlowResult) = try verifySyntheticClipFlows(
            runner: runner, modelsURL: modelsURL, oracleURL: oracleURL
        )
        let result = try verifyThreeFrameRecurrence(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            flows: [firstFlowResult.backwardFlow, secondFlowResult.backwardFlow]
        )
        print("Three-frame backward_1 recurrence: PASS")
        print("Chain: 3 feature extracts → first/second-order temporal alignment → 2 offset/DCNv2 stages → 3 dependent backbones")
        print("GPU median:            \(String(format: "%.3f", result.medianMilliseconds)) ms")
        print("GPU best/worst:        \(String(format: "%.3f", result.minimumMilliseconds)) / \(String(format: "%.3f", result.maximumMilliseconds)) ms")
        print("Samples:               \(result.iterations)")
        print("Output elements:       \(result.elementCount)")
        print("Output maximum:        \(result.maximumMagnitude)")
        print("Second-order flow max: \(result.secondOrderFlowMaximum)")
        print("Adjacent flow checksums: \(String(format: "%.6f", firstFlowResult.backwardChecksum)) / \(String(format: "%.6f", secondFlowResult.backwardChecksum))")
        print("Repeat max error:      \(result.repeatMaximumError)")
        print("Output checksum:       \(String(format: "%.6f", result.checksum))")
    }
    if let firstPassIndex = CommandLine.arguments.firstIndex(of: "--three-frame-first-pass") {
        guard CommandLine.arguments.indices.contains(firstPassIndex + 3) else {
            throw DeformConvError.commandFailed(
                "--three-frame-first-pass requires MetalML, DeformConv, and SPyNetOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("three-frame first pass requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[firstPassIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[firstPassIndex + 2], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[firstPassIndex + 3], isDirectory: true
        )
        let (firstFlow, secondFlow) = try verifySyntheticClipFlows(
            runner: runner, modelsURL: modelsURL, oracleURL: oracleURL
        )
        let backward = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "backward_1", direction: .backward,
            flows: [firstFlow.backwardFlow, secondFlow.backwardFlow]
        )
        let forward = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "forward_1", direction: .forward,
            flows: [firstFlow.forwardFlow, secondFlow.forwardFlow],
            priorBranchFrames: [backward.propagatedFrames],
            inputSpatialFrames: backward.spatialFrames
        )
        let combined = backward.medianMilliseconds + forward.medianMilliseconds
        print("Three-frame first propagation pass: PASS")
        print("Chain: backward_1 recurrence → persistent per-frame features → forward_1 recurrence")
        print("Backward median:       \(String(format: "%.3f", backward.medianMilliseconds)) ms")
        print("Forward median:        \(String(format: "%.3f", forward.medianMilliseconds)) ms")
        print("Staged combined:       \(String(format: "%.3f", combined)) ms")
        print("Backward/forward max:  \(backward.maximumMagnitude) / \(forward.maximumMagnitude)")
        print("Second-order flow max: \(backward.secondOrderFlowMaximum) / \(forward.secondOrderFlowMaximum)")
        print("Repeat max error:      \(max(backward.repeatMaximumError, forward.repeatMaximumError))")
        print("Output checksums:      \(String(format: "%.6f", backward.checksum)) / \(String(format: "%.6f", forward.checksum))")
    }
    if let fourPassIndex = CommandLine.arguments.firstIndex(of: "--three-frame-four-pass") {
        guard CommandLine.arguments.indices.contains(fourPassIndex + 4) else {
            throw DeformConvError.commandFailed(
                "--three-frame-four-pass requires MetalML, DeformConv, SPyNetOracle, and FullModelOracle directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("three-frame four-pass graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[fourPassIndex + 1], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[fourPassIndex + 2], isDirectory: true
        )
        let oracleURL = URL(
            fileURLWithPath: CommandLine.arguments[fourPassIndex + 3], isDirectory: true
        )
        let fullOracleURL = URL(
            fileURLWithPath: CommandLine.arguments[fourPassIndex + 4], isDirectory: true
        )
        let (firstFlow, secondFlow) = try verifySyntheticClipFlows(
            runner: runner, modelsURL: modelsURL, oracleURL: oracleURL
        )
        let backwardFlows = [firstFlow.backwardFlow, secondFlow.backwardFlow]
        let forwardFlows = [firstFlow.forwardFlow, secondFlow.forwardFlow]
        let backward1 = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "backward_1", direction: .backward, flows: backwardFlows
        )
        let forward1 = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "forward_1", direction: .forward, flows: forwardFlows,
            priorBranchFrames: [backward1.propagatedFrames],
            inputSpatialFrames: backward1.spatialFrames
        )
        let backward2 = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "backward_2", direction: .backward, flows: backwardFlows,
            priorBranchFrames: [backward1.propagatedFrames, forward1.propagatedFrames],
            inputSpatialFrames: backward1.spatialFrames
        )
        let forward2 = try verifyThreeFrameRecurrence(
            device: runner.device, modelsURL: modelsURL, weightsURL: weightsURL,
            branchName: "forward_2", direction: .forward, flows: forwardFlows,
            priorBranchFrames: [
                backward1.propagatedFrames, forward1.propagatedFrames,
                backward2.propagatedFrames,
            ],
            inputSpatialFrames: backward1.spatialFrames
        )
        let branches = [backward1, forward1, backward2, forward2]
        let total = branches.reduce(0.0) { $0 + $1.medianMilliseconds }
        let reconstruction = try verifyReconstructedClip(
            device: runner.device,
            modelsURL: modelsURL,
            spatialFrames: backward1.spatialFrames,
            branchFrames: branches.map(\.propagatedFrames),
            inputFrames: backward1.inputFrames
        )
        let fused = try verifyFusedFourPassRecurrence(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            backwardFlows: backwardFlows,
            forwardFlows: forwardFlows,
            inputFrames: backward1.inputFrames,
            stagedBranchFrames: branches.map(\.propagatedFrames),
            stagedRestoredFrames: reconstruction.restoredFrames
        )
        let fullOracle = try compareFullModelOracle(
            restoredFrames: fused.restoredFrames, oracleURL: fullOracleURL
        )
        guard fullOracle.maximumAbsoluteError <= 0.02,
              fullOracle.meanAbsoluteError <= 0.0005,
              fullOracle.percentile99AbsoluteError <= 0.002,
              fullOracle.psnr >= 60
        else {
            throw DeformConvError.commandFailed(
                "full-model oracle mismatch (max=\(fullOracle.maximumAbsoluteError), "
                    + "p99=\(fullOracle.percentile99AbsoluteError), psnr=\(fullOracle.psnr))"
            )
        }
        print("Three-frame fused input-to-restored graph: PASS")
        print("Chain: bicubic → two bidirectional SPyNet pairs → feature extraction → four propagation passes → reconstruction")
        print("Branch medians: \(branches.map { String(format: "%.3f", $0.medianMilliseconds) }.joined(separator: " / ")) ms")
        print("Branch ranges:  \(branches.map { String(format: "%.3f–%.3f", $0.minimumMilliseconds, $0.maximumMilliseconds) }.joined(separator: " / ")) ms")
        print("Branch P10–P90: \(branches.map { String(format: "%.3f–%.3f", $0.statistics.percentile10, $0.statistics.percentile90) }.joined(separator: " / ")) ms")
        print("Branch stddev:  \(branches.map { String(format: "%.3f", $0.statistics.standardDeviation) }.joined(separator: " / ")) ms")
        print("Branch samples: \(branches.map(\.iterations))")
        print("Staged total:   \(String(format: "%.3f", total)) ms")
        print("Fused input→restored median: \(String(format: "%.3f", fused.statistics.median)) ms")
        print("Fused end-to-end P10–P90/stddev: \(String(format: "%.3f–%.3f / %.3f", fused.statistics.percentile10, fused.statistics.percentile90, fused.statistics.standardDeviation)) ms")
        print("Fused end-to-end best/worst/samples: \(String(format: "%.3f / %.3f / %d", fused.statistics.minimum, fused.statistics.maximum, fused.statistics.samples.count))")
        print("Propagation staged/repeat error: \(fused.propagationStagedMaximumError) / \(fused.propagationRepeatMaximumError)")
        print("Restored staged/repeat/residual error: \(fused.restoredStagedMaximumError) / \(fused.restoredRepeatMaximumError) / \(fused.residualMaximumError)")
        print("Flow oracle/repeat error: \(fused.flowOracleMaximumError) / \(fused.flowRepeatMaximumError)")
        print("Fused flow checksums: \(fused.flowChecksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
        print("Fused propagation checksums: \(fused.propagationChecksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
        print("Fused frame checksums: \(fused.restoredChecksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
        print("PyTorch full-model oracle: PASS (\(fullOracle.elementCount) values)")
        print("PyTorch max/mean/P99 error: \(String(format: "%.7f / %.7f / %.7f", fullOracle.maximumAbsoluteError, fullOracle.meanAbsoluteError, fullOracle.percentile99AbsoluteError))")
        print("PyTorch RMSE/PSNR: \(String(format: "%.7f / %.2f dB", fullOracle.rootMeanSquaredError, fullOracle.psnr))")
        let maximumErrorFrame = fullOracle.maximumErrorIndex / (3 * 256 * 256)
        let maximumErrorWithinFrame = fullOracle.maximumErrorIndex % (3 * 256 * 256)
        let maximumErrorChannel = maximumErrorWithinFrame / (256 * 256)
        let maximumErrorSpatial = maximumErrorWithinFrame % (256 * 256)
        print("PyTorch maximum error location: frame \(maximumErrorFrame), channel \(maximumErrorChannel), y/x \(maximumErrorSpatial / 256)/\(maximumErrorSpatial % 256)")
        print("PyTorch FP16 max error: \(fullOracle.fp16MaximumError)")
        print("Separate SPyNet oracle medians: \(String(format: "%.3f / %.3f", firstFlow.medianMilliseconds, secondFlow.medianMilliseconds)) ms")
        print("Backward flow checksums: \(String(format: "%.6f / %.6f", firstFlow.backwardChecksum, secondFlow.backwardChecksum))")
        print("Forward flow checksums:  \(String(format: "%.6f / %.6f", firstFlow.forwardChecksum, secondFlow.forwardChecksum))")
        print("Separate 3-frame reconstruction oracle: \(String(format: "%.3f", reconstruction.medianMilliseconds)) ms")
        print("Reconstruction best/worst: \(String(format: "%.3f", reconstruction.minimumMilliseconds)) / \(String(format: "%.3f", reconstruction.maximumMilliseconds)) ms")
        print("Reconstruction P10–P90/stddev: \(String(format: "%.3f–%.3f / %.3f", reconstruction.statistics.percentile10, reconstruction.statistics.percentile90, reconstruction.statistics.standardDeviation)) ms")
        print("Propagation + reconstruction: \(String(format: "%.3f", total + reconstruction.medianMilliseconds)) ms")
        print("Output maxima:  \(branches.map(\.maximumMagnitude))")
        print("Flow2 maxima:   \(branches.map(\.secondOrderFlowMaximum))")
        print("Repeat error:   \(branches.map(\.repeatMaximumError).max() ?? 0)")
        print("Checksums:      \(branches.map { String(format: "%.6f", $0.checksum) }.joined(separator: " / "))")
        print("Frame checksums: \(reconstruction.checksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
        print("Frame repeat/residual error: \(reconstruction.repeatMaximumError) / \(reconstruction.residualMaximumError)")
    }
    if let variableIndex = CommandLine.arguments.firstIndex(of: "--variable-clip") {
        guard CommandLine.arguments.indices.contains(variableIndex + 5),
              let frameCount = Int(CommandLine.arguments[variableIndex + 1]),
              frameCount >= 3
        else {
            throw DeformConvError.commandFailed(
                "--variable-clip requires frame count, MetalML, DeformConv, SPyNetOracle, and FullModelOracle"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("variable clip graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[variableIndex + 2], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[variableIndex + 3], isDirectory: true
        )
        let spynetOracleURL = URL(
            fileURLWithPath: CommandLine.arguments[variableIndex + 4], isDirectory: true
        )
        let fullOracleURL = URL(
            fileURLWithPath: CommandLine.arguments[variableIndex + 5], isDirectory: true
        )
        let inputFrames = (0..<frameCount).map(makeJasnaSyntheticFrame)
        let flowOracle: VariableSyntheticClipFlows? = frameCount <= 5
            ? try verifyVariableSyntheticClipFlows(
                runner: runner,
                modelsURL: modelsURL,
                oracleURL: spynetOracleURL,
                frameCount: frameCount
            )
            : nil
        let fused = try verifyFusedFourPassRecurrence(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            backwardFlows: flowOracle?.backward ?? [],
            forwardFlows: flowOracle?.forward ?? [],
            inputFrames: inputFrames,
            stagedBranchFrames: [],
            stagedRestoredFrames: []
        )
        let fullOracle = try compareFullModelOracle(
            restoredFrames: fused.restoredFrames, oracleURL: fullOracleURL
        )
        let maximumAllowedError: Float = frameCount <= 5 ? 0.02 : 0.03
        let percentile99AllowedError: Float = frameCount <= 5 ? 0.002 : 0.003
        guard fullOracle.maximumAbsoluteError <= maximumAllowedError,
              fullOracle.meanAbsoluteError <= 0.0005,
              fullOracle.percentile99AbsoluteError <= percentile99AllowedError,
              fullOracle.psnr >= 60
        else {
            throw DeformConvError.commandFailed(
                "variable full-model oracle mismatch (max=\(fullOracle.maximumAbsoluteError), "
                    + "mean=\(fullOracle.meanAbsoluteError), "
                    + "p99=\(fullOracle.percentile99AbsoluteError), psnr=\(fullOracle.psnr))"
            )
        }
        let schedule = try TemporalSchedule(frameCount: frameCount)
        let framesPerSecond = Double(frameCount) * 1_000 / fused.statistics.median
        print("Variable \(frameCount)-frame input-to-restored graph: PASS")
        print("Fused median: \(String(format: "%.3f", fused.statistics.median)) ms")
        print("Fused P10–P90/stddev: \(String(format: "%.3f–%.3f / %.3f", fused.statistics.percentile10, fused.statistics.percentile90, fused.statistics.standardDeviation)) ms")
        print("Fused best/worst/samples: \(String(format: "%.3f / %.3f / %d", fused.statistics.minimum, fused.statistics.maximum, fused.statistics.samples.count))")
        print("In-clip restored-frame rate: \(String(format: "%.1f", framesPerSecond)) FPS")
        print("Persistent clip tensors: \(String(format: "%.2f", Double(schedule.persistentTensorBytes) / 1_048_576)) MiB")
        if fused.flowOracleCompared {
            print("Flow oracle/repeat error: \(fused.flowOracleMaximumError) / \(fused.flowRepeatMaximumError)")
        } else {
            print("Flow repeat error: \(fused.flowRepeatMaximumError) (full PyTorch output is the external oracle)")
        }
        print("Propagation/frame repeat error: \(fused.propagationRepeatMaximumError) / \(fused.restoredRepeatMaximumError)")
        print("Residual maximum error: \(fused.residualMaximumError)")
        print("PyTorch values: \(fullOracle.elementCount)")
        print("PyTorch max/mean/P99 error: \(String(format: "%.7f / %.7f / %.7f", fullOracle.maximumAbsoluteError, fullOracle.meanAbsoluteError, fullOracle.percentile99AbsoluteError))")
        print("PyTorch RMSE/PSNR: \(String(format: "%.7f / %.2f dB", fullOracle.rootMeanSquaredError, fullOracle.psnr))")
        let maximumErrorFrame = fullOracle.maximumErrorIndex / (3 * 256 * 256)
        let maximumErrorWithinFrame = fullOracle.maximumErrorIndex % (3 * 256 * 256)
        let maximumErrorChannel = maximumErrorWithinFrame / (256 * 256)
        let maximumErrorSpatial = maximumErrorWithinFrame % (256 * 256)
        print("PyTorch maximum error location: frame \(maximumErrorFrame), channel \(maximumErrorChannel), y/x \(maximumErrorSpatial / 256)/\(maximumErrorSpatial % 256)")
        if let flowOracle {
            print("Separate SPyNet pair medians: \(flowOracle.pairs.map { String(format: "%.3f", $0.medianMilliseconds) }.joined(separator: " / ")) ms")
        }
        print("Frame checksums: \(fused.restoredChecksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
    }
    if let productionIndex = CommandLine.arguments.firstIndex(of: "--single-run-clip") {
        guard CommandLine.arguments.indices.contains(productionIndex + 3),
              let frameCount = Int(CommandLine.arguments[productionIndex + 1]),
              frameCount >= 3
        else {
            throw DeformConvError.commandFailed(
                "--single-run-clip requires frame count, MetalML, and DeformConv directories"
            )
        }
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("single-run clip graph requires macOS 27")
        }
        let modelsURL = URL(
            fileURLWithPath: CommandLine.arguments[productionIndex + 2], isDirectory: true
        )
        let weightsURL = URL(
            fileURLWithPath: CommandLine.arguments[productionIndex + 3], isDirectory: true
        )
        let fused = try verifyFusedFourPassRecurrence(
            device: runner.device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            backwardFlows: [],
            forwardFlows: [],
            inputFrames: (0..<frameCount).map(makeJasnaSyntheticFrame),
            stagedBranchFrames: [],
            stagedRestoredFrames: [],
            warmupCount: 0,
            measurementCount: 1
        )
        print("Production single-run \(frameCount)-frame graph: PASS")
        print("GPU timeline:       \(String(format: "%.3f", fused.statistics.median)) ms")
        print("Executions:         \(fused.statistics.samples.count)")
        print("Restored frames:    \(fused.restoredFrames.count)")
        print("Output elements:    \(fused.restoredFrames.reduce(0) { $0 + $1.count })")
        print("Flow/frame repeat:  \(fused.flowRepeatMaximumError) / \(fused.restoredRepeatMaximumError)")
        print("Frame checksums:    \(fused.restoredChecksums.map { String(format: "%.6f", $0) }.joined(separator: " / "))")
    }
    if let scheduleIndex = CommandLine.arguments.firstIndex(of: "--schedule") {
        let frameCount = CommandLine.arguments.indices.contains(scheduleIndex + 1)
            ? Int(CommandLine.arguments[scheduleIndex + 1]) ?? 5
            : 5
        let schedule = try TemporalSchedule(frameCount: frameCount)
        print("BasicVSR++ temporal schedule: \(frameCount) frames")
        print("Optical flows: \(schedule.flowCountPerDirection) forward + \(schedule.flowCountPerDirection) backward")
        for branch in schedule.branches {
            let branchSteps = schedule.steps.filter { $0.branch.name == branch.name }
            print("\(branch.name): frames \(branchSteps.map(\.frameIndex)), flows \(branchSteps.map(\.flowIndex))")
            print("  packages: \(branch.offsetPackage) → custom DCNv2 → \(branch.backbonePackage)")
            print("  backbone input: \(branch.backboneInputChannels) channels")
        }
        print("Persistent FP16 tensors: \(String(format: "%.2f", Double(schedule.persistentTensorBytes) / 1_048_576)) MiB")
    }
    if let graphIndex = CommandLine.arguments.firstIndex(of: "--validate-package-graph") {
        guard CommandLine.arguments.indices.contains(graphIndex + 1) else {
            throw DeformConvError.commandFailed("--validate-package-graph requires the MetalML directory")
        }
        let directoryURL = URL(fileURLWithPath: CommandLine.arguments[graphIndex + 1], isDirectory: true)
        let result = try validateMetalMLPackageGraph(device: runner.device, directoryURL: directoryURL)
        print("Metal ML temporal package graph: PASS")
        print("Packages validated: \(result.packageCount)")
        print("Bindings validated: \(result.bindingCount)")
    }
    if let arenaIndex = CommandLine.arguments.firstIndex(of: "--allocate-frame-graph") {
        guard #available(macOS 27.0, *) else {
            throw DeformConvError.commandFailed("buffer-backed frame graph requires macOS 27")
        }
        let frameCount = CommandLine.arguments.indices.contains(arenaIndex + 1)
            ? Int(CommandLine.arguments[arenaIndex + 1]) ?? 5
            : 5
        let schedule = try TemporalSchedule(frameCount: frameCount)
        let arena = try TemporalTensorArena(device: runner.device, schedule: schedule)
        print("Temporal tensor arena: PASS")
        print("Frames:          \(frameCount)")
        print("Tensor slots:    \(arena.slots.count)")
        print("Allocated bytes: \(arena.allocatedBytes)")
        print("Allocated MiB:   \(String(format: "%.2f", Double(arena.allocatedBytes) / 1_048_576))")
        print("Shared usage:    compute + machineLearning")
    }
    if let weightIndex = CommandLine.arguments.firstIndex(of: "--validate-deform-weights") {
        guard CommandLine.arguments.indices.contains(weightIndex + 1) else {
            throw DeformConvError.commandFailed("--validate-deform-weights requires a directory")
        }
        let directoryURL = URL(fileURLWithPath: CommandLine.arguments[weightIndex + 1], isDirectory: true)
        let sets = try loadDeformConvWeightSets(directoryURL: directoryURL)
        print("Jasna DCNv2 checkpoint weights: PASS")
        for set in sets {
            _ = try set.makeBuffers(device: runner.device)
            let checksum = set.data.withUnsafeBytes { bytes -> Double in
                let values = bytes.bindMemory(to: UInt16.self)
                var result = 0.0
                for index in stride(from: 0, to: values.count, by: 257) {
                    result += Double(Float(Float16(bitPattern: values[index])))
                }
                return result
            }
            print("\(set.direction): \(set.data.count) bytes, checksum \(String(format: "%.6f", checksum))")
        }
    }
} catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
