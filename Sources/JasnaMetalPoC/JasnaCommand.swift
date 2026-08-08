enum JasnaCommand: String, CaseIterable, Sendable {
    case selfTest = "--self-test"
    case benchmark = "--benchmark"
    case planSBSVideo = "--plan-sbs-video"
    case inspectSBSVideo = "--inspect-sbs-video"
    case transcodeSBS30 = "--transcode-sbs-30"
    case transcodeSBS30Tiled = "--transcode-sbs-30-tiled"
    case restoreSBSVideo = "--restore-sbs-video"
    case restoreSBSWindow = "--restore-sbs-window"
    case restoreSBSEye = "--restore-sbs-eye"
    case restoreEyeVideo = "--restore-eye-video"
    case restoreEyeWindows = "--restore-eye-windows"
    case restoreEyeWindowsSparseBatch = "--restore-eye-windows-sparse-batch"
    case restoreEyeWindowsSparse = "--restore-eye-windows-sparse"
    case diagnoseSBSTile = "--diagnose-sbs-tile"
    case benchmarkRealWeights = "--benchmark-real-weights"
    case metalMLProbe = "--metal-ml-probe"
    case metalMLBenchmark = "--metal-ml-benchmark"
    case metalMLInterop = "--metal-ml-interop"
    case propagationSmoke = "--propagation-smoke"
    case propagationSuite = "--propagation-suite"
    case reconstructFrame = "--reconstruct-frame"
    case zeroCopyFrame = "--zero-copy-frame"
    case zeroCopyFrameGrouped = "--zero-copy-frame-grouped"
    case zeroCopyFrameStaged = "--zero-copy-frame-staged"
    case zeroCopyFrameFused = "--zero-copy-frame-fused"
    case spynetPair = "--spynet-pair"
    case frameWithSPyNet = "--frame-with-spynet"
    case temporalInputs = "--temporal-inputs"
    case threeFrameRecurrence = "--three-frame-recurrence"
    case threeFrameFirstPass = "--three-frame-first-pass"
    case threeFrameFourPass = "--three-frame-four-pass"
    case variableClip = "--variable-clip"
    case singleRunClip = "--single-run-clip"
    case schedule = "--schedule"
    case validatePackageGraph = "--validate-package-graph"
    case allocateFrameGraph = "--allocate-frame-graph"
    case validateDeformWeights = "--validate-deform-weights"
}

struct JasnaCommandLine {
    let arguments: [String]

    init(arguments: [String] = CommandLine.arguments) {
        self.arguments = arguments
    }

    var hasNoArguments: Bool { arguments.count == 1 }
    var indices: Range<Array<String>.Index> { arguments.indices }

    subscript(index: Int) -> String { arguments[index] }

    func index(of command: JasnaCommand) -> Int? {
        arguments.firstIndex(of: command.rawValue)
    }

    func contains(_ command: JasnaCommand) -> Bool {
        index(of: command) != nil
    }

    func dispatch(
        _ command: JasnaCommand,
        whenNoArguments: Bool = false,
        handler: (Int) async throws -> Void
    ) async rethrows {
        if let index = index(of: command) {
            try await handler(index)
        } else if whenNoArguments && hasNoArguments {
            try await handler(0)
        }
    }

    func dispatch(
        firstOf commands: [JasnaCommand],
        handler: (Int) async throws -> Void
    ) async rethrows {
        guard let index = commands.lazy.compactMap({ index(of: $0) }).first else { return }
        try await handler(index)
    }
}
