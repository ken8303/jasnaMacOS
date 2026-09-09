import Foundation

struct ReusePlan {
    let usesPerInput: Int
    let inputsPerTrial = 4
    var outputsPerTrial: Int { inputsPerTrial * usesPerInput }

    init(usesPerInput: Int) throws {
        guard [1, 2, 4, 8].contains(usesPerInput) else { throw LabError.invalid("Reuse must be 1, 2, 4, or 8") }
        self.usesPerInput = usesPerInput
    }

    func uploads(for mode: SamplingMode) -> Int {
        switch mode {
        case .cpu, .cpuSIMD, .cpuParallel: 0
        case .upload: outputsPerTrial
        case .resident, .grouped: inputsPerTrial
        }
    }

    func submissions(for mode: SamplingMode) -> Int {
        switch mode {
        case .cpu, .cpuSIMD, .cpuParallel: 0
        case .upload, .resident: outputsPerTrial
        case .grouped: usesPerInput
        }
    }
}

func consumeFourOutputs(_ frames: [[SIMD4<Float>]], pixelsPerFrame: Int) throws -> Double {
    try require(pixelsPerFrame > 0 && frames.count == 4 && frames.allSatisfy { $0.count == pixelsPerFrame },
                "Incorrect number or shape of output images")
    let checksum = frames.reduce(0.0) { sum, pixels in
        sum + Double(pixels[0].x) + Double(pixels[pixelsPerFrame / 2].y) + Double(pixels[pixelsPerFrame - 1].z)
    }
    try require(checksum.isFinite, "Non-finite output checksum")
    return checksum
}

struct ReuseMeasurement: Encodable {
    let mode: SamplingMode
    let trialWall: TimingSummary
    let perOutputWall: TimingSummary
    let trialGPU: TimingSummary?
    let perOutputGPU: TimingSummary?
    let gpuTimestampSamples: Int
    let expectedUploadsPerTrial: Int
    let expectedSubmissionsPerTrial: Int
    let observedUploadCounts: [Int]
    let observedSubmissionCounts: [Int]
    let maxReferenceError: Double
    let maxDifferenceFromUpload: Double?
    let checksum: Double
}

struct ReuseBenchmark: Encodable {
    let sourceSize: String
    let outputSize: String
    let usesPerInput: Int
    let inputsPerTrial: Int
    let outputsPerTrial: Int
    let warmupTrials: Int
    let measuredTrials: Int
    let fixtureBanks: Int
    let singleUploadGPUResourceBytes: Int
    let residentGPUResourceBytes: Int
    let combinedGPUResourceBytes: Int
    let hostFixturePayloadBytes: Int
    let oracleArrayPayloadBytes: Int
    let oneFourOutputPayloadBytes: Int
    let measurements: [ReuseMeasurement]
}

private struct ReuseInputBank {
    let sources: [Raster]
    let coordinates: [[SIMD2<Float>]]
    let reference: [[SIMD4<Float>]]

    init(width: Int, height: Int, frameOffset: Int) {
        let frames = (0..<4).map { Raster.generated(width: 1_024, height: 1_024, frame: frameOffset + $0) }
        let points = frames.enumerated().map {
            movingCoordinates(source: $0.element, width: width, height: height, frame: frameOffset + $0.offset)
        }
        sources = frames
        coordinates = points
        reference = frames.indices.map { referenceSample(frames[$0], coordinates: points[$0]) }
    }
}

private struct ReuseObservation {
    let checksum: Double
    let gpuMS: Double?
    let outputCount: Int
    let uploads: Int
    let submissions: Int
    let maxError: Double
    let maxDifferenceFromUpload: Double
}

private struct ReuseSamples {
    var wall = [Double](), gpu = [Double]()
    var uploads = [Int](), submissions = [Int]()
    var checksum = 0.0, maxError = 0.0, maxDifferenceFromUpload = 0.0
}

func reuseBenchmark(context: MetalContext, width: Int, height: Int, usesPerInput: Int) throws -> ReuseBenchmark {
    let plan = try ReusePlan(usesPerInput: usesPerInput)
    let count = width * height, warmups = 4, trials = 24
    let banks = [0, 8].map { ReuseInputBank(width: width, height: height, frameOffset: $0) }
    let uploader = try MetalSampler(context: context, width: 1_024, height: 1_024, count: count)
    let residents = try (0..<4).map { _ in try MetalSampler(context: context, width: 1_024, height: 1_024, count: count) }
    let uploadedReferences = try banks.map {
        try sampleFourInputs(.upload, sources: $0.sources, coordinates: $0.coordinates,
                             uploader: uploader, residents: residents).frames
    }
    var records = Dictionary(uniqueKeysWithValues: SamplingMode.allCases.map { ($0, ReuseSamples()) })
    var expectedChecksums = Dictionary(uniqueKeysWithValues: SamplingMode.allCases.map { ($0, [Double]()) })

    func trial(_ mode: SamplingMode, bankIndex: Int, verifyPixels: Bool) throws -> ReuseObservation {
        let bank = banks[bankIndex]
        let beforeUploads = uploader.uploadCount + residents.reduce(0) { $0 + $1.uploadCount }
        let beforeSubmissions = context.submissionCount
        // This upload is deliberately INSIDE the caller's stopwatch, once per
        // resident input per trial. A previous trial's prepared data is not free.
        if mode == .resident || mode == .grouped {
            for index in bank.sources.indices {
                try residents[index].upload(bank.sources[index], coordinates: bank.coordinates[index])
            }
        }
        var checksum = 0.0, gpuTimes = [Double](), outputCount = 0
        var maxError = 0.0, difference = 0.0
        for _ in 0..<plan.usesPerInput {
            let output = try sampleFourInputs(mode, sources: bank.sources, coordinates: bank.coordinates,
                                              uploader: uploader, residents: residents)
            // All outputs are copied to CPU arrays and consumed on EVERY use.
            // Only one four-image output batch is retained at a time.
            checksum += try consumeFourOutputs(output.frames, pixelsPerFrame: count)
            outputCount += output.frames.count
            if let gpuMS = output.gpuMS { gpuTimes.append(gpuMS) }
            if verifyPixels {
                for index in bank.sources.indices {
                    maxError = max(maxError, try maximumError(output.frames[index], bank.reference[index]))
                    if !mode.isCPU {
                        difference = max(difference, try maximumError(output.frames[index], uploadedReferences[bankIndex][index]))
                    }
                }
            }
        }
        return ReuseObservation(
            checksum: checksum, gpuMS: gpuTimes.count == plan.usesPerInput ? gpuTimes.reduce(0, +) : nil,
            outputCount: outputCount,
            uploads: uploader.uploadCount + residents.reduce(0) { $0 + $1.uploadCount } - beforeUploads,
            submissions: context.submissionCount - beforeSubmissions,
            maxError: maxError, maxDifferenceFromUpload: difference
        )
    }

    func validateWorkload(_ observation: ReuseObservation, mode: SamplingMode) throws {
        try require(observation.outputCount == plan.outputsPerTrial, "Reuse trial omitted output images")
        try require(observation.uploads == plan.uploads(for: mode), "Reuse trial upload count is incorrect")
        try require(observation.submissions == plan.submissions(for: mode), "Reuse trial submission count is incorrect")
    }

    func validatePixels(_ observation: ReuseObservation, mode: SamplingMode) throws {
        try validateWorkload(observation, mode: mode)
        let tolerance = mode.isCPU ? 0.000_001 : 1.0 / 255
        try require(observation.maxError < tolerance && observation.maxDifferenceFromUpload == 0,
                    "Reuse quality gate failed for \(mode.rawValue): \(observation.maxError)")
    }

    // Full-pixel gates on every use of both banks, before collecting any timing.
    for mode in SamplingMode.allCases {
        for bankIndex in banks.indices {
            let observation = try trial(mode, bankIndex: bankIndex, verifyPixels: true)
            try validatePixels(observation, mode: mode)
            expectedChecksums[mode]!.append(observation.checksum)
            records[mode]!.maxError = max(records[mode]!.maxError, observation.maxError)
            records[mode]!.maxDifferenceFromUpload = max(records[mode]!.maxDifferenceFromUpload, observation.maxDifferenceFromUpload)
        }
        try require(expectedChecksums[mode]![0] != expectedChecksums[mode]![1],
                    "Fixture banks must have different checksums to detect stale input")
    }
    for index in 0..<(warmups + trials) {
        let bankIndex = index % banks.count
        for mode in samplingOrder(round: index) {
            let timedTrial = try timed { try trial(mode, bankIndex: bankIndex, verifyPixels: false) }
            let observation = timedTrial.value
            try validateWorkload(observation, mode: mode)
            try require(observation.checksum == expectedChecksums[mode]![bankIndex],
                        "Reuse trial checksum changed or stale inputs were sampled")
            if index >= warmups {
                records[mode]!.wall.append(timedTrial.milliseconds)
                if let gpuMS = observation.gpuMS { records[mode]!.gpu.append(gpuMS) }
                records[mode]!.uploads.append(observation.uploads)
                records[mode]!.submissions.append(observation.submissions)
                records[mode]!.checksum += observation.checksum
            }
        }
    }
    // Recheck every pixel, slot, and use after the timed loops too.
    for mode in SamplingMode.allCases {
        for bankIndex in banks.indices {
            let observation = try trial(mode, bankIndex: bankIndex, verifyPixels: true)
            try validatePixels(observation, mode: mode)
            records[mode]!.maxError = max(records[mode]!.maxError, observation.maxError)
            records[mode]!.maxDifferenceFromUpload = max(records[mode]!.maxDifferenceFromUpload, observation.maxDifferenceFromUpload)
        }
    }
    let measurements = try SamplingMode.allCases.map { mode in
        let record = records[mode]!
        return ReuseMeasurement(
            mode: mode, trialWall: try TimingSummary(record.wall),
            perOutputWall: try perImageTimings(record.wall, imagesPerRound: plan.outputsPerTrial),
            trialGPU: record.gpu.isEmpty ? nil : try TimingSummary(record.gpu),
            perOutputGPU: record.gpu.isEmpty ? nil : try perImageTimings(record.gpu, imagesPerRound: plan.outputsPerTrial),
            gpuTimestampSamples: record.gpu.count, expectedUploadsPerTrial: plan.uploads(for: mode),
            expectedSubmissionsPerTrial: plan.submissions(for: mode), observedUploadCounts: record.uploads,
            observedSubmissionCounts: record.submissions, maxReferenceError: record.maxError,
            maxDifferenceFromUpload: mode.isCPU ? nil : record.maxDifferenceFromUpload, checksum: record.checksum
        )
    }
    let residentBytes = residents.reduce(0) { $0 + $1.reusableGPUResourceBytes }
    let outputBytes = plan.inputsPerTrial * count * MemoryLayout<SIMD4<Float>>.stride
    return ReuseBenchmark(
        sourceSize: "1024x1024", outputSize: "\(width)x\(height)", usesPerInput: plan.usesPerInput,
        inputsPerTrial: plan.inputsPerTrial, outputsPerTrial: plan.outputsPerTrial,
        warmupTrials: warmups, measuredTrials: trials, fixtureBanks: banks.count,
        singleUploadGPUResourceBytes: uploader.reusableGPUResourceBytes, residentGPUResourceBytes: residentBytes,
        combinedGPUResourceBytes: uploader.reusableGPUResourceBytes + residentBytes,
        hostFixturePayloadBytes: banks.reduce(0) { sum, bank in
            sum + bank.sources.reduce(0) { $0 + $1.bytes.count }
                + bank.coordinates.reduce(0) { $0 + $1.count * MemoryLayout<SIMD2<Float>>.stride }
        },
        oracleArrayPayloadBytes: 2 * banks.count * outputBytes, oneFourOutputPayloadBytes: outputBytes,
        measurements: measurements
    )
}
