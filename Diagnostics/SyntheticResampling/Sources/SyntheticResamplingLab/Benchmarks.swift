import Foundation

enum SamplingMode: String, CaseIterable, Codable, Hashable {
    case cpu = "plain-swift-cpu"
    case cpuSIMD = "cpu-simd-serial"
    case cpuParallel = "cpu-simd-parallel4"
    case upload = "metal-upload-each-image"
    case resident = "metal-resident-separate-submissions"
    case grouped = "metal-resident-group4"

    var isCPU: Bool { self == .cpu || self == .cpuSIMD || self == .cpuParallel }

    var label: String {
        switch self {
        case .cpu: "CPU scalar"
        case .cpuSIMD: "CPU SIMD"
        case .cpuParallel: "CPU SIMD×4"
        case .upload: "Metal upload"
        case .resident: "resident"
        case .grouped: "resident group4"
        }
    }
}

func samplingOrder(round: Int) -> [SamplingMode] {
    precondition(round >= 0)
    let modes = SamplingMode.allCases
    let offset = round % modes.count
    return (0..<modes.count).map { modes[($0 + offset) % modes.count] }
}

func perImageTimings(_ batchTimes: [Double], imagesPerRound: Int) throws -> TimingSummary {
    guard imagesPerRound > 0 else { throw LabError.invalid("Invalid image count for timing normalization") }
    return try TimingSummary(batchTimes.map { $0 / Double(imagesPerRound) })
}

struct BackendMeasurement: Encodable {
    let mode: SamplingMode
    let batchWall: TimingSummary
    let perImageWall: TimingSummary
    let batchGPU: TimingSummary?
    let perImageGPU: TimingSummary?
    let gpuTimestampSamples: Int
    let maxReferenceError: Double
    let maxDifferenceFromUpload: Double?
    let checksum: Double
}

struct Benchmark: Encodable {
    let sourceSize: String
    let outputSize: String
    let imagesPerRound: Int
    let warmupRounds: Int
    let measuredRounds: Int
    let residentInitialUploadMS: Double
    let residentUploadCountBefore: Int
    let residentUploadCountAfter: Int
    let timedUploadModeUploadCount: Int
    let singleUploadGPUResourceBytes: Int
    let residentGPUResourceBytes: Int
    let combinedGPUResourceBytes: Int
    let hostFixturePayloadBytes: Int
    let oneOutputArrayBytes: Int
    let measurements: [BackendMeasurement]
}

private struct MeasurementSamples {
    var wall = [Double]()
    var gpu = [Double]()
    var checksum = 0.0
    var maxError = 0.0
    var maxDifferenceFromUpload = 0.0
}

// Shared by the steady-state and upload-inclusive experiments. Every invocation
// returns all four images, regardless of how many command buffers were used.
func sampleFourInputs(_ mode: SamplingMode, sources: [Raster], coordinates: [[SIMD2<Float>]],
                      uploader: MetalSampler, residents: [MetalSampler]) throws -> MetalGroupOutput {
    try require(sources.count == 4 && coordinates.count == 4 && residents.count == 4,
                "Sampling round requires four inputs")
    switch mode {
    case .cpu:
        return MetalGroupOutput(
            frames: sources.indices.map { cpuSample(sources[$0], coordinates: coordinates[$0]) }, gpuMS: nil
        )
    case .cpuSIMD, .cpuParallel:
        return MetalGroupOutput(frames: try sources.indices.map {
            try optimizedCPUSample(sources[$0], coordinates: coordinates[$0], workers: mode == .cpuParallel ? 4 : 1)
        }, gpuMS: nil)
    case .upload, .resident:
        var frames = [[SIMD4<Float>]](), gpuTimes = [Double]()
        for index in sources.indices {
            let output: MetalOutput
            if mode == .upload {
                output = try uploader.sample(sources[index], coordinates: coordinates[index])
            } else {
                output = try residents[index].sampleResident()
            }
            frames.append(output.pixels)
            if let gpuMS = output.gpuMS { gpuTimes.append(gpuMS) }
        }
        // A partial set of timestamps is not a complete GPU duration.
        return MetalGroupOutput(frames: frames, gpuMS: gpuTimes.count == 4 ? gpuTimes.reduce(0, +) : nil)
    case .grouped:
        return try MetalSampler.sampleResidentGroup(residents)
    }
}

func benchmark(context: MetalContext, sourceWidth: Int, sourceHeight: Int, width: Int, height: Int) throws -> Benchmark {
    let count = width * height, images = 4, warmups = 4, rounds = 24
    let sources = (0..<images).map { Raster.generated(width: sourceWidth, height: sourceHeight, frame: $0) }
    let coordinates = sources.enumerated().map {
        movingCoordinates(source: $0.element, width: width, height: height, frame: $0.offset)
    }
    let expected = sources.indices.map { referenceSample(sources[$0], coordinates: coordinates[$0]) }
    let uploader = try MetalSampler(context: context, width: sourceWidth, height: sourceHeight, count: count)
    let residents = try sources.map { _ in
        try MetalSampler(context: context, width: sourceWidth, height: sourceHeight, count: count)
    }
    let preparation = try timed {
        for index in sources.indices { try residents[index].upload(sources[index], coordinates: coordinates[index]) }
    }
    var records = Dictionary(uniqueKeysWithValues: SamplingMode.allCases.map { ($0, MeasurementSamples()) })

    func execute(_ mode: SamplingMode) throws -> MetalGroupOutput {
        try sampleFourInputs(mode, sources: sources, coordinates: coordinates, uploader: uploader, residents: residents)
    }

    func validateAllModes() throws {
        let uploaded = try execute(.upload).frames
        for mode in SamplingMode.allCases {
            let frames = mode == .upload ? uploaded : try execute(mode).frames
            try require(frames.count == images, "Incorrect output count for \(mode.rawValue)")
            var error = 0.0, difference = 0.0
            for index in sources.indices {
                error = max(error, try maximumError(frames[index], expected[index]))
                if !mode.isCPU {
                    difference = max(difference, try maximumError(frames[index], uploaded[index]))
                }
            }
            let tolerance = mode.isCPU ? 0.000_001 : 1.0 / 255
            try require(error < tolerance, "\(mode.rawValue) reference error \(error) exceeded \(tolerance)")
            try require(difference == 0, "Resident output differs from upload-per-call output")
            var record = records[mode]!
            record.maxError = max(record.maxError, error)
            record.maxDifferenceFromUpload = max(record.maxDifferenceFromUpload, difference)
            records[mode] = record
        }
    }

    // Quality gates precede all timing and are repeated afterwards. Every output
    // slot contains a different generated frame; reusing stale slots cannot pass.
    try validateAllModes()
    let residentUploadsBefore = residents.reduce(0) { $0 + $1.uploadCount }
    let uploaderCountBefore = uploader.uploadCount
    for round in 0..<(warmups + rounds) {
        for mode in samplingOrder(round: round) {
            let result = try timed { try execute(mode) }
            // Consume all four outputs outside the stopwatch. No omitted frames
            // or GPU-only/discarded-output timings are compared with CPU output.
            let checksum = result.value.frames.reduce(0.0) { sum, pixels in
                sum + Double(pixels[0].x) + Double(pixels[count / 2].y) + Double(pixels[count - 1].z)
            }
            try require(checksum.isFinite, "Non-finite benchmark checksum")
            if round >= warmups {
                records[mode]!.wall.append(result.milliseconds)
                if let gpuMS = result.value.gpuMS { records[mode]!.gpu.append(gpuMS) }
                records[mode]!.checksum += checksum
            }
        }
    }
    let timedUploads = uploader.uploadCount - uploaderCountBefore
    let residentUploadsAfter = residents.reduce(0) { $0 + $1.uploadCount }
    try require(residentUploadsBefore == images && residentUploadsAfter == residentUploadsBefore,
                "Resident benchmark unexpectedly uploaded input again")
    try require(timedUploads == (warmups + rounds) * images, "Upload-mode workload changed")
    try validateAllModes()

    let measurements = try SamplingMode.allCases.map { mode in
        let record = records[mode]!
        return BackendMeasurement(
            mode: mode, batchWall: try TimingSummary(record.wall),
            perImageWall: try perImageTimings(record.wall, imagesPerRound: images),
            batchGPU: record.gpu.isEmpty ? nil : try TimingSummary(record.gpu),
            perImageGPU: record.gpu.isEmpty ? nil : try perImageTimings(record.gpu, imagesPerRound: images),
            gpuTimestampSamples: record.gpu.count, maxReferenceError: record.maxError,
            maxDifferenceFromUpload: mode.isCPU ? nil : record.maxDifferenceFromUpload,
            checksum: record.checksum
        )
    }
    let residentBytes = residents.reduce(0) { $0 + $1.reusableGPUResourceBytes }
    return Benchmark(
        sourceSize: "\(sourceWidth)x\(sourceHeight)", outputSize: "\(width)x\(height)", imagesPerRound: images,
        warmupRounds: warmups, measuredRounds: rounds, residentInitialUploadMS: preparation.milliseconds,
        residentUploadCountBefore: residentUploadsBefore, residentUploadCountAfter: residentUploadsAfter,
        timedUploadModeUploadCount: timedUploads, singleUploadGPUResourceBytes: uploader.reusableGPUResourceBytes,
        residentGPUResourceBytes: residentBytes, combinedGPUResourceBytes: residentBytes + uploader.reusableGPUResourceBytes,
        hostFixturePayloadBytes: sources.reduce(0) { $0 + $1.bytes.count }
            + coordinates.reduce(0) { $0 + $1.count * MemoryLayout<SIMD2<Float>>.stride },
        oneOutputArrayBytes: count * MemoryLayout<SIMD4<Float>>.stride, measurements: measurements
    )
}

func validateResidentSafety(context: MetalContext) throws -> [String] {
    func rejects(_ operation: () throws -> Void) throws {
        var rejected = false
        do { try operation() } catch LabError.invalid { rejected = true }
        try require(rejected, "Resident-input safety check did not reject invalid use")
    }
    let sampler = try MetalSampler(context: context, width: 4, height: 3, count: 2)
    let source = Raster.generated(width: 4, height: 3, frame: 1)
    let coordinates: [SIMD2<Float>] = [SIMD2(0.5, 0.5), SIMD2(2, 1)]
    try rejects { _ = try MetalSampler.sampleResidentGroup([]) }
    try rejects { _ = try sampler.sampleResident() }
    try sampler.upload(source, coordinates: coordinates)
    try rejects { _ = try MetalSampler.sampleResidentGroup([sampler, sampler]) }
    try rejects { try sampler.upload(source, coordinates: [SIMD2(.nan, 0), .zero]) }
    try rejects { _ = try sampler.sampleResident() }
    try sampler.upload(source, coordinates: coordinates)
    let output = try sampler.sampleResident()
    try require(try maximumError(output.pixels, referenceSample(source, coordinates: coordinates)) < 1.0 / 255,
                "Valid re-upload did not recover after invalid input")
    return ["empty-group-rejected", "unprepared-input-rejected", "aliased-output-slots-rejected",
            "nonfinite-upload-rejected", "stale-input-after-failure-rejected", "valid-reupload-passed"]
}
