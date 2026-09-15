import Foundation
import Testing
@testable import JasnaMetalPoC

@available(macOS 27.0, *)
@Test func restorationIdentityIncludesSupplementalBatchModels() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "jasna-model-identity-test-\(UUID().uuidString)", isDirectory: true
    )
    let source = directory.appendingPathComponent("source.mov")
    let models = directory.appendingPathComponent("models", isDirectory: true)
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    let batch2A = directory.appendingPathComponent("batch2-a", isDirectory: true)
    let batch2B = directory.appendingPathComponent("batch2-b", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    for url in [models, weights, batch2A, batch2B] {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    }
    try Data("source".utf8).write(to: source)
    try Data("batch-a".utf8).write(to: batch2A.appendingPathComponent("model.bin"))
    try Data("batch-b".utf8).write(to: batch2B.appendingPathComponent("model.bin"))

    let identityA = SideBySideRestoration.restorationCacheIdentity(
        sourceURLs: [source], modelsURL: models, weightsURL: weights,
        additionalModelURLs: [batch2A]
    )
    let identityB = SideBySideRestoration.restorationCacheIdentity(
        sourceURLs: [source], modelsURL: models, weightsURL: weights,
        additionalModelURLs: [batch2B]
    )

    #expect(identityA != identityB)
    #expect(
        SideBySideRestoration.configuredAdditionalModelURLs(environment: [:]).isEmpty
    )
    #expect(
        SideBySideRestoration.configuredAdditionalModelURLs(
            environment: ["JASNA_BATCH2_MODELS_DIR": batch2A.path]
        ).first?.standardizedFileURL == batch2A.standardizedFileURL
    )
}

@available(macOS 27.0, *)
@Test func restorationResumeUsesOnlyTilesCompleteInEveryFrame() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "jasna-resume-test-\(UUID().uuidString)", isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }

    let bytesPerTile = 16
    let sizes = [bytesPerTile * 5, bytesPerTile * 3 + 7, bytesPerTile * 4]
    let urls = try sizes.enumerated().map { index, size in
        let url = directory.appendingPathComponent("frame-\(index).fp16")
        try Data(repeating: UInt8(index), count: size).write(to: url)
        return url
    }

    let completed = try SideBySideRestoration.recoverableTileCount(
        cacheURLs: urls,
        bytesPerTile: bytesPerTile,
        tileCount: 100
    )

    #expect(completed == 3)
}


@available(macOS 27.0, *)
@Test func restorationResumePrefersCompletedTilesMarkerClampedBySize() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "jasna-resume-marker-test-\(UUID().uuidString)", isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }

    let bytesPerTile = 16
    // Size says 5 complete tiles; marker claims 4.
    let url = directory.appendingPathComponent("frame-0.fp16")
    try Data(repeating: 0, count: bytesPerTile * 5).write(to: url)
    try Data("4\n".utf8).write(to: directory.appendingPathComponent("completed-tiles.txt"))

    let completed = try SideBySideRestoration.recoverableTileCount(
        cacheURLs: [url],
        bytesPerTile: bytesPerTile,
        tileCount: 100
    )
    #expect(completed == 4)

    // Marker above size must clamp to size-based count.
    try Data("9\n".utf8).write(to: directory.appendingPathComponent("completed-tiles.txt"))
    let clamped = try SideBySideRestoration.recoverableTileCount(
        cacheURLs: [url],
        bytesPerTile: bytesPerTile,
        tileCount: 100
    )
    #expect(clamped == 5)
}

@available(macOS 27.0, *)
@Test func restorationResumeCapsCompletedTilesAtPlanSize() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "jasna-resume-cap-test-\(UUID().uuidString)", isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }

    let url = directory.appendingPathComponent("frame-0.fp16")
    try Data(repeating: 0, count: 160).write(to: url)

    let completed = try SideBySideRestoration.recoverableTileCount(
        cacheURLs: [url],
        bytesPerTile: 16,
        tileCount: 7
    )

    #expect(completed == 7)
}

@available(macOS 27.0, *)
@Test func recurrenceFallbackMakesBalancedChunksWithoutShortTails() throws {
    #expect(
        try SideBySideRestoration.temporalChunkRanges(
            frameCount: 30, maximumFramesPerChunk: 10
        ) == [0..<10, 10..<20, 20..<30]
    )
    #expect(
        try SideBySideRestoration.temporalChunkRanges(
            frameCount: 29, maximumFramesPerChunk: 10
        ) == [0..<10, 10..<20, 20..<29]
    )
    #expect(
        try SideBySideRestoration.temporalChunkRanges(
            frameCount: 11, maximumFramesPerChunk: 5
        ) == [0..<4, 4..<8, 8..<11]
    )
    #expect(
        try SideBySideRestoration.temporalChunkRanges(
            frameCount: 5, maximumFramesPerChunk: 3
        ) == [0..<5]
    )
}

@available(macOS 27.0, *)
@Test func encoderSegmentsStopBeforeExistingLegacyOutputs() {
    #expect(SideBySideRestoration.defaultEncoderWindowsPerSegment == 4)
    #expect(
        SideBySideRestoration.encoderSegmentEnd(
            windowIndex: 0,
            windowCount: 30,
            maximumWindows: 5,
            hasExistingOutput: { _ in false }
        ) == 5
    )
    #expect(
        SideBySideRestoration.encoderSegmentEnd(
            windowIndex: 5,
            windowCount: 30,
            maximumWindows: 5,
            hasExistingOutput: { $0 == 8 }
        ) == 8
    )
    #expect(
        SideBySideRestoration.encoderSegmentEnd(
            windowIndex: 28,
            windowCount: 30,
            maximumWindows: 5,
            hasExistingOutput: { _ in false }
        ) == 30
    )
    #expect(
        SideBySideRestoration.encoderSegmentEnd(
            windowIndex: 0,
            windowCount: 120,
            maximumWindows: SideBySideRestoration.defaultEncoderWindowsPerSegment,
            hasExistingOutput: { $0 == 3 || $0 == 10 }
        ) == 3
    )
    #expect(
        SideBySideRestoration.encoderSegmentEnd(
            windowIndex: 0,
            windowCount: 120,
            maximumWindows: SideBySideRestoration.defaultEncoderWindowsPerSegment,
            hasExistingOutput: { _ in false }
        ) == 4
    )
}

@available(macOS 27.0, *)
@Test func restorationWindowRangeBoundsEachMetalProcess() throws {
    #expect(
        try SideBySideRestoration.restorationWindowRange(
            windowCount: 30,
            environment: [:]
        ) == 0..<30
    )
    #expect(
        try SideBySideRestoration.restorationWindowRange(
            windowCount: 30,
            environment: ["JASNA_WINDOW_START": "6", "JASNA_WINDOW_COUNT": "6"]
        ) == 6..<12
    )
    #expect(
        try SideBySideRestoration.restorationWindowRange(
            windowCount: 30,
            environment: ["JASNA_WINDOW_START": "24", "JASNA_WINDOW_COUNT": "20"]
        ) == 24..<30
    )
    #expect(throws: (any Error).self) {
        try SideBySideRestoration.restorationWindowRange(
            windowCount: 30,
            environment: ["JASNA_WINDOW_START": "30", "JASNA_WINDOW_COUNT": "6"]
        )
    }
    #expect(throws: (any Error).self) {
        try SideBySideRestoration.restorationWindowRange(
            windowCount: 30,
            environment: ["JASNA_WINDOW_START": "0", "JASNA_WINDOW_COUNT": "0"]
        )
    }
}


@available(macOS 27.0, *)
@Test func inMemoryRegionFrameCacheCopyValuesReusesCallerBuffer() throws {
    let cache = try SideBySideRestoration.InMemoryRegionFrameCache(
        frameCount: 1, regionCount: 1
    )
    var stored = [Float16](repeating: 0, count: SideBySideRestoration.tileElements)
    stored[0] = 9.5
    stored[stored.count - 1] = -1.25
    try cache.store(stored, frame: 0, region: 0)

    var scratch = [Float16](repeating: 7, count: SideBySideRestoration.tileElements)
    try cache.copyValues(frame: 0, region: 0, into: &scratch)
    #expect(scratch == stored)
}

@available(macOS 27.0, *)
@Test func inMemoryRegionFrameCacheKeepsFramesAndRegionsIndependent() throws {
    let cache = try SideBySideRestoration.InMemoryRegionFrameCache(
        frameCount: 2, regionCount: 2
    )
    var first = [Float16](repeating: 0, count: SideBySideRestoration.tileElements)
    var second = [Float16](repeating: 0, count: SideBySideRestoration.tileElements)
    first[0] = 1.25
    first[first.count - 1] = -2.5
    second[0] = 3.5
    second[second.count - 1] = 4.75

    try cache.store(first, frame: 0, region: 1)
    try cache.store(second, frame: 1, region: 0)

    #expect(try cache.values(frame: 0, region: 1) == first)
    #expect(try cache.values(frame: 1, region: 0) == second)
    #expect(try cache.values(frame: 0, region: 0).allSatisfy { $0 == 0 })
}
