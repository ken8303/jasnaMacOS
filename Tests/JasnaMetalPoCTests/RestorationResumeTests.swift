import Foundation
import Testing
@testable import JasnaMetalPoC

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
