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
