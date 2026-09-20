import Foundation
import Testing
@testable import JasnaAppSupport

@Test func parsesAndNormalizesSupportedTimeFormats() throws {
    #expect(try MosaicTimeRangeParser.parseTimecode("01:02:03") == 3_723)
    #expect(try MosaicTimeRangeParser.parseTimecode("12:30") == 750)
    #expect(try MosaicTimeRangeParser.parseTimecode("45") == 45)
    #expect(MosaicTimeRange.format(3_723) == "01:02:03")
}

@Test func mergesOverlappingAndAdjacentRangesForTheRolloutScript() throws {
    let argument = try MosaicTimeRangeParser.commandArgument(for: [
        ("00:12:00", "00:13:00"),
        ("00:12:30", "00:14:00"),
        ("00:20:00", "00:21:00"),
    ])

    #expect(argument == "00:12:00-00:14:00,00:20:00-00:21:00")
}

@Test func parsesSingleFieldRangeExpressionsAndIgnoresEmptyRows() throws {
    let argument = try MosaicTimeRangeParser.commandArgument(for: [
        "11:00-30:00",
        "  ",
        "00:29:30 - 00:31:00",
    ])

    #expect(argument == "00:11:00-00:31:00")
}

@Test func emptyRangeExpressionsSelectFullVideo() throws {
    #expect(try MosaicTimeRangeParser.optionalCommandArgument(for: ["", "  "]) == nil)
    #expect(
        try MosaicTimeRangeParser.optionalCommandArgument(for: ["11:00-30:00"])
            == "00:11:00-00:30:00"
    )
}

@Test func rejectsInvalidAndReversedRanges() {
    #expect(throws: MosaicTimeRangeError.invalidTimecode("12:99")) {
        try MosaicTimeRangeParser.parseTimecode("12:99")
    }
    let overflowingTime = String(Int.max) + ":00"
    #expect(throws: MosaicTimeRangeError.invalidTimecode(overflowingTime)) {
        try MosaicTimeRangeParser.parseTimecode(overflowingTime)
    }
    #expect(throws: MosaicTimeRangeError.endNotAfterStart(1)) {
        try MosaicTimeRangeParser.parse([("00:10:00", "00:09:59")])
    }
    #expect(throws: MosaicTimeRangeError.noRanges) {
        try MosaicTimeRangeParser.parseExpressions([""])
    }
    #expect(throws: MosaicTimeRangeError.invalidRange("11:00")) {
        try MosaicTimeRangeParser.parseExpressions(["11:00"])
    }
}

@Test func locatesAProjectRootFromANestedDirectory() throws {
    let fileManager = FileManager.default
    let temporary = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let nested = temporary.appendingPathComponent("a/b", isDirectory: true)
    try fileManager.createDirectory(at: nested, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: temporary) }
    try Data().write(to: temporary.appendingPathComponent("Package.swift"))
    let scriptDirectory = temporary.appendingPathComponent("script", isDirectory: true)
    try fileManager.createDirectory(at: scriptDirectory, withIntermediateDirectories: true)
    let script = scriptDirectory.appendingPathComponent("restore_vr_rollout.sh")
    try Data("#!/bin/bash\n".utf8).write(to: script)
    try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

    let located = ProjectRootLocator.locate(
        environment: [:], currentDirectory: nested, executableURL: nil
    )
    #expect(located?.standardizedFileURL.path == temporary.standardizedFileURL.path)
}

@Test func locatesABundledRuntimeInsideAppResources() throws {
    let temporaryRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let resources = temporaryRoot.appendingPathComponent("Contents/Resources", isDirectory: true)
    let runtime = resources.appendingPathComponent("Runtime", isDirectory: true)
    let scriptDirectory = runtime.appendingPathComponent("script", isDirectory: true)
    try FileManager.default.createDirectory(at: scriptDirectory, withIntermediateDirectories: true)
    FileManager.default.createFile(
        atPath: runtime.appendingPathComponent("Package.swift").path,
        contents: Data()
    )
    let launcher = scriptDirectory.appendingPathComponent("restore_vr_rollout.sh")
    FileManager.default.createFile(atPath: launcher.path, contents: Data("#!/bin/sh\n".utf8))
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755], ofItemAtPath: launcher.path
    )
    defer { try? FileManager.default.removeItem(at: temporaryRoot) }

    #expect(
        ProjectRootLocator.locate(
            environment: [:],
            currentDirectory: URL(fileURLWithPath: "/"),
            executableURL: nil,
            resourceURL: resources
        ) == runtime.standardizedFileURL
    )
}

@Test func projectRootSearchTerminatesForMissingExternalVolumePath() {
    let missingInstalledApp = URL(
        fileURLWithPath: "/Volumes/External/Jasna VR Restoration.app/Contents/MacOS",
        isDirectory: true
    )

    #expect(
        ProjectRootLocator.locate(
            environment: [:],
            currentDirectory: missingInstalledApp,
            executableURL: nil,
            resourceURL: nil
        ) == nil
    )
}

@Test func streamDecoderPreservesScalarsSplitAcrossPipeReads() {
    var decoder = UTF8StreamDecoder()
    let bytes = Array("start 測試 end".utf8)
    let split = bytes.firstIndex(of: 0xE6)! + 2

    let first = decoder.decode(Data(bytes[..<split]))
    let second = decoder.decode(Data(bytes[split...]))

    #expect(first + second + decoder.finish() == "start 測試 end")
}

@Test func streamDecoderFlushesMalformedTrailingBytesWithoutDroppingEarlierText() {
    var decoder = UTF8StreamDecoder()
    let text = decoder.decode(Data([0x4F, 0x4B, 0x20, 0xE6]))

    #expect(text == "OK ")
    #expect(decoder.finish() == "�")
}
