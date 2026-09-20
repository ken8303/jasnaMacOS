import Foundation
import Testing
@testable import JasnaAppSupport

private final class LogErrorCount: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = 0
    func increment() { lock.withLock { storage += 1 } }
    var value: Int { lock.withLock { storage } }
}

@Test func unavailableLogReportsOnceAndStillCompletesClose() async {
    let errors = LogErrorCount()
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathComponent("missing-parent/test.log")
    let writer = QueuedLogWriter(url: url) { _ in errors.increment() }
    for _ in 0..<100 { writer.append("Still visible in the UI\n") }
    await withCheckedContinuation { continuation in
        writer.close { continuation.resume() }
    }
    #expect(errors.value == 1)
}

@Test func queuedLogWriterPreservesOrderAndDrainsBeforeClose() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("test.log")
    try Data("existing\n".utf8).write(to: file)
    let writer = QueuedLogWriter(url: file) { message in
        Issue.record("Unexpected log error: \(message)")
    }
    for index in 0..<100 { writer.append("line \(index) 😀\n") }
    await withCheckedContinuation { continuation in
        writer.close { continuation.resume() }
    }
    let expected = "existing\n" + (0..<100).map { "line \($0) 😀\n" }.joined()
    #expect(try String(contentsOf: file, encoding: .utf8) == expected)
}

@Test func oversizedLogBacklogStopsOnceAndPreservesExistingFile() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("bounded.log")
    try Data("existing\n".utf8).write(to: file)
    let errors = LogErrorCount()
    let writer = QueuedLogWriter(url: file, maximumPendingBytes: 64) { _ in errors.increment() }
    writer.append(String(repeating: "x", count: 65))
    for _ in 0..<100 { writer.append("later\n") }
    await withCheckedContinuation { continuation in
        writer.close { continuation.resume() }
    }
    #expect(errors.value == 1)
    #expect(try String(contentsOf: file, encoding: .utf8) == "existing\n")
}
