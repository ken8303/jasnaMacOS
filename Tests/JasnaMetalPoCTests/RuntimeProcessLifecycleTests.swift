import Foundation
import Testing
@testable import JasnaMetalPoC

@Test
func runtimeProcessLifecycleRecordsRequestedIdentifier() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let pidFile = directory.appendingPathComponent("runtime.pid")
    try RuntimeProcessLifecycle.recordProcessIdentifierIfRequested(
        environment: ["JASNA_RUNTIME_PID_FILE": pidFile.path],
        processIdentifier: 12_345
    )

    #expect(try String(contentsOf: pidFile, encoding: .utf8) == "12345\n")
}

@Test
func runtimeProcessLifecycleDoesNothingWithoutRequestedPath() throws {
    try RuntimeProcessLifecycle.recordProcessIdentifierIfRequested(
        environment: [:],
        processIdentifier: 12_345
    )
}
