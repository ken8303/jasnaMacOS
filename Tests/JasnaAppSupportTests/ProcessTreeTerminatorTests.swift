import Darwin
import Foundation
import Testing
@testable import JasnaAppSupport

@Test func processTreeReturnsDeepestChildrenBeforeTheirParents() {
    let relationships = [
        ProcessRelationship(processIdentifier: 20, parentIdentifier: 10),
        ProcessRelationship(processIdentifier: 30, parentIdentifier: 20),
        ProcessRelationship(processIdentifier: 40, parentIdentifier: 10),
        ProcessRelationship(processIdentifier: 50, parentIdentifier: 999),
    ]

    #expect(
        ProcessTreeTerminator.descendantProcessIdentifiers(
            rootIdentifier: 10, relationships: relationships
        ) == [30, 20, 40]
    )
}

@Test func processTreeRejectsCyclesAndUnsafeRootIdentifiers() {
    let relationships = [
        ProcessRelationship(processIdentifier: 20, parentIdentifier: 10),
        ProcessRelationship(processIdentifier: 10, parentIdentifier: 20),
    ]

    #expect(
        ProcessTreeTerminator.descendantProcessIdentifiers(
            rootIdentifier: 10, relationships: relationships
        ) == [20]
    )
    #expect(
        ProcessTreeTerminator.descendantProcessIdentifiers(
            rootIdentifier: 1, relationships: relationships
        ).isEmpty
    )
}

@Test func processTreeEscalatesWhenTheRootIgnoresTermination() throws {
    let process = Process()
    let readyPipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-c", "trap '' TERM; echo READY; while :; do /bin/sleep 1; done"]
    process.standardOutput = readyPipe
    process.standardError = FileHandle.nullDevice
    try process.run()
    defer {
        if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }
    }
    let ready = readyPipe.fileHandleForReading.readData(ofLength: 6)
    #expect(String(decoding: ready, as: UTF8.self) == "READY\n")
    #expect(
        ProcessTreeTerminator.processIdentities().contains {
            $0.processIdentifier == process.processIdentifier
        }
    )

    ProcessTreeTerminator.terminate(
        rootIdentifier: process.processIdentifier,
        forceAfter: 0.05
    )
    let deadline = Date().addingTimeInterval(2)
    while process.isRunning, Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
    }

    #expect(!process.isRunning)
}

@Test func delayedTerminationRejectsAReusedProcessIdentifier() {
    let original = [
        ProcessIdentity(processIdentifier: 42, parentIdentifier: 10, startMarker: "first"),
        ProcessIdentity(processIdentifier: 43, parentIdentifier: 10, startMarker: "stable"),
    ]
    let current = [
        ProcessIdentity(processIdentifier: 42, parentIdentifier: 999, startMarker: "reused"),
        ProcessIdentity(processIdentifier: 43, parentIdentifier: 1, startMarker: "stable"),
    ]

    #expect(
        ProcessTreeTerminator.matchingOriginalProcessIdentifiers(
            original: original, current: current
        ) == [43]
    )
}
