import Foundation
import Testing
@testable import JasnaAppSupport

@Test func fileIdentityRecognizesLinksAndAllowsDistinctOutputs() throws {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: directory) }
    let source = directory.appendingPathComponent("source.mp4")
    let symbolic = directory.appendingPathComponent("symbolic.mp4")
    let hard = directory.appendingPathComponent("hard.mp4")
    let copy = directory.appendingPathComponent("copy.mp4")
    let newOutput = directory.appendingPathComponent("new.mp4")
    try Data("fixture".utf8).write(to: source)
    try manager.createSymbolicLink(at: symbolic, withDestinationURL: source)
    try manager.linkItem(at: source, to: hard)
    try manager.copyItem(at: source, to: copy)
    #expect(try FileIdentity.refersToSameFile(source, source))
    #expect(try FileIdentity.refersToSameFile(source, symbolic))
    #expect(try FileIdentity.refersToSameFile(source, hard))
    #expect(try !FileIdentity.refersToSameFile(source, copy))
    #expect(try !FileIdentity.refersToSameFile(source, newOutput))
    #expect(try String(contentsOf: source, encoding: .utf8) == "fixture")
}
