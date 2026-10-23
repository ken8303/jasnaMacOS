import Foundation
import Testing
@testable import JasnaAppSupport

@Test func sourceQueueUsesSourceFolderAndRestoredSuffix() {
    #expect(RestorationSourceQueue.outputURL(for: URL(fileURLWithPath: "/tmp/clip.mp4")).path == "/tmp/clip_restored.mp4")
    #expect(RestorationSourceQueue.outputURL(for: URL(fileURLWithPath: "/tmp/clip.MOV")).path == "/tmp/clip_restored.mov")
    #expect(RestorationSourceQueue.outputURL(for: URL(fileURLWithPath: "/tmp/clip.mkv")).path == "/tmp/clip_restored.mp4")
}

@Test func sourceQueueProtectsSelectedSourcesAndDuplicateOutputs() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = directory.appendingPathComponent("clip.mp4")
    let other = directory.appendingPathComponent("clip.mkv")
    let restored = directory.appendingPathComponent("clip_restored.mp4")
    let movie = directory.appendingPathComponent("clip.mov")
    for url in [source, other, restored, movie] { try Data("fixture".utf8).write(to: url) }
    #expect(throws: (any Error).self) { try RestorationSourceQueue.validate([source, restored]) }
    #expect(throws: (any Error).self) { try RestorationSourceQueue.validate([source, other]) }
    #expect(throws: (any Error).self) { try RestorationSourceQueue.validate([source, movie]) }
    try RestorationSourceQueue.validate([source])
    #expect(RestorationSourceQueue.accepts(source))
    #expect(!RestorationSourceQueue.accepts(directory))
    let text = directory.appendingPathComponent("notes.txt")
    try Data("fixture".utf8).write(to: text)
    #expect(!RestorationSourceQueue.accepts(text))
}
