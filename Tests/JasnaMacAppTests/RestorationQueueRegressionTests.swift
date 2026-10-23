import Foundation
import Darwin
import Testing
import JasnaAppSupport
@testable import JasnaMacApp

@Suite(.serialized) @MainActor
struct RestorationQueueRegressionTests {
    @Test func stopAtProcessExitBoundary() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("script"), withIntermediateDirectories: true)
        let script = root.appendingPathComponent("script/restore_vr_rollout.sh")
        try "#!/bin/bash\ntouch \"$1.started\"\nsleep 0.1\ntouch \"$2\"\n".write(to: script, atomically: true, encoding: .utf8)
        let sources = ["first.mp4", "second.mp4"].map { root.appendingPathComponent($0) }
        for source in sources { try Data("fixture".utf8).write(to: source) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources(sources)
        session.autoShutdownAfterCompletion = true
        session.start(performanceProfile: .fast, backend: .metal)
        // Hold the main actor while the child exits; completion is queued on it.
        usleep(500_000)
        session.stop()
        for _ in 0..<200 {
            if !session.isRunning { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!FileManager.default.fileExists(atPath: sources[1].path + ".started"))
        #expect(session.state == .ready)
        #expect(session.shutdownCountdownSeconds == nil)
        session.cancelAutomaticShutdown()
    }

    @Test func caseInsensitiveRestartCollision() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sources = ["clip.mp4", "CLIP.mov"].map { root.appendingPathComponent($0) }
        for source in sources { try Data("fixture".utf8).write(to: source) }
        let caseSensitive = try root.resourceValues(forKeys: [.volumeSupportsCaseSensitiveNamesKey]).volumeSupportsCaseSensitiveNames ?? true
        guard !caseSensitive else { return }
        #expect(throws: (any Error).self) { try RestorationSourceQueue.validate(sources) }
    }

    @Test func clearingQueueResetsProgress() throws {
        let session = RestorationSession()
        var tracker = RestorationProgressTracker()
        session.progress = tracker.markCompleted()
        session.activity = "Restoration completed successfully."
        session.clearSources()
        #expect(session.progress.overallFraction == 0)
    }
}
