import Foundation
import Testing
@testable import JasnaMacApp

@Suite(.serialized)
@MainActor
struct RestorationQueueTests {
    private func fixture(_ script: String) throws -> (URL, [URL]) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let scripts = root.appendingPathComponent("script")
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        let path = scripts.appendingPathComponent("restore_vr_rollout.sh")
        try ("#!/bin/bash\n" + script).write(to: path, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path.path)
        let sources = ["first.mp4", "second.mp4", "third.mp4"].map { root.appendingPathComponent($0) }
        for source in sources { try Data("fixture".utf8).write(to: source) }
        return (root, sources)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<500 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Queue did not reach expected state")
    }

    @Test func sequentialCompletionAndFinalShutdownOnly() async throws {
        let (root, sources) = try fixture("echo run >> \"$1.calls\"\nsleep 0.2\ntouch \"$2\"\n")
        defer { try? FileManager.default.removeItem(at: root) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources(sources + [sources[0]])
        #expect(session.sourceURLs.count == 3)
        session.autoShutdownAfterCompletion = true
        session.start(performanceProfile: .fast, backend: .metal)
        try await waitUntil { session.activeSourceIndex == 1 }
        #expect(session.isRunning)
        #expect(session.shutdownCountdownSeconds == nil)
        try await waitUntil { session.state == .completed }
        #expect(session.activeSourceIndex == 2)
        #expect(session.shutdownCountdownSeconds != nil)
        session.cancelAutomaticShutdown()
        for source in sources {
            let calls = try String(contentsOf: URL(fileURLWithPath: source.path + ".calls"), encoding: .utf8)
            #expect(calls == "run\n")
            #expect(FileManager.default.fileExists(atPath: source.deletingPathExtension().path + "_restored.mp4"))
        }
    }

    @Test func failureStopsQueueAndOneTimeResumeRetriesOnlyFailedVideo() async throws {
        let (root, sources) = try fixture("""
        echo "$JASNA_ALLOW_IMPLEMENTATION_RESUME" >> "$1.calls"
        if [[ "$1" == *second.mp4 && "$JASNA_ALLOW_IMPLEMENTATION_RESUME" != 1 ]]; then
          echo 'only the implementation changed; set JASNA_ALLOW_IMPLEMENTATION_RESUME=1 once to reuse compatible work'
          exit 1
        fi
        touch "$2"
        """)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources(sources)
        session.start(performanceProfile: .fast, backend: .metal)
        try await waitUntil { session.state == .failed }
        #expect(session.activeSourceIndex == 1)
        #expect(session.implementationResumeAvailable)
        #expect(!FileManager.default.fileExists(atPath: sources[2].path + ".calls"))
        session.start(performanceProfile: .fast, backend: .metal, allowImplementationResume: true)
        try await waitUntil { session.state == .completed }
        let first = try String(contentsOf: URL(fileURLWithPath: sources[0].path + ".calls"), encoding: .utf8)
        let second = try String(contentsOf: URL(fileURLWithPath: sources[1].path + ".calls"), encoding: .utf8)
        let third = try String(contentsOf: URL(fileURLWithPath: sources[2].path + ".calls"), encoding: .utf8)
        #expect(first == "0\n")
        #expect(second == "0\n1\n")
        #expect(third == "0\n")
    }

    @Test func stopDoesNotStartNextVideo() async throws {
        let (root, sources) = try fixture("touch \"$1.started\"\nsleep 30\ntouch \"$2\"\n")
        defer { try? FileManager.default.removeItem(at: root) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources(sources)
        session.start(performanceProfile: .fast, backend: .metal)
        try await waitUntil { FileManager.default.fileExists(atPath: sources[0].path + ".started") }
        session.stop()
        try await waitUntil { !session.isRunning }
        #expect(session.activeSourceIndex == 0)
        #expect(!FileManager.default.fileExists(atPath: sources[1].path + ".started"))
        #expect(session.shutdownCountdownSeconds == nil)
    }
    @Test func freshStartIsOfferedForConfigurationConflictAndAppliesOnlyToFailedVideo() async throws {
        let (root, sources) = try fixture("""
        echo "$JASNA_START_FRESH:$JASNA_ALLOW_IMPLEMENTATION_RESUME:$JASNA_RESTORATION_BACKEND" >> "$1.calls"
        if [[ "$1" == *second.mp4 && "$JASNA_START_FRESH" != 1 ]]; then
          echo 'error: this output path belongs to a different test configuration'
          exit 1
        fi
        touch "$2"
        """)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources(sources)
        session.start(performanceProfile: .fast, backend: .mlx)
        try await waitUntil { session.state == .failed }
        #expect(session.freshStartAvailable)
        #expect(!session.implementationResumeAvailable)
        #expect(session.activeSourceIndex == 1)
        session.start(performanceProfile: .fast, backend: .mlx, allowImplementationResume: true, startFresh: true)
        try await waitUntil { session.state == .completed }
        #expect(!session.freshStartAvailable)
        let calls = try sources.map { try String(contentsOf: URL(fileURLWithPath: $0.path + ".calls"), encoding: .utf8) }
        #expect(calls == ["0:0:mlx\n", "0:0:mlx\n1:0:mlx\n", "0:0:mlx\n"])
    }

    @Test func changingSourcesClearsPreviousRunDisplay() async throws {
        let (root, sources) = try fixture("touch \"$2\"\n")
        defer { try? FileManager.default.removeItem(at: root) }
        let session = RestorationSession(projectRootURL: root)
        session.addSources([sources[0]])
        session.start(performanceProfile: .fast, backend: .metal)
        try await waitUntil { session.state == .completed }
        #expect(session.progress.overallFraction == 1)
        session.addSources([sources[1]])
        #expect(session.progress.overallFraction == 0)
        #expect(session.logText.isEmpty)
        #expect(session.state == .ready)
        #expect(!session.activity.contains("completed"))
        session.removeSource(sources[0])
        #expect(session.inputURL == sources[1])
        session.clearSources()
        #expect(session.sourceURLs.isEmpty)
        #expect(session.activity.contains("Drop source videos"))
        try await Task.sleep(for: .milliseconds(300))
        #expect(session.logText.isEmpty)
        #expect(session.progress.overallFraction == 0)
    }

}
