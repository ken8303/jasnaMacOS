import AppKit
import Foundation
import JasnaAppSupport
import Observation
import OSLog

private let runtimeLogger = Logger(
    subsystem: Bundle.main.bundleIdentifier ?? "JasnaMacApp",
    category: "RestorationRuntime"
)

@MainActor
@Observable
final class RestorationSession {
    enum State: Equatable {
        case ready
        case running
        case stopping
        case completed
        case failed

        var title: String {
            switch self {
            case .ready: "Ready"
            case .running: "Restoring video"
            case .stopping: "Stopping"
            case .completed: "Completed"
            case .failed: "Needs attention"
            }
        }
    }

    var inputURL: URL?
    var outputURL: URL?
    var ranges = [MosaicTimeRangeDraft()]
    var state: State = .ready
    var activity = "Choose a source video and output file. Leave mosaic times blank for full video."
    var logText = ""
    var validationMessage: String?
    var logWarning: String?
    var autoShutdownAfterCompletion = false {
        didSet {
            if !autoShutdownAfterCompletion, shutdownCountdownSeconds != nil {
                cancelAutomaticShutdown()
            }
        }
    }
    var progress = RestorationProgressSnapshot.initial
    var shutdownCountdownSeconds: Int?

    @ObservationIgnored private var process: Process?
    @ObservationIgnored private var outputPipe: Pipe?
    @ObservationIgnored private var outputCoordinator: ProcessOutputCoordinator?
    @ObservationIgnored private var uiLogWriter: QueuedLogWriter?
    @ObservationIgnored private var logSessionID = UUID()
    @ObservationIgnored private var pendingVisibleLog = ""
    @ObservationIgnored private var logRefreshTask: Task<Void, Never>?
    @ObservationIgnored private var outputDecoder = UTF8StreamDecoder()
    @ObservationIgnored private var progressTracker = RestorationProgressTracker()
    @ObservationIgnored private var shutdownTask: Task<Void, Never>?
    // The complete transcript is persisted by QueuedLogWriter. Keep only a small tail in
    // SwiftUI so frequent updates do not repeatedly lay out hundreds of thousands of glyphs.
    @ObservationIgnored private let maximumVisibleLogCharacters = 64_000

    var isRunning: Bool { state == .running || state == .stopping }
    var canStart: Bool { inputURL != nil && outputURL != nil && !isRunning }
    var canRevealOutput: Bool {
        guard let outputURL else { return false }
        return FileManager.default.fileExists(atPath: outputURL.path)
    }

    var normalizedRangePreview: String? {
        do {
            return try MosaicTimeRangeParser.optionalCommandArgument(
                for: ranges.map(\.value)
            ) ?? "Full video"
        } catch {
            return nil
        }
    }

    func chooseInput() {
        guard let selected = FilePanelService.chooseInputVideo() else { return }
        inputURL = selected
        validationMessage = nil
        if outputURL == nil {
            outputURL = selected.deletingLastPathComponent()
                .appendingPathComponent(
                    selected.deletingPathExtension().lastPathComponent + "-restored.mp4"
                )
        }
    }

    func chooseOutput() {
        let suggestedName = inputURL.map {
            $0.deletingPathExtension().lastPathComponent + "-restored.mp4"
        } ?? "restored-vr.mp4"
        guard let selected = FilePanelService.chooseOutputVideo(suggestedName: suggestedName) else {
            return
        }
        outputURL = selected
        validationMessage = nil
    }

    func addRange() {
        ranges.append(MosaicTimeRangeDraft())
    }

    func removeRange(id: UUID) {
        guard ranges.count > 1 else { return }
        ranges.removeAll { $0.id == id }
        validationMessage = nil
    }

    func start(performanceProfile: RestorationPerformanceProfile) {
        guard !isRunning else { return }
        // Cancel the previous run's countdown even if validating this new request fails.
        shutdownTask?.cancel()
        shutdownTask = nil
        shutdownCountdownSeconds = nil
        do {
            let request = try validatedRequest(performanceProfile: performanceProfile)
            try launch(request)
        } catch {
            state = .failed
            validationMessage = error.localizedDescription
            activity = error.localizedDescription
        }
    }

    func stop() {
        guard let process, process.isRunning else { return }
        state = .stopping
        activity = "Stopping restoration safely…"
        appendLog("\nUser requested stop. Waiting for the current process to exit…\n")
        ProcessTreeTerminator.terminate(rootIdentifier: process.processIdentifier)
    }

    func revealOutput() {
        guard let outputURL, canRevealOutput else { return }
        NSWorkspace.shared.activateFileViewerSelecting([outputURL])
    }

    func cancelAutomaticShutdown() {
        shutdownTask?.cancel()
        shutdownTask = nil
        shutdownCountdownSeconds = nil
        activity = "Restoration completed successfully. Automatic shutdown cancelled."
    }

    private struct Request {
        let rootURL: URL
        let inputURL: URL
        let outputURL: URL
        let ranges: String?
        let autoShutdownAfterCompletion: Bool
        let performanceProfile: RestorationPerformanceProfile
    }

    private func validatedRequest(
        performanceProfile: RestorationPerformanceProfile
    ) throws -> Request {
        guard let inputURL else {
            throw ValidationError("Choose a source side-by-side video.")
        }
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("The selected source video no longer exists.")
        }
        guard let outputURL else { throw ValidationError("Choose an output video file.") }
        guard try !FileIdentity.refersToSameFile(inputURL, outputURL) else {
            throw ValidationError("The output file must be different from the source video.")
        }
        guard ["mov", "mp4"].contains(outputURL.pathExtension.lowercased()) else {
            throw ValidationError("Choose a .mov or .mp4 output file.")
        }
        let ranges = try MosaicTimeRangeParser.optionalCommandArgument(
            for: self.ranges.map(\.value)
        )
        guard let rootURL = ProjectRootLocator.locate() else {
            throw ValidationError(
                "The Jasna project could not be found. Launch the app from the project folder "
                    + "or set JASNA_PROJECT_ROOT."
            )
        }
        return Request(
            rootURL: rootURL,
            inputURL: inputURL,
            outputURL: outputURL,
            ranges: ranges,
            autoShutdownAfterCompletion: autoShutdownAfterCompletion,
            performanceProfile: performanceProfile
        )
    }

    private func launch(_ request: Request) throws {
        let scriptURL = request.rootURL.appendingPathComponent("script/restore_vr_rollout.sh")
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var arguments = ["bash", scriptURL.path, request.inputURL.path, request.outputURL.path]
        if let ranges = request.ranges { arguments.append(ranges) }
        process.arguments = arguments
        process.currentDirectoryURL = request.rootURL
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        if let runtimeURL = Bundle.main.resourceURL?
            .appendingPathComponent("Runtime", isDirectory: true),
           FileManager.default.fileExists(atPath: runtimeURL.path)
        {
            let bundledBin = runtimeURL.appendingPathComponent("bin", isDirectory: true).path
            let existingPath = environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
            environment["PATH"] = "\(bundledBin):\(existingPath)"
        }
        if let bundledEngine = Bundle.main.url(forAuxiliaryExecutable: "JasnaMetalPoC") {
            environment["JASNA_APP_BINARY"] = bundledEngine.path
        }
        environment.merge(request.performanceProfile.processEnvironment) { _, requested in requested }
        process.environment = environment
        // GUI-launched child processes must never inherit a terminal as stdin. FFmpeg otherwise
        // receives SIGTTIN and silently pauses when it probes stdin from a background process group.
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = pipe
        process.standardError = pipe

        logText = ""
        logRefreshTask?.cancel()
        logRefreshTask = nil
        pendingVisibleLog = ""
        outputDecoder = UTF8StreamDecoder()
        progress = progressTracker.reset()
        shutdownTask?.cancel()
        shutdownTask = nil
        shutdownCountdownSeconds = nil
        openUILog(beside: request.outputURL)
        validationMessage = nil
        appendLog(
            "Jasna VR restoration\nSource: \(request.inputURL.path)\n"
                + "Output: \(request.outputURL.path)\nMosaic ranges: "
                + "\(request.ranges ?? "full video (automatic detection)")\n"
                + "Performance: \(request.performanceProfile.logDescription)\n"
                + "Compatible interrupted work retains its recorded model batch.\n"
                + "Restart files: always preserved\n"
                + "Automatic shutdown after success: "
                + "\(request.autoShutdownAfterCompletion ? "enabled" : "disabled")\n\n"
        )
        let coordinator = ProcessOutputCoordinator(
            onData: { [weak self] data in
                DispatchQueue.main.async { [weak self] in self?.receiveOutput(data) }
            },
            onCompletion: { [weak self] status in
                DispatchQueue.main.async { [weak self] in self?.processFinished(status: status) }
            }
        )
        outputCoordinator = coordinator
        pipe.fileHandleForReading.readabilityHandler = { handle in
            coordinator.consumeAvailableData(from: handle)
        }
        process.terminationHandler = { finishedProcess in
            let status = finishedProcess.terminationStatus
            coordinator.finish(status: status, readingRemainingFrom: pipe.fileHandleForReading)
        }
        do {
            try process.run()
            runtimeLogger.info(
                "Started \(request.performanceProfile.rawValue, privacy: .public) restoration profile with output-local runtime scratch"
            )
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            outputCoordinator = nil
            uiLogWriter?.close()
            uiLogWriter = nil
            throw error
        }
        self.process = process
        outputPipe = pipe
        state = .running
        activity = "Starting the validated restoration workflow…"
    }

    private func appendLog(_ text: String) {
        uiLogWriter?.append(text)
        pendingVisibleLog.append(text)
        if pendingVisibleLog.count > maximumVisibleLogCharacters {
            pendingVisibleLog = String(pendingVisibleLog.suffix(maximumVisibleLogCharacters))
        }
        guard logRefreshTask == nil else { return }
        logRefreshTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            self?.flushVisibleLog()
        }
    }

    private func flushVisibleLog() {
        logRefreshTask?.cancel()
        logRefreshTask = nil
        let text = pendingVisibleLog
        pendingVisibleLog = ""
        guard !text.isEmpty else { return }
        logText.append(text)
        if logText.count > maximumVisibleLogCharacters {
            logText.removeFirst(logText.count - maximumVisibleLogCharacters)
        }
        if let latest = text.split(whereSeparator: \.isNewline).last {
            let line = latest.trimmingCharacters(in: .whitespacesAndNewlines)
            if !line.isEmpty { activity = line }
        }
    }

    private func receiveOutput(_ data: Data) {
        let text = outputDecoder.decode(data)
        if !text.isEmpty {
            progress = progressTracker.consume(text)
            appendLog(text)
        }
    }

    private func openUILog(beside outputURL: URL) {
        uiLogWriter?.close()
        logWarning = nil
        let sessionID = UUID()
        logSessionID = sessionID
        let logURL = outputURL.deletingPathExtension().appendingPathExtension("jasna-ui.log")
        uiLogWriter = QueuedLogWriter(url: logURL) { [weak self] message in
            runtimeLogger.error("UI log write failed: \(message, privacy: .public)")
            Task { @MainActor [weak self] in
                guard let self, self.logSessionID == sessionID else { return }
                self.logWarning = "Could not save the UI log at \(logURL.path). "
                    + "The saved log may be incomplete; live output is still shown below. \(message)"
            }
        }
        appendLog("\n===== UI session \(Date().formatted()) =====\n")
    }

    private func processFinished(status: Int32) {
        runtimeLogger.info("Restoration process finished with status \(status, privacy: .public)")
        let finalText = outputDecoder.finish()
        if !finalText.isEmpty {
            progress = progressTracker.consume(finalText)
            appendLog(finalText)
        }
        progress = progressTracker.finish()
        flushVisibleLog()
        outputPipe = nil
        outputCoordinator = nil
        process = nil
        if state == .stopping {
            // A child may exit successfully just as Stop is requested. User cancellation
            // still takes precedence over success-triggered actions such as shutdown.
            state = .ready
            activity = "Restoration stopped. Existing work files are preserved for resume."
        } else if status == 0 {
            state = .completed
            progress = progressTracker.markCompleted()
            if autoShutdownAfterCompletion {
                appendLog("\nOutput validation passed. Automatic shutdown scheduled in 60 seconds.\n")
                flushVisibleLog()
                scheduleAutomaticShutdown()
            } else {
                activity = "Restoration completed successfully. Restart files were preserved."
            }
        } else {
            state = .failed
            activity = "Restoration stopped with exit code \(status). Review the log below."
        }
        uiLogWriter?.close()
        uiLogWriter = nil
    }

    private func scheduleAutomaticShutdown() {
        shutdownTask?.cancel()
        shutdownCountdownSeconds = 60
        activity = "Restoration completed. This Mac will shut down in 60 seconds."
        shutdownTask = Task { [weak self] in
            for remaining in stride(from: 59, through: 0, by: -1) {
                try? await Task.sleep(for: .seconds(1))
                guard !Task.isCancelled else { return }
                self?.shutdownCountdownSeconds = remaining
                self?.activity = "Restoration completed. This Mac will shut down in \(remaining) seconds."
            }
            guard !Task.isCancelled,
                  self?.state == .completed,
                  self?.autoShutdownAfterCompletion == true else { return }
            do {
                try await SystemShutdownService.requestShutdown()
            } catch {
                guard !Task.isCancelled else { return }
                self?.shutdownCountdownSeconds = nil
                self?.validationMessage = "Automatic shutdown failed: \(error.localizedDescription)"
                self?.activity = "Restoration completed, but macOS did not accept the shutdown request."
            }
        }
    }

    private struct ValidationError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
