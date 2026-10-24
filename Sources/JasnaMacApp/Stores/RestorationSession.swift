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

    private(set) var sourceURLs: [URL] = []
    private(set) var activeSourceIndex = 0
    var inputURL: URL? { sourceURLs.indices.contains(activeSourceIndex) ? sourceURLs[activeSourceIndex] : nil }
    var outputURL: URL? { inputURL.map { RestorationSourceQueue.outputURL(for: $0) } }
    var selectedBackend: RestorationBackend = .metal
    var ranges = [MosaicTimeRangeDraft()]
    var state: State = .ready
    var activity = "Drop source videos here. Restored files are saved beside each source. Leave mosaic times blank for full video."
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

    @ObservationIgnored private var pendingRequests: [Request] = []
    @ObservationIgnored private let projectRootURL: URL?
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

    init(projectRootURL: URL? = nil) {
        self.projectRootURL = projectRootURL
    }

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
        addSources(FilePanelService.chooseInputVideos())
    }

    func addSources(_ urls: [URL]) {
        guard !isRunning, !urls.isEmpty else { return }
        var accepted = sourceURLs
        var rejected = [String]()
        for candidate in urls {
            let url = candidate.resolvingSymlinksInPath().standardizedFileURL
            guard RestorationSourceQueue.accepts(url) else {
                rejected.append(candidate.lastPathComponent)
                continue
            }
            if accepted.contains(where: { (try? FileIdentity.refersToSameFile($0, url)) == true }) { continue }
            accepted.append(url)
        }
        if accepted != sourceURLs {
            sourceURLs = accepted
            resetForSourceChange()
        }
        validationMessage = rejected.isEmpty ? nil : "These files are not supported videos: " + rejected.joined(separator: ", ")
    }

    func removeSource(_ url: URL) {
        guard !isRunning else { return }
        sourceURLs.removeAll { $0 == url }
        resetForSourceChange()
    }

    func clearSources() {
        guard !isRunning else { return }
        sourceURLs = []
        resetForSourceChange()
    }

    private func resetForSourceChange() {
        shutdownTask?.cancel()
        shutdownTask = nil
        shutdownCountdownSeconds = nil
        logRefreshTask?.cancel()
        logRefreshTask = nil
        pendingVisibleLog = ""
        logText = ""
        logSessionID = UUID()
        uiLogWriter?.close()
        uiLogWriter = nil
        logWarning = nil
        outputDecoder = UTF8StreamDecoder()
        progress = progressTracker.reset()
        pendingRequests = []
        activeSourceIndex = 0
        state = .ready
        implementationResumeAvailable = false
        freshStartAvailable = false
        validationMessage = nil
        activity = sourceURLs.isEmpty
            ? "Drop source videos here. Restored files are saved beside each source."
            : "Ready to restore \(sourceURLs.count) video(s). Results will be saved beside each source."
    }

    func selectBackend(_ backend: RestorationBackend) {
        selectedBackend = backend
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

    private(set) var implementationResumeAvailable = false
    private(set) var freshStartAvailable = false

    func start(
        performanceProfile: RestorationPerformanceProfile,
        backend: RestorationBackend,
        detectorMode: DetectorExecutionMode = .stableFP32,
        restorationKernel: MLXRestorationKernel = .reference,
        allowImplementationResume: Bool = false,
        startFresh: Bool = false
    ) {
        guard !isRunning else { return }
        let freshAttempt = startFresh && freshStartAvailable
        let resumeAfterUpdate = !freshAttempt && allowImplementationResume && implementationResumeAvailable
        // Cancel the previous run's countdown even if validating this new request fails.
        shutdownTask?.cancel()
        shutdownTask = nil
        shutdownCountdownSeconds = nil
        do {
            try RestorationSourceQueue.validate(sourceURLs)
            if state == .completed { activeSourceIndex = 0 }
            pendingRequests = try sourceURLs.dropFirst(activeSourceIndex).map { source in
                try validatedRequest(
                    inputURL: source,
                    performanceProfile: performanceProfile,
                    backend: backend,
                    detectorMode: detectorMode,
                    restorationKernel: restorationKernel
                )
            }
            guard !pendingRequests.isEmpty else { throw ValidationError("Add at least one source video.") }
            let request = pendingRequests.removeFirst()
            try launch(request, allowImplementationResume: resumeAfterUpdate, startFresh: freshAttempt)
        } catch {
            pendingRequests = []
            state = .failed
            validationMessage = error.localizedDescription
            activity = error.localizedDescription
        }
    }

    func stop() {
        // Completion can already be queued on the main actor after the child exits.
        // Record cancellation before checking the process so it cannot launch another video.
        guard isRunning else { return }
        pendingRequests = []
        state = .stopping
        activity = "Stopping restoration safely…"
        appendLog("\nUser requested stop. Waiting for the current process to exit…\n")
        if let process, process.isRunning {
            ProcessTreeTerminator.terminate(rootIdentifier: process.processIdentifier)
        }
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
        let backend: RestorationBackend
        let detectorMode: DetectorExecutionMode
        let restorationKernel: MLXRestorationKernel
    }

    private func validatedRequest(
        inputURL: URL,
        performanceProfile: RestorationPerformanceProfile,
        backend: RestorationBackend,
        detectorMode: DetectorExecutionMode,
        restorationKernel: MLXRestorationKernel
    ) throws -> Request {
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("The selected source video no longer exists.")
        }
        let outputURL = RestorationSourceQueue.outputURL(for: inputURL)
        guard try !FileIdentity.refersToSameFile(inputURL, outputURL) else {
            throw ValidationError("The output file must be different from the source video.")
        }
        guard ["mov", "mp4"].contains(outputURL.pathExtension.lowercased()) else {
            throw ValidationError("Choose a .mov or .mp4 output file.")
        }
        let ranges = try MosaicTimeRangeParser.optionalCommandArgument(
            for: self.ranges.map(\.value)
        )
        guard let rootURL = projectRootURL ?? ProjectRootLocator.locate() else {
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
            performanceProfile: performanceProfile,
            backend: backend,
            detectorMode: detectorMode,
            restorationKernel: restorationKernel
        )
    }

    private func launch(_ request: Request, allowImplementationResume: Bool, startFresh: Bool = false) throws {
        guard try !FileIdentity.refersToSameFile(request.inputURL, request.outputURL) else {
            throw ValidationError("The output file must be different from the source video.")
        }
        let scriptURL = request.rootURL.appendingPathComponent("script/restore_vr_rollout.sh")
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var arguments = ["bash", scriptURL.path, request.inputURL.path, request.outputURL.path]
        if let ranges = request.ranges { arguments.append(ranges) }
        process.arguments = arguments
        process.currentDirectoryURL = request.outputURL.deletingLastPathComponent()
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
        environment.merge(request.backend.processEnvironment) { _, requested in requested }
        environment.merge(request.detectorMode.processEnvironment) { _, requested in requested }
        environment.merge(request.restorationKernel.processEnvironment) { _, requested in requested }
        // This opt-in belongs to this attempt only. The shell still verifies all other settings.
        environment["JASNA_ALLOW_IMPLEMENTATION_RESUME"] = allowImplementationResume ? "1" : "0"
        // Fresh start applies only to the failed video, never the remaining queue.
        environment["JASNA_START_FRESH"] = startFresh ? "1" : "0"
        process.environment = environment
        // GUI-launched child processes must never inherit a terminal as stdin. FFmpeg otherwise
        // receives SIGTTIN and silently pauses when it probes stdin from a background process group.
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = pipe
        process.standardError = pipe

        logText = ""
        implementationResumeAvailable = false
        freshStartAvailable = false
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
            "Jasna VR restoration\nQueue: \(activeSourceIndex + 1)/\(sourceURLs.count)\nSource: \(request.inputURL.path)\n"
                + "Output: \(request.outputURL.path)\nMosaic ranges: "
                + "\(request.ranges ?? "full video (automatic detection)")\n"
                + "Performance: \(request.performanceProfile.logDescription)\n"
                + "Restoration model: \(request.backend.title)\n"
                + "MLX detector mode: \(request.detectorMode.title)\n"
                + (request.backend == .mlx
                    ? "MLX restoration kernel: \(request.restorationKernel.title)\n"
                    : "")
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
        if status == 0, state != .stopping, !pendingRequests.isEmpty {
            uiLogWriter?.close()
            uiLogWriter = nil
            activeSourceIndex += 1
            let next = pendingRequests.removeFirst()
            do {
                try launch(next, allowImplementationResume: false)
            } catch {
                pendingRequests = []
                state = .failed
                validationMessage = error.localizedDescription
                activity = error.localizedDescription
            }
            return
        }
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
            pendingRequests = []
            state = .failed
            implementationResumeAvailable = logText.contains(
                "only the implementation changed; set JASNA_ALLOW_IMPLEMENTATION_RESUME=1 once to reuse compatible work"
            )
            freshStartAvailable = logText.contains("error: this output path belongs to a different test configuration")
                || logText.contains("error: a previous fresh start was interrupted")
            activity = freshStartAvailable
                ? "Saved work uses different settings. Start Fresh preserves it in a backup folder."
                : "Restoration stopped with exit code \(status). Review the log below."
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
