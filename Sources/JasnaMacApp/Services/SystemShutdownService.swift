import Foundation

enum SystemShutdownService {
    static func requestShutdown() async throws {
        try await Task.detached(priority: .utility) {
            try runShutdownRequest()
        }.value
    }

    private static func runShutdownRequest() throws {
        let process = Process()
        let errorPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", "tell application \"System Events\" to shut down"]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = errorPipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let message = String(
                decoding: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            throw ShutdownError(message: message.isEmpty ? "macOS rejected the shutdown request" : message)
        }
    }

    private struct ShutdownError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
}
