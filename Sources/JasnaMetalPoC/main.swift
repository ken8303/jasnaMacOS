import Foundation

do {
    RuntimePerformanceTelemetry.reportIfEnabled(phase: "process start")
    defer {
        RuntimePerformanceTelemetry.reportIfEnabled(phase: "process end")
        RuntimeMemoryTelemetry.reportIfEnabled()
    }
    try RuntimeProcessLifecycle.recordProcessIdentifierIfRequested()
    try await runJasnaCLI()
} catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
