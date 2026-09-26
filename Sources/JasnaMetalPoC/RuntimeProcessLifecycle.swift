import Foundation

enum RuntimeProcessLifecycle {
    static func recordProcessIdentifierIfRequested(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        processIdentifier: Int32 = ProcessInfo.processInfo.processIdentifier
    ) throws {
        guard let path = environment["JASNA_RUNTIME_PID_FILE"], !path.isEmpty else {
            return
        }
        try Data("\(processIdentifier)\n".utf8).write(
            to: URL(fileURLWithPath: path),
            options: .atomic
        )
    }
}
