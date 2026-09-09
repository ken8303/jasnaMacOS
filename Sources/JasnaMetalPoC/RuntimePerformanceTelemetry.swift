import Foundation

enum RuntimePerformanceTelemetry {
    static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    static func message(
        phase: String,
        thermalState: ProcessInfo.ThermalState,
        lowPowerModeEnabled: Bool
    ) -> String {
        "Runtime performance state (\(phase)): thermal "
            + "\(thermalStateName(thermalState)), low power mode "
            + (lowPowerModeEnabled ? "enabled" : "disabled")
    }

    static func reportIfEnabled(
        phase: String,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        processInfo: ProcessInfo = .processInfo
    ) {
        guard environment["JASNA_LOG_PERFORMANCE_STATE"] == "1" else { return }
        let line = message(
            phase: phase,
            thermalState: processInfo.thermalState,
            lowPowerModeEnabled: processInfo.isLowPowerModeEnabled
        )
        // Restoration subprocess logs capture stderr. Keep this diagnostic in
        // the same persistent output-local log as peak-memory telemetry.
        FileHandle.standardError.write(Data("\(line)\n".utf8))
    }
}
