import Darwin
import Foundation

enum RuntimeMemoryTelemetry {
    static func peakResidentBytes() -> UInt64? {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0, usage.ru_maxrss > 0 else {
            return nil
        }
        // Darwin reports ru_maxrss in bytes. Linux reports KiB, but this target is macOS-only.
        return UInt64(usage.ru_maxrss)
    }

    static func formattedGiB(bytes: UInt64) -> String {
        String(format: "%.2f GiB", Double(bytes) / 1_073_741_824)
    }

    static func reportIfEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        guard environment["JASNA_LOG_PEAK_MEMORY"] == "1" else { return }
        let message: String
        if let bytes = peakResidentBytes() {
            message = "Runtime memory: peak resident \(formattedGiB(bytes: bytes)) "
                + "(\(bytes) bytes)\n"
        } else {
            message = "Runtime memory: peak resident unavailable\n"
        }
        FileHandle.standardError.write(Data(message.utf8))
    }
}
