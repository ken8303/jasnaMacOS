import Foundation

/// The file handle is owned exclusively by this queue; writes and closure stay ordered.
public final class QueuedLogWriter: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.jasna.ui-log", qos: .utility)
    private var handle: FileHandle?
    private var failed = false
    private let onError: @Sendable (String) -> Void
    private let admissionLock = NSLock()
    private var acceptingWrites = true
    private var pendingBytes = 0
    private var pendingChunks = 0
    private let maximumPendingBytes: Int

    public init(
        url: URL,
        maximumPendingBytes: Int = 4 * 1024 * 1024,
        onError: @escaping @Sendable (String) -> Void
    ) {
        self.maximumPendingBytes = max(1, maximumPendingBytes)
        self.onError = onError
        queue.async { [self] in
            do {
                if !FileManager.default.fileExists(atPath: url.path) {
                    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                        throw CocoaError(.fileWriteUnknown)
                    }
                }
                handle = try FileHandle(forWritingTo: url)
                try handle?.seekToEnd()
            } catch { reportFailure(error) }
        }
    }

    public func append(_ text: String) {
        guard !text.isEmpty else { return }
        let data = Data(text.utf8)
        let byteCount = data.count
        admissionLock.lock()
        defer { admissionLock.unlock() }
        guard acceptingWrites else { return }
        guard byteCount <= maximumPendingBytes - pendingBytes, pendingChunks < 4096 else {
            acceptingWrites = false
            queue.async { [self] in
                reportFailure(LogBacklogError())
            }
            return
        }
        pendingBytes += byteCount
        pendingChunks += 1
        queue.async { [self] in
            defer {
                admissionLock.withLock {
                    pendingBytes -= byteCount
                    pendingChunks -= 1
                }
            }
            guard !failed else { return }
            do { try handle?.write(contentsOf: data) }
            catch { reportFailure(error) }
        }
    }

    public func close(completion: @escaping @Sendable () -> Void = {}) {
        admissionLock.lock()
        defer { admissionLock.unlock() }
        acceptingWrites = false
        queue.async { [self] in
            do { try handle?.close() }
            catch { reportFailure(error) }
            handle = nil
            completion()
        }
    }

    private func reportFailure(_ error: Error) {
        guard !failed else { return }
        failed = true
        admissionLock.withLock { acceptingWrites = false }
        try? handle?.close()
        handle = nil
        onError(error.localizedDescription)
    }

    private struct LogBacklogError: LocalizedError {
        var errorDescription: String? {
            "File logging stopped because queued output exceeded its memory limit. "
                + "The saved UI log is incomplete. Check the output drive."
        }
    }
}
