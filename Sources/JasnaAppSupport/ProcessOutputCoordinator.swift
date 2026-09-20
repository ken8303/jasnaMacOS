import Foundation

/// Serializes pipe reads with process termination so the final output cannot race completion.
public final class ProcessOutputCoordinator: @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false
    private let onData: @Sendable (Data) -> Void
    private let onCompletion: @Sendable (Int32) -> Void

    public init(
        onData: @escaping @Sendable (Data) -> Void,
        onCompletion: @escaping @Sendable (Int32) -> Void
    ) {
        self.onData = onData
        self.onCompletion = onCompletion
    }

    public func consumeAvailableData(from handle: FileHandle) {
        lock.lock()
        defer { lock.unlock() }
        guard !completed else { return }
        let data = handle.availableData
        if !data.isEmpty { onData(data) }
    }

    public func finish(status: Int32, readingRemainingFrom handle: FileHandle) {
        lock.lock()
        defer { lock.unlock() }
        guard !completed else { return }
        handle.readabilityHandler = nil
        let remaining = handle.readDataToEndOfFile()
        if !remaining.isEmpty { onData(remaining) }
        completed = true
        onCompletion(status)
    }
}
