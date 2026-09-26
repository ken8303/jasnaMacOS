import Foundation
import Testing
@testable import JasnaAppSupport

private final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = [String]()

    func append(_ event: String) {
        lock.withLock { storage.append(event) }
    }

    var events: [String] { lock.withLock { storage } }
}

@Test func processOutputCoordinatorDeliversTailBeforeCompletion() throws {
    let pipe = Pipe()
    let recorder = EventRecorder()
    let coordinator = ProcessOutputCoordinator(
        onData: { data in
            recorder.append(String(decoding: data, as: UTF8.self))
        },
        onCompletion: { status in
            recorder.append("finished:\(status)")
        }
    )

    try pipe.fileHandleForWriting.write(contentsOf: Data("first\n".utf8))
    coordinator.consumeAvailableData(from: pipe.fileHandleForReading)
    try pipe.fileHandleForWriting.write(contentsOf: Data("tail\n".utf8))
    try pipe.fileHandleForWriting.close()
    coordinator.finish(status: 0, readingRemainingFrom: pipe.fileHandleForReading)

    #expect(recorder.events == ["first\n", "tail\n", "finished:0"])
}
