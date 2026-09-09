import Foundation
import Testing
@testable import JasnaAppSupport

@Test func progressRecoversAfterCarriageReturnsAndOversizedRecords() {
    var tracker = RestorationProgressTracker()
    _ = tracker.consume(String(repeating: "x", count: 100_000))
    let result = tracker.consume("\rframe=100\rJASNA_PROGRESS|1|1|1|Ready\r\n")
    #expect(result.phases[0].fraction == 1)
    let invalid = tracker.consume("JASNA_PROGRESS|2|nan|1|Invalid\n")
    #expect(invalid.phases[1].fraction == 0)
    #expect(invalid.overallFraction.isFinite)
}

@Test func restorationProgressHandlesSplitRecordsAndEstimatesCompletion() {
    let start = Date(timeIntervalSince1970: 1_000)
    var tracker = RestorationProgressTracker()
    _ = tracker.reset(at: start)

    _ = tracker.consume("JASNA_PROGRESS|1|1|1|Prepared\nJASNA_PRO", at: start.addingTimeInterval(10))
    let result = tracker.consume(
        "GRESS|2|3|12|Detecting\n",
        at: start.addingTimeInterval(20)
    )

    #expect(result.phases[0].fraction == 1)
    #expect(result.phases[1].fraction == 0.25)
    #expect(result.phases[2].fraction == 0)
    #expect(abs(result.overallFraction - 0.075) < 0.000_001)
    #expect(result.estimatedCompletion == nil)
    let measured = tracker.consume("JASNA_PROGRESS|2|6|12|Detecting\n", at: start.addingTimeInterval(30))
    #expect(measured.estimatedCompletion == start.addingTimeInterval(50))
    let nextPhase = tracker.consume("JASNA_PROGRESS|3|0|100|Next\n", at: start.addingTimeInterval(31))
    #expect(nextPhase.estimatedCompletion == nil)
}

@Test func rapidCachedProgressDoesNotProduceAnETA() {
    var tracker = RestorationProgressTracker()
    let start = Date(timeIntervalSince1970: 1000)
    _ = tracker.consume("JASNA_PROGRESS|3|0|100|Start\n", at: start)
    let cached = tracker.consume("JASNA_PROGRESS|3|90|100|Cached\n", at: start.addingTimeInterval(1))
    #expect(cached.estimatedCompletion == nil)
    let measured = tracker.consume("JASNA_PROGRESS|3|95|100|Next\n", at: start.addingTimeInterval(11))
    #expect(abs(measured.estimatedCompletion!.timeIntervalSince(start) - 21) < 0.0001)
}

@Test func restorationProgressNeverMovesBackward() {
    var tracker = RestorationProgressTracker()
    _ = tracker.reset()
    _ = tracker.consume("JASNA_PROGRESS|3|8|10|Restoring\n")
    let result = tracker.consume("JASNA_PROGRESS|3|2|10|Restoring\n")
    #expect(result.phases[2].fraction == 0.8)
}

@Test func restorationProgressCompletesEveryPhase() {
    var tracker = RestorationProgressTracker()
    let result = tracker.markCompleted()
    #expect(result.overallFraction == 1)
    #expect(result.phases.allSatisfy { $0.fraction == 1 })
    #expect(result.estimatedCompletion == nil)
}
