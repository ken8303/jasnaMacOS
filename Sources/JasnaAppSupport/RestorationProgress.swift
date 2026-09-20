import Foundation

public struct RestorationPhaseProgress: Equatable, Identifiable, Sendable {
    public let id: Int
    public let title: String
    public let fraction: Double

    public init(id: Int, title: String, fraction: Double) {
        self.id = id
        self.title = title
        self.fraction = min(max(fraction, 0), 1)
    }
}

public struct RestorationProgressSnapshot: Equatable, Sendable {
    public let phases: [RestorationPhaseProgress]
    public let overallFraction: Double
    public let estimatedCompletion: Date?

    public static let initial = RestorationProgressSnapshot(
        phases: RestorationProgressTracker.phaseTitles.enumerated().map {
            RestorationPhaseProgress(id: $0.offset + 1, title: $0.element, fraction: 0)
        },
        overallFraction: 0,
        estimatedCompletion: nil
    )
}

/// Consumes the workflow's stable `JASNA_PROGRESS|phase|current|total|label` records.
/// Keeping the parser in AppSupport makes progress reporting testable without launching the UI.
public struct RestorationProgressTracker: Sendable {
    public static let phaseTitles = [
        "Prepare video",
        "Detect mosaics",
        "Restore video",
        "Assemble and verify",
    ]

    private static let phaseWeights = [0.05, 0.10, 0.80, 0.05]
    private var fractions = Array(repeating: 0.0, count: phaseTitles.count)
    private var pendingText = ""
    private var pendingUTF8Bytes = 0
    private var discardingOversizedLine = false
    private static let maximumRecordLength = 4096
    private var activePhase = 0
    private var measurementDate: Date?
    private var measurementFraction = 0.0
    private var phaseEstimate: Date?
    private var warmupUntil: Date?

    public init() {}

    public mutating func reset(at date: Date = Date()) -> RestorationProgressSnapshot {
        fractions = Array(repeating: 0, count: Self.phaseTitles.count)
        pendingText = ""
        pendingUTF8Bytes = 0
        discardingOversizedLine = false
        activePhase = 0
        measurementDate = nil
        measurementFraction = 0
        phaseEstimate = nil
        warmupUntil = nil
        return snapshot(at: date)
    }

    public mutating func consume(
        _ text: String,
        at date: Date = Date()
    ) -> RestorationProgressSnapshot {
        // FFmpeg status uses carriage returns. Parse once, with bounded storage,
        // even if a child writes an enormous unterminated diagnostic line.
        for character in text {
            if character.isNewline {
                if !discardingOversizedLine { consumeLine(pendingText, at: date) }
                pendingText = ""
                pendingUTF8Bytes = 0
                discardingOversizedLine = false
            } else if !discardingOversizedLine {
                let characterBytes = character.utf8.count
                if pendingUTF8Bytes + characterBytes > Self.maximumRecordLength {
                    pendingText = ""
                    pendingUTF8Bytes = 0
                    discardingOversizedLine = true
                } else {
                    pendingText.append(character)
                    pendingUTF8Bytes += characterBytes
                }
            }
        }
        return snapshot(at: date)
    }

    public mutating func finish(at date: Date = Date()) -> RestorationProgressSnapshot {
        if !discardingOversizedLine && !pendingText.isEmpty { consumeLine(pendingText, at: date) }
        pendingText = ""
        pendingUTF8Bytes = 0
        discardingOversizedLine = false
        return snapshot(at: date)
    }

    public mutating func markCompleted(at date: Date = Date()) -> RestorationProgressSnapshot {
        fractions = Array(repeating: 1, count: Self.phaseTitles.count)
        phaseEstimate = nil
        return snapshot(at: date)
    }

    private mutating func consumeLine(_ line: String, at date: Date) {
        guard line.hasPrefix("JASNA_PROGRESS|") else { return }
        let fields = line.split(separator: "|", omittingEmptySubsequences: false)
        guard fields.count >= 5,
              let phase = Int(fields[1]),
              let current = Double(fields[2]),
              let total = Double(fields[3]),
              phase >= 1, phase <= fractions.count,
              current.isFinite, total.isFinite, total > 0
        else { return }
        guard phase >= activePhase else { return }
        let fraction = min(max(current / total, 0), 1)
        if phase != activePhase {
            activePhase = phase
            measurementDate = date
            measurementFraction = fraction
            phaseEstimate = nil
            warmupUntil = date.addingTimeInterval(2)
        } else if let measurementDate, fraction > measurementFraction {
            let elapsed = date.timeIntervalSince(measurementDate)
            if elapsed >= 5 {
                let remaining = elapsed * (1 - fraction) / (fraction - measurementFraction)
                phaseEstimate = remaining.isFinite ? date.addingTimeInterval(remaining) : nil
                self.measurementDate = date
                measurementFraction = fraction
            } else if let warmupUntil, date <= warmupUntil {
                // Fast replay of cached progress cannot establish a useful processing rate.
                self.measurementDate = date
                measurementFraction = fraction
                phaseEstimate = nil
            }
        }
        if fraction >= 1 { phaseEstimate = nil }

        for completedIndex in 0..<(phase - 1) {
            fractions[completedIndex] = 1
        }
        fractions[phase - 1] = max(fractions[phase - 1], min(max(current / total, 0), 1))
    }

    private func snapshot(at date: Date) -> RestorationProgressSnapshot {
        let overall = zip(fractions, Self.phaseWeights).reduce(0) { partial, pair in
            partial + pair.0 * pair.1
        }
        // Do not extrapolate across phases with different costs or keep a missed ETA on screen.
        let estimate = phaseEstimate.flatMap { $0 > date ? $0 : nil }
        return RestorationProgressSnapshot(
            phases: zip(Self.phaseTitles.indices, Self.phaseTitles).map { index, title in
                RestorationPhaseProgress(id: index + 1, title: title, fraction: fractions[index])
            },
            overallFraction: overall,
            estimatedCompletion: estimate
        )
    }
}
