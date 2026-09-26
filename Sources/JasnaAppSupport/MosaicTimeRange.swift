import Foundation

public struct MosaicTimeRange: Equatable, Sendable {
    public let startSeconds: Int
    public let endSeconds: Int

    public init(startSeconds: Int, endSeconds: Int) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
    }

    public var commandArgument: String {
        "\(Self.format(startSeconds))-\(Self.format(endSeconds))"
    }

    public static func format(_ seconds: Int) -> String {
        let safeSeconds = max(0, seconds)
        let hours = safeSeconds / 3_600
        let minutes = (safeSeconds % 3_600) / 60
        let remainingSeconds = safeSeconds % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, remainingSeconds)
    }
}

public enum MosaicTimeRangeError: LocalizedError, Equatable {
    case noRanges
    case invalidRange(String)
    case invalidTimecode(String)
    case endNotAfterStart(Int)

    public var errorDescription: String? {
        switch self {
        case .noRanges:
            "Add at least one time range containing mosaic."
        case .invalidRange(let value):
            "“\(value)” is not a valid range. Use start-end, for example 11:00-30:00."
        case .invalidTimecode(let value):
            "“\(value)” is not a valid time. Use HH:MM:SS, MM:SS, or seconds."
        case .endNotAfterStart(let row):
            "Range \(row) must end after it starts."
        }
    }
}

public enum MosaicTimeRangeParser {
    public static func optionalCommandArgument(for expressions: [String]) throws -> String? {
        let hasRange = expressions.contains {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard hasRange else { return nil }
        return try commandArgument(for: expressions)
    }

    public static func parseExpressions(_ expressions: [String]) throws -> [MosaicTimeRange] {
        let nonempty = expressions.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty }
        guard !nonempty.isEmpty else { throw MosaicTimeRangeError.noRanges }

        let drafts = try nonempty.map { expression -> (start: String, end: String) in
            let components = expression.split(
                separator: "-", omittingEmptySubsequences: false
            )
            guard components.count == 2 else {
                throw MosaicTimeRangeError.invalidRange(expression)
            }
            let start = String(components[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let end = String(components[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !start.isEmpty, !end.isEmpty else {
                throw MosaicTimeRangeError.invalidRange(expression)
            }
            return (start, end)
        }
        return try parse(drafts)
    }

    public static func commandArgument(for expressions: [String]) throws -> String {
        try parseExpressions(expressions).map(\.commandArgument).joined(separator: ",")
    }

    public static func parse(
        _ drafts: [(start: String, end: String)]
    ) throws -> [MosaicTimeRange] {
        guard !drafts.isEmpty else { throw MosaicTimeRangeError.noRanges }
        let ranges = try drafts.enumerated().map { index, draft in
            let start = try parseTimecode(draft.start)
            let end = try parseTimecode(draft.end)
            guard end > start else {
                throw MosaicTimeRangeError.endNotAfterStart(index + 1)
            }
            return MosaicTimeRange(startSeconds: start, endSeconds: end)
        }.sorted { $0.startSeconds < $1.startSeconds }

        var merged = [MosaicTimeRange]()
        for range in ranges {
            guard let previous = merged.last, range.startSeconds <= previous.endSeconds else {
                merged.append(range)
                continue
            }
            merged[merged.count - 1] = MosaicTimeRange(
                startSeconds: previous.startSeconds,
                endSeconds: max(previous.endSeconds, range.endSeconds)
            )
        }
        return merged
    }

    public static func commandArgument(
        for drafts: [(start: String, end: String)]
    ) throws -> String {
        try parse(drafts).map(\.commandArgument).joined(separator: ",")
    }

    public static func parseTimecode(_ value: String) throws -> Int {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let components = trimmed.split(separator: ":", omittingEmptySubsequences: false)
        guard (1...3).contains(components.count),
              components.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) })
        else { throw MosaicTimeRangeError.invalidTimecode(value) }
        let numbers = components.compactMap { Int($0) }
        guard numbers.count == components.count else {
            throw MosaicTimeRangeError.invalidTimecode(value)
        }
        switch numbers.count {
        case 1:
            return numbers[0]
        case 2:
            guard numbers[1] < 60 else {
                throw MosaicTimeRangeError.invalidTimecode(value)
            }
            let (minutes, multiplyOverflow) = numbers[0].multipliedReportingOverflow(by: 60)
            let (total, addOverflow) = minutes.addingReportingOverflow(numbers[1])
            guard !multiplyOverflow, !addOverflow else {
                throw MosaicTimeRangeError.invalidTimecode(value)
            }
            return total
        case 3:
            guard numbers[1] < 60, numbers[2] < 60 else {
                throw MosaicTimeRangeError.invalidTimecode(value)
            }
            let (hours, hoursOverflow) = numbers[0].multipliedReportingOverflow(by: 3_600)
            let (minutes, minutesOverflow) = numbers[1].multipliedReportingOverflow(by: 60)
            let (partial, partialOverflow) = hours.addingReportingOverflow(minutes)
            let (total, totalOverflow) = partial.addingReportingOverflow(numbers[2])
            guard !hoursOverflow, !minutesOverflow, !partialOverflow, !totalOverflow else {
                throw MosaicTimeRangeError.invalidTimecode(value)
            }
            return total
        default:
            throw MosaicTimeRangeError.invalidTimecode(value)
        }
    }
}
