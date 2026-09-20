import Foundation

/// Decodes arbitrary process-pipe chunks without losing text when a UTF-8
/// scalar is split across two reads.
public struct UTF8StreamDecoder: Sendable {
    private var pending = Data()

    public init() {}

    public mutating func decode(_ data: Data) -> String {
        guard !data.isEmpty else { return "" }
        pending.append(data)
        let suffixLength = incompleteUTF8SuffixLength(in: pending)
        let prefixCount = pending.count - suffixLength
        guard prefixCount > 0 else { return "" }
        let prefix = pending.prefix(prefixCount)
        pending = Data(pending.suffix(suffixLength))
        return String(decoding: prefix, as: UTF8.self)
    }

    public mutating func finish() -> String {
        defer { pending.removeAll(keepingCapacity: false) }
        return String(decoding: pending, as: UTF8.self)
    }

    private func incompleteUTF8SuffixLength(in data: Data) -> Int {
        guard let last = data.last, last >= 0x80 else { return 0 }
        let bytes = [UInt8](data.suffix(4))
        var leadIndex = bytes.count - 1
        while leadIndex > 0, bytes[leadIndex] & 0xC0 == 0x80 {
            leadIndex -= 1
        }
        let lead = bytes[leadIndex]
        let requiredLength: Int
        switch lead {
        case 0xC2...0xDF: requiredLength = 2
        case 0xE0...0xEF: requiredLength = 3
        case 0xF0...0xF4: requiredLength = 4
        default: return 0
        }
        let availableLength = bytes.count - leadIndex
        return availableLength < requiredLength ? availableLength : 0
    }
}
