import Foundation

public enum FileIdentity {
    /// Compare resolved paths and, for existing files, their filesystem identity.
    /// Both volume and inode must match: different volumes can reuse inode numbers.
    public static func refersToSameFile(_ first: URL, _ second: URL) throws -> Bool {
        let first = first.resolvingSymlinksInPath().standardizedFileURL
        let second = second.resolvingSymlinksInPath().standardizedFileURL
        if first == second { return true }
        let firstAttributes = try FileManager.default.attributesOfItem(atPath: first.path)
        let secondAttributes: [FileAttributeKey: Any]
        do {
            secondAttributes = try FileManager.default.attributesOfItem(atPath: second.path)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            return false // A new output filename is expected and allowed.
        }
        guard let firstVolume = firstAttributes[.systemNumber] as? NSNumber,
              let secondVolume = secondAttributes[.systemNumber] as? NSNumber,
              let firstInode = firstAttributes[.systemFileNumber] as? NSNumber,
              let secondInode = secondAttributes[.systemFileNumber] as? NSNumber else {
            throw CocoaError(.fileReadUnknown)
        }
        return firstVolume == secondVolume && firstInode == secondInode
    }
}
