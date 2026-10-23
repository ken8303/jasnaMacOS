import Foundation
import UniformTypeIdentifiers

public enum RestorationSourceQueue {
    public static func outputURL(for source: URL) -> URL {
        let suffix = source.pathExtension.lowercased() == "mov" ? "mov" : "mp4"
        return source.deletingLastPathComponent().appendingPathComponent(
            source.deletingPathExtension().lastPathComponent + "_restored." + suffix
        )
    }

    public static func accepts(_ url: URL) -> Bool {
        guard url.isFileURL,
              let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .contentTypeKey]),
              values.isRegularFile == true else { return false }
        return values.contentType?.conforms(to: .movie) == true
            || ["mp4", "mov", "mkv", "m4v", "avi", "webm", "mts", "m2ts"].contains(url.pathExtension.lowercased())
    }

    public static func validate(_ sources: [URL]) throws {
        for (index, source) in sources.enumerated() {
            let output = outputURL(for: source)
            for other in sources {
                if try FileIdentity.refersToSameFile(other, output) {
                    throw QueueError("The output for \(source.lastPathComponent) would overwrite a selected source video. Remove one of these files.")
                }
            }
            for previous in sources.prefix(index) {
                let previousOutput = outputURL(for: previous)
                let samePath = try sameDestination(output, previousOutput)
                let sameRestartPath = try sameDestination(output.deletingPathExtension(), previousOutput.deletingPathExtension())
                var sameExistingFile = false
                if FileManager.default.fileExists(atPath: output.path),
                   FileManager.default.fileExists(atPath: previousOutput.path) {
                    sameExistingFile = try FileIdentity.refersToSameFile(output, previousOutput)
                }
                if samePath || sameRestartPath || sameExistingFile {
                    throw QueueError("\(source.lastPathComponent) and \(previous.lastPathComponent) would share an output or restart filename. Rename one source file.")
                }
            }
        }
    }

    // Outputs may not exist yet, so inode checks alone cannot detect collisions.
    // Compare existing parent directories by identity and names using that volume's rules.
    private static func sameDestination(_ first: URL, _ second: URL) throws -> Bool {
        let first = first.resolvingSymlinksInPath().standardizedFileURL
        let second = second.resolvingSymlinksInPath().standardizedFileURL
        if first == second { return true }
        let parent = first.deletingLastPathComponent()
        guard try FileIdentity.refersToSameFile(parent, second.deletingLastPathComponent()) else { return false }
        let caseSensitive = try parent.resourceValues(forKeys: [.volumeSupportsCaseSensitiveNamesKey])
            .volumeSupportsCaseSensitiveNames ?? false
        let firstName = first.lastPathComponent.precomposedStringWithCanonicalMapping
        let secondName = second.lastPathComponent.precomposedStringWithCanonicalMapping
        return caseSensitive ? firstName == secondName
            : firstName.compare(secondName, options: .caseInsensitive, locale: Locale(identifier: "en_US_POSIX")) == .orderedSame
    }

    private struct QueueError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
