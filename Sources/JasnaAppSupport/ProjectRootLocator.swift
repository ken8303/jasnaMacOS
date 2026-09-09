import Foundation

public enum ProjectRootLocator {
    public static func locate(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        ),
        executableURL: URL? = Bundle.main.executableURL,
        resourceURL: URL? = Bundle.main.resourceURL
    ) -> URL? {
        var candidates = [URL]()
        if let configured = environment["JASNA_PROJECT_ROOT"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured, isDirectory: true))
        }
        // A packaged app carries its complete runtime here. Check it before the
        // launch working directory: Finder commonly launches an app with `/`
        // or another unrelated directory as its current directory.
        if let resourceURL {
            candidates.append(resourceURL.appendingPathComponent("Runtime", isDirectory: true))
        }
        if let executableURL {
            candidates.append(executableURL.deletingLastPathComponent())
        }
        candidates.append(currentDirectory)
        for candidate in candidates {
            if let root = firstProjectRoot(startingAt: candidate) {
                return root
            }
        }
        return nil
    }

    public static func isProjectRoot(_ url: URL) -> Bool {
        let fileManager = FileManager.default
        return fileManager.fileExists(
            atPath: url.appendingPathComponent("Package.swift").path
        ) && fileManager.isExecutableFile(
            atPath: url.appendingPathComponent("script/restore_vr_rollout.sh").path
        )
    }

    private static func firstProjectRoot(startingAt start: URL) -> URL? {
        var candidate = start.standardizedFileURL
        var visitedPaths = Set<String>()
        // A valid absolute file URL has far fewer ancestors. The bound and
        // cycle check defend against Foundation returning alternating root
        // representations on beta systems or unusual mounted volumes.
        for _ in 0..<256 {
            guard visitedPaths.insert(candidate.path).inserted else { return nil }
            if isProjectRoot(candidate) { return candidate }
            let parent = candidate.deletingLastPathComponent().standardizedFileURL
            guard parent.path != candidate.path else { return nil }
            candidate = parent
        }
        return nil
    }
}
