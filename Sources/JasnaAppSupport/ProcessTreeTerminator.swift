import Darwin
import Dispatch
import Foundation

public struct ProcessRelationship: Equatable, Sendable {
    public let processIdentifier: Int32
    public let parentIdentifier: Int32

    public init(processIdentifier: Int32, parentIdentifier: Int32) {
        self.processIdentifier = processIdentifier
        self.parentIdentifier = parentIdentifier
    }
}

struct ProcessIdentity: Equatable, Hashable, Sendable {
    let processIdentifier: Int32
    let parentIdentifier: Int32
    let startMarker: String
}

public enum ProcessTreeTerminator {
    // Use a dedicated serial queue instead of the shared utility pool. Long CPU-heavy
    // restoration jobs can saturate that pool and delay the promised force-stop.
    private static let terminationQueue = DispatchQueue(
        label: "JasnaAppSupport.ProcessTreeTerminator",
        qos: .userInitiated
    )

    public static func descendantProcessIdentifiers(
        rootIdentifier: Int32,
        relationships: [ProcessRelationship]
    ) -> [Int32] {
        guard rootIdentifier > 1 else { return [] }
        let children = Dictionary(grouping: relationships, by: \.parentIdentifier)
        var visited: Set<Int32> = [rootIdentifier]
        var result = [Int32]()

        func appendDescendants(of parent: Int32) {
            for relationship in children[parent, default: []] {
                let child = relationship.processIdentifier
                guard child > 1, visited.insert(child).inserted else { continue }
                appendDescendants(of: child)
                result.append(child)
            }
        }
        appendDescendants(of: rootIdentifier)
        return result
    }

    public static func terminate(
        rootIdentifier: Int32,
        forceAfter gracePeriod: TimeInterval = 5
    ) {
        guard rootIdentifier > 1,
              rootIdentifier != ProcessInfo.processInfo.processIdentifier
        else { return }
        let initialProcesses = processIdentities()
        let descendants = descendantProcessIdentifiers(
            rootIdentifier: rootIdentifier,
            relationships: initialProcesses.map {
                ProcessRelationship(
                    processIdentifier: $0.processIdentifier,
                    parentIdentifier: $0.parentIdentifier
                )
            }
        )
        for identifier in descendants {
            _ = Darwin.kill(identifier, SIGTERM)
        }
        _ = Darwin.kill(rootIdentifier, SIGTERM)
        guard gracePeriod >= 0 else { return }
        let originalTargets = initialProcesses.filter {
            $0.processIdentifier == rootIdentifier || descendants.contains($0.processIdentifier)
        }
        terminationQueue.asyncAfter(deadline: .now() + gracePeriod) {
            forceTerminateSurvivors(
                rootIdentifier: rootIdentifier,
                originalTargets: originalTargets
            )
        }
    }

    private static func forceTerminateSurvivors(
        rootIdentifier: Int32,
        originalTargets: [ProcessIdentity]
    ) {
        let currentProcesses = processIdentities()
        var targets = matchingOriginalProcessIdentifiers(
            original: originalTargets,
            current: currentProcesses
        )
        let originalRoot = originalTargets.first { $0.processIdentifier == rootIdentifier }
        if let originalRoot, containsSameProcess(originalRoot, in: currentProcesses) {
            targets += descendantProcessIdentifiers(
                rootIdentifier: rootIdentifier,
                relationships: currentProcesses.map {
                    ProcessRelationship(
                        processIdentifier: $0.processIdentifier,
                        parentIdentifier: $0.parentIdentifier
                    )
                }
            )
        }
        var signaled = Set<Int32>()
        for identifier in targets where signaled.insert(identifier).inserted {
            _ = Darwin.kill(identifier, SIGKILL)
        }
        if let originalRoot, containsSameProcess(originalRoot, in: currentProcesses) {
            _ = Darwin.kill(rootIdentifier, SIGKILL)
        }
    }

    static func matchingOriginalProcessIdentifiers(
        original: [ProcessIdentity],
        current: [ProcessIdentity]
    ) -> [Int32] {
        return original.filter { containsSameProcess($0, in: current) }.map(\.processIdentifier)
    }

    private static func containsSameProcess(
        _ process: ProcessIdentity,
        in candidates: [ProcessIdentity]
    ) -> Bool {
        candidates.contains {
            $0.processIdentifier == process.processIdentifier
                && $0.startMarker == process.startMarker
        }
    }

    static func processIdentities() -> [ProcessIdentity] {
        let estimatedCount = proc_listallpids(nil, 0)
        guard estimatedCount > 0 else { return [] }
        // Leave headroom for processes created between the sizing and fill calls.
        var identifiers = [pid_t](
            repeating: 0,
            count: Int(estimatedCount) + 64
        )
        let returnedCount = identifiers.withUnsafeMutableBytes { buffer in
            proc_listallpids(buffer.baseAddress, Int32(buffer.count))
        }
        guard returnedCount > 0 else { return [] }

        let infoSize = Int32(MemoryLayout<proc_bsdinfo>.stride)
        return identifiers.prefix(Int(returnedCount)).compactMap { identifier in
            guard identifier > 1 else { return nil }
            var info = proc_bsdinfo()
            let bytes = withUnsafeMutablePointer(to: &info) { pointer in
                proc_pidinfo(identifier, PROC_PIDTBSDINFO, 0, pointer, infoSize)
            }
            guard bytes == infoSize else { return nil }
            return ProcessIdentity(
                processIdentifier: identifier,
                parentIdentifier: pid_t(info.pbi_ppid),
                startMarker: "\(info.pbi_start_tvsec):\(info.pbi_start_tvusec)"
            )
        }
    }
}
