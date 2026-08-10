import Foundation

@available(macOS 27.0, *)
extension SideBySideRestoration {
    struct WindowResult {
        let cacheDirectory: URL
        let cacheURLs: [URL]
        let gpuMilliseconds: Double
        let cacheBytes: Int
    }

    struct ResumableWindowCache {
        let directory: URL
        let urls: [URL]
        let completedTiles: Int
    }

    static func hasInterruptedWindowCache() throws -> Bool {
        guard let configuredWorkPath = ProcessInfo.processInfo.environment["JASNA_WORK_DIR"] else {
            return false
        }
        let workURL = URL(fileURLWithPath: configuredWorkPath, isDirectory: true)
        guard FileManager.default.fileExists(atPath: workURL.path) else { return false }
        return try FileManager.default.contentsOfDirectory(
            at: workURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).contains { $0.lastPathComponent.hasPrefix("window-") }
    }

    static func archiveInterruptedOutput(_ outputURL: URL) throws -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        let stem = outputURL.deletingPathExtension().lastPathComponent
        let suffix = outputURL.pathExtension
        let archivedName = "\(stem).interrupted-\(formatter.string(from: Date()))"
            + (suffix.isEmpty ? "" : ".\(suffix)")
        let archivedURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(archivedName)
        try FileManager.default.moveItem(at: outputURL, to: archivedURL)
        return archivedURL
    }

    static func resumableWindowCache(
        in workDirectory: URL,
        windowIndex: Int,
        outputCount: Int,
        tileCount: Int,
        cacheVariant: String?
    ) throws -> ResumableWindowCache? {
        let prefix = cacheDirectoryPrefix(windowIndex: windowIndex, cacheVariant: cacheVariant)
        let candidates = try FileManager.default.contentsOfDirectory(
            at: workDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            url.lastPathComponent.hasPrefix(prefix)
                && (cacheVariant != nil || !url.lastPathComponent.contains("-sparse-"))
                && (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        var best: ResumableWindowCache?
        for directory in candidates {
            let urls = (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            guard urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else {
                continue
            }
            let completed = try recoverableTileCount(
                cacheURLs: urls,
                bytesPerTile: tileBytes,
                tileCount: tileCount
            )
            guard completed > 0 else { continue }
            if best == nil || completed > best!.completedTiles {
                best = ResumableWindowCache(
                    directory: directory,
                    urls: urls,
                    completedTiles: completed
                )
            }
        }
        return best
    }

    static func cacheDirectoryPrefix(
        windowIndex: Int,
        cacheVariant: String?
    ) -> String {
        if let cacheVariant {
            return "window-\(windowIndex + 1)-\(cacheVariant)-"
        }
        return "window-\(windowIndex + 1)-"
    }

    static func sparseCacheVariant(tiles: [VideoTile]) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for tile in tiles {
            for value in [tile.x, tile.y, tile.width, tile.height] {
                var littleEndian = UInt64(value).littleEndian
                withUnsafeBytes(of: &littleEndian) { bytes in
                    for byte in bytes {
                        hash ^= UInt64(byte)
                        hash &*= 1_099_511_628_211
                    }
                }
            }
        }
        return String(format: "sparse-%016llx", hash)
    }

    static func sparseRegionCacheVariant(
        regions: [MosaicRegion],
        projection: VRMosaicProjection = .raw,
        restorationIdentity: String = ""
    ) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in "\(projection.rawValue):\(restorationIdentity)".utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        for region in regions {
            let values = [
                region.startFrame, region.endFrame,
                region.x, region.y, region.width, region.height,
                region.effectiveBlendX, region.effectiveBlendY,
                region.effectiveBlendWidth, region.effectiveBlendHeight,
                region.subdivisionGroup ?? 0,
            ]
            for value in values {
                var littleEndian = UInt64(value).littleEndian
                withUnsafeBytes(of: &littleEndian) { bytes in
                    for byte in bytes {
                        hash ^= UInt64(byte)
                        hash &*= 1_099_511_628_211
                    }
                }
            }
        }
        return String(format: "crop-v4-%@-%016llx", projection.rawValue, hash)
    }

    static func restorationCacheIdentity(
        sourceURLs: [URL],
        modelsURL: URL,
        weightsURL: URL
    ) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        let fileManager = FileManager.default
        let roots = sourceURLs + [modelsURL, weightsURL]
        var entries = [URL]()
        for root in roots {
            entries.append(root.standardizedFileURL)
            if let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [
                    .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
                ],
                options: [.skipsHiddenFiles]
            ) {
                entries.append(contentsOf: enumerator.compactMap { $0 as? URL })
            }
        }
        for url in entries.sorted(by: { $0.path < $1.path }) {
            let values = try? url.resourceValues(
                forKeys: [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
            )
            guard values?.isRegularFile == true || roots.contains(url) else { continue }
            let metadata = "\(url.standardizedFileURL.path)\u{0}"
                + "\(values?.fileSize ?? -1)\u{0}"
                + "\(values?.contentModificationDate?.timeIntervalSince1970 ?? -1)\u{0}"
            for byte in metadata.utf8 {
                hash ^= UInt64(byte)
                hash &*= 1_099_511_628_211
            }
        }
        return String(format: "%016llx", hash)
    }

    static func recoverableTileCount(
        cacheURLs: [URL], bytesPerTile: Int, tileCount: Int
    ) throws -> Int {
        guard !cacheURLs.isEmpty, bytesPerTile > 0, tileCount >= 0 else { return 0 }
        let sizes = try cacheURLs.map {
            try $0.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        }
        return min(tileCount, (sizes.min() ?? 0) / bytesPerTile)
    }
}
