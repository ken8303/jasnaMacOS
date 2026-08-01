import AVFoundation
import CoreVideo
import Foundation
import Metal

struct SideBySideRestorationResult: Sendable {
    let input: VideoAssetInfo
    let output: VideoAssetInfo
    let tileCount: Int
    let frameCount: Int
    let windowCount: Int
    let gpuMilliseconds: Double
    let cacheBytes: Int
}

enum SideBySideEye: String, Sendable {
    case left
    case right
}

@available(macOS 27.0, *)
enum SideBySideRestoration {
    private static let tileElements = 3 * SideBySideVideoPlan.modelTileSize
        * SideBySideVideoPlan.modelTileSize
    private static let tileBytes = tileElements * MemoryLayout<Float16>.stride

    private struct PreparedRegionRestoration: Sendable {
        let regionIndex: Int
        let localStart: Int
        let activeFrameCount: Int
        let inputFrames: [[Float16]]
        let context: String
    }

    private struct CompletedRegionRestoration: Sendable {
        let prepared: PreparedRegionRestoration
        let frames: [[Float16]]
        let gpuMilliseconds: Double
        let wallMilliseconds: Double
    }

    private final class RegionRestorationBatch: @unchecked Sendable {
        private let device: MTLDevice
        private let modelsURL: URL
        private let weightsURL: URL
        private let work: [PreparedRegionRestoration]
        private let lock = NSLock()
        private var completed: [CompletedRegionRestoration?]
        private var failures: [(any Error)?]

        init(
            device: MTLDevice,
            modelsURL: URL,
            weightsURL: URL,
            work: [PreparedRegionRestoration]
        ) {
            self.device = device
            self.modelsURL = modelsURL
            self.weightsURL = weightsURL
            self.work = work
            completed = [CompletedRegionRestoration?](repeating: nil, count: work.count)
            failures = [(any Error)?](repeating: nil, count: work.count)
        }

        func execute(_ index: Int) {
            do {
                let item = work[index]
                let started = ContinuousClock.now
                let restored = try SideBySideRestoration.restoreTileWithFallback(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    inputFrames: item.inputFrames,
                    context: item.context
                )
                let elapsed = started.duration(to: .now).components
                let wallMilliseconds = Double(elapsed.seconds) * 1_000
                    + Double(elapsed.attoseconds) / 1_000_000_000_000_000
                lock.lock()
                completed[index] = CompletedRegionRestoration(
                    prepared: item,
                    frames: restored.frames,
                    gpuMilliseconds: restored.gpuMilliseconds,
                    wallMilliseconds: wallMilliseconds
                )
                lock.unlock()
            } catch {
                lock.lock()
                failures[index] = error
                lock.unlock()
            }
        }

        func results() throws -> [CompletedRegionRestoration] {
            lock.lock()
            defer { lock.unlock() }
            if let failure = failures.compactMap({ $0 }).first { throw failure }
            guard completed.allSatisfy({ $0 != nil }) else {
                throw DeformConvError.commandFailed("parallel mosaic restoration was incomplete")
            }
            return completed.compactMap { $0 }
        }
    }

    static func restoreVideo(
        device: MTLDevice,
        inputURL: URL,
        outputURL: URL,
        modelsURL: URL,
        weightsURL: URL
    ) async throws -> SideBySideRestorationResult {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds
        )
        return try await restorePlannedVideo(
            device: device,
            inputURL: inputURL,
            outputURL: outputURL,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            inputInfo: inputInfo,
            plan: plan,
            cropX: 0,
            description: "side-by-side"
        )
    }

    static func restoreEyeVideo(
        device: MTLDevice,
        inputURL: URL,
        eye: SideBySideEye,
        outputURL: URL,
        modelsURL: URL,
        weightsURL: URL
    ) async throws -> SideBySideRestorationResult {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        guard inputInfo.dimensions.width.isMultiple(of: 2) else {
            throw DeformConvError.commandFailed("SBS input width must be even")
        }
        let eyeWidth = inputInfo.dimensions.width / 2
        let plan = try SideBySideVideoPlan(
            width: eyeWidth,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        return try await restorePlannedVideo(
            device: device,
            inputURL: inputURL,
            outputURL: outputURL,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            inputInfo: inputInfo,
            plan: plan,
            cropX: eye == .left ? 0 : eyeWidth,
            description: "\(eye.rawValue) eye"
        )
    }

    static func restoreSingleEyeVideo(
        device: MTLDevice,
        inputURL: URL,
        outputURL: URL,
        modelsURL: URL,
        weightsURL: URL
    ) async throws -> SideBySideRestorationResult {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        return try await restorePlannedVideo(
            device: device,
            inputURL: inputURL,
            outputURL: outputURL,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            inputInfo: inputInfo,
            plan: plan,
            cropX: 0,
            description: "single eye"
        )
    }

    static func restoreSingleEyeVideoWindows(
        device: MTLDevice,
        inputURL: URL,
        windowsDirectoryURL: URL,
        modelsURL: URL,
        weightsURL: URL
    ) async throws -> Int {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        try FileManager.default.createDirectory(
            at: windowsDirectoryURL, withIntermediateDirectories: true
        )
        report(
            "Restartable single-eye plan: \(plan.dimensions.width)×\(plan.dimensions.height), "
                + "\(plan.frameRate.outputFrameCount) frames, "
                + "\(plan.temporalWindowCount) independently encoded windows"
        )
        let decoder = try await FrameDecoder(
            inputURL: inputURL,
            plan: plan,
            sourceDimensions: inputInfo.dimensions,
            cropX: 0
        )
        var completedWindows = 0
        for (windowIndex, outputCount) in plan.temporalWindowFrameCounts.enumerated() {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            let outputURL = windowsDirectoryURL.appendingPathComponent(
                String(format: "window-%05d.mov", windowIndex + 1)
            )
            if await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: outputCount
            ) {
                report(
                    "Window \(windowIndex + 1)/\(plan.temporalWindowCount): "
                        + "encoded output already complete"
                )
                completedWindows += 1
                continue
            }
            if FileManager.default.fileExists(atPath: outputURL.path) {
                let archived = try archiveInterruptedOutput(outputURL)
                report(
                    "Window \(windowIndex + 1)/\(plan.temporalWindowCount): "
                        + "archived incomplete output at \(archived.path)"
                )
            }

            report(
                "Window \(windowIndex + 1)/\(plan.temporalWindowCount): decoding "
                    + "\(outputCount) output frames from frame \(windowStart)"
            )
            var decoded = try (0..<outputCount).map {
                try decoder.copyFrame(outputIndex: windowStart + $0)
            }
            let attachments = decoded.map { CVBufferCopyAttachments($0, .shouldPropagate) }
            while decoded.count < 3 {
                guard let last = decoded.last else { throw DeformConvError.invalidShape }
                decoded.append(last)
            }
            let window = try processWindow(
                device: device,
                plan: plan,
                tiles: plan.tiles,
                decodedFrames: decoded,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: plan.temporalWindowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: nil
            )
            let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
            try await writer.appendCachedFrames(
                cacheURLs: window.cacheURLs,
                attachments: attachments,
                startFrame: 0,
                progressStartFrame: windowStart,
                plan: plan
            )
            try await writer.finish()
            guard await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: outputCount
            ) else {
                throw DeformConvError.commandFailed(
                    "encoded window \(windowIndex + 1) failed validation"
                )
            }
            try FileManager.default.removeItem(at: window.cacheDirectory)
            completedWindows += 1
            report(
                "Window \(windowIndex + 1)/\(plan.temporalWindowCount): "
                    + "encoded, validated, and cache removed"
            )
        }
        return completedWindows
    }

    static func restoreSparseSingleEyeVideoWindows(
        device: MTLDevice,
        inputURL: URL,
        windowsDirectoryURL: URL,
        manifestURL: URL,
        modelsURL: URL,
        weightsURL: URL,
        projection: VRMosaicProjection = .raw
    ) async throws -> Int {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        let manifest = try MosaicRegionManifest.load(from: manifestURL)
        try manifest.validate(for: plan)
        let windowFrameCounts = stride(
            from: 0,
            to: manifest.frameCount,
            by: SideBySideVideoPlan.temporalWindowFrames
        ).map {
            min(SideBySideVideoPlan.temporalWindowFrames, manifest.frameCount - $0)
        }
        try FileManager.default.createDirectory(
            at: windowsDirectoryURL, withIntermediateDirectories: true
        )
        report(
            "Sparse single-eye plan: \(plan.dimensions.width)×\(plan.dimensions.height), "
                + "\(manifest.frameCount) frames, \(manifest.regions.count) regions, "
                + "VR projection \(projection.rawValue)"
        )
        let decoder = try await FrameDecoder(
            inputURL: inputURL,
            plan: plan,
            sourceDimensions: inputInfo.dimensions,
            cropX: 0
        )
        var completedWindows = 0
        var restoredRegionWindows = 0
        var skippedTileWindows = 0
        for (windowIndex, outputCount) in windowFrameCounts.enumerated() {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            let frameRange = windowStart..<(windowStart + outputCount)
            let activeRegions = manifest.regions(intersecting: frameRange)
            let outputURL = windowsDirectoryURL.appendingPathComponent(
                String(format: "window-%05d.mov", windowIndex + 1)
            )
            if await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: outputCount
            ) {
                report(
                    "Window \(windowIndex + 1)/\(windowFrameCounts.count): "
                        + "encoded output already complete"
                )
                completedWindows += 1
                continue
            }
            if FileManager.default.fileExists(atPath: outputURL.path) {
                let archived = try archiveInterruptedOutput(outputURL)
                report(
                    "Window \(windowIndex + 1)/\(windowFrameCounts.count): "
                        + "archived incomplete output at \(archived.path)"
                )
            }

            report(
                "Window \(windowIndex + 1)/\(windowFrameCounts.count): decoding "
                    + "\(outputCount) frames; mosaic regions \(activeRegions.count), "
                    + "model crops \(activeRegions.count)"
            )
            let baseFrames = try (0..<outputCount).map {
                try decoder.copyFrame(outputIndex: windowStart + $0)
            }
            let attachments = baseFrames.map { CVBufferCopyAttachments($0, .shouldPropagate) }
            var modelFrames = baseFrames
            while modelFrames.count < 3 {
                guard let last = modelFrames.last else { throw DeformConvError.invalidShape }
                modelFrames.append(last)
            }
            let cacheVariant = sparseRegionCacheVariant(
                regions: activeRegions, projection: projection
            )
            let samplingMaps = activeRegions.map {
                MosaicCropSamplingMap(
                    region: $0,
                    eyeWidth: plan.dimensions.width,
                    eyeHeight: plan.dimensions.height,
                    projection: projection
                )
            }
            let window = try processRegionWindow(
                device: device,
                plan: plan,
                regions: activeRegions,
                decodedFrames: modelFrames,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: windowFrameCounts.count,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: cacheVariant,
                projection: projection,
                samplingMaps: samplingMaps
            )
            let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
            try await writer.appendRegionCachedFrames(
                cacheURLs: window.cacheURLs,
                attachments: attachments,
                startFrame: 0,
                progressStartFrame: windowStart,
                progressFrameCount: manifest.frameCount,
                plan: plan,
                baseFrames: baseFrames,
                regions: activeRegions,
                projection: projection,
                samplingMaps: samplingMaps
            )
            try await writer.finish()
            guard await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: outputCount
            ) else {
                throw DeformConvError.commandFailed(
                    "encoded sparse window \(windowIndex + 1) failed validation"
                )
            }
            try FileManager.default.removeItem(at: window.cacheDirectory)
            if activeRegions.isEmpty {
                skippedTileWindows += 1
            } else {
                restoredRegionWindows += activeRegions.count
            }
            completedWindows += 1
            report(
                "Window \(windowIndex + 1)/\(windowFrameCounts.count): "
                    + "sparse output encoded, validated, and cache removed"
            )
        }
        report(
            "Sparse restoration completed: \(completedWindows) windows, "
                + "\(restoredRegionWindows) mosaic crops restored, "
                + "\(skippedTileWindows) clean windows bypassed"
        )
        return completedWindows
    }

    private static func validWindowOutput(
        _ url: URL,
        dimensions: VideoDimensions,
        frameCount: Int
    ) async -> Bool {
        guard FileManager.default.fileExists(atPath: url.path), frameCount > 0 else {
            return false
        }
        do {
            let info = try await SideBySideVideoIO.inspect(url: url)
            return info.dimensions == dimensions
                && abs(info.nominalFramesPerSecond - 30) < 0.01
                && abs(info.durationSeconds - Double(frameCount) / 30) < 0.01
        } catch {
            return false
        }
    }

    private static func restorePlannedVideo(
        device: MTLDevice,
        inputURL: URL,
        outputURL: URL,
        modelsURL: URL,
        weightsURL: URL,
        inputInfo: VideoAssetInfo,
        plan: SideBySideVideoPlan,
        cropX: Int,
        description: String
    ) async throws -> SideBySideRestorationResult {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            guard try hasInterruptedWindowCache() else {
                throw DeformConvError.commandFailed("output already exists: \(outputURL.path)")
            }
            let archivedURL = try archiveInterruptedOutput(outputURL)
            report("Archived interrupted output at \(archivedURL.path)")
        }
        report(
            "Restoration plan (\(description)): "
                + "\(plan.dimensions.width)×\(plan.dimensions.height), "
                + "\(plan.frameRate.outputFrameCount) frames, "
                + "\(plan.temporalWindowCount) windows, \(plan.tiles.count) tiles/window"
        )
        if let workDirectory = ProcessInfo.processInfo.environment["JASNA_WORK_DIR"] {
            report("Persistent work directory: \(workDirectory)")
        }
        let decoder = try await FrameDecoder(
            inputURL: inputURL,
            plan: plan,
            sourceDimensions: inputInfo.dimensions,
            cropX: cropX
        )
        let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
        var totalGPU: Double = 0
        var peakCacheBytes = 0
        var windows = 0

        for (windowIndex, outputCount) in plan.temporalWindowFrameCounts.enumerated() {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            report(
                "Window \(windowIndex + 1)/\(plan.temporalWindowCount): decoding "
                    + "\(outputCount) output frames from frame \(windowStart)"
            )
            var decoded = try (0..<outputCount).map {
                try decoder.copyFrame(outputIndex: windowStart + $0)
            }
            let attachments = decoded.map { CVBufferCopyAttachments($0, .shouldPropagate) }
            while decoded.count < 3 {
                guard let last = decoded.last else { throw DeformConvError.invalidShape }
                decoded.append(last)
            }
            let window = try processWindow(
                device: device,
                plan: plan,
                tiles: plan.tiles,
                decodedFrames: decoded,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: plan.temporalWindowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: nil
            )
            defer { try? FileManager.default.removeItem(at: window.cacheDirectory) }
            try await writer.appendCachedFrames(
                cacheURLs: window.cacheURLs,
                attachments: attachments,
                startFrame: windowStart,
                plan: plan
            )
            try FileManager.default.removeItem(at: window.cacheDirectory)
            report("Window \(windowIndex + 1)/\(plan.temporalWindowCount): encoded and cache removed")
            totalGPU += window.gpuMilliseconds
            peakCacheBytes = max(peakCacheBytes, window.cacheBytes)
            windows += 1
        }
        try await writer.finish()
        report("Restoration writer completed \(plan.frameRate.outputFrameCount) frames")

        let outputInfo = try await SideBySideVideoIO.inspect(url: outputURL)
        guard outputInfo.dimensions == plan.dimensions,
              abs(outputInfo.nominalFramesPerSecond - 30) < 0.01
        else {
            throw DeformConvError.commandFailed("restored video metadata validation failed")
        }
        return SideBySideRestorationResult(
            input: inputInfo,
            output: outputInfo,
            tileCount: plan.tiles.count,
            frameCount: plan.frameRate.outputFrameCount,
            windowCount: windows,
            gpuMilliseconds: totalGPU,
            cacheBytes: peakCacheBytes
        )
    }

    static func diagnoseTile(
        device: MTLDevice,
        inputURL: URL,
        tileNumber: Int,
        modelsURL: URL,
        weightsURL: URL
    ) async throws {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds
        )
        guard plan.tiles.indices.contains(tileNumber - 1),
              let outputCount = plan.temporalWindowFrameCounts.first
        else { throw DeformConvError.invalidShape }
        let tile = plan.tiles[tileNumber - 1]
        report(
            "Diagnosing tile \(tileNumber)/\(plan.tiles.count) at eye \(tile.eyeIndex), "
                + "x \(tile.x), y \(tile.y) with \(outputCount) frames"
        )
        let decoder = try await FrameDecoder(inputURL: inputURL, plan: plan)
        var decoded = try (0..<outputCount).map { try decoder.copyFrame(outputIndex: $0) }
        while decoded.count < 3 {
            guard let last = decoded.last else { throw DeformConvError.invalidShape }
            decoded.append(last)
        }
        let tileFrames = try decoded.map {
            try TilePixelPipeline.extractPlanarRGB(from: $0, tile: tile)
        }
        let result = try autoreleasepool {
            try restoreTileWithFallback(
                device: device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                inputFrames: tileFrames,
                context: "Diagnostic tile \(tileNumber)/\(plan.tiles.count)"
            )
        }
        report(
            "Diagnostic tile \(tileNumber)/\(plan.tiles.count): PASS, "
                + "\(result.frames.count) frames, GPU "
                + "\(String(format: "%.3f", result.gpuMilliseconds)) ms"
        )
    }

    private struct WindowResult {
        let cacheDirectory: URL
        let cacheURLs: [URL]
        let gpuMilliseconds: Double
        let cacheBytes: Int
    }

    private struct ResumableWindowCache {
        let directory: URL
        let urls: [URL]
        let completedTiles: Int
    }

    private static func processWindow(
        device: MTLDevice,
        plan: SideBySideVideoPlan,
        tiles: [VideoTile],
        decodedFrames: [CVPixelBuffer],
        outputCount: Int,
        windowIndex: Int,
        windowCount: Int,
        modelsURL: URL,
        weightsURL: URL,
        cacheVariant: String?
    ) throws -> WindowResult {
        let cacheBytes = tiles.count * outputCount * tileBytes
        let configuredWorkPath = ProcessInfo.processInfo.environment["JASNA_WORK_DIR"]
        let temporaryURL = configuredWorkPath.map {
            URL(fileURLWithPath: $0, isDirectory: true)
        } ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(
            at: temporaryURL, withIntermediateDirectories: true
        )
        let available = try temporaryURL.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
        if let available, Int64(cacheBytes) + 1_073_741_824 > available {
            throw DeformConvError.commandFailed(
                "insufficient temporary storage for restored tiles: need at least "
                    + "\(cacheBytes + 1_073_741_824) bytes, available \(available)"
            )
        }
        let resumed = configuredWorkPath == nil ? nil : try resumableWindowCache(
            in: temporaryURL,
            windowIndex: windowIndex,
            outputCount: outputCount,
            tileCount: tiles.count,
            cacheVariant: cacheVariant
        )
        let prefix = cacheDirectoryPrefix(windowIndex: windowIndex, cacheVariant: cacheVariant)
        let directory = resumed?.directory ?? temporaryURL.appendingPathComponent(
            "\(prefix)\(UUID().uuidString)", isDirectory: true
        )
        if resumed == nil {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: false
            )
        }
        do {
            let urls = resumed?.urls ?? (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            var handles = try urls.map { url -> FileHandle in
                if resumed == nil {
                    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                        throw DeformConvError.commandFailed("failed creating tile cache: \(url.path)")
                    }
                }
                let handle = try FileHandle(forUpdating: url)
                if let resumed {
                    let safeOffset = UInt64(resumed.completedTiles * tileBytes)
                    try handle.truncate(atOffset: safeOffset)
                    try handle.seek(toOffset: safeOffset)
                }
                return handle
            }
            defer { for handle in handles { try? handle.close() } }
            var gpuMilliseconds: Double = 0
            let progressInterval = max(1, tiles.count / 20)
            report(
                "Window \(windowIndex + 1)/\(windowCount): restoring \(tiles.count) tiles; "
                    + "cache \(String(format: "%.2f", Double(cacheBytes) / 1_073_741_824)) GiB"
            )
            let completedTiles = resumed?.completedTiles ?? 0
            if completedTiles == tiles.count, !tiles.isEmpty {
                report(
                    "Window \(windowIndex + 1)/\(windowCount): all tiles recovered from "
                        + directory.path
                )
            } else if completedTiles > 0 {
                report(
                    "Window \(windowIndex + 1)/\(windowCount): resuming at tile "
                        + "\(completedTiles + 1)/\(tiles.count) from \(directory.path)"
                )
            }
            for tileIndex in completedTiles..<tiles.count {
                let tile = tiles[tileIndex]
                let tileGPU = try autoreleasepool { () throws -> Double in
                    let tileFrames = try decodedFrames.map {
                        try TilePixelPipeline.extractPlanarRGB(from: $0, tile: tile)
                    }
                    let context = "Window \(windowIndex + 1)/\(windowCount): tile "
                        + "\(tileIndex + 1)/\(tiles.count) at eye \(tile.eyeIndex), "
                        + "x \(tile.x), y \(tile.y)"
                    let restored = try restoreTileWithFallback(
                        device: device,
                        modelsURL: modelsURL,
                        weightsURL: weightsURL,
                        inputFrames: tileFrames,
                        context: context
                    )
                    for frame in 0..<outputCount {
                        try restored.frames[frame].withUnsafeBytes { bytes in
                            try handles[frame].write(contentsOf: Data(bytes))
                        }
                    }
                    return restored.gpuMilliseconds
                }
                gpuMilliseconds += tileGPU
                let completed = tileIndex + 1
                if completed.isMultiple(of: 8) || completed == tiles.count {
                    for handle in handles { try handle.synchronize() }
                    try Data("\(completed)\n".utf8).write(
                        to: directory.appendingPathComponent("completed-tiles.txt"),
                        options: .atomic
                    )
                }
                if tileIndex == 0
                    || tileIndex + 1 == tiles.count
                    || (tileIndex + 1).isMultiple(of: progressInterval) {
                    report(
                        "Window \(windowIndex + 1)/\(windowCount): tile "
                            + "\(tileIndex + 1)/\(tiles.count), GPU "
                            + "\(String(format: "%.3f", gpuMilliseconds)) ms"
                    )
                }
            }
            for handle in handles { try handle.close() }
            handles.removeAll()
            return WindowResult(
                cacheDirectory: directory,
                cacheURLs: urls,
                gpuMilliseconds: gpuMilliseconds,
                cacheBytes: cacheBytes
            )
        } catch {
            if configuredWorkPath == nil {
                try? FileManager.default.removeItem(at: directory)
            } else {
                report("Preserving failed window cache at \(directory.path)")
            }
            throw error
        }
    }

    private static func processRegionWindow(
        device: MTLDevice,
        plan: SideBySideVideoPlan,
        regions: [MosaicRegion],
        decodedFrames: [CVPixelBuffer],
        outputCount: Int,
        windowIndex: Int,
        windowCount: Int,
        modelsURL: URL,
        weightsURL: URL,
        cacheVariant: String,
        projection: VRMosaicProjection,
        samplingMaps: [MosaicCropSamplingMap]
    ) throws -> WindowResult {
        guard samplingMaps.count == regions.count else {
            throw DeformConvError.invalidShape
        }
        let cacheBytes = regions.count * outputCount * tileBytes
        let configuredWorkPath = ProcessInfo.processInfo.environment["JASNA_WORK_DIR"]
        let workURL = configuredWorkPath.map {
            URL(fileURLWithPath: $0, isDirectory: true)
        } ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: workURL, withIntermediateDirectories: true)
        let resumed = configuredWorkPath == nil ? nil : try resumableWindowCache(
            in: workURL,
            windowIndex: windowIndex,
            outputCount: outputCount,
            tileCount: regions.count,
            cacheVariant: cacheVariant
        )
        let prefix = cacheDirectoryPrefix(windowIndex: windowIndex, cacheVariant: cacheVariant)
        let directory = resumed?.directory ?? workURL.appendingPathComponent(
            "\(prefix)\(UUID().uuidString)", isDirectory: true
        )
        if resumed == nil {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        }
        do {
            let urls = resumed?.urls ?? (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            var handles = try urls.map { url -> FileHandle in
                if resumed == nil {
                    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                        throw DeformConvError.commandFailed("failed creating crop cache: \(url.path)")
                    }
                }
                let handle = try FileHandle(forUpdating: url)
                if let resumed {
                    let safeOffset = UInt64(resumed.completedTiles * tileBytes)
                    try handle.truncate(atOffset: safeOffset)
                    try handle.seek(toOffset: safeOffset)
                }
                return handle
            }
            defer { for handle in handles { try? handle.close() } }
            let completedRegions = resumed?.completedTiles ?? 0
            var gpuMilliseconds: Double = 0
            let configuredCheckpointInterval = Int(
                ProcessInfo.processInfo.environment["JASNA_REGION_CHECKPOINT_INTERVAL"] ?? ""
            )
            let checkpointInterval = configuredWorkPath == nil
                ? max(1, regions.count)
                : max(1, configuredCheckpointInterval ?? 5)
            // The retained 30-frame graph makes concurrent graph construction
            // unnecessary after the first crop. Building two first-use Metal ML
            // graphs concurrently is also unstable in the macOS 27 beta runtime.
            let regionConcurrency = 1
            report(
                "Window \(windowIndex + 1)/\(windowCount): restoring "
                    + "\(regions.count) tight mosaic crops; cache "
                    + "\(String(format: "%.2f", Double(cacheBytes) / 1_073_741_824)) GiB; "
                    + "concurrency \(regionConcurrency)"
            )
            var nextRegion = completedRegions
            while nextRegion < regions.count {
                let batchEnd = min(regions.count, nextRegion + regionConcurrency)
                let work = try (nextRegion..<batchEnd).map { regionIndex in
                    let windowStartFrame = windowIndex * SideBySideVideoPlan.temporalWindowFrames
                    let region = regions[regionIndex]
                    let localStart = max(0, region.startFrame - windowStartFrame)
                    let localEnd = min(outputCount, region.endFrame - windowStartFrame)
                    guard localStart < localEnd else {
                        throw DeformConvError.commandFailed(
                            "mosaic crop does not intersect its assigned window"
                        )
                    }
                    let samplingMap = samplingMaps[regionIndex]
                    let activeFrames = decodedFrames[localStart..<localEnd]
                    var cropFrames = try activeFrames.map {
                        try samplingMap.extractPlanarRGB(from: $0)
                    }
                    let activeFrameCount = cropFrames.count
                    while cropFrames.count < 3 {
                        guard let last = cropFrames.last else {
                            throw DeformConvError.invalidShape
                        }
                        cropFrames.append(last)
                    }
                    let context = "Window \(windowIndex + 1)/\(windowCount): mosaic crop "
                        + "\(regionIndex + 1)/\(regions.count), x \(region.x), y \(region.y), "
                        + "size \(region.width)×\(region.height), frames "
                        + "\(region.startFrame)..<\(region.endFrame)"
                    return PreparedRegionRestoration(
                        regionIndex: regionIndex,
                        localStart: localStart,
                        activeFrameCount: activeFrameCount,
                        inputFrames: cropFrames,
                        context: context
                    )
                }
                let batch = RegionRestorationBatch(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    work: work
                )
                if work.count == 1 {
                    batch.execute(0)
                } else {
                    DispatchQueue.concurrentPerform(iterations: work.count) { index in
                        batch.execute(index)
                    }
                }
                for restored in try batch.results() {
                    let prepared = restored.prepared
                    for frame in 0..<outputCount {
                        if frame >= prepared.localStart
                            && frame < prepared.localStart + prepared.activeFrameCount
                        {
                            let values = restored.frames[
                                min(
                                    frame - prepared.localStart,
                                    prepared.activeFrameCount - 1
                                )
                            ]
                            try values.withUnsafeBytes { bytes in
                                try handles[frame].write(contentsOf: Data(bytes))
                            }
                        } else {
                            try handles[frame].seek(
                                toOffset: UInt64((prepared.regionIndex + 1) * tileBytes)
                            )
                        }
                    }
                    gpuMilliseconds += restored.gpuMilliseconds
                    let completedCount = prepared.regionIndex + 1
                    let shouldCheckpoint = completedCount == regions.count
                        || completedCount.isMultiple(of: checkpointInterval)
                    if shouldCheckpoint {
                        let completedBytes = UInt64(completedCount * tileBytes)
                        for handle in handles {
                            try handle.truncate(atOffset: completedBytes)
                            try handle.synchronize()
                        }
                        try Data("\(completedCount)\n".utf8).write(
                            to: directory.appendingPathComponent("completed-tiles.txt"),
                            options: .atomic
                        )
                    }
                    report(
                        "Window \(windowIndex + 1)/\(windowCount): mosaic crop "
                            + "\(completedCount)/\(regions.count), GPU "
                            + "\(String(format: "%.3f", gpuMilliseconds)) ms cumulative, "
                            + "crop wall \(String(format: "%.3f", restored.wallMilliseconds)) ms"
                    )
                }
                nextRegion = batchEnd
            }
            for handle in handles { try handle.close() }
            handles.removeAll()
            return WindowResult(
                cacheDirectory: directory,
                cacheURLs: urls,
                gpuMilliseconds: gpuMilliseconds,
                cacheBytes: cacheBytes
            )
        } catch {
            if configuredWorkPath == nil {
                try? FileManager.default.removeItem(at: directory)
            } else {
                report("Preserving failed crop cache at \(directory.path)")
            }
            throw error
        }
    }

    private static func restoreTileWithFallback(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]],
        context: String
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        do {
            return try restoreTileFrames(
                device: device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                inputFrames: inputFrames,
                maximumFramesPerChunk: inputFrames.count
            )
        } catch {
            guard let modelError = error as? DeformConvError,
                  modelError.isRecoverableNumericalFailure
            else { throw error }
            report(
                "\(context) failed its full temporal window (\(error)); "
                    + "retrying shorter recurrence chunks"
            )
            var lastError: Error = error
            for chunkSize in [10, 5, 3] where chunkSize < inputFrames.count {
                do {
                    let recovered = try restoreTileFrames(
                        device: device,
                        modelsURL: modelsURL,
                        weightsURL: weightsURL,
                        inputFrames: inputFrames,
                        maximumFramesPerChunk: chunkSize
                    )
                    report(
                        "\(context) recovered with at most \(chunkSize) frames "
                            + "per recurrence chunk"
                    )
                    return recovered
                } catch {
                    guard let modelError = error as? DeformConvError,
                          modelError.isRecoverableNumericalFailure
                    else { throw error }
                    lastError = error
                    report("\(context) also failed with \(chunkSize)-frame chunks (\(error))")
                }
            }
            do {
                let recovered = try restoreTileFramesIndependently(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    inputFrames: inputFrames
                )
                report("\(context) recovered with independent zero-motion frame triplets")
                return recovered
            } catch {
                guard let modelError = error as? DeformConvError,
                      modelError.isRecoverableNumericalFailure
                else { throw error }
                lastError = error
                report("\(context) also failed independent frame recovery (\(error))")
            }
            guard inputFrames.joined().allSatisfy({ Float($0).isFinite }) else {
                throw DeformConvError.commandFailed(
                    "\(context) has non-finite input pixels after model failure: \(lastError)"
                )
            }
            report(
                "WARNING: \(context) is using finite input-pixel passthrough because all "
                    + "Metal recurrence recovery modes failed (\(lastError))"
            )
            return (inputFrames, 0)
        }
    }

    private static func restoreTileFramesIndependently(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]]
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        var frames = [[Float16]]()
        frames.reserveCapacity(inputFrames.count)
        var gpuMilliseconds: Double = 0
        for frame in inputFrames {
            let recovered = try autoreleasepool {
                try verifyFusedFourPassRecurrence(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    backwardFlows: [],
                    forwardFlows: [],
                    inputFrames: [frame, frame, frame],
                    stagedBranchFrames: [],
                    stagedRestoredFrames: [],
                    warmupCount: 0,
                    measurementCount: 1,
                    collectDiagnostics: false
                )
            }
            guard recovered.restoredFrames.count == 3 else {
                throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
            }
            frames.append(recovered.restoredFrames[1])
            gpuMilliseconds += recovered.statistics.median
        }
        return (frames, gpuMilliseconds)
    }

    private static func restoreTileFrames(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]],
        maximumFramesPerChunk: Int
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        let ranges = try temporalChunkRanges(
            frameCount: inputFrames.count,
            maximumFramesPerChunk: maximumFramesPerChunk
        )
        var frames = [[Float16]]()
        frames.reserveCapacity(inputFrames.count)
        var gpuMilliseconds: Double = 0
        for range in ranges {
            let result = try autoreleasepool {
                try verifyFusedFourPassRecurrence(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    backwardFlows: [],
                    forwardFlows: [],
                    inputFrames: Array(inputFrames[range]),
                    stagedBranchFrames: [],
                    stagedRestoredFrames: [],
                    warmupCount: 0,
                    measurementCount: 1,
                    collectDiagnostics: false
                )
            }
            guard result.restoredFrames.count == range.count else {
                throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
            }
            frames.append(contentsOf: result.restoredFrames)
            gpuMilliseconds += result.statistics.median
        }
        return (frames, gpuMilliseconds)
    }

    static func temporalChunkRanges(
        frameCount: Int, maximumFramesPerChunk: Int
    ) throws -> [Range<Int>] {
        guard frameCount >= 3, maximumFramesPerChunk >= 3 else {
            throw DeformConvError.invalidShape
        }
        let requestedChunks = (frameCount + maximumFramesPerChunk - 1)
            / maximumFramesPerChunk
        let chunkCount = max(1, min(requestedChunks, frameCount / 3))
        let baseSize = frameCount / chunkCount
        let largerChunkCount = frameCount % chunkCount
        var start = 0
        return (0..<chunkCount).map { index in
            let size = baseSize + (index < largerChunkCount ? 1 : 0)
            defer { start += size }
            return start..<(start + size)
        }
    }

    private static func hasInterruptedWindowCache() throws -> Bool {
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

    private static func archiveInterruptedOutput(_ outputURL: URL) throws -> URL {
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

    private static func resumableWindowCache(
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

    private static func cacheDirectoryPrefix(
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
        projection: VRMosaicProjection = .raw
    ) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in projection.rawValue.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        for region in regions {
            let values = [
                region.startFrame, region.endFrame,
                region.x, region.y, region.width, region.height,
                region.effectiveBlendX, region.effectiveBlendY,
                region.effectiveBlendWidth, region.effectiveBlendHeight,
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
        return String(format: "crop-v3-%@-%016llx", projection.rawValue, hash)
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

    private static func report(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        FileHandle.standardOutput.write(Data("[\(timestamp)] \(message)\n".utf8))
    }

    private final class FrameDecoder {
        private let reader: AVAssetReader
        private let output: AVAssetReaderTrackOutput
        private let dimensions: VideoDimensions
        private let cropX: Int
        private var previous: CMSampleBuffer?
        private var next: CMSampleBuffer?

        init(
            inputURL: URL,
            plan: SideBySideVideoPlan,
            sourceDimensions: VideoDimensions? = nil,
            cropX: Int = 0
        ) async throws {
            let asset = AVURLAsset(url: inputURL)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw DeformConvError.commandFailed("video has no video track")
            }
            let naturalSize = try await track.load(.naturalSize)
            let expectedSource = sourceDimensions ?? plan.dimensions
            guard Int(abs(naturalSize.width).rounded()) == expectedSource.width,
                  Int(abs(naturalSize.height).rounded()) == expectedSource.height,
                  cropX >= 0,
                  cropX + plan.dimensions.width <= expectedSource.width,
                  plan.dimensions.height == expectedSource.height
            else {
                throw DeformConvError.commandFailed(
                    "video dimensions/crop do not match the restoration plan; "
                        + "rotated tracks are not supported yet"
                )
            }
            reader = try AVAssetReader(asset: asset)
            output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                ]
            )
            dimensions = plan.dimensions
            self.cropX = cropX
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw DeformConvError.commandFailed("video reader rejected BGRA output")
            }
            reader.add(output)
            guard reader.startReading() else {
                throw reader.error ?? DeformConvError.commandFailed("video reader failed to start")
            }
            next = output.copyNextSampleBuffer()
        }

        deinit { reader.cancelReading() }

        func copyFrame(outputIndex: Int) throws -> CVPixelBuffer {
            let target = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(candidate), target) < 0 {
                previous = candidate
                next = output.copyNextSampleBuffer()
            }
            guard let sample = Self.closest(previous: previous, next: next, to: target),
                  let source = CMSampleBufferGetImageBuffer(sample)
            else {
                if let error = reader.error as NSError?,
                   error.domain == AVFoundationErrorDomain,
                   error.code == AVError.Code.decoderNotFound.rawValue {
                    throw DeformConvError.commandFailed(
                        "Apple VideoToolbox cannot decode this source. For 8K Main 10 at "
                            + "59.94 fps, first run script/prepare_8k_30fps.sh, then restore "
                            + "the prepared 30 fps file. Underlying error: \(error)"
                    )
                }
                throw reader.error
                    ?? DeformConvError.commandFailed("decoder ended before frame \(outputIndex)")
            }
            return try Self.copyBGRA(source, dimensions: dimensions, cropX: cropX)
        }

        private static func closest(
            previous: CMSampleBuffer?, next: CMSampleBuffer?, to target: CMTime
        ) -> CMSampleBuffer? {
            guard let previous else { return next }
            guard let next else { return previous }
            let priorDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(previous), target)
            ))
            let nextDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(next), target)
            ))
            return priorDistance <= nextDistance ? previous : next
        }

        private static func copyBGRA(
            _ source: CVPixelBuffer, dimensions: VideoDimensions, cropX: Int
        ) throws -> CVPixelBuffer {
            var optionalDestination: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                nil,
                dimensions.width,
                dimensions.height,
                kCVPixelFormatType_32BGRA,
                [
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                ] as CFDictionary,
                &optionalDestination
            )
            guard status == kCVReturnSuccess, let destination = optionalDestination else {
                throw DeformConvError.commandFailed("failed allocating decoded frame copy")
            }
            CVPixelBufferLockBaseAddress(source, .readOnly)
            CVPixelBufferLockBaseAddress(destination, [])
            defer {
                CVPixelBufferUnlockBaseAddress(destination, [])
                CVPixelBufferUnlockBaseAddress(source, .readOnly)
            }
            guard let sourceBase = CVPixelBufferGetBaseAddress(source),
                  let destinationBase = CVPixelBufferGetBaseAddress(destination)
            else { throw DeformConvError.commandFailed("decoded frame is not CPU accessible") }
            for row in 0..<dimensions.height {
                memcpy(
                    destinationBase.advanced(by: row * CVPixelBufferGetBytesPerRow(destination)),
                    sourceBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(source) + cropX * 4
                    ),
                    dimensions.width * 4
                )
            }
            CVBufferPropagateAttachments(source, destination)
            return destination
        }
    }

    private final class RestoredFrameWriter {
        private struct CompositedRegionFrame {
            let localFrame: Int
            let pixelBuffer: CVPixelBuffer
            let wallSeconds: Double
        }

        private final class RegionFrameCompositeBatch: @unchecked Sendable {
            private let localFrames: [Int]
            private let cacheURLs: [URL]
            private let baseFrames: [CVPixelBuffer]
            private let outputBuffers: [CVPixelBuffer]
            private let dimensions: VideoDimensions
            private let regions: [MosaicRegion]
            private let projection: VRMosaicProjection
            private let samplingMaps: [MosaicCropSamplingMap]
            private let progressStartFrame: Int
            private let metalCompositor: MetalMosaicCompositor?
            private let lock = NSLock()
            private var completed: [CompositedRegionFrame?]
            private var failures: [(any Error)?]

            init(
                localFrames: [Int],
                cacheURLs: [URL],
                baseFrames: [CVPixelBuffer],
                outputBuffers: [CVPixelBuffer],
                dimensions: VideoDimensions,
                regions: [MosaicRegion],
                projection: VRMosaicProjection,
                samplingMaps: [MosaicCropSamplingMap],
                progressStartFrame: Int,
                metalCompositor: MetalMosaicCompositor?
            ) {
                self.localFrames = localFrames
                self.cacheURLs = cacheURLs
                self.baseFrames = baseFrames
                self.outputBuffers = outputBuffers
                self.dimensions = dimensions
                self.regions = regions
                self.projection = projection
                self.samplingMaps = samplingMaps
                self.progressStartFrame = progressStartFrame
                self.metalCompositor = metalCompositor
                completed = [CompositedRegionFrame?](repeating: nil, count: localFrames.count)
                failures = [(any Error)?](repeating: nil, count: localFrames.count)
            }

            func execute(_ index: Int) {
                do {
                    let started = Date()
                    let localFrame = localFrames[index]
                    let cache = try FileHandle(forReadingFrom: cacheURLs[index])
                    defer { try? cache.close() }
                    var compositeInputs = [MetalMosaicCompositeInput]()
                    compositeInputs.reserveCapacity(regions.count)
                    for (regionIndex, region) in regions.enumerated() {
                        let absoluteFrame = progressStartFrame + localFrame
                        guard region.frameRange.contains(absoluteFrame) else {
                            try cache.seek(toOffset: UInt64((regionIndex + 1) * tileBytes))
                            continue
                        }
                        guard let data = try cache.read(upToCount: tileBytes),
                              data.count == tileBytes
                        else {
                            throw DeformConvError.commandFailed(
                                "restored mosaic-crop cache is truncated"
                            )
                        }
                        var values = [Float16](repeating: 0, count: tileElements)
                        _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                        let original = projection == .fisheye
                            ? try samplingMaps[regionIndex].extractPlanarRGB(
                                from: baseFrames[index]
                            )
                            : nil
                        compositeInputs.append(
                            MetalMosaicCompositeInput(
                                region: region,
                                restored: values,
                                original: original ?? [],
                                samples: samplingMaps[regionIndex].compositeSamples
                            )
                        )
                    }
                    if projection == .fisheye, let metalCompositor {
                        var usedMetal = false
                        do {
                            try metalCompositor.composite(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: outputBuffers[index],
                                dimensions: dimensions,
                                inputs: compositeInputs
                            )
                            usedMetal = true
                        } catch {
                            report(
                                "WARNING: Metal mosaic compositing failed; using CPU for frame "
                                    + "\(progressStartFrame + localFrame + 1) (\(error))"
                            )
                            try Self.compositeOnCPU(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: outputBuffers[index],
                                dimensions: dimensions,
                                inputs: compositeInputs,
                                projection: projection
                            )
                        }
                        if usedMetal,
                           localFrame == 0,
                           ProcessInfo.processInfo.environment[
                            "JASNA_VERIFY_METAL_COMPOSITOR"
                           ] == "1"
                        {
                            let cpuReference = try Self.makePixelBuffer(dimensions: dimensions)
                            try Self.compositeOnCPU(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: cpuReference,
                                dimensions: dimensions,
                                inputs: compositeInputs,
                                projection: projection
                            )
                            let difference = try Self.pixelDifference(
                                outputBuffers[index], cpuReference, dimensions: dimensions
                            )
                            report(
                                "Metal compositor raw check: max byte error "
                                    + "\(difference.maximum), differing bytes "
                                    + "\(difference.differing)/\(difference.total)"
                            )
                        }
                    } else {
                        try Self.compositeOnCPU(
                            basePixelBuffer: baseFrames[index],
                            outputPixelBuffer: outputBuffers[index],
                            dimensions: dimensions,
                            inputs: compositeInputs,
                            projection: projection
                        )
                    }
                    let result = CompositedRegionFrame(
                        localFrame: localFrame,
                        pixelBuffer: outputBuffers[index],
                        wallSeconds: Date().timeIntervalSince(started)
                    )
                    lock.lock()
                    completed[index] = result
                    lock.unlock()
                } catch {
                    lock.lock()
                    failures[index] = error
                    lock.unlock()
                }
            }

            private static func compositeOnCPU(
                basePixelBuffer: CVPixelBuffer,
                outputPixelBuffer: CVPixelBuffer,
                dimensions: VideoDimensions,
                inputs: [MetalMosaicCompositeInput],
                projection: VRMosaicProjection
            ) throws {
                var accumulator = try MosaicRegionFrameAccumulator(
                    basePixelBuffer: basePixelBuffer, dimensions: dimensions
                )
                for input in inputs {
                    try accumulator.composite(
                        region: input.region,
                        planarRGB: input.restored,
                        originalPlanarRGB: projection == .fisheye ? input.original : nil,
                        projection: projection
                    )
                }
                try accumulator.writeBGRA(to: outputPixelBuffer)
            }

            private static func makePixelBuffer(
                dimensions: VideoDimensions
            ) throws -> CVPixelBuffer {
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    nil,
                    dimensions.width,
                    dimensions.height,
                    kCVPixelFormatType_32BGRA,
                    [
                        kCVPixelBufferMetalCompatibilityKey as String: true,
                        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                    ] as CFDictionary,
                    &optionalBuffer
                )
                guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
                    throw DeformConvError.commandFailed(
                        "failed allocating Metal compositor verification frame"
                    )
                }
                return buffer
            }

            private static func pixelDifference(
                _ lhs: CVPixelBuffer,
                _ rhs: CVPixelBuffer,
                dimensions: VideoDimensions
            ) throws -> (maximum: Int, differing: Int, total: Int) {
                CVPixelBufferLockBaseAddress(lhs, .readOnly)
                CVPixelBufferLockBaseAddress(rhs, .readOnly)
                defer {
                    CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
                    CVPixelBufferUnlockBaseAddress(lhs, .readOnly)
                }
                guard let lhsBase = CVPixelBufferGetBaseAddress(lhs),
                      let rhsBase = CVPixelBufferGetBaseAddress(rhs)
                else {
                    throw DeformConvError.commandFailed(
                        "Metal compositor verification frame is not CPU accessible"
                    )
                }
                let packedRowBytes = dimensions.width * 4
                var maximum = 0
                var differing = 0
                for row in 0..<dimensions.height {
                    let lhsRow = lhsBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(lhs)
                    ).assumingMemoryBound(to: UInt8.self)
                    let rhsRow = rhsBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(rhs)
                    ).assumingMemoryBound(to: UInt8.self)
                    for column in 0..<packedRowBytes {
                        let difference = abs(Int(lhsRow[column]) - Int(rhsRow[column]))
                        maximum = max(maximum, difference)
                        if difference != 0 { differing += 1 }
                    }
                }
                return (maximum, differing, packedRowBytes * dimensions.height)
            }

            func results() throws -> [CompositedRegionFrame] {
                lock.lock()
                defer { lock.unlock() }
                if let failure = failures.compactMap({ $0 }).first { throw failure }
                guard completed.allSatisfy({ $0 != nil }) else {
                    throw DeformConvError.commandFailed(
                        "parallel mosaic compositing was incomplete"
                    )
                }
                return completed.compactMap { $0 }.sorted { $0.localFrame < $1.localFrame }
            }
        }

        private let writer: AVAssetWriter
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor
        private let metalCompositor: MetalMosaicCompositor?

        init(device: MTLDevice, outputURL: URL, plan: SideBySideVideoPlan) throws {
            metalCompositor = ProcessInfo.processInfo.environment["JASNA_METAL_COMPOSITOR"] == "0"
                ? nil : try? MetalMosaicCompositor(device: device)
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
            let bitRate = min(160_000_000, max(8_000_000, plan.dimensions.pixelCount * 5 / 2))
            input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: plan.dimensions.width,
                    AVVideoHeightKey: plan.dimensions.height,
                    AVVideoCompressionPropertiesKey: [
                        AVVideoAverageBitRateKey: bitRate,
                        AVVideoExpectedSourceFrameRateKey: 30,
                        AVVideoMaxKeyFrameIntervalKey: 60,
                        AVVideoAllowFrameReorderingKey: false,
                    ],
                ]
            )
            adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: plan.dimensions.width,
                    kCVPixelBufferHeightKey as String: plan.dimensions.height,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                ]
            )
            guard writer.canAdd(input) else {
                throw DeformConvError.commandFailed("video writer rejected restored frames")
            }
            writer.add(input)
            guard writer.startWriting() else {
                throw writer.error ?? DeformConvError.commandFailed("video writer failed to start")
            }
            writer.startSession(atSourceTime: .zero)
        }

        func appendCachedFrames(
            cacheURLs: [URL],
            attachments: [CFDictionary?],
            startFrame: Int,
            progressStartFrame: Int? = nil,
            plan: SideBySideVideoPlan,
            tiles: [VideoTile]? = nil,
            baseFrames: [CVPixelBuffer]? = nil,
            regions: [MosaicRegion] = []
        ) async throws {
            let compositeTiles = tiles ?? plan.tiles
            guard baseFrames == nil || baseFrames?.count == cacheURLs.count else {
                throw DeformConvError.invalidShape
            }
            report(
                "Compositing and encoding \(cacheURLs.count) cached frame(s) "
                    + "starting at output frame \(startFrame)"
            )
            for localFrame in cacheURLs.indices {
                let frameStarted = Date()
                let progressFrame = (progressStartFrame ?? startFrame) + localFrame
                report(
                    "Compositing output frame \(progressFrame + 1)/"
                        + "\(plan.frameRate.outputFrameCount)"
                )
                let cache = try FileHandle(forReadingFrom: cacheURLs[localFrame])
                var restoredTiles = [(VideoTile, [Float16])]()
                restoredTiles.reserveCapacity(compositeTiles.count)
                for tile in compositeTiles {
                    guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                        try? cache.close()
                        throw DeformConvError.commandFailed("restored tile cache is truncated")
                    }
                    var values = [Float16](repeating: 0, count: tileElements)
                    _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                    restoredTiles.append((tile, values))
                }
                try cache.close()
                while !input.isReadyForMoreMediaData {
                    if writer.status == .failed {
                        throw writer.error ?? DeformConvError.commandFailed("video writer failed")
                    }
                    try await Task.sleep(for: .milliseconds(1))
                }
                guard let pool = adaptor.pixelBufferPool else {
                    throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
                }
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer)
                guard status == kCVReturnSuccess, let pixelBuffer = optionalBuffer else {
                    throw DeformConvError.commandFailed("failed allocating restored output frame")
                }
                if let frameAttachments = attachments[localFrame] {
                    CVBufferSetAttachments(pixelBuffer, frameAttachments, .shouldPropagate)
                }
                if let baseFrames {
                    var accumulator = try SparseTileFrameAccumulator(
                        basePixelBuffer: baseFrames[localFrame],
                        dimensions: plan.dimensions
                    )
                    for (tile, values) in restoredTiles {
                        try accumulator.accumulate(tile: tile, planarRGB: values)
                    }
                    try accumulator.writeBGRA(to: pixelBuffer, regions: regions)
                } else {
                    var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                    for (tile, values) in restoredTiles {
                        try accumulator.accumulate(tile: tile, planarRGB: values)
                    }
                    try accumulator.writeBGRA(to: pixelBuffer)
                }
                let outputFrame = startFrame + localFrame
                let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                    throw writer.error
                        ?? DeformConvError.commandFailed("failed encoding frame \(outputFrame)")
                }
                report(
                    "Queued output frame \(progressFrame + 1)/"
                        + "\(plan.frameRate.outputFrameCount) in "
                        + "\(String(format: "%.3f", Date().timeIntervalSince(frameStarted))) s"
                )
            }
        }

        func appendRegionCachedFrames(
            cacheURLs: [URL],
            attachments: [CFDictionary?],
            startFrame: Int,
            progressStartFrame: Int,
            progressFrameCount: Int? = nil,
            plan: SideBySideVideoPlan,
            baseFrames: [CVPixelBuffer],
            regions: [MosaicRegion],
            projection: VRMosaicProjection = .raw,
            samplingMaps: [MosaicCropSamplingMap]
        ) async throws {
            guard cacheURLs.count == baseFrames.count,
                  attachments.count == cacheURLs.count,
                  samplingMaps.count == regions.count
            else { throw DeformConvError.invalidShape }
            report(
                "Compositing \(regions.count) restored mosaic crops into "
                    + "\(cacheURLs.count) frame(s)"
            )
            let configuredConcurrency = Int(
                ProcessInfo.processInfo.environment["JASNA_COMPOSITE_CONCURRENCY"] ?? ""
            )
            let automaticConcurrency = ProcessInfo.processInfo.physicalMemory
                >= UInt64(16 * 1_073_741_824) ? 2 : 1
            let compositeConcurrency = min(
                2, max(1, configuredConcurrency ?? automaticConcurrency)
            )
            report("Frame compositing concurrency: \(compositeConcurrency)")
            report(
                "Fisheye compositor: "
                    + (projection == .fisheye && metalCompositor != nil ? "Metal" : "CPU")
            )
            guard let pool = adaptor.pixelBufferPool else {
                throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
            }
            var nextFrame = 0
            while nextFrame < cacheURLs.count {
                let batchEnd = min(cacheURLs.count, nextFrame + compositeConcurrency)
                let localFrames = Array(nextFrame..<batchEnd)
                var outputBuffers = [CVPixelBuffer]()
                outputBuffers.reserveCapacity(localFrames.count)
                for localFrame in localFrames {
                    var optionalBuffer: CVPixelBuffer?
                    let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer)
                    guard status == kCVReturnSuccess, let pixelBuffer = optionalBuffer else {
                        throw DeformConvError.commandFailed(
                            "failed allocating restored output frame"
                        )
                    }
                    if let frameAttachments = attachments[localFrame] {
                        CVBufferSetAttachments(pixelBuffer, frameAttachments, .shouldPropagate)
                    }
                    outputBuffers.append(pixelBuffer)
                }
                let batch = RegionFrameCompositeBatch(
                    localFrames: localFrames,
                    cacheURLs: localFrames.map { cacheURLs[$0] },
                    baseFrames: localFrames.map { baseFrames[$0] },
                    outputBuffers: outputBuffers,
                    dimensions: plan.dimensions,
                    regions: regions,
                    projection: projection,
                    samplingMaps: samplingMaps,
                    progressStartFrame: progressStartFrame,
                    metalCompositor: metalCompositor
                )
                if localFrames.count == 1 {
                    batch.execute(0)
                } else {
                    DispatchQueue.concurrentPerform(iterations: localFrames.count) {
                        batch.execute($0)
                    }
                }
                for composited in try batch.results() {
                    while !input.isReadyForMoreMediaData {
                        if writer.status == .failed {
                            throw writer.error
                                ?? DeformConvError.commandFailed("video writer failed")
                        }
                        try await Task.sleep(for: .milliseconds(1))
                    }
                    let outputFrame = startFrame + composited.localFrame
                    let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                    guard adaptor.append(composited.pixelBuffer, withPresentationTime: time) else {
                        throw writer.error
                            ?? DeformConvError.commandFailed(
                                "failed encoding frame \(outputFrame)"
                            )
                    }
                    report(
                        "Queued crop-restored output frame "
                            + "\(progressStartFrame + composited.localFrame + 1)/"
                            + "\(progressFrameCount ?? plan.frameRate.outputFrameCount); "
                            + "composite \(String(format: "%.3f", composited.wallSeconds)) s"
                    )
                }
                nextFrame = batchEnd
            }
        }

        func finish() async throws {
            input.markAsFinished()
            await withCheckedContinuation { continuation in
                writer.finishWriting { continuation.resume() }
            }
            guard writer.status == .completed else {
                throw writer.error ?? DeformConvError.commandFailed("video writer did not complete")
            }
        }
    }
}
