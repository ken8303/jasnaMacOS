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
    static let tileElements = 3 * SideBySideVideoPlan.modelTileSize
        * SideBySideVideoPlan.modelTileSize
    static let tileBytes = tileElements * MemoryLayout<Float16>.stride


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


    static func encoderSegmentEnd(
        windowIndex: Int,
        windowCount: Int,
        maximumWindows: Int,
        hasExistingOutput: (Int) -> Bool
    ) -> Int {
        let preferredEnd = min(windowCount, windowIndex + max(1, maximumWindows))
        return ((windowIndex + 1)..<preferredEnd).first(where: hasExistingOutput)
            ?? preferredEnd
    }

    static func validWindowOutput(
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


    static func report(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        FileHandle.standardOutput.write(Data("[\(timestamp)] \(message)\n".utf8))
    }

}
