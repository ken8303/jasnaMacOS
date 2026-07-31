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
        let writer = try RestoredFrameWriter(outputURL: outputURL, plan: plan)
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
                decodedFrames: decoded,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: plan.temporalWindowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL
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
        decodedFrames: [CVPixelBuffer],
        outputCount: Int,
        windowIndex: Int,
        windowCount: Int,
        modelsURL: URL,
        weightsURL: URL
    ) throws -> WindowResult {
        let cacheBytes = plan.tiles.count * outputCount * tileBytes
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
            tileCount: plan.tiles.count
        )
        let directory = resumed?.directory ?? temporaryURL.appendingPathComponent(
            "window-\(windowIndex + 1)-\(UUID().uuidString)", isDirectory: true
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
            let progressInterval = max(1, plan.tiles.count / 20)
            report(
                "Window \(windowIndex + 1)/\(windowCount): restoring \(plan.tiles.count) tiles; "
                    + "cache \(String(format: "%.2f", Double(cacheBytes) / 1_073_741_824)) GiB"
            )
            let completedTiles = resumed?.completedTiles ?? 0
            if completedTiles == plan.tiles.count {
                report(
                    "Window \(windowIndex + 1)/\(windowCount): all tiles recovered from "
                        + directory.path
                )
            } else if completedTiles > 0 {
                report(
                    "Window \(windowIndex + 1)/\(windowCount): resuming at tile "
                        + "\(completedTiles + 1)/\(plan.tiles.count) from \(directory.path)"
                )
            }
            for tileIndex in completedTiles..<plan.tiles.count {
                let tile = plan.tiles[tileIndex]
                let tileGPU = try autoreleasepool { () throws -> Double in
                    let tileFrames = try decodedFrames.map {
                        try TilePixelPipeline.extractPlanarRGB(from: $0, tile: tile)
                    }
                    let context = "Window \(windowIndex + 1)/\(windowCount): tile "
                        + "\(tileIndex + 1)/\(plan.tiles.count) at eye \(tile.eyeIndex), "
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
                if completed.isMultiple(of: 8) || completed == plan.tiles.count {
                    for handle in handles { try handle.synchronize() }
                    try Data("\(completed)\n".utf8).write(
                        to: directory.appendingPathComponent("completed-tiles.txt"),
                        options: .atomic
                    )
                }
                if tileIndex == 0
                    || tileIndex + 1 == plan.tiles.count
                    || (tileIndex + 1).isMultiple(of: progressInterval) {
                    report(
                        "Window \(windowIndex + 1)/\(windowCount): tile "
                            + "\(tileIndex + 1)/\(plan.tiles.count), GPU "
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
                    measurementCount: 1
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
                    measurementCount: 1
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
        tileCount: Int
    ) throws -> ResumableWindowCache? {
        let prefix = "window-\(windowIndex + 1)-"
        let candidates = try FileManager.default.contentsOfDirectory(
            at: workDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            url.lastPathComponent.hasPrefix(prefix)
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
        private let writer: AVAssetWriter
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor

        init(outputURL: URL, plan: SideBySideVideoPlan) throws {
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
            plan: SideBySideVideoPlan
        ) async throws {
            for localFrame in cacheURLs.indices {
                let cache = try FileHandle(forReadingFrom: cacheURLs[localFrame])
                var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                for tile in plan.tiles {
                    guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                        try? cache.close()
                        throw DeformConvError.commandFailed("restored tile cache is truncated")
                    }
                    var values = [Float16](repeating: 0, count: tileElements)
                    _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                    try accumulator.accumulate(tile: tile, planarRGB: values)
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
                try accumulator.writeBGRA(to: pixelBuffer)
                let outputFrame = startFrame + localFrame
                let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                    throw writer.error
                        ?? DeformConvError.commandFailed("failed encoding frame \(outputFrame)")
                }
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
