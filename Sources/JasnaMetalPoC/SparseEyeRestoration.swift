import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    static let defaultEncoderWindowsPerSegment = 120

    static func elapsedMilliseconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: .now).components
        return Double(elapsed.seconds) * 1_000
            + Double(elapsed.attoseconds) / 1_000_000_000_000_000
    }

    struct PreparedRegionRestoration: Sendable {
        let regionIndex: Int
        let localStart: Int
        let activeFrameCount: Int
        let inputFrames: [[Float16]]
        let context: String
    }

    struct CompletedRegionRestoration: Sendable {
        let prepared: PreparedRegionRestoration
        let frames: [[Float16]]
        let gpuMilliseconds: Double
        let wallMilliseconds: Double
    }

    final class RegionRestorationBatch: @unchecked Sendable {
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
                let wallMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                    since: started
                )
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

    static func restoreSparseSingleEyeVideoWindows(
        device: MTLDevice,
        inputURL: URL,
        windowsDirectoryURL: URL,
        manifestURL: URL,
        modelsURL: URL,
        weightsURL: URL,
        projection: VRMosaicProjection = .raw,
        workDirectoryURL: URL? = nil
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
        let encoderWindowsPerSegment = min(
            windowFrameCounts.count,
            max(
                1,
                Int(
                    ProcessInfo.processInfo.environment[
                        "JASNA_ENCODER_WINDOWS_PER_SEGMENT"
                    ] ?? ""
                ) ?? defaultEncoderWindowsPerSegment
            )
        )
        report(
            "HEVC encoder segment: up to \(encoderWindowsPerSegment) recurrence window(s)"
        )
        var completedWindows = 0
        var restoredRegionWindows = 0
        var skippedTileWindows = 0
        var windowIndex = 0
        while windowIndex < windowFrameCounts.count {
            let outputURL = windowsDirectoryURL.appendingPathComponent(
                String(format: "window-%05d.mov", windowIndex + 1)
            )
            let segmentEnd = encoderSegmentEnd(
                windowIndex: windowIndex,
                windowCount: windowFrameCounts.count,
                maximumWindows: encoderWindowsPerSegment
            ) { candidate in
                let candidateURL = windowsDirectoryURL.appendingPathComponent(
                    String(format: "window-%05d.mov", candidate + 1)
                )
                return FileManager.default.fileExists(atPath: candidateURL.path)
            }
            let segmentFrameCount = windowFrameCounts[windowIndex..<segmentEnd].reduce(0, +)
            if await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: segmentFrameCount
            ) {
                report(
                    "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                        + "encoded segment already complete"
                )
                completedWindows += segmentEnd - windowIndex
                windowIndex = segmentEnd
                continue
            }
            if segmentEnd > windowIndex + 1,
               await validWindowOutput(
                   outputURL,
                   dimensions: plan.dimensions,
                   frameCount: windowFrameCounts[windowIndex]
               )
            {
                report(
                    "Window \(windowIndex + 1)/\(windowFrameCounts.count): "
                        + "legacy one-window output already complete"
                )
                completedWindows += 1
                windowIndex += 1
                continue
            }
            if FileManager.default.fileExists(atPath: outputURL.path) {
                let archived = try archiveInterruptedOutput(outputURL)
                report(
                    "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                        + "archived incomplete output at \(archived.path)"
                )
            }

            let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
            var segmentCacheDirectories = [URL]()
            var segmentOutputFrame = 0
            for currentWindowIndex in windowIndex..<segmentEnd {
                let outputCount = windowFrameCounts[currentWindowIndex]
                let windowStart = currentWindowIndex * SideBySideVideoPlan.temporalWindowFrames
                let frameRange = windowStart..<(windowStart + outputCount)
                let activeRegions = manifest.regions(intersecting: frameRange)
                report(
                    "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): decoding "
                        + "\(outputCount) frames; mosaic regions \(activeRegions.count), "
                        + "model crops \(activeRegions.count)"
                )
                let baseFrames = try (0..<outputCount).map {
                    try decoder.copyFrame(outputIndex: windowStart + $0)
                }
                let attachments = baseFrames.map {
                    CVBufferCopyAttachments($0, .shouldPropagate)
                }
                var modelFrames = baseFrames
                while modelFrames.count < 3 {
                    guard let last = modelFrames.last else {
                        throw DeformConvError.invalidShape
                    }
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
                    windowIndex: currentWindowIndex,
                    windowCount: windowFrameCounts.count,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    cacheVariant: cacheVariant,
                    projection: projection,
                    samplingMaps: samplingMaps,
                    workDirectoryURL: workDirectoryURL
                )
                segmentCacheDirectories.append(window.cacheDirectory)
                let compositeStarted = ContinuousClock.now
                try await writer.appendRegionCachedFrames(
                    cacheURLs: window.cacheURLs,
                    attachments: attachments,
                    startFrame: segmentOutputFrame,
                    progressStartFrame: windowStart,
                    progressFrameCount: manifest.frameCount,
                    plan: plan,
                    baseFrames: baseFrames,
                    regions: activeRegions,
                    projection: projection,
                    samplingMaps: samplingMaps
                )
                let compositeMilliseconds = elapsedMilliseconds(since: compositeStarted)
                report(
                    "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                        + "compositor/writer "
                        + "\(String(format: "%.3f", compositeMilliseconds)) ms"
                )
                segmentOutputFrame += outputCount
                if activeRegions.isEmpty {
                    skippedTileWindows += 1
                } else {
                    restoredRegionWindows += activeRegions.count
                }
            }
            let finishStarted = ContinuousClock.now
            try await writer.finish()
            let finishMilliseconds = elapsedMilliseconds(since: finishStarted)
            report(
                "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                    + "encoder finish "
                    + "\(String(format: "%.3f", finishMilliseconds)) ms"
            )
            guard await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: segmentFrameCount
            ) else {
                throw DeformConvError.commandFailed(
                    "encoded sparse segment \(windowIndex + 1)-\(segmentEnd) failed validation"
                )
            }
            for cacheDirectory in segmentCacheDirectories {
                try FileManager.default.removeItem(at: cacheDirectory)
            }
            completedWindows += segmentEnd - windowIndex
            report(
                "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                    + "sparse segment encoded, validated, and caches removed"
            )
            windowIndex = segmentEnd
        }
        report(
            "Sparse restoration completed: \(completedWindows) windows, "
                + "\(restoredRegionWindows) mosaic crops restored, "
                + "\(skippedTileWindows) clean windows bypassed"
        )
        return completedWindows
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
        samplingMaps: [MosaicCropSamplingMap],
        workDirectoryURL: URL? = nil
    ) throws -> WindowResult {
        guard samplingMaps.count == regions.count else {
            throw DeformConvError.invalidShape
        }
        let cacheBytes = regions.count * outputCount * tileBytes
        let configuredWorkPath = workDirectoryURL?.path
            ?? ProcessInfo.processInfo.environment["JASNA_WORK_DIR"]
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
            var extractionMilliseconds: Double = 0
            var graphWallMilliseconds: Double = 0
            var cacheWriteMilliseconds: Double = 0
            var restoredModelFrames = 0
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
                let extractionStarted = ContinuousClock.now
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
                extractionMilliseconds += elapsedMilliseconds(since: extractionStarted)
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
                    let cacheWriteStarted = ContinuousClock.now
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
                    graphWallMilliseconds += restored.wallMilliseconds
                    restoredModelFrames += prepared.activeFrameCount
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
                    cacheWriteMilliseconds += elapsedMilliseconds(since: cacheWriteStarted)
                    report(
                        "Window \(windowIndex + 1)/\(windowCount): mosaic crop "
                            + "\(completedCount)/\(regions.count), GPU "
                            + "\(String(format: "%.3f", gpuMilliseconds)) ms cumulative, "
                            + "crop wall \(String(format: "%.3f", restored.wallMilliseconds)) ms"
                    )
                }
                nextRegion = batchEnd
            }
            report(
                "Window \(windowIndex + 1)/\(windowCount): sparse hot-path phases: "
                    + "\(restoredModelFrames) model frames, extraction "
                    + "\(String(format: "%.3f", extractionMilliseconds)) ms, graph wall "
                    + "\(String(format: "%.3f", graphWallMilliseconds)) ms, GPU "
                    + "\(String(format: "%.3f", gpuMilliseconds)) ms, cache writes "
                    + "\(String(format: "%.3f", cacheWriteMilliseconds)) ms"
            )
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

    static func restoreTileWithFallback(
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
}
